import csv
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from common import load_json, logger, save_json
from glint.visualization.chest_ct.load_dataset import build_ct_rate_raw_path
from glint.zs_eval.chest_ct.utils import (
    calc_crg_score,
    logit_df_to_prob_df,
    process_pos_neg_prompts,
    write_csv_rows,
)
from glint.zs_eval.distribution_utils import (
    compute_shard_indices,
    gather_results,
    is_distributed,
    is_main_process,
)


def _strip_image_ext(image_id: str) -> str:
    name = os.path.basename(image_id)
    for ext in (".nii.gz", ".npz", ".npy"):
        if name.endswith(ext):
            return name[: -len(ext)]
    return os.path.splitext(name)[0]


def normalize_pixel_values_videos(pixel_values_videos):
    s = pixel_values_videos.shape
    num_channels = 3
    pixel_values_videos = pixel_values_videos.reshape(-1, num_channels, s[-2], s[-1])

    pixel_values_videos = (pixel_values_videos + 1) / 2

    image_mean = torch.tensor([0.485, 0.456, 0.406], device=pixel_values_videos.device).view(1, 3, 1, 1)
    image_std = torch.tensor([0.229, 0.224, 0.225], device=pixel_values_videos.device).view(1, 3, 1, 1)
    pixel_values_videos = (pixel_values_videos - image_mean) / image_std

    return pixel_values_videos


class InferCT3DDataset(torch.utils.data.Dataset):
    def __init__(self, data, monai_cfg, dataset_name, orientd=True, raw_spacing=None):
        from glint.preprocess.chest_ct.monai.transform import build_dataset

        self.data = data
        self._monai_ds = build_dataset(
            [{"image": d["3d_raw_image_path"]} for d in data],
            monai_cfg,
            f"{dataset_name}/test",
            orientd=orientd,
            raw_spacing=raw_spacing,
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        instance = self.data[idx]
        pixel_values_videos = self._monai_ds[idx]["image"].squeeze(0)  # (1, D, H, W) -> (D, H, W)
        pixel_values_videos = normalize_pixel_values_videos(pixel_values_videos)
        return {
            "pixel_values_videos": pixel_values_videos,
            "image_id": instance["image_id"],
            "labels": instance.get("labels", []),
        }


@torch.no_grad()
def eval_classification(
    model,
    tokenizer,
    dataset: InferCT3DDataset,
    dataset_name: str,
    batch_size: int,
    num_workers: int,
    save_root_dir: str,
    split: str = "test",
    binary: bool = True,
    optimize_thrsh: str = None,  # ["eval", "test", None]
):
    image_ids = []
    logit_list = []
    gt_list = []

    ct_rate_label_name = [
        "Medical material",
        "Arterial wall calcification",
        "Cardiomegaly",
        "Pericardial effusion",
        "Coronary artery wall calcification",
        "Hiatal hernia",
        "Lymphadenopathy",
        "Emphysema",
        "Atelectasis",
        "Lung nodule",
        "Lung opacity",
        "Pulmonary fibrotic sequela",
        "Pleural effusion",
        "Mosaic attenuation pattern",
        "Peribronchial thickening",
        "Consolidation",
        "Bronchiectasis",
        "Interlobular septal thickening",
    ]

    radchest_label_name = [
        "atelectasis",
        "peribronchial thickening",
        "bronchiectasis",
        "emphysema",
        "pleural effusion",
        "interlobular septal thickening",
        "cardiomegaly",
        "pericardial effusion",
        "arterial wall calcification",
        "consolidation",
        "pulmonary fibrotic sequela",
        "hiatal hernia",
        "lymphadenopathy",
        "lung nodule",
        "lung opacity",
        "medical material",
        "Coronary artery wall calcification",
    ]

    if dataset_name == "CT_RATE":
        label_name = ct_rate_label_name
    elif dataset_name == "RAD_ChestCT":
        label_name = radchest_label_name
    else:
        raise ValueError(f"Unsupported zs_eval dataset: {dataset_name}")

    pred_csv = os.path.join(save_root_dir, f"{dataset_name}_{split}.csv")
    gt_csv = os.path.join(save_root_dir, f"{dataset_name}_{split}_label.csv")
    if os.path.exists(pred_csv) and os.path.exists(gt_csv):
        if not is_main_process():
            return None
        gt_df = pd.read_csv(gt_csv)
        logit_df = pd.read_csv(pred_csv)
    else:
        os.makedirs(save_root_dir, exist_ok=True)

        if is_distributed():
            start_idx, end_idx = compute_shard_indices(dataset)
            subset = Subset(dataset, range(start_idx, end_idx))
        else:
            subset = dataset

        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
        )

        if is_main_process():
            print(f"Processing dataset type: {dataset_name}")

        it = tqdm(loader) if is_main_process() else loader
        for batch in it:
            image_ids.extend(batch["image_id"])

            imgs = batch["pixel_values_videos"].to(model.device).to(model.dtype)
            labels = batch["labels"]
            text_batch = process_pos_neg_prompts(labels, tokenizer, model)

            out = model.compute_logits(
                pixel_values=imgs,
                enc_sents=[text_batch["encoded_sentences"]],
            )

            logits = out["logits"]
            logit_list.append(logits)

            batch_labels = torch.stack(
                [torch.tensor([int(x) for x in labels[k]], device=model.device) for k in label_name],
                dim=1,
            )
            gt_list.append(batch_labels)

        local_logits = torch.cat(logit_list, dim=0).float().detach().cpu()
        local_gts = torch.cat(gt_list, dim=0).cpu()

        if is_distributed():
            all_logits = gather_results(local_logits)
            all_gts = gather_results(local_gts)
            image_ids = gather_results(image_ids)

            if not is_main_process():
                return None

            logits = all_logits.numpy()
            gts = all_gts.numpy()
        else:
            logits = local_logits.numpy()
            gts = local_gts.numpy()

        logit_cols = label_name + [f"no {finding}" for finding in label_name]
        logit_df = pd.DataFrame(logits, columns=logit_cols)
        logit_df = pd.concat([pd.Series(image_ids, name="VolumeName"), logit_df], axis=1)
        gt_df = pd.DataFrame(gts, columns=label_name)
        gt_df = pd.concat([pd.Series(image_ids, name="VolumeName"), gt_df], axis=1)

        logit_df.to_csv(pred_csv, index=False)
        gt_df.to_csv(gt_csv, index=False)

    prob_df = logit_df_to_prob_df(logit_df, label_name, binary)

    if dataset_name == "RAD_ChestCT":
        if "Coronary artery wall calcification" in gt_df.columns:
            assert (gt_df["arterial wall calcification"] == gt_df["Coronary artery wall calcification"]).all()
            gt_df.drop(["Coronary artery wall calcification"], axis=1, inplace=True)
        gt_df["calcification"] = gt_df["arterial wall calcification"]
        gt_df.drop(["arterial wall calcification"], axis=1, inplace=True)
        prob_df["calcification"] = prob_df[["arterial wall calcification", "Coronary artery wall calcification"]].max(
            axis=1
        )

    dir_name = "metrics"
    if optimize_thrsh:
        dir_name = dir_name + f"_{optimize_thrsh}thrsh"
    else:
        dir_name = dir_name + "_thrsh0.5"
    if binary:
        dir_name = dir_name + "_binary"
    metric_dir = os.path.join(save_root_dir, dir_name)

    if optimize_thrsh == "eval" and split == "test":
        eval_thrsh = os.path.join(metric_dir, f"{dataset_name}_eval_thrsh.json").replace(dataset_name, "CT_RATE")
        if not os.path.exists(eval_thrsh):
            print(f"{eval_thrsh} not saved")
            return
        thrsh_dict = load_json(eval_thrsh)
    else:
        thrsh_dict = dict()
        if split == "eval" and optimize_thrsh == "test":
            print("No need to compute eval when optimizing tresholds for test set")
            return
    os.makedirs(metric_dir, exist_ok=True)
    result = dict()
    findings = gt_df.columns[1:].tolist()
    csv_file = os.path.join(metric_dir, f"{dataset_name}_{split}_cf_matrix.csv")
    fw = open(csv_file, mode="w", newline="")
    writer = csv.writer(fw)
    all_gt = list()
    all_pred = list()
    for finding in findings:
        gt = gt_df[finding].values
        prob = prob_df[finding].values
        pred_label = None
        if optimize_thrsh is None:
            thrsh = 0.5
        elif split == optimize_thrsh:
            p, r, thresholds = precision_recall_curve(gt, prob)
            f_scores = 2 * p * r / (p + r)
            optimal_idx = np.nanargmax(f_scores)
            thrsh = thresholds[optimal_idx]
            thrsh_key = finding[0].upper() + finding[1:]
            thrsh_dict[thrsh_key] = float(thrsh)
        else:
            assert optimize_thrsh == "eval" and split == "test"
            thrsh_key = finding[0].upper() + finding[1:]
            if thrsh_key not in thrsh_dict.keys():
                assert finding.lower() == "calcification"
                pred1 = (
                    prob_df["Coronary artery wall calcification"] >= thrsh_dict["Coronary artery wall calcification"]
                )
                pred2 = prob_df["arterial wall calcification"] >= thrsh_dict["Arterial wall calcification"]
                pred_label = (pred1 | pred2).astype(int)
                thrsh = "-"
            else:
                thrsh = thrsh_dict[thrsh_key]

        if pred_label is None:
            pred_label = (prob >= thrsh).astype(int)

        prec = precision_score(gt, pred_label, zero_division=0)
        rec = recall_score(gt, pred_label, zero_division=0)
        f1 = f1_score(gt, pred_label, zero_division=0)
        wf1 = f1_score(gt, pred_label, zero_division=0, average="weighted")
        acc = accuracy_score(gt, pred_label)
        bacc = balanced_accuracy_score(gt, pred_label)
        auc = roc_auc_score(gt, prob)

        result[finding] = {
            "AUC": auc,
            "precision": prec,
            "recall": rec,
            "F1": f1,
            "weighted F1": wf1,
            "accuracy": acc,
            "balanced accuracy": bacc,
        }

        write_csv_rows(
            writer=writer,
            finding=finding,
            gt=gt,
            pred_label=pred_label,
            thrsh=thrsh,
            auroc={"finding": auc},
        )

        all_gt.append(gt)
        all_pred.append(pred_label)

    metrics = [
        "AUC",
        "precision",
        "recall",
        "F1",
        "accuracy",
        "weighted F1",
        "balanced accuracy",
    ]
    result["average (Macro)"] = {
        metric: float(np.mean([result[finding][metric] for finding in findings])) for metric in metrics
    }

    writer.writerow([])
    writer.writerow(["Finding"] + metrics)
    for finding, metric_dict in result.items():
        writer.writerow([finding] + [metric_dict[metric] for metric in metrics])

    crg_result = calc_crg_score(
        np.stack(all_gt),
        np.stack(all_pred),
    )
    writer.writerow([])
    writer.writerow(["CRG Score", crg_result["CRG"]])
    fw.close()

    result["CRG"] = crg_result
    save_json(result, os.path.join(metric_dir, f"metrics_{split}.json"))

    if split == optimize_thrsh:
        save_json(thrsh_dict, os.path.join(metric_dir, f"{dataset_name}_{split}_thrsh.json"))

    result["main_metric"] = {"AUC": result["average (Macro)"]["AUC"]}

    return result


def load_zs_eval_instances(dataset_name: str, split: str, data_root_dir: str, cfg: dict) -> list[dict]:
    dataset_cfg = cfg["dataset"][dataset_name]
    json_path = os.path.join(data_root_dir, dataset_cfg[split])
    input_json = load_json(json_path)

    if dataset_name == "CT_RATE":
        raw_root = dataset_cfg.get("raw_root", "CT-RATE/dataset")
        return [
            {
                "3d_raw_image_path": build_ct_rate_raw_path(
                    data_root_dir, entry["image_id"].replace(".nii.gz", ""), raw_root=raw_root
                ),
                "image_id": _strip_image_ext(entry["image_id"]),
                "labels": entry.get("labels", []),
            }
            for entry in input_json
        ]
    if dataset_name == "RAD_ChestCT":
        return [
            {
                "3d_raw_image_path": os.path.join(data_root_dir, entry["image_id"]),
                "image_id": _strip_image_ext(entry["image_id"]),
                "labels": entry.get("labels", []),
            }
            for entry in input_json
        ]
    raise ValueError(f"Unsupported zs_eval dataset: {dataset_name}")


class Inference:
    def __init__(
        self,
        cls_dataset,
        det_dataset,
        seg_dataset,
        data_root_dir,
        cfg,
    ):
        self.cls_dataset = cls_dataset
        self.det_dataset = det_dataset
        self.seg_dataset = seg_dataset
        self.data_root_dir = data_root_dir
        if "inference" in cfg:
            self.batch_size = cfg["inference"]["batch_size"]
            self.num_workers = cfg["inference"]["num_workers"]
        else:
            self.batch_size = cfg["train"]["per_device_eval_batch_size"]
            self.num_workers = cfg["train"]["dataloader_num_workers"]
        self.cfg = cfg

        self.monai_cfg = None
        use_monai = cfg.get("dataset", {}).get("use_monai", False)
        if use_monai:
            from glint.preprocess.chest_ct.monai.transform import load_config

            self.monai_cfg = load_config(
                cfg["dataset"]["monai_config"],
                data_root=cfg.get("dataset", {}).get("data_root", ""),
            )

    @torch.no_grad()
    def classification(
        self,
        model,
        tokenizer,
        save_root_dir,
        split="test",
        binary=True,
        optimize_thrsh=None,  # ["eval", "test", None]
    ):
        performances = {}
        for ds_name in self.cls_dataset:
            if ds_name not in ("CT_RATE", "RAD_ChestCT"):
                raise ValueError(f"Unsupported dataset: {ds_name}")
            if ds_name == "RAD_ChestCT" and split == "eval":
                continue

            instances = load_zs_eval_instances(ds_name, split, self.data_root_dir, self.cfg)
            if not instances:
                continue

            ds_cfg = self.cfg["dataset"][ds_name]
            dataset = InferCT3DDataset(
                instances,
                monai_cfg=self.monai_cfg,
                dataset_name=ds_name,
                orientd=ds_cfg.get("orientd", True),
                raw_spacing=ds_cfg.get("raw_spacing"),
            )
            performances[ds_name] = eval_classification(
                model,
                tokenizer,
                dataset,
                ds_name,
                self.batch_size,
                self.num_workers,
                os.path.join(save_root_dir, ds_name),
                split,
                binary,
                optimize_thrsh,
            )

        if is_main_process() and performances:
            logger.info(f"[zs_eval] classification metrics: {performances}")

    @torch.no_grad()
    def segmentation(
        self,
        model,
        tokenizer,
        save_root_dir,
        threshold_step=0.01,
    ):
        from .segmentation_utils import eval_segmentation_rexgrounding

        performances = {}

        for ds_name in self.seg_dataset or []:
            if ds_name == "ReXGroundingCT":
                result = eval_segmentation_rexgrounding(
                    model=model,
                    tokenizer=tokenizer,
                    data_root_dir=self.data_root_dir,
                    cfg=self.cfg,
                    save_root_dir=os.path.join(save_root_dir, ds_name),
                    monai_cfg=self.monai_cfg,
                    threshold_step=threshold_step,
                )
                if result is not None:
                    performances[ds_name] = result
            else:
                raise ValueError(f"Unsupported seg dataset: {ds_name}")

        if is_main_process() and performances:
            logger.info(f"[zs_eval] segmentation metrics: {performances}")
            save_json(performances, os.path.join(save_root_dir, "result.json"))
