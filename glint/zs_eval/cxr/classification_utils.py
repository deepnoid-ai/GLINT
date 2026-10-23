import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from tqdm import tqdm

from ..distribution_utils import (
    create_multi_gpu_dataloader,
    gather_results,
    is_distributed,
    is_main_process,
)
from .dataset import InferDataset, collate_fn
from .label_loaders import (
    load_chestxdet10_labels,
    load_chestxray14_labels,
    load_chexpert5_labels,
    load_openi_labels,
    load_padchest_labels,
)
from .metrics import compute_classification_metrics, compute_long_tail_auc
from .utils import get_cxr14_labels, get_openi_labels, write_cf_matrix


def _ensure_period(text):
    text = text.strip()
    if not text.endswith("."):
        text += "."
    return text


def process_class_prompts(text_prompt, tokenizer, model):
    text_prompt_list = []
    neg_text_prompt_list = []
    for i in range(len(text_prompt)):
        prompt = _ensure_period(text_prompt[str(i)][0])
        text_prompt_list.append(prompt)
        neg_text_prompt_list.append(prompt.replace("There is", "There is no").replace("There are", "There are no"))

    text_batch = tokenizer(
        text_prompt_list,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(model.device)
    neg_text_batch = tokenizer(
        neg_text_prompt_list,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(model.device)

    return {
        "encoded_key_phrases": text_batch,
        "encoded_negative_phrases": neg_text_batch,
    }


def calculate_similarities(
    image_paths,
    text_batch,
    model,
    image_processor,
    batch_size,
    num_workers,
    image_root_dir,
):
    with torch.no_grad():
        dataset = InferDataset(image_processor, image_paths, image_root_dir)
        loader = create_multi_gpu_dataloader(dataset, batch_size, num_workers, collate_fn, image_processor)

        logit_list = []

        it = tqdm(loader) if is_main_process() else loader

        for image in it:
            image = image.to(model.device, model.dtype)
            logits = model.compute_logits(
                pixel_values=image,
                enc_sents=[text_batch["encoded_key_phrases"]],
                encoded_negative_phrases=[text_batch["encoded_negative_phrases"]],
            )["logits"]
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)
            logit_list.append(logits)

    local_class_similarities = torch.cat(logit_list, dim=0)
    local_class_similarities = local_class_similarities.float().detach().cpu()

    if is_distributed():
        gathered_similarities = gather_results(local_class_similarities)
        if gathered_similarities is not None:
            global_class_similarities = gathered_similarities
        else:
            return None
    else:
        global_class_similarities = local_class_similarities

    return global_class_similarities.numpy()


def prompt_to_finding(text):
    if text.lower() == "the lungs are clear.":
        return "normal"
    elif text.lower() == "there is lung nodule.":
        return "lung nodule"
    elif text.lower() == "there is pulmonary mass.":
        return "pulmonary mass"
    elif text.lower() == "there is nodule or mass.":
        return "nodule/mass"
    else:
        return text.replace("There is ", "").replace(".", "").strip()


def eval_classification(
    model,
    image_processor,
    tokenizer,
    sel_datasets,
    batch_size,
    num_workers,
    save_root_dir,
    image_paths,
    text_paths,
    dataset_paths,
):
    performances = {}
    for image_path, text_path, sel_dataset in zip(image_paths, text_paths, sel_datasets):
        ds_paths = dataset_paths[sel_dataset]
        image_root_dir = ds_paths["image_root"]
        df = pd.read_csv(image_path)
        image_set = df["Path"].tolist()

        with open(text_path, "r") as f:
            text_prompt = json.load(f)

        text_batch = process_class_prompts(text_prompt, tokenizer, model)

        similarities = calculate_similarities(
            image_set,
            text_batch,
            model,
            image_processor,
            batch_size,
            num_workers,
            image_root_dir,
        )

        if similarities is None:
            continue

        header = [prompt_to_finding(text_prompt[str(index)][0]) for index in range(len(text_prompt))]
        sim_df = pd.DataFrame(similarities, columns=header)

        if is_main_process():
            sim_df.to_csv(os.path.join(save_root_dir, sel_dataset) + ".csv", index=False)

            label_path = dataset_paths[sel_dataset]["label_path"]
            performance = cal_performance(similarities, label_path, sel_dataset)
            performances.update(performance)

            cf_dir = os.path.join(save_root_dir, "cf_matrix")
            os.makedirs(cf_dir, exist_ok=True)
            if sel_dataset == "OpenI":
                openi_label = get_openi_labels(label_path, sim_df.columns)
                write_cf_matrix(
                    sim_df,
                    openi_label,
                    os.path.join(cf_dir, f"{sel_dataset}_cf_matrix.csv"),
                    auroc=performance[sel_dataset].get("Findings AUC"),
                )
            elif sel_dataset == "ChestXray14":
                cxr14_label = get_cxr14_labels(label_path, sim_df.columns)
                write_cf_matrix(
                    sim_df,
                    cxr14_label,
                    os.path.join(cf_dir, f"{sel_dataset}_cf_matrix.csv"),
                    auroc=performance[sel_dataset].get("Findings AUC"),
                )

    return performances


def _eval_standard(similarities, labels, class_names):
    result = compute_classification_metrics(labels, similarities, class_names)
    for name, val in result["Findings AUC"].items():
        print(f"{name}: {val}")
    print(f"Total AUC: {result['Total AUC']}")
    print(f"Micro AUPRC: {result['Micro AUPRC']:.4f}, Macro AUPRC: {result['Macro AUPRC']:.4f}")
    return result


def _eval_openi(similarities, label_path, **kwargs):
    labels, class_names = load_openi_labels(label_path)
    result = _eval_standard(similarities, labels, class_names)
    head, medium, tail = compute_long_tail_auc(labels, similarities)
    result.update({"Head AUC": head, "Medium AUC": medium, "Tail AUC": tail})
    return result


def _eval_padchest(similarities, label_path, **kwargs):
    labels, class_names = load_padchest_labels(label_path)
    result = _eval_standard(similarities, labels, class_names)
    head, medium, tail = compute_long_tail_auc(labels, similarities)
    result.update({"Head AUC": head, "Medium AUC": medium, "Tail AUC": tail})

    class_freq = np.sum(labels, axis=0)
    tail_indices = np.where(class_freq <= 10)[0]
    if len(tail_indices) > 0:
        tail_aucs = [float(roc_auc_score(labels[:, i], similarities[:, i])) for i in tail_indices]
        tail_auprcs = []
        for i in tail_indices:
            p, r, _ = precision_recall_curve(labels[:, i], similarities[:, i])
            tail_auprcs.append(auc(r, p))
        result["PadChest20 AUROC"] = float(np.mean(tail_aucs))
        result["PadChest20 AUPRC"] = float(np.mean(tail_auprcs))

    return result


def _eval_chestxray14(similarities, label_path, **kwargs):
    labels, class_names = load_chestxray14_labels(label_path)
    return _eval_standard(similarities, labels, class_names)


def _eval_chexpert(similarities, label_path, **kwargs):
    labels, class_names = load_chexpert5_labels(label_path)
    return _eval_standard(similarities, labels, class_names)


def _eval_chestxdet10(similarities, label_path, **kwargs):
    labels, class_names = load_chestxdet10_labels(label_path)
    return _eval_standard(similarities, labels, class_names)


def _eval_chestxdet_cls(similarities, label_path, **kwargs):
    from .datasets.chestx_det import CHESTX_DET_EVAL_FINDINGS, load_chestx_det_json

    data_list = load_chestx_det_json(label_path, image_dir="")
    all_labels = []
    for d in data_list:
        syms = d["label"]
        if "Nodule" in syms or "Mass" in syms:
            syms = syms + ["Nodule Mass"]
        all_labels.append(syms)

    from sklearn.preprocessing import MultiLabelBinarizer

    mlb = MultiLabelBinarizer(classes=CHESTX_DET_EVAL_FINDINGS)
    labels = mlb.fit_transform(all_labels).astype(np.float32)
    return _eval_standard(similarities, labels, CHESTX_DET_EVAL_FINDINGS)


CLASSIFICATION_EVALUATORS = {
    "OpenI": _eval_openi,
    "PadChest": _eval_padchest,
    "ChestXray14": _eval_chestxray14,
    "Chexpert": _eval_chexpert,
    "ChestXDet10": _eval_chestxdet10,
    "ChestXDet": _eval_chestxdet_cls,
}


def cal_performance(similarities, label_path, sel_dataset):
    evaluator = CLASSIFICATION_EVALUATORS.get(sel_dataset)
    if evaluator is None:
        print(f"Warning: no evaluator for {sel_dataset}")
        return {}

    print(sel_dataset)
    result = evaluator(similarities, label_path)
    return {sel_dataset: result}
