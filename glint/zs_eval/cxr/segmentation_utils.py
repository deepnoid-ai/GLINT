import os

import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from ..distribution_utils import (
    compute_shard_indices,
    gather_results,
    is_distributed,
    is_main_process,
)
from .datasets.chestx_det import (
    CHESTX_DET_EVAL_FINDINGS,
    load_chestx_det_json,
)
from .utils import finding_to_prompt, interpolate_similarity_scores


def polygon2mask(polygons, height, width):
    mask = np.zeros([height, width])

    for polygon in polygons:
        pts = np.asarray(polygon, dtype=np.int32)

        if pts.ndim != 2 or pts.shape[1] != 2:
            pts = pts.reshape(-1, 2).astype(np.int32)
        mask = cv2.fillPoly(mask, [pts], 1)

    return mask


def rle2mask(rle, width, height):
    mask = np.zeros(width * height)
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]
    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position : current_position + lengths[index]] = 1
        current_position += lengths[index]

    return mask.reshape(width, height).T


def read_from_dicom(img_path):
    dcm = pydicom.dcmread(img_path)
    x = dcm.pixel_array
    x = cv2.convertScaleAbs(x, alpha=(255.0 / x.max()))

    if dcm.PhotometricInterpretation == "MONOCHROME1":
        x = cv2.bitwise_not(x)

    return Image.fromarray(x).convert("RGB")


def compute_specificity(negative_probs, threshold):
    if isinstance(negative_probs, list):
        neg = np.stack([np.asarray(np.squeeze(p)) for p in negative_probs])
    else:
        neg = np.squeeze(np.asarray(negative_probs))

    true_negatives = ((neg > threshold).astype(np.int64).sum(-1).sum(-1) == 0).sum()
    spec = float(true_negatives / len(negative_probs))

    return spec


def calculate_dice_per_finding(pos_probs, pos_masks, ts, smooth=1e-5):
    if not pos_probs or not pos_masks:
        return 0.0, 0.0

    pos_probs_np = np.stack(pos_probs).squeeze()
    pos_masks_np = np.stack(pos_masks).squeeze()
    if pos_probs_np.ndim == 2:
        pos_probs_np = pos_probs_np[np.newaxis]
        pos_masks_np = pos_masks_np[np.newaxis]

    n = pos_probs_np.shape[0]
    ts_arr = np.asarray(ts)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Threshold search: {n} samples, device={device}")
    probs_flat = torch.from_numpy(pos_probs_np.reshape(n, -1)).to(device)
    target_flat = torch.from_numpy(pos_masks_np.astype(np.float32).reshape(n, -1)).to(device)
    gt_sum = target_flat.sum(dim=1)

    mean_dice = torch.zeros(len(ts_arr), device=device)
    for i, t in enumerate(tqdm(ts_arr, desc="  threshold search")):
        pred = (probs_flat > t).float()
        intersection = (pred * target_flat).sum(dim=1)
        pred_sum = pred.sum(dim=1)
        dice = (2 * intersection + smooth) / (pred_sum + gt_sum + smooth)
        mean_dice[i] = dice.mean()

    best_idx = int(mean_dice.argmax().cpu())
    return float(mean_dice[best_idx].cpu()), float(ts_arr[best_idx])


def calculate_dice_combined(pos_probs, pos_masks, neg_probs, ts, smooth=1e-5, empty_score=1.0):
    if not pos_probs or not pos_masks:
        return 0.0, 0.0

    _, best_threshold = calculate_dice_per_finding(pos_probs, pos_masks, ts, smooth=smooth)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dice_scores = []

    for prob, mask in zip(pos_probs, pos_masks):
        p = torch.from_numpy(np.asarray(np.squeeze(prob)).ravel()).to(device)
        t = torch.from_numpy(np.asarray(np.squeeze(mask)).astype(np.float32).ravel()).to(device)
        pred = (p > best_threshold).float()
        intersection = (pred * t).sum()
        union = pred.sum() + t.sum()
        dice_scores.append(float((2.0 * intersection / union).cpu()) if union > 0 else empty_score)

    for prob in neg_probs:
        p = torch.from_numpy(np.asarray(np.squeeze(prob)).ravel()).to(device)
        pred_sum = (p > best_threshold).float().sum()
        dice_scores.append(empty_score if pred_sum == 0 else 0.0)

    combined_dice = float(np.mean(dice_scores))

    return combined_dice, best_threshold


def compute_dice_and_threshold_by_finding(finding_pos_probs, finding_pos_masks, ts, labels):
    dice_by_finding = {}
    threshold_by_finding = {}
    for finding in tqdm(labels, total=len(labels), desc=f"Calculating dice and threshold by [{len(labels)}]-findings"):
        probs = finding_pos_probs.get(finding)
        masks = finding_pos_masks.get(finding)
        if not probs or not masks:
            dice_by_finding[finding] = None
            threshold_by_finding[finding] = None
            continue
        if isinstance(probs, list):
            probs = [p.squeeze() if hasattr(p, "squeeze") else p for p in probs]
            masks = [m.squeeze() if hasattr(m, "squeeze") else m for m in masks]
        best_dice, best_t = calculate_dice_per_finding(probs, masks, ts)
        dice_by_finding[finding] = best_dice
        threshold_by_finding[finding] = best_t
    valid_dice = [v for v in dice_by_finding.values() if v is not None]
    mean_dice = float(np.mean(valid_dice)) if valid_dice else 0.0
    return dice_by_finding, threshold_by_finding, mean_dice


def compute_combined_dice_by_finding(finding_pos_probs, finding_pos_masks, finding_neg_probs, ts, labels):
    combined_dice_by_finding = {}
    specificity_by_finding = {}
    for finding in tqdm(labels, total=len(labels), desc=f"Calculating combined dice by [{len(labels)}]-findings"):
        probs = finding_pos_probs.get(finding)
        masks = finding_pos_masks.get(finding)
        neg = finding_neg_probs.get(finding, [])
        if not probs or not masks:
            combined_dice_by_finding[finding] = None
            specificity_by_finding[finding] = None
            continue
        if isinstance(probs, list):
            probs = [p.squeeze() if hasattr(p, "squeeze") else p for p in probs]
            masks = [m.squeeze() if hasattr(m, "squeeze") else m for m in masks]
        if isinstance(neg, list):
            neg = [n.squeeze() if hasattr(n, "squeeze") else n for n in neg]

        combined_dice, best_t = calculate_dice_combined(probs, masks, neg, ts)
        combined_dice_by_finding[finding] = combined_dice

        if neg:
            specificity_by_finding[finding] = compute_specificity(neg, best_t)
        else:
            specificity_by_finding[finding] = None

    valid_dice = [v for v in combined_dice_by_finding.values() if v is not None]
    mean_combined_dice = float(np.mean(valid_dice)) if valid_dice else 0.0
    valid_spec = [v for v in specificity_by_finding.values() if v is not None]
    mean_specificity = float(np.mean(valid_spec)) if valid_spec else 0.0
    return combined_dice_by_finding, specificity_by_finding, mean_combined_dice, mean_specificity


def compute_auroc_by_finding(finding_pos_probs, finding_pos_masks, labels):
    auroc_by_finding = {}
    for finding in labels:
        probs = finding_pos_probs.get(finding)
        masks = finding_pos_masks.get(finding)
        if not probs or not masks:
            auroc_by_finding[finding] = None
            continue
        if isinstance(probs, list):
            probs = [p.squeeze() if hasattr(p, "squeeze") else p for p in probs]
            masks = [m.squeeze() if hasattr(m, "squeeze") else m for m in masks]
        all_probs = np.concatenate([p.reshape(-1) for p in probs])
        all_masks = np.concatenate([m.reshape(-1).astype(np.int64) for m in masks])
        if np.unique(all_masks).size < 2:
            auroc_by_finding[finding] = None
            continue
        auroc_by_finding[finding] = float(roc_auc_score(all_masks, all_probs))
    valid_auroc = [v for v in auroc_by_finding.values() if v is not None]
    mean_auroc = float(np.mean(valid_auroc)) if valid_auroc else 0.0
    return auroc_by_finding, mean_auroc


def process_image_and_text(image, text, image_processor, tokenizer, model):
    processed_image = image_processor(image)
    processed_image = torch.FloatTensor(np.array(processed_image["pixel_values"])).to(model.device, model.dtype)
    tokenized_text = tokenizer(
        text,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(model.device)
    output = model.compute_logits(processed_image, [tokenized_text])
    del processed_image, tokenized_text
    return output


@torch.no_grad()
def eval_segmentation_siim(
    model,
    image_processor,
    tokenizer,
    data,
    text,
    image_root_dir,
    compute_pixel_level_auroc,
    threshold_step=0.01,
):
    data["class"] = data[" EncodedPixels"].apply(lambda x: x != " -1")
    imgids = data.ImageId.unique().tolist()

    start_idx, end_idx = compute_shard_indices(imgids)
    shard = imgids[start_idx:end_idx]

    logits = []
    labels = []

    pos_probs, pos_masks = [], []
    neg_probs = []
    prob_list, mask_list = [], []

    ts = np.arange(0, 1 + threshold_step, threshold_step)

    it = tqdm(shard) if is_main_process() else shard

    for imgid in it:
        imgid_df = data.groupby("ImageId").get_group(imgid)

        dicom_col = "dicom_path" if "dicom_path" in imgid_df.columns else "ours_path"
        image = read_from_dicom(os.path.join(image_root_dir, imgid_df[dicom_col].tolist()[0]))
        (width, height) = image.size

        encoded_labels = imgid_df[" EncodedPixels"].tolist()
        mask = np.zeros([height, width])
        if encoded_labels[0] != " -1":
            for encoded_label in encoded_labels:
                mask += rle2mask(encoded_label, height, width)
        mask_t = torch.from_numpy(mask).long().unsqueeze(0)
        labels.append((mask_t.sum() > 0).long().cpu().numpy())

        processed_image = image_processor(image)
        processed_image = torch.FloatTensor(np.array(processed_image["pixel_values"])).to(model.device, model.dtype)

        tokenized_text = tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(model.device)

        output = model.compute_logits(processed_image, [tokenized_text])
        similarity_scores = output["similarity_scores"]

        logits.append(output["logits"].detach().float().cpu().numpy())

        similarity_scores = similarity_scores.detach().float().cpu()
        recon_feature = interpolate_similarity_scores(similarity_scores, (height, width), image_processor)
        prob = recon_feature.detach().cpu().float().numpy()

        if mask_t.sum() > 0:
            pos_probs.append(prob)
            pos_masks.append(mask_t)
        else:
            neg_probs.append(prob)

        if compute_pixel_level_auroc:
            prob_list.append(prob)
            mask_list.append(mask_t.numpy())

        del output, processed_image, tokenized_text

    if is_distributed():
        logits = gather_results(logits)
        labels = gather_results(labels)

        pos_probs = gather_results(pos_probs)
        pos_masks = gather_results(pos_masks)
        neg_probs = gather_results(neg_probs)

        if compute_pixel_level_auroc:
            prob_list = gather_results(prob_list)
            mask_list = gather_results(mask_list)

    result = {}

    if not is_distributed() or is_main_process():
        labels_np = np.stack(labels)
        logits_np = np.stack(logits).reshape(-1)
        classification_auc = roc_auc_score(labels_np, logits_np)
        result["auc"] = classification_auc

        print(f"classification auc: {classification_auc}")

        best_dice, best_threshold = 0.0, 0.0
        if len(pos_probs) > 0:
            best_dice, best_threshold = calculate_dice_per_finding(pos_probs, pos_masks, ts)

        result["dice"] = float(best_dice)
        result["best_threshold"] = best_threshold
        print(f"dice score (positive only): {best_dice}")
        print(f"best threshold: {best_threshold}")

        if len(neg_probs) > 0:
            specificity = compute_specificity(neg_probs, best_threshold)
        else:
            specificity = 1.0
        result["specificity"] = specificity
        print(f"specificity: {specificity}")

        if len(pos_probs) > 0 and len(neg_probs) > 0:
            combined_dice, _ = calculate_dice_combined(pos_probs, pos_masks, neg_probs, ts)
            result["dice_combined"] = float(combined_dice)
            print(f"dice score (pos+neg combined): {combined_dice}")
        else:
            result["dice_combined"] = result["dice"]

        if compute_pixel_level_auroc:
            all_probs = np.stack(prob_list).reshape(-1)
            all_masks = np.stack(mask_list).reshape(-1).astype(np.int64)
            pixel_level_auroc = roc_auc_score(all_masks, all_probs)
            result["pixel_level_auroc"] = pixel_level_auroc
            print(f"pixel-level auroc: {pixel_level_auroc}")

        print("SIIM Result")
        print(result)
        result["main_metric"] = {"auc": result["auc"], "dice": result["dice"]}
        return result
    else:
        return None


@torch.no_grad()
def eval_segmentation_rsna_medklip(
    model,
    image_processor,
    tokenizer,
    data,
    text,
    image_root_dir,
    compute_pixel_level_auroc,
    threshold_step=0.01,
):
    image_paths = data["img_path"].tolist()
    bboxs = data["boxes"].tolist()

    image_paths = np.array(image_paths)
    start_idx, end_idx = compute_shard_indices(image_paths)
    shard = image_paths[start_idx:end_idx]
    bbox_shard = bboxs[start_idx:end_idx]

    logits = []
    labels = []

    ts = np.arange(0, 1 + threshold_step, threshold_step)
    pos_probs, pos_masks = [], []
    neg_probs = []
    prob_list, mask_list = [], []

    it = tqdm(zip(shard, bbox_shard)) if is_main_process() else zip(shard, bbox_shard)
    for img_path, bbox in it:
        image = read_from_dicom(os.path.join(image_root_dir, img_path))
        (width, height) = image.size

        mask = np.zeros([height, width])

        if not pd.isna(bbox) and bbox != "nan":
            boxes = bbox.split("|")
            for box in boxes:
                cc = box.split(";")
                mask[
                    int(float(cc[1])) : (int(float(cc[1])) + int(float(cc[3]))),
                    int(float(cc[0])) : (int(float(cc[0])) + int(float(cc[2]))),
                ] = 1

        mask_t = torch.from_numpy(mask).long().unsqueeze(0)
        labels.append((mask_t.sum() > 0).long().cpu().numpy())

        processed_image = image_processor(image)
        processed_image = torch.FloatTensor(np.array(processed_image["pixel_values"])).to(model.device, model.dtype)

        tokenized_text = tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(model.device)
        output = model.compute_logits(processed_image, [tokenized_text])
        similarity_scores = output["similarity_scores"]

        logits.append(output["logits"].detach().float().cpu().numpy())

        similarity_scores = similarity_scores.float().cpu()
        recon_feature = interpolate_similarity_scores(similarity_scores, (height, width), image_processor)
        prob = recon_feature.detach().cpu().float().numpy()

        if mask_t.sum() > 0:
            pos_probs.append(prob)
            pos_masks.append(mask_t)
        else:
            neg_probs.append(prob)

        if compute_pixel_level_auroc:
            prob_list.append(prob)
            mask_list.append(mask_t)

        del output, processed_image, tokenized_text

    if is_distributed():
        logits = gather_results(logits)
        labels = gather_results(labels)
        pos_probs = gather_results(pos_probs)
        pos_masks = gather_results(pos_masks)
        neg_probs = gather_results(neg_probs)
        if compute_pixel_level_auroc:
            prob_list = gather_results(prob_list)
            mask_list = gather_results(mask_list)

    result = {}

    if not is_distributed() or is_main_process():
        labels_np = np.stack(labels)
        logits_np = np.stack(logits).reshape(-1)
        classification_auc = roc_auc_score(labels_np, logits_np)
        result["auc"] = classification_auc

        print(f"classification auc: {classification_auc}")

        best_dice, best_threshold = 0.0, 0.0
        if len(pos_probs) > 0:
            best_dice, best_threshold = calculate_dice_per_finding(pos_probs, pos_masks, ts)

        result["dice"] = float(best_dice)
        result["best_threshold"] = best_threshold
        print(f"dice score (positive only): {best_dice}")
        print(f"best threshold: {best_threshold}")

        if len(neg_probs) > 0:
            specificity = compute_specificity(neg_probs, best_threshold)
        else:
            specificity = 1.0
        result["specificity"] = specificity
        print(f"specificity: {specificity}")

        if len(pos_probs) > 0 and len(neg_probs) > 0:
            combined_dice, _ = calculate_dice_combined(pos_probs, pos_masks, neg_probs, ts)
            result["dice_combined"] = float(combined_dice)
            print(f"dice score (pos+neg combined): {combined_dice}")
        else:
            result["dice_combined"] = result["dice"]

        if compute_pixel_level_auroc:
            all_probs = np.stack(prob_list).reshape(-1)
            all_masks = np.stack(mask_list).reshape(-1).astype(np.int64)

            pixel_level_auroc = roc_auc_score(all_masks, all_probs)
            print(f"pixel-level auroc: {pixel_level_auroc}")

            result["pixel_level_auroc"] = pixel_level_auroc

        print("RSNA Result")
        print(result)
        result["main_metric"] = {"auc": result["auc"], "dice": result["dice"]}

        return result

    else:
        return None


@torch.no_grad()
def eval_segmentation_chestx_det(
    model,
    image_processor,
    tokenizer,
    data_path,
    image_dir,
    compute_pixel_level_auroc,
    threshold_step=0.01,
):
    def _empty_dict(labels):
        return {f: [] for f in labels}

    data_list = load_chestx_det_json(data_path, image_dir)
    finding_to_pos_samples = _empty_dict(CHESTX_DET_EVAL_FINDINGS)
    finding_to_neg_samples = _empty_dict(CHESTX_DET_EVAL_FINDINGS)
    for item in data_list:
        for finding in CHESTX_DET_EVAL_FINDINGS:
            polygons = item["masks"].get(finding, [])
            if polygons:
                finding_to_pos_samples[finding].append((item["image_path"], polygons))
            else:
                finding_to_neg_samples[finding].append(item["image_path"])

    pos_probs = _empty_dict(CHESTX_DET_EVAL_FINDINGS)
    pos_masks = _empty_dict(CHESTX_DET_EVAL_FINDINGS)
    neg_probs = _empty_dict(CHESTX_DET_EVAL_FINDINGS)
    prob_list = _empty_dict(CHESTX_DET_EVAL_FINDINGS)
    mask_list = _empty_dict(CHESTX_DET_EVAL_FINDINGS)

    ts = np.arange(0, 1 + threshold_step, threshold_step)

    for finding in CHESTX_DET_EVAL_FINDINGS:
        text = finding_to_prompt(finding)

        pos_samples = finding_to_pos_samples[finding]
        pos_paths = [s[0] for s in pos_samples]
        start_idx, end_idx = compute_shard_indices(pos_paths)
        pos_shard = list(
            zip(
                [s[0] for s in pos_samples[start_idx:end_idx]],
                [s[1] for s in pos_samples[start_idx:end_idx]],
            )
        )

        it = tqdm(pos_shard, desc=f"{finding} (pos)") if is_main_process() else pos_shard
        for image_path, masks_by_finding in it:
            image = Image.open(image_path).convert("RGB")
            height, width = image.size[1], image.size[0]

            mask = polygon2mask(masks_by_finding, height, width)
            mask_t = torch.from_numpy(mask).long().unsqueeze(0)

            output = process_image_and_text(image, text, image_processor, tokenizer, model)
            similarity_scores = output["similarity_scores"].float().cpu()
            recon_feature = interpolate_similarity_scores(similarity_scores, (height, width), image_processor)
            prob = recon_feature.detach().cpu().float().numpy()

            pos_probs[finding].append(prob)
            pos_masks[finding].append(mask_t.numpy())

            if compute_pixel_level_auroc:
                prob_list[finding].append(prob)
                mask_list[finding].append(mask_t.numpy())

            del output

        neg_paths = finding_to_neg_samples[finding]
        start_idx, end_idx = compute_shard_indices(neg_paths)
        neg_shard = neg_paths[start_idx:end_idx]

        it = tqdm(neg_shard, desc=f"{finding} (neg)") if is_main_process() else neg_shard
        for image_path in it:
            image = Image.open(image_path).convert("RGB")
            height, width = image.size[1], image.size[0]

            output = process_image_and_text(image, text, image_processor, tokenizer, model)
            similarity_scores = output["similarity_scores"].float().cpu()
            recon_feature = interpolate_similarity_scores(similarity_scores, (height, width), image_processor)
            prob = recon_feature.detach().cpu().float().numpy()

            neg_probs[finding].append(prob)

            if compute_pixel_level_auroc:
                mask_t = torch.zeros(1, height, width).long()
                prob_list[finding].append(prob)
                mask_list[finding].append(mask_t.numpy())

            del output

        if is_distributed():
            pos_probs[finding] = gather_results(pos_probs[finding])
            pos_masks[finding] = gather_results(pos_masks[finding])
            neg_probs[finding] = gather_results(neg_probs[finding])

            if compute_pixel_level_auroc:
                prob_list[finding] = gather_results(prob_list[finding])
                mask_list[finding] = gather_results(mask_list[finding])

    result = {}

    if not is_distributed() or is_main_process():
        dice_by_finding, threshold_by_finding, mean_dice = compute_dice_and_threshold_by_finding(
            pos_probs, pos_masks, ts, CHESTX_DET_EVAL_FINDINGS
        )

        flat_probs, flat_masks = [], []
        for f in CHESTX_DET_EVAL_FINDINGS:
            probs = prob_list.get(f) or []
            masks = mask_list.get(f) or []
            if not probs or not masks:
                continue

            if isinstance(probs, list):
                flat_probs.extend(np.asarray(p).squeeze().ravel() for p in probs)
                flat_masks.extend(np.asarray(m).squeeze().astype(np.int64).ravel() for m in masks)
            else:
                flat_probs.append(np.asarray(probs).squeeze().ravel())
                flat_masks.append(np.asarray(masks).squeeze().astype(np.int64).ravel())

        if flat_probs and flat_masks:
            y_score = np.concatenate(flat_probs)
            y_true = np.concatenate(flat_masks)
            if np.unique(y_true).size >= 2:
                result["pixel_level_auroc"] = float(roc_auc_score(y_true, y_score))

        result["mean_dice"] = mean_dice
        result["dice"] = dice_by_finding
        result["best_threshold"] = threshold_by_finding

        combined_dice_by_finding, specificity_by_finding, mean_combined_dice, mean_specificity = (
            compute_combined_dice_by_finding(pos_probs, pos_masks, neg_probs, ts, CHESTX_DET_EVAL_FINDINGS)
        )
        result["dice_combined"] = combined_dice_by_finding
        result["mean_dice_combined"] = mean_combined_dice
        result["specificity"] = specificity_by_finding
        result["mean_specificity"] = mean_specificity

        print("ChestX_Det Result")
        print(f"mean dice: {mean_dice}")
        print(f"mean dice (combined): {mean_combined_dice}")
        print(f"mean specificity: {mean_specificity}")
        print(f"dice: {dice_by_finding}")
        print(f"dice_combined: {combined_dice_by_finding}")
        print(f"specificity: {specificity_by_finding}")
        print(f"best threshold: {threshold_by_finding}")

        main_metric = {
            "mean_dice": mean_dice,
        }
        if compute_pixel_level_auroc and "pixel_level_auroc" in result:
            print(f"pixel-level auroc: {result['pixel_level_auroc']}")
            main_metric["pixel_level_auroc"] = result["pixel_level_auroc"]
        result["main_metric"] = main_metric
        return result

    return None
