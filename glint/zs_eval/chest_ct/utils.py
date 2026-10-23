import os

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from transformers import Trainer

from glint.model import load_model


def load_pretrained_model(
    resume_from_checkpoint,
    config_path=os.path.join("glint", "configs", "ct_model.yaml"),
    **kwargs,
):
    config = OmegaConf.load(config_path)
    config = OmegaConf.to_container(config, resolve=True)

    models = load_model(config["model"], output_dir=None)
    trainer = Trainer(model=models["model"])
    trainer._load_from_checkpoint(resume_from_checkpoint)

    model = trainer.model
    model.eval()
    model.to(torch.float32)
    return {"model": model, "tokenizer": models["tokenizer"]}


def process_pos_neg_prompts(text_prompt, tokenizer, model):
    pos_prompt_list = []
    neg_prompt_list = []
    for key in text_prompt:
        pos_prompt_list.append(f"There is {key}")
        neg_prompt_list.append(f"There is no {key}")
    prompt_list = pos_prompt_list + neg_prompt_list
    text_batch = tokenizer(
        prompt_list,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(model.device)

    return {"encoded_sentences": text_batch}


def logit_df_to_prob_df(logit_df, label_name, binary=True):
    logits = torch.tensor(logit_df.iloc[:, 1:].values, dtype=torch.float32)

    B = logits.size(0)
    C = logits.size(1) // 2
    if binary:
        logits = logits.reshape(B, 2, C)
        probs = torch.softmax(logits, dim=1)[:, 0]
    else:
        probs = torch.sigmoid(logits[:, :C])

    prob_df = pd.DataFrame(probs.numpy(), columns=label_name)
    prob_df.insert(0, "VolumeName", logit_df["VolumeName"])
    return prob_df


def write_csv_rows(writer, finding, gt, pred_label, thrsh, auroc=None, f_index=1):
    matrix = confusion_matrix(gt, pred_label)
    tn, fp = matrix[0]
    fn, tp = matrix[1]
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    f_score = (1 + f_index**2) * tp / ((1 + f_index**2) * tp + fp + (f_index**2) * fn)

    writer.writerow([finding, "PRED(-)", "PRED(+)", "", "Thrsh", thrsh])
    writer.writerow(["GT(-)", tn, fp, tn + fp, "Precision", precision])
    writer.writerow(["GT(+)", fn, tp, fn + tp, "Recall", recall])
    writer.writerow(["", tn + fn, fp + tp, "", "Specificity", specificity])
    writer.writerow(["", "", "", "", f"F{f_index} score", f_score])
    if auroc is not None:
        if finding in auroc:
            writer.writerow(["", "", "", "", "AUROC", auroc[finding]])
    writer.writerow([])

    wf1 = f1_score(gt, pred_label, average="weighted")
    acc = balanced_accuracy_score(gt, pred_label)

    return {
        "acc": acc,
        "f1": wf1,
        "prec": precision,
        "spec": specificity,
        "sens": recall,
    }


# Adapted from https://github.com/forithmus/VLM3D-Dockers/blob/main/ct_challenges/reportgen_evaluation/src/crg_score.py
# (no upstream license; rights remain with the original authors)
def calc_crg_score(gt, pred_label):
    TP = ((pred_label == 1) & (gt == 1)).sum()
    FN = ((pred_label == 0) & (gt == 1)).sum()
    FP = ((pred_label == 1) & (gt == 0)).sum()

    X = gt.size
    A = int(gt.sum())

    if A == 0:
        raise ValueError("Ground-truth contains zero positive labels; CRG undefined.")

    r = (X - A) / (2 * A)
    U = (X - A) / 2
    s = r * TP - r * FN - FP
    crg = U / (2 * U - s) if (2 * U - s) != 0 else 0

    return {
        "TP": int(TP),
        "FN": int(FN),
        "FP": int(FP),
        "X": int(X),
        "A": int(A),
        "r": float(r),
        "U": float(U),
        "score_s": float(s),
        "CRG": float(crg),
    }


def load_volume_data(volume_path, image_size=None, nii_info=None):
    if isinstance(volume_path, torch.Tensor):
        return volume_path

    elif volume_path.endswith(".npy"):
        volume_data = torch.FloatTensor(np.load(volume_path, mmap_mode="r"))
    elif volume_path.endswith(".nii.gz"):
        raise ValueError("nii.gz loading requires monai_cfg; use the monai-based loader instead")
    else:
        raise ValueError(f"Unsupported file type: {volume_path}")
    return volume_data


def _get_vision_spatial_config(vision_cfg):
    image_size = getattr(vision_cfg, "image_size", None) or getattr(vision_cfg, "img_size", 256)
    patch_size = getattr(vision_cfg, "patch_size", 16)
    input_slices = getattr(vision_cfg, "input_slices", 240)
    tubelet_size = getattr(vision_cfg, "tubelet_size", 2)
    frame_length = vision_cfg.num_channels * tubelet_size
    return image_size, patch_size, input_slices, frame_length


def reshape_patch_similarity_to_volume(patch_similarity, model):
    image_size, patch_size, input_slices, frame_length = _get_vision_spatial_config(model.config.vision_config)

    return patch_similarity.reshape(
        input_slices // frame_length,
        image_size // patch_size,
        image_size // patch_size,
    )


def interpolate_similarity_volume(similarity_volume, target_shape):
    volume = (
        torch.nn.functional.interpolate(
            similarity_volume.unsqueeze(0).unsqueeze(0),
            size=target_shape,
            mode="trilinear",
            align_corners=False,
        )
        .squeeze(0)
        .squeeze(0)
    )
    return volume


@torch.no_grad()
def model_inference_similarity_map(volume_data, model, text, tokenizer, target_shape=None, nii_info=None):
    similarity_logits, patch_similarity_logits = model_inferece_text(
        volume_data, model, text, tokenizer, nii_info=nii_info
    )

    similarity_prob = similarity_logits.sigmoid()[0].item()
    patch_prob = patch_similarity_logits[0]

    similarity_volume = reshape_patch_similarity_to_volume(patch_prob, model)

    if target_shape is not None:
        similarity_map = interpolate_similarity_volume(similarity_volume, target_shape)
    else:
        similarity_map = similarity_volume

    return similarity_prob, similarity_map


@torch.no_grad()
def model_inferece_text(volume_path, model, text, tokenizer, nii_info=None, rgb_z_normalize=True):
    image_size = _get_vision_spatial_config(model.config.vision_config)[0]
    volume_data = load_volume_data(volume_path, image_size, nii_info)

    s = volume_data.shape
    volume_data = volume_data.reshape(
        -1,
        model.config.vision_config.num_channels,
        s[-2],
        s[-1],
    )
    volume_data = volume_data.unsqueeze(0).to(model.device)

    if rgb_z_normalize:
        assert (
            model.config.vision_config.num_channels == 3
        ), "num_channels must be 3 for normalization with V-JEPA2.1 RGB pretrained model"

        volume_data = (volume_data + 1) / 2
        image_mean = torch.tensor([0.485, 0.456, 0.406], device=volume_data.device).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225], device=volume_data.device).view(1, 3, 1, 1)
        volume_data = (volume_data - image_mean) / image_std

    text = tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(model.device)

    model.eval()
    output = model.compute_logits(
        pixel_values=volume_data,
        enc_sents=[text],
    )
    similarity_logits = output["logits"]
    patch_similarity_logits = output["similarity_scores"][0]

    return similarity_logits, patch_similarity_logits
