import csv
import os
import shutil

import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from sklearn.metrics import confusion_matrix, precision_recall_curve
from transformers import Trainer

from common import save_json
from glint.model import load_model

FINDINGS_ALIAS = {
    "nodule": "lung nodule",
    "lung nodule": "nodule",
    "mass": "pulmonary mass",
    "pulmonary mass": "mass",
    "nodule/mass": "nodule or mass",
    "nodule or mass": "nodule/mass",
    "pneumoperitonuem": "pneumoperitoneum",
    "pneumoperitoneum": "pneumoperitonuem",
    "normal": "The lungs are clear",
    "the lungs are clear": "normal",
}


def load_pretrained_model(
    resume_from_checkpoint,
    config_path=os.path.join("glint", "configs", "cxr_model.yaml"),
):
    config = OmegaConf.load(config_path)
    config = OmegaConf.to_container(config, resolve=True)

    models = load_model(config["model"], output_dir=None)
    trainer = Trainer(model=models["model"])
    trainer._load_from_checkpoint(resume_from_checkpoint)

    model = trainer.model
    model.eval()
    model.to(torch.float32)

    return model, models["image_processor"], models["tokenizer"]


@torch.no_grad()
def model_inference(image, text, model, image_processor, tokenizer):
    image = load_image(image)

    (width, height) = image.size

    image_size = (height, width)

    image_processor_outputs = image_processor(image)

    processed_image = torch.FloatTensor(np.array(image_processor_outputs["pixel_values"])).to(model.device)

    tokenized_text = tokenizer(
        text,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(model.device)

    output = model.compute_logits(processed_image, [tokenized_text])
    logits = output["logits"]
    similarity_prob = logits.sigmoid()

    similarity_scores = output["similarity_scores"]

    similarity_scores = similarity_scores.view(-1)

    similarity_scores = interpolate_similarity_scores(similarity_scores, image_size, image_processor)
    similarity_map = similarity_scores[0]

    return similarity_prob, similarity_map


def interpolate_similarity_scores(similarity_scores, origin_size, image_processor):
    (height, width) = origin_size
    patch_size = int(similarity_scores.shape[-1] ** 0.5)
    scores = similarity_scores.view(1, 1, patch_size, patch_size)

    if image_processor.__class__.__name__ == "AspectRatioPadWrapper":
        padded_size = max(height, width)
        interpolated_scores = F.interpolate(
            scores,
            size=(padded_size, padded_size),
            mode="bilinear",
            align_corners=False,
        )

        pad_left = (padded_size - width) // 2
        pad_top = (padded_size - height) // 2

        interpolated_scores = interpolated_scores[
            :, :, pad_top : pad_top + height, pad_left : pad_left + width
        ].contiguous()

        interpolated_scores = interpolated_scores.squeeze(1)

    elif (image_processor.__class__.__name__ == "BlipImageProcessor") or (
        image_processor.__class__.__name__ == "DINOv3ViTImageProcessorFast"
    ):
        interpolated_scores = F.interpolate(
            scores,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        interpolated_scores = interpolated_scores.squeeze(1)

    elif image_processor.__class__.__name__ == "BitImageProcessor":
        shortest = min(height, width)

        interpolated_scores = F.interpolate(
            scores,
            size=(shortest, shortest),
            mode="bilinear",
            align_corners=False,
        )

        cropped_left = (width - shortest) // 2
        cropped_top = (height - shortest) // 2

        original_size_map = torch.ones(height, width) * -999
        original_size_map[cropped_top : cropped_top + shortest, cropped_left : cropped_left + shortest] = (
            interpolated_scores.view(shortest, shortest)
        )

        interpolated_scores = original_size_map
        interpolated_scores = interpolated_scores.unsqueeze(0)
    elif image_processor.__class__.__name__ == "M3AEImageProcessor":
        padded_size = max(height, width)

        padded_size_map = torch.ones(padded_size, padded_size) * -999

        cropped_size = int(padded_size * 224 / 256)

        interpolated_scores = F.interpolate(
            scores,
            size=(cropped_size, cropped_size),
            mode="bilinear",
            align_corners=False,
        )

        left = (padded_size - cropped_size) // 2
        top = (padded_size - cropped_size) // 2

        padded_size_map[top : top + cropped_size, left : left + cropped_size] = interpolated_scores.view(
            cropped_size, cropped_size
        )

        pad_left = (padded_size - width) // 2
        pad_top = (padded_size - height) // 2

        interpolated_scores = padded_size_map[pad_top : pad_top + height, pad_left : pad_left + width]
        interpolated_scores = interpolated_scores.unsqueeze(0)

    elif image_processor.__class__.__name__ == "SamProcessor":
        pad_h = image_processor.image_processor.pad_size["height"]
        pad_w = image_processor.image_processor.pad_size["width"]
        longest_edge = image_processor.image_processor.size["longest_edge"]

        scale = longest_edge * 1.0 / max(height, width)
        resized_h = int(height * scale + 0.5)
        resized_w = int(width * scale + 0.5)

        interpolated_scores = F.interpolate(
            scores,
            size=(pad_h, pad_w),
            mode="bilinear",
            align_corners=False,
        )

        interpolated_scores = interpolated_scores[:, :, :resized_h, :resized_w]

        interpolated_scores = F.interpolate(
            interpolated_scores,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

        interpolated_scores = interpolated_scores.squeeze(1)

    return interpolated_scores


@torch.no_grad()
def model_inference_multiple_text(image, text_list, model, image_processor, tokenizer):
    image = load_image(image)
    (width, height) = image.size

    image_size = (height, width)

    image_processor_outputs = image_processor(image, return_tensors="pt")

    processed_image = torch.FloatTensor(np.array(image_processor_outputs["pixel_values"])).to(model.device)

    tokenized_text = tokenizer(
        text_list,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(model.device)

    output = model.compute_logits(processed_image, [tokenized_text])

    logits = output["logits"]
    similarity_probs = logits.sigmoid()

    similarity_scores = output["similarity_scores"]

    similarity_scores = interpolate_similarity_scores_multiple_text(similarity_scores, image_size, image_processor)
    similarity_maps = similarity_scores

    return similarity_probs, similarity_maps


def interpolate_similarity_scores_multiple_text(similarity_scores, origin_size, image_processor):
    (height, width) = origin_size
    patch_size = int(similarity_scores.shape[-1] ** 0.5)
    scores = similarity_scores.view(1, -1, patch_size, patch_size)

    if image_processor.__class__.__name__ == "AspectRatioPadWrapper":
        padded_size = max(height, width)

        interpolated_scores = F.interpolate(
            scores,
            size=(padded_size, padded_size),
            mode="bilinear",
            align_corners=False,
        )

        pad_left = (padded_size - width) // 2
        pad_top = (padded_size - height) // 2

        interpolated_scores = interpolated_scores[:, :, pad_top : pad_top + height, pad_left : pad_left + width]

        interpolated_scores = interpolated_scores.squeeze(0).contiguous()

    elif (image_processor.__class__.__name__ == "BlipImageProcessor") or (
        image_processor.__class__.__name__ == "DINOv3ViTImageProcessorFast"
    ):
        interpolated_scores = F.interpolate(
            scores,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        interpolated_scores = interpolated_scores.squeeze(0)

    elif image_processor.__class__.__name__ == "BitImageProcessor":
        shortest = min(height, width)

        interpolated_scores = F.interpolate(
            scores,
            size=(shortest, shortest),
            mode="bilinear",
            align_corners=False,
        )

        cropped_left = (width - shortest) // 2
        cropped_top = (height - shortest) // 2

        original_size_map = torch.ones(height, width) * -999
        original_size_map[cropped_top : cropped_top + shortest, cropped_left : cropped_left + shortest] = (
            interpolated_scores.view(shortest, shortest)
        )

        interpolated_scores = original_size_map
        interpolated_scores = interpolated_scores.unsqueeze(0)

    return interpolated_scores


def dicom_to_pil_image(input_file_path, save_dir=None):
    dcm_file = pydicom.dcmread(input_file_path)
    raw_image = dcm_file.pixel_array

    raw_image = raw_image - raw_image.min()
    normalized_image = raw_image / raw_image.max()
    rescaled_image = (normalized_image * 255).astype(np.uint8)

    if dcm_file.PhotometricInterpretation == "MONOCHROME1":
        rescaled_image = cv2.bitwise_not(rescaled_image)

    if len(rescaled_image.shape) != 2:
        rescaled_image = Image.fromarray(rescaled_image).convert("L")
        rescaled_image = np.array(rescaled_image)

    final_image = cv2.equalizeHist(rescaled_image)

    image = Image.fromarray(final_image)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        shutil.copy2(input_file_path, save_dir)

    return image


def load_image(image, save_dir=None):
    if isinstance(image, str):
        if image.lower().endswith(".dcm"):
            image = dicom_to_pil_image(image, save_dir)
        elif image.lower().endswith(".png") or image.lower().endswith(".jpg") or image.lower().endswith(".jpeg"):
            image = Image.open(image)
        else:
            raise ValueError(f"Invalid image type: {image}")
    elif not isinstance(image, Image.Image):
        raise ValueError(f"Invalid image type: {type(image)}")

    return image


def write_cf_matrix(df, gt_df, result_file, thrsh_dict=None, drop_first_col=False, auroc=None, f_index=1):
    if thrsh_dict is None:
        thrsh_dict = dict()
        return_dict = True
    else:
        return_dict = False
    findings = gt_df.columns[1:] if drop_first_col else gt_df.columns
    with open(result_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        for finding in findings:
            if finding in df.columns:
                df_key = finding
            elif finding.lower() in FINDINGS_ALIAS.keys():
                df_key = FINDINGS_ALIAS[finding.lower()]
                assert df_key in df.columns
            else:
                raise KeyError("Key Error: ", finding)
            gt = gt_df[finding].values
            logits = torch.tensor(df[df_key].values, dtype=torch.float)
            pred = logits.sigmoid().numpy()

            precision, recall, thresholds = precision_recall_curve(gt, pred)

            f_scores = (1 + f_index**2) * (precision * recall) / ((f_index**2) * precision + recall)
            optimal_idx = np.nanargmax(f_scores)
            optimal_threshold = thresholds[optimal_idx]
            if return_dict:
                thrsh_dict[finding] = float(optimal_threshold)

            write_csv_rows(writer, finding, gt, pred, thrsh_dict[finding], auroc, f_index)

    if return_dict:
        save_json(
            thrsh_dict,
            result_file.replace("_cf_matrix", "_thrsh").replace(".csv", ".json"),
        )
        return thrsh_dict


def write_csv_rows(writer, finding, gt, pred, thrsh, auroc=None, f_index=1):
    matrix = confusion_matrix(gt, (pred >= thrsh).astype(int))
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


def get_openi_labels(label_file_path, columns):
    pathologies = [
        "Atelectasis",
        "Cardiomegaly",
        "Effusion",
        "Infiltration",
        "Mass",
        "Nodule",
        "Pneumonia",
        "Pneumothorax",
        "Edema",
        "Emphysema",
        "Fibrosis",
        "Pleural_Thickening",
        "Hernia",
        "Fracture",
        "Opacity",
        "Lesion",
        "Calcified Granuloma",
        "Granuloma",
        "No_Finding",
    ]

    mapping = dict()
    mapping["Pleural_Thickening"] = ["pleural thickening"]
    mapping["Infiltration"] = ["Infiltrate"]
    mapping["Atelectasis"] = ["Atelectases"]

    csv = pd.read_csv(label_file_path)
    csv = csv.replace(np.nan, "-1")

    gt = []
    for pathology in pathologies:
        mask = csv["labels_automatic"].str.contains(pathology.lower())
        if pathology in mapping:
            for syn in mapping[pathology]:
                mask |= csv["labels_automatic"].str.contains(syn.lower())
        gt.append(mask.values)

    gt = np.asarray(gt).T
    gt = gt.astype(np.float32)

    return pd.DataFrame(gt[:, :-1], columns=columns, dtype=np.int64)


def get_cxr14_labels(label_file_path, columns):
    csv_head = [
        "path",
        "Atelectasis",
        "Cardiomegaly",
        "Effusion",
        "Infiltration",
        "Lung Mass",
        "Lung Nodule",
        "Pneumonia",
        "Pneumothorax",
        "Consolidation",
        "Edema",
        "Emphysema",
        "Fibrosis",
        "Pleural Thickening",
        "Hernia",
    ]
    df_test = pd.read_csv(label_file_path, sep=" ", names=csv_head)

    key = csv_head[1:]

    label = df_test[key].values

    return pd.DataFrame(label, columns=columns, dtype=np.int64)


def finding_to_prompt(finding_name):
    finding_name = finding_name.lower().strip().replace("_", " ")

    if finding_name == "no finding":
        sentence = "The lungs are clear."
        return sentence

    if finding_name == "nodule mass":
        sentence = "There is nodule and mass."
        return sentence

    sentence = f"There is {finding_name}."

    if finding_name in [
        "multiple masses and nodules",
        "diffuse nodular lesions",
    ]:
        sentence = sentence.replace("There is", "There are")

    return sentence
