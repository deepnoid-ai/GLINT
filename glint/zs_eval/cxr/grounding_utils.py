import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from common import load_json

from ..distribution_utils import (
    compute_shard_indices,
    create_multi_gpu_dataloader,
    is_distributed,
    is_main_process,
)
from .classification_utils import process_class_prompts
from .dataset import InferDataset, collate_fn


def get_similarity_scores(
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

        similarity_scores = []

        it = tqdm(loader) if is_main_process() else loader

        for images in it:
            images = images.to(model.device, model.dtype)

            outputs = model.compute_logits(
                pixel_values=images,
                enc_sents=[text_batch["encoded_key_phrases"]],
                use_negative_logits=False,
            )

            batch_similarity_scores = outputs["similarity_scores"]

            similarity_scores.append(batch_similarity_scores)

        local_similarity_scores = torch.cat(similarity_scores, dim=0).float().detach()

        return local_similarity_scores


def chestXDet10_eval_grounding(
    model,
    image_processor,
    tokenizer,
    batch_size,
    num_workers,
    image_path,
    text_path,
    test_path,
    image_root_dir,
):
    image = pd.read_csv(image_path)
    text_prompt = load_json(text_path)

    finding_mapping = {
        "atelectasis": "atelectasis",
        "tissue calcification": "calcification",
        "pulmonary consolidation": "consolidation",
        "pleural effusion": "effusion",
        "pulmonary emphysema": "emphysema",
        "fibrosis": "fibrosis",
        "bone fracture": "fracture",
        "pulmonary mass": "mass",
        "lung nodule": "nodule",
        "pneumothorax": "pneumothorax",
    }

    finding_classes = [
        finding_mapping[text[0].replace("There is ", "").replace(".", "").lower()] for k, text in text_prompt.items()
    ]

    finding_indices = {finding_class: i for i, finding_class in enumerate(finding_classes)}

    image_list = image["Path"].tolist()
    text_batch = process_class_prompts(text_prompt, tokenizer, model)

    similarity_scores = get_similarity_scores(
        image_list,
        text_batch,
        model,
        image_processor,
        batch_size,
        num_workers,
        image_root_dir,
    )

    start_idx, end_idx = compute_shard_indices(image_list)
    local_image_list = image_list[start_idx:end_idx]

    image_size_list = []
    for image_path in local_image_list:
        image = Image.open(os.path.join(image_root_dir, image_path))
        (width, height) = image.size
        image_size = (height, width)
        image_size_list.append(image_size)

    local_bbox_labels = load_json(test_path)[start_idx:end_idx]

    correct_counts = torch.zeros(len(finding_classes), dtype=torch.long, device=model.device)
    total_counts = torch.zeros(len(finding_classes), dtype=torch.long, device=model.device)

    for bbox_label, similarity_score, image_size in zip(local_bbox_labels, similarity_scores, image_size_list):
        syms = bbox_label["syms"]
        boxes = bbox_label["boxes"]

        image_finding_box = defaultdict(list)
        for sym, box in zip(syms, boxes):
            image_finding_box[sym].append(box)

        for finding_class, _boxes in image_finding_box.items():
            finding_index = finding_indices[finding_class.lower()]
            finding_similarity_score = similarity_score[finding_index, :]

            (x_index, y_index) = get_grounding_point(finding_similarity_score, image_size, image_processor)
            hit = is_point_in_bbox(_boxes, (x_index, y_index))
            correct_counts[finding_index] += int(hit)
            total_counts[finding_index] += 1

    if is_distributed():
        torch.distributed.all_reduce(correct_counts, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(total_counts, op=torch.distributed.ReduceOp.SUM)

    if is_main_process():
        result = {}
        for fc, i in finding_indices.items():
            result[fc] = correct_counts[i].item() / max(total_counts[i].item(), 1)

        mps = float(torch.mean(torch.tensor([result[fc] for fc in finding_classes])))

        result["mean_pointing_score"] = mps

        print("ChestXDet10 Grounding Result")
        for finding_class, pointing_score in result.items():
            print(f"{finding_class}: {pointing_score:.4f}")

        result["main_metric"] = {"mean_pointing_score": mps}
        return result

    else:
        return None


def get_grounding_point(similarity_score, image_size, image_processor):
    (height, width) = image_size

    similarity_score = similarity_score.view(int(np.sqrt(len(similarity_score))), int(np.sqrt(len(similarity_score))))
    if image_processor.__class__.__name__ == "AspectRatioPadWrapper":
        padded_size = max(height, width)

        similarity_score = F.interpolate(
            similarity_score.unsqueeze(0).unsqueeze(0),
            size=(padded_size, padded_size),
            mode="bilinear",
            align_corners=False,
        )

        pad_left = (padded_size - width) // 2
        pad_top = (padded_size - height) // 2

        similarity_score = similarity_score[:, :, pad_top : pad_top + height, pad_left : pad_left + width].contiguous()

    elif (image_processor.__class__.__name__ == "BlipImageProcessor") or (
        image_processor.__class__.__name__ == "DINOv3ViTImageProcessorFast"
    ):
        similarity_score = F.interpolate(
            similarity_score.unsqueeze(0).unsqueeze(0),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
    elif image_processor.__class__.__name__ == "BitImageProcessor":
        shortest = min(height, width)

        similarity_score = F.interpolate(
            similarity_score.unsqueeze(0).unsqueeze(0),
            size=(shortest, shortest),
            mode="bilinear",
            align_corners=False,
        )

        cropped_left = (width - shortest) // 2
        cropped_top = (height - shortest) // 2

        original_size_map = torch.ones(height, width) * -999
        original_size_map[cropped_top : cropped_top + shortest, cropped_left : cropped_left + shortest] = (
            similarity_score.view(shortest, shortest)
        )

        similarity_score = original_size_map
    elif image_processor.__class__.__name__ == "M3AEImageProcessor":
        padded_size = max(height, width)

        padded_size_map = torch.ones(padded_size, padded_size) * -999

        cropped_size = int(padded_size * 224 / 256)

        similarity_score = F.interpolate(
            similarity_score.unsqueeze(0).unsqueeze(0),
            size=(cropped_size, cropped_size),
            mode="bilinear",
            align_corners=False,
        )

        left = (padded_size - cropped_size) // 2
        top = (padded_size - cropped_size) // 2

        padded_size_map[top : top + cropped_size, left : left + cropped_size] = similarity_score.view(
            cropped_size, cropped_size
        )

        pad_left = (padded_size - width) // 2
        pad_top = (padded_size - height) // 2

        similarity_score = padded_size_map[pad_top : pad_top + height, pad_left : pad_left + width].contiguous()

    else:
        raise NotImplementedError(f"Image processor {type(image_processor)} is not supported")

    values, indices = similarity_score.view(-1).max(dim=0)

    h_index, w_index = torch.unravel_index(indices, (height, width))

    x_index = w_index.item()
    y_index = h_index.item()

    return (x_index, y_index)


def is_point_in_bbox(bbox_list, point):
    x, y = point
    for bbox in bbox_list:
        x_min, y_min, x_max, y_max = bbox
        if x_min <= x <= x_max and y_min <= y <= y_max:
            return True
    return False


def eval_refer_grounding(
    model,
    image_processor,
    tokenizer,
    data,
    image_root_dir,
):
    start_idx, end_idx = compute_shard_indices(data)
    shard = data[start_idx:end_idx]

    local_correct = 0
    local_total = 0

    it = tqdm(shard) if is_main_process() else shard

    for d in it:
        image_path = d["image"]

        image = Image.open(os.path.join(image_root_dir, image_path))
        text = d["det"][0]["name"]
        label = d["det"][0]["label"][0]
        (width, height) = image.size
        image_size = (height, width)

        pixel_values = image_processor(image)
        pixel_values = torch.FloatTensor(np.array(pixel_values["pixel_values"])).to(model.device, model.dtype)

        encoded_key_phrases = tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(model.device)

        outputs = model.compute_logits(pixel_values, [encoded_key_phrases])
        similarity_scores = outputs["similarity_scores"]

        (x_index, y_index) = get_grounding_point(similarity_scores.view(-1), image_size, image_processor)

        local_correct += is_point_in_bbox([label], (x_index, y_index))
        local_total += 1

    if is_distributed():
        t = torch.tensor([local_correct, local_total], device=model.device, dtype=torch.long)
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
        global_correct, global_total = t.tolist()
    else:
        global_correct, global_total = local_correct, local_total

    if is_main_process():
        acc = global_correct / max(global_total, 1)
        print(f"Accuracy of MS-CXR: {acc}")

        return {"main_metric": {"accuracy": acc}, "accuracy": acc}

    else:
        return None
