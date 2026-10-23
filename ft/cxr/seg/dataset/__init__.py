import random
from copy import deepcopy

import cv2
import numpy as np
import PIL
import torch
from torch.utils.data import ConcatDataset

from common import logger
from glint.zs_eval.cxr.segmentation_utils import rle2mask

from .chestx_det import load_chestx_det_dataset, polygon2mask
from .data_collator import collate_fn  # noqa: F401
from .siim import load_siim_dataset


def input_json_file_load(name, split, cfg):
    logger.info(f"load dataset: {name}, {split}")

    if name == "SIIM":
        return load_siim_dataset(name, split, cfg)
    elif name == "ChestX_Det":
        return load_chestx_det_dataset(name, split, cfg)
    else:
        raise NotImplementedError(f"Dataset {name} is not supported")


def subsample_dataset(dataset, split_ratio, seed=42, split_name="train"):
    if split_ratio >= 1.0:
        logger.info(f"{split_name}: Using full dataset ({len(dataset)} samples)")
        return dataset

    random.seed(seed)
    np.random.seed(seed)

    total_samples = len(dataset)
    target_samples = int(total_samples * split_ratio)

    indices = list(range(total_samples))
    random.shuffle(indices)
    selected_indices = indices[:target_samples]

    subsampled_dataset = [dataset[i] for i in selected_indices]

    logger.info(f"{split_name}: Subsampled {target_samples}/{total_samples} samples ({split_ratio * 100:.1f}%)")

    return subsampled_dataset


def load_datasets(cfg, train, inference, image_processor):
    datasets = {}

    split_ratio = cfg.get("train_split_ratio", 1.0)
    seed = cfg.get("seed", 42)

    logger.info(f"Dataset split ratio for training: {split_ratio * 100:.1f}%")
    if train:
        train_datasets = []
        for dataset_name in cfg["train"]:
            dataset_data = input_json_file_load(dataset_name, "train", cfg)
            dataset_data = subsample_dataset(dataset_data, split_ratio, seed, "train")
            logger.info(f"train: {dataset_name} - {split_ratio * 100:.1f}% ({len(dataset_data)} samples)")
            train_datasets.append(CXRFtDataset(dataset_data, "train", cfg, image_processor))
        datasets["train"] = ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]

    eval_datasets = []
    for dataset_name in cfg["eval"]:
        dataset_data = input_json_file_load(dataset_name, "eval", cfg)
        logger.info(f"eval: {dataset_name} ({len(dataset_data)} samples)")
        eval_datasets.append(CXRFtDataset(dataset_data, "eval", cfg, image_processor))
    datasets["eval"] = ConcatDataset(eval_datasets) if len(eval_datasets) > 1 else eval_datasets[0]

    if inference:
        test_datasets = []
        for dataset_name in cfg["test"]:
            dataset_data = input_json_file_load(dataset_name, "test", cfg)
            logger.info(f"test: {dataset_name} ({len(dataset_data)} samples)")
            test_datasets.append(CXRFtDataset(dataset_data, "test", cfg, image_processor))
        datasets["test"] = ConcatDataset(test_datasets) if len(test_datasets) > 1 else test_datasets[0]

    return datasets


class CXRFtDataset(torch.utils.data.Dataset):
    def __init__(self, data, split, cfg, original_image_processor):
        self.data = data
        self.split = split
        self.cfg = cfg

        self.target_mask_size = cfg.get("target_mask_size", 256)
        self.target_image_size = self._extract_target_size(original_image_processor)
        self.image_processor = self.modify_image_processor(original_image_processor)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_instance = self.data[idx]
        item_dict = {}

        image = PIL.Image.open(data_instance["image"])
        image = image.convert("RGB")
        width, height = image.size

        image = image.resize((self.target_image_size, self.target_image_size))

        pixel_values = self.image_processor(image, return_tensors="pt")["pixel_values"][0]

        item_dict["pixel_values"] = pixel_values

        mask_instance = data_instance.get("mask")
        if isinstance(mask_instance, str):
            mask = PIL.Image.open(mask_instance)
            mask = mask.convert("L")
            mask = mask.resize((self.target_mask_size, self.target_mask_size))
            mask = np.array(mask)
            mask = mask > 128
            mask = mask.astype(np.uint8)
            item_dict["masks"] = torch.LongTensor(mask).unsqueeze(0)

        elif isinstance(mask_instance, dict):
            num_classes = len(mask_instance)
            mask = np.zeros([num_classes, height, width])

            for class_idx, polygons in enumerate(mask_instance.values()):
                if polygons:
                    if num_classes == 1 and isinstance(polygons[0], str):
                        for poly in polygons:
                            mask[0, ...] += rle2mask(poly, height=height, width=width)

                    else:
                        mask[class_idx, ...] += polygon2mask(polygons, height=height, width=width)

            mask = np.clip(mask, 0, 1)

            resized_mask = np.zeros([num_classes, self.target_mask_size, self.target_mask_size])
            for i in range(num_classes):
                single_mask = mask[i, ...].astype(np.uint8)
                single_mask = cv2.resize(
                    single_mask,
                    (self.target_mask_size, self.target_mask_size),
                    cv2.INTER_NEAREST,
                )
                resized_mask[i, ...] = single_mask

            item_dict["masks"] = torch.LongTensor(resized_mask)

        else:
            raise NotImplementedError(f"Unsupported mask type: {type(mask_instance)}")

        return item_dict

    def modify_image_processor(self, original_image_processor):
        image_processor = deepcopy(original_image_processor)
        if hasattr(image_processor, "do_resize"):
            image_processor.do_resize = False
        if hasattr(image_processor, "do_center_crop"):
            image_processor.do_center_crop = False
        return image_processor

    def _extract_target_size(self, image_processor):
        target_size = None

        if hasattr(image_processor, "crop_size"):
            crop_size = image_processor.crop_size
            if crop_size:
                if isinstance(crop_size, dict):
                    if "height" in crop_size:
                        target_size = crop_size["height"]
                    elif "width" in crop_size:
                        target_size = crop_size["width"]
                    else:
                        target_size = next(iter(crop_size.values()))
                elif isinstance(crop_size, int):
                    target_size = int(crop_size)
                else:
                    raise NotImplementedError(f"Unsupported crop_size type for image_processor: {image_processor}")

        if target_size is None and hasattr(image_processor, "size"):
            size = image_processor.size
            if isinstance(size, dict):
                if "shortest_edge" in size:
                    target_size = size["shortest_edge"]
                elif "height" in size:
                    target_size = size["height"]
                elif "width" in size:
                    target_size = size["width"]
                else:
                    target_size = next(iter(size.values()))
            elif isinstance(size, int):
                target_size = int(size)
            elif isinstance(size, (list, tuple)) and len(size) > 0:
                target_size = int(size[0])

        return target_size
