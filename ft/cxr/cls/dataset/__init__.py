import random

import cv2
import numpy as np
import PIL
import torch
from torch.utils.data import ConcatDataset

from common import logger

from .data_collator import collate_fn  # noqa: F401
from .nih import load_nih_dataset
from .rsna import load_rsna_dataset
from .siim import load_siim_dataset
from .vindr import load_vindr_dataset


def input_json_file_load(name, split, cfg):
    logger.info(f"load dataset: {name}, {split}")

    if name == "NIH":
        return load_nih_dataset(name, split, cfg)
    elif name == "RSNA":
        return load_rsna_dataset(name, split, cfg)
    elif name == "SIIM":
        return load_siim_dataset(name, split, cfg)
    elif name == "VinDr":
        return load_vindr_dataset(name, split, cfg)
    else:
        raise NotImplementedError


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
    def __init__(self, data, split, cfg, image_processor):
        self.data = data
        self.split = split
        self.cfg = cfg

        self.image_processor = image_processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_instance = self.data[idx]
        item_dict = {}

        image = PIL.Image.open(data_instance["image"])
        image = image.convert("L")
        if "I" in image.mode:
            image = PIL.Image.fromarray(
                cv2.normalize(
                    np.array(image),
                    None,
                    0,
                    255,
                    norm_type=cv2.NORM_MINMAX,
                    dtype=cv2.CV_8U,
                )
            )

        image = image.convert("RGB")

        pixel_values = self.image_processor(image, return_tensors="pt")["pixel_values"][0]

        item_dict["pixel_values"] = pixel_values

        if data_instance.get("labels") is not None:
            item_dict["labels"] = torch.LongTensor(data_instance["labels"])

        return item_dict
