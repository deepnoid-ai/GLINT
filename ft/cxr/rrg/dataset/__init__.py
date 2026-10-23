import random
from collections import Counter

import cv2
import numpy as np
import PIL
import torch
from torch.utils.data import ConcatDataset

from common import logger

from .data_collator import collate_fn  # noqa: F401
from .mimic_cxr import load_mimic_cxr_dataset


# Adapted from https://github.com/cuhksz-nlp/R2Gen/blob/main/modules/tokenizers.py
# (no upstream license; rights remain with the original authors)
def create_vocabulary(total_tokens, threshold=3):
    counter = Counter(total_tokens)
    vocab = [k for k, v in counter.items() if v >= threshold] + ["<unk>"]
    vocab.sort()
    token2idx, idx2token = {}, {}
    for idx, token in enumerate(vocab):
        token2idx[token] = idx + 1
        idx2token[idx + 1] = token

    return token2idx, idx2token


def input_json_file_load(name, split, cfg):
    logger.info(f"load dataset: {name}, {split}")

    if name == "mimic_cxr":
        return load_mimic_cxr_dataset(name, split, cfg)
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
    token2idx, idx2token = None, None

    split_ratio = cfg.get("train_split_ratio", 1.0)
    seed = cfg.get("seed", 42)
    dataset_class = {"CXRDataset": CXRDataset}[cfg.get("dataset_class", "CXRDataset")]

    logger.info(f"Dataset split ratio for training: {split_ratio * 100:.1f}%")
    if train:
        train_data_per_ds = {}
        for dataset_name in cfg["train"]:
            dataset_data = input_json_file_load(dataset_name, "train", cfg)
            dataset_data = subsample_dataset(dataset_data, split_ratio, seed, "train")
            logger.info(f"train: {dataset_name} - {split_ratio * 100:.1f}% ({len(dataset_data)} samples)")
            train_data_per_ds[dataset_name] = dataset_data

        total_tokens = []
        max_seq_length = 0
        for dataset_data in train_data_per_ds.values():
            for data_instance in dataset_data:
                tokens = data_instance["report"].split()
                max_seq_length = max(max_seq_length, len(tokens))
                total_tokens.extend(tokens)

        total_tokens.sort()
        token2idx, idx2token = create_vocabulary(total_tokens, threshold=3)

        train_datasets = []
        for dataset_name, dataset_data in train_data_per_ds.items():
            train_datasets.append(dataset_class(dataset_data, "train", cfg, image_processor))
        datasets["train"] = ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]

    eval_datasets = []
    for dataset_name in cfg["eval"]:
        dataset_data = input_json_file_load(dataset_name, "eval", cfg)
        logger.info(f"eval: {dataset_name} ({len(dataset_data)} samples)")
        eval_datasets.append(dataset_class(dataset_data, "eval", cfg, image_processor))
    datasets["eval"] = ConcatDataset(eval_datasets) if len(eval_datasets) > 1 else eval_datasets[0]

    if inference:
        test_datasets = []
        for dataset_name in cfg["test"]:
            dataset_data = input_json_file_load(dataset_name, "test", cfg)
            logger.info(f"test: {dataset_name} ({len(dataset_data)} samples)")
            test_datasets.append(dataset_class(dataset_data, "test", cfg, image_processor))
        datasets["test"] = ConcatDataset(test_datasets) if len(test_datasets) > 1 else test_datasets[0]

    if token2idx is not None and idx2token is not None:
        datasets["vocabulary"] = {"token2idx": token2idx, "idx2token": idx2token}
    return datasets


class CXRDataset(torch.utils.data.Dataset):
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

        # 16/32-bit grayscale -> uint8
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

        item_dict["report"] = data_instance["report"]

        return item_dict
