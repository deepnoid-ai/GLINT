import random

import numpy as np
import torch
from monai.data import PersistentDataset
from torch.utils.data import ConcatDataset

from common import logger
from ft.chest_ct.processed_volumes import load_processed_volume
from glint.preprocess.chest_ct.monai.transform import build_transform, load_config

from .ct_rate import load_ct_rate_dataset
from .data_collator import collate_fn  # noqa: F401


def input_json_file_load(name, split, cfg):
    logger.info(f"load dataset: {name}, {split}")

    if name == "CT_RATE":
        return load_ct_rate_dataset(name, split, cfg)
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


def load_datasets(cfg, train, inference):
    datasets = {}
    token2idx, idx2token = None, None

    split_ratio = cfg.get("train_split_ratio", 1.0)
    seed = cfg.get("seed", 42)

    monai_cfg = None
    if cfg.get("monai_config"):
        monai_cfg = load_config(cfg["monai_config"], data_root=cfg.get("data_root", ""))
        logger.info(f"MONAI mode enabled: monai_cfg={monai_cfg}")

    logger.info(f"Dataset split ratio for training: {split_ratio * 100:.1f}%")
    if train:
        train_datasets = []
        for dataset_name in cfg["train"]:
            result = input_json_file_load(dataset_name, "train", cfg)
            # the train split also returns the vocabulary: (data_list, token2idx, idx2token)
            if isinstance(result, tuple):
                dataset_data, token2idx, idx2token = result
            else:
                dataset_data = result
            dataset_data = subsample_dataset(dataset_data, split_ratio, seed, "train")
            logger.info(f"train: {dataset_name} - {split_ratio * 100:.1f}% ({len(dataset_data)} samples)")
            train_datasets.append(
                CTFtDataset(dataset_data, "train", cfg, monai_cfg=monai_cfg, dataset_name=dataset_name)
            )
        datasets["train"] = ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]

    eval_datasets = []
    for dataset_name in cfg["eval"]:
        dataset_data = input_json_file_load(dataset_name, "eval", cfg)
        logger.info(f"eval: {dataset_name} ({len(dataset_data)} samples)")
        eval_datasets.append(CTFtDataset(dataset_data, "eval", cfg, monai_cfg=monai_cfg, dataset_name=dataset_name))
    datasets["eval"] = ConcatDataset(eval_datasets) if len(eval_datasets) > 1 else eval_datasets[0]

    if inference:
        test_datasets = []
        for dataset_name in cfg["test"]:
            dataset_data = input_json_file_load(dataset_name, "test", cfg)
            logger.info(f"test: {dataset_name} ({len(dataset_data)} samples)")
            test_datasets.append(CTFtDataset(dataset_data, "test", cfg, monai_cfg=monai_cfg, dataset_name=dataset_name))
        datasets["test"] = ConcatDataset(test_datasets) if len(test_datasets) > 1 else test_datasets[0]

    if token2idx is not None and idx2token is not None:
        datasets["vocabulary"] = {"token2idx": token2idx, "idx2token": idx2token}

    return datasets


class CTFtDataset(torch.utils.data.Dataset):
    def __init__(self, data, split, cfg, monai_cfg, dataset_name="CT"):
        self.data = data
        self.split = split
        self.cfg = cfg

        self._monai_ds = None
        if all("3d_image_path" in d for d in data):
            logger.info(f"  CTFtDataset preprocessed volumes rrg ({split}): {len(data)} samples")
            return
        if monai_cfg is None:
            raise ValueError("Set dataset.monai_config (e.g. chest_ct_monai) or dataset.processed_volumes")

        transform = build_transform(monai_cfg)

        cache_dir = monai_cfg.cache_dir(f"{dataset_name}/{split}")
        cache_dir.mkdir(parents=True, exist_ok=True)

        monai_data = [{"image": d["3d_raw_image_path"]} for d in data]
        self._monai_ds = PersistentDataset(
            data=monai_data,
            transform=transform,
            cache_dir=str(cache_dir),
        )
        logger.info(f"  CTFtDataset MONAI rrg ({split}): {len(data)} samples, cache={cache_dir}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_instance = self.data[idx]
        item_dict = {}

        if self._monai_ds is None:
            pixel_values_videos = load_processed_volume(
                data_instance["3d_image_path"], self.cfg.get("center_crop_size")
            )  # (D, H, W)
        else:
            monai_item = self._monai_ds[idx]
            pixel_values_videos = monai_item["image"].squeeze(0)  # (1, D, H, W) -> (D, H, W)

        if self.cfg.get("rgb_norm", True):
            pixel_values_videos = normalize_pixel_values_videos(pixel_values_videos)
        else:
            pixel_values_videos = pixel_values_videos.unsqueeze(0)

        item_dict["pixel_values"] = pixel_values_videos
        item_dict["report"] = data_instance["report"]

        return item_dict


def normalize_pixel_values_videos(pixel_values_videos):
    s = pixel_values_videos.shape
    num_channels = 3
    pixel_values_videos = pixel_values_videos.reshape(-1, num_channels, s[-2], s[-1])

    pixel_values_videos = (pixel_values_videos + 1) / 2

    # normalize for V-JEPA2 RGB pretrained model
    image_mean = torch.tensor([0.485, 0.456, 0.406], device=pixel_values_videos.device).view(1, 3, 1, 1)
    image_std = torch.tensor([0.229, 0.224, 0.225], device=pixel_values_videos.device).view(1, 3, 1, 1)
    pixel_values_videos = (pixel_values_videos - image_mean) / image_std

    return pixel_values_videos
