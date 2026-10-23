import os

import cv2
import numpy as np
import PIL
import torch
from torch.utils.data import ConcatDataset

from common import logger
from glint.dataset.cxr.mimic_cxr import load_mimic_cxr_dataset


def load_datasets(cfg, train, inference, image_processor, **kwargs):
    dataset = {}

    def _load(name, split):
        logger.info(f"load dataset: {name}, {split}")
        return load_mimic_cxr_dataset(name, split, cfg)

    if train:
        train_datasets = [CXRDataset(_load(n, "train"), "train", cfg, image_processor) for n in cfg["train"]]
        dataset["train"] = ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]

        eval_datasets = [CXRDataset(_load(n, "eval"), "eval", cfg, image_processor) for n in cfg["eval"]]
        dataset["eval"] = ConcatDataset(eval_datasets) if len(eval_datasets) > 1 else eval_datasets[0]

    if inference:
        test_datasets = [CXRDataset(_load(n, "test"), "test", cfg, image_processor) for n in cfg["test"]]
        dataset["test"] = ConcatDataset(test_datasets) if len(test_datasets) > 1 else test_datasets[0]

    return dataset


class CXRDataset(torch.utils.data.Dataset):
    def __init__(self, data, split, cfg, image_processor):
        self.data = data
        self.image_processor = image_processor

        self.dfr_cache_dir = cfg.get("dfr_cache_dir", None)
        self.dfr_cache_layers = cfg.get("dfr_cache_layers", None)
        if self.dfr_cache_dir:
            logger.info(f"  CXRDataset DFR cache ({split}): {self.dfr_cache_dir}, layers={self.dfr_cache_layers}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_instance = self.data[idx]
        image = PIL.Image.open(data_instance["image"])

        if "I" in image.mode:
            image = PIL.Image.fromarray(
                cv2.normalize(np.array(image), None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            )
        image = image.convert("RGB")

        pixel_values = self.image_processor(image, return_tensors="pt")["pixel_values"][0]

        item_dict = {
            "pixel_values": pixel_values,
            "sentences": data_instance["sentences"],
        }

        if self.dfr_cache_dir:
            from safetensors.torch import load_file

            image_id = os.path.splitext(os.path.basename(data_instance["image"]))[0]
            cached = load_file(os.path.join(self.dfr_cache_dir, f"{image_id}.safetensors"))
            cached = {int(k): v for k, v in cached.items()}
            if self.dfr_cache_layers is not None:
                cached = {k: v for k, v in cached.items() if k in self.dfr_cache_layers}
            item_dict["cached_teacher_features"] = cached

        return item_dict
