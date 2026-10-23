import os

import torch
from torch.utils.data import ConcatDataset

from common import logger


def input_json_file_load(name, split, cfg):
    logger.info(f"load dataset: {name}, {split}")
    from glint.dataset.chest_ct.ct_rate import load_ct_rate_dataset

    return load_ct_rate_dataset(name, split, cfg)


def load_datasets(cfg, train, inference, **kwargs):
    from glint.preprocess.chest_ct.monai.transform import load_config

    dataset = {}
    monai_cfg = load_config(cfg["monai_config"], data_root=cfg.get("data_root", ""))
    logger.info(f"MONAI mode enabled: monai_cfg={monai_cfg}")

    if train:
        train_datasets = []
        for dataset_name in cfg["train"]:
            dataset_data = input_json_file_load(dataset_name, "train", cfg)
            logger.info(f"train: {dataset_name} ({len(dataset_data)} samples)")
            train_datasets.append(
                CT_Dataset(dataset_data, "train", cfg, monai_cfg=monai_cfg, dataset_name=dataset_name)
            )
        dataset["train"] = ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]

        eval_datasets = []
        for dataset_name in cfg["eval"]:
            dataset_data = input_json_file_load(dataset_name, "eval", cfg)
            logger.info(f"eval: {dataset_name} ({len(dataset_data)} samples)")
            eval_datasets.append(CT_Dataset(dataset_data, "eval", cfg, monai_cfg=monai_cfg, dataset_name=dataset_name))
        dataset["eval"] = ConcatDataset(eval_datasets) if len(eval_datasets) > 1 else eval_datasets[0]

    if inference:
        test_datasets = []
        for dataset_name in cfg["test"]:
            dataset_data = input_json_file_load(dataset_name, "test", cfg)
            logger.info(f"test: {dataset_name} ({len(dataset_data)} samples)")
            test_datasets.append(CT_Dataset(dataset_data, "test", cfg, monai_cfg=monai_cfg, dataset_name=dataset_name))
        dataset["test"] = ConcatDataset(test_datasets) if len(test_datasets) > 1 else test_datasets[0]

    return dataset


class CT_Dataset(torch.utils.data.Dataset):
    def __init__(self, data, split, cfg, monai_cfg, dataset_name="CT"):
        from glint.preprocess.chest_ct.monai.transform import build_dataset

        self.data = data
        self.cfg = cfg

        self.dfr_cache_dir = cfg.get("dfr_cache_dir", None)
        self.dfr_cache_layers = cfg.get("dfr_cache_layers", None)
        if self.dfr_cache_dir:
            logger.info(f"  CT_Dataset DFR cache ({split}): {self.dfr_cache_dir}, layers={self.dfr_cache_layers}")

        self._monai_ds = build_dataset(
            [{"image": d["3d_raw_image_path"]} for d in data], monai_cfg, f"{dataset_name}/{split}"
        )
        logger.info(f"  CT_Dataset MONAI ({split}): {len(data)} samples, cache={monai_cfg.cache_dir(dataset_name)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_instance = self.data[idx]
        pixel_values_videos = self._monai_ds[idx]["image"].squeeze(0)
        pixel_values_videos = normalize_pixel_values_videos(pixel_values_videos)

        item_dict = {
            "pixel_values": pixel_values_videos,
            "sentences": data_instance["sentences"],
        }

        if self.dfr_cache_dir:
            from safetensors.torch import load_file

            cache_path = os.path.join(self.dfr_cache_dir, f"{data_instance['image_id']}.safetensors")
            cached = load_file(cache_path)
            cached = {int(k): v for k, v in cached.items()}
            if self.dfr_cache_layers is not None:
                cached = {k: v for k, v in cached.items() if k in self.dfr_cache_layers}
            item_dict["cached_teacher_features"] = cached

        return item_dict


def normalize_pixel_values_videos(pixel_values_videos):
    s = pixel_values_videos.shape
    num_channels = 3
    pixel_values_videos = pixel_values_videos.reshape(-1, num_channels, s[-2], s[-1])

    pixel_values_videos = (pixel_values_videos + 1) / 2

    image_mean = torch.tensor([0.485, 0.456, 0.406], device=pixel_values_videos.device).view(1, 3, 1, 1)
    image_std = torch.tensor([0.229, 0.224, 0.225], device=pixel_values_videos.device).view(1, 3, 1, 1)
    pixel_values_videos = (pixel_values_videos - image_mean) / image_std

    return pixel_values_videos
