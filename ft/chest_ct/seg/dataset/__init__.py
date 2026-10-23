"""CT segmentation datasets (MSD / nnU-Net raw layout or datalist JSON) with PersistentDataset caching."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from monai.data import PersistentDataset
from torch.utils.data import ConcatDataset

from common import logger

from .data_collator import collate_fn  # noqa: F401
from .transforms import build_cache_transforms, build_eval_transform, build_train_augmentation


def resolve_path(path: str | Path | None, data_root: str | Path | None) -> str | None:
    """Join a relative path with ``dataset.data_root``; absolute paths pass through."""
    if path is None or os.path.isabs(str(path)) or not data_root:
        return None if path is None else str(path)
    return os.path.join(str(data_root), str(path))


def _load_split_json(split_json_path: str | Path | None, fold: int | None = None) -> dict | None:
    """Load a flat {"train", "val"[, "test"]} or 5-fold {"fold0": {...}, ...} split JSON."""
    if split_json_path is None:
        return None
    path = Path(split_json_path)
    if not path.exists():
        logger.warning(f"Split JSON not found: {path}, using default split")
        return None
    with open(path) as f:
        data = json.load(f)

    if "fold0" in data:
        if fold is None:
            fold = 0
            logger.warning("5-fold split detected but no fold specified, defaulting to fold0")
        fold_key = f"fold{fold}"
        if fold_key not in data:
            raise ValueError(f"Fold {fold_key} not found in {path}. Available: {list(data.keys())}")
        logger.info(f"Using {fold_key} from {path.name}")
        return data[fold_key]

    return data


def load_raw_test_cases(ds_cfg: dict, data_root: str | Path | None = None) -> list[dict]:
    """Raw-path test cases for sliding-window inference, from an MSD split_json or a datalist_json."""
    datalist_json = resolve_path(ds_cfg.get("datalist_json"), data_root)
    if datalist_json:
        payload = _load_datalist_splits(datalist_json)
        ds_name = ds_cfg.get("name", Path(datalist_json).stem)
        if "test" not in payload:
            raise ValueError(f"{ds_name}: datalist_json {datalist_json} has no 'test' key")
        cases: list[dict] = []
        for e in payload["test"]:
            img = resolve_path(e["image"], data_root)
            lbl = resolve_path(e["label"], data_root)
            if not Path(img).exists() or not Path(lbl).exists():
                logger.warning(f"{ds_name}: skip {e.get('case_id', e['pid'])} (missing file)")
                continue
            cases.append(
                {
                    "image": img,
                    "label": lbl,
                    "case_id": e.get("case_id", e.get("pid")),
                    "dataset_name": ds_name,
                }
            )
        logger.info(f"{ds_name}: {len(cases)} test cases (datalist)")
        return cases

    raw_dir = Path(resolve_path(ds_cfg["raw_dir"], data_root))
    ds_name = ds_cfg.get("name", raw_dir.name)
    split_json_path = resolve_path(ds_cfg.get("split_json"), data_root)
    if not split_json_path:
        raise ValueError(f"{ds_name}: split_json is required (need 'test' key)")

    with open(split_json_path) as f:
        split = json.load(f)
    if "test" not in split:
        raise ValueError(
            f"{ds_name}: split_json {split_json_path} has no 'test' key — "
            "regenerate with splits_holdout712.json (3-way)"
        )

    ds_json_path = raw_dir / "dataset.json"
    file_ending = ".nii.gz"
    if ds_json_path.exists():
        with open(ds_json_path) as f:
            file_ending = json.load(f).get("file_ending", ".nii.gz")

    cases = []
    for case_id in split["test"]:
        img_path = raw_dir / "imagesTr" / f"{case_id}_0000{file_ending}"
        if not img_path.exists():
            img_path = raw_dir / "imagesTr" / f"{case_id}{file_ending}"
        lbl_path = raw_dir / "labelsTr" / f"{case_id}{file_ending}"
        if not img_path.exists() or not lbl_path.exists():
            logger.warning(f"{ds_name}: skip {case_id} (missing image or label)")
            continue
        cases.append(
            {
                "image": str(img_path),
                "label": str(lbl_path),
                "case_id": case_id,
                "dataset_name": ds_name,
            }
        )
    logger.info(f"{ds_name}: {len(cases)} test cases")
    return cases


def _load_dataset_json(raw_dir: Path) -> dict:
    json_path = raw_dir / "dataset.json"
    if not json_path.exists():
        raise FileNotFoundError(f"dataset.json not found: {json_path}")
    with open(json_path) as f:
        return json.load(f)


def _load_datalist_splits(datalist_json: str | Path) -> dict:
    path = Path(datalist_json)
    if not path.exists():
        raise FileNotFoundError(f"datalist_json not found: {path}")
    with open(path) as f:
        return json.load(f)


def _build_data_dicts_from_datalist(
    datalist_json: str | Path, split: str, data_root: str | Path | None = None
) -> list[dict]:
    payload = _load_datalist_splits(datalist_json)
    split_key = "val" if split == "eval" else split
    entries = payload.get(split_key, [])

    data_dicts: list[dict] = []
    for e in entries:
        img = resolve_path(e["image"], data_root)
        lbl = resolve_path(e["label"], data_root)
        if not Path(img).exists():
            logger.warning(f"Image not found, skipping: {img}")
            continue
        if not Path(lbl).exists():
            logger.warning(f"Label not found, skipping: {lbl}")
            continue
        data_dicts.append({"image": img, "label": lbl})

    logger.info(f"  [datalist:{Path(datalist_json).name}] {split}: {len(data_dicts)} volumes")
    return data_dicts


def _build_data_dicts(
    raw_dir: str | Path,
    split: str,
    split_json: dict | None = None,
) -> list[dict]:
    raw_dir = Path(raw_dir)
    ds_json = _load_dataset_json(raw_dir)
    file_ending = ds_json.get("file_ending", ".nii.gz")

    if split_json is not None:
        split_key = "val" if split == "eval" else split
        case_ids = split_json.get(split_key, [])
    else:
        labels_dir = raw_dir / "labelsTr"
        if labels_dir.exists():
            case_ids = sorted(
                f.name.replace(file_ending, "") for f in labels_dir.iterdir() if f.name.endswith(file_ending)
            )
        else:
            num_training = ds_json.get("numTraining", 0)
            case_ids = [f"case_{i:04d}" for i in range(num_training)]

    data_dicts = []
    for case_id in case_ids:
        # nnU-Net: {case_id}_0000.nii.gz, MSD: {case_id}.nii.gz
        img_path = raw_dir / "imagesTr" / f"{case_id}_0000{file_ending}"
        if not img_path.exists():
            img_path = raw_dir / "imagesTr" / f"{case_id}{file_ending}"
        lbl_path = raw_dir / "labelsTr" / f"{case_id}{file_ending}"

        if not img_path.exists():
            logger.warning(f"Image not found, skipping: {img_path}")
            continue
        if not lbl_path.exists():
            logger.warning(f"Label not found, skipping: {lbl_path}")
            continue

        data_dicts.append({"image": str(img_path), "label": str(lbl_path)})

    logger.info(f"  [{raw_dir.name}] {split}: {len(data_dicts)} volumes")
    return data_dicts


class CTSegDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dicts: list[dict],
        cache_dir: str | Path,
        cache_transform,
        split: str = "train",
        augmentation=None,
    ):
        self.split = split
        self.augmentation = augmentation

        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        self._persistent_ds = PersistentDataset(
            data=data_dicts,
            transform=cache_transform,
            cache_dir=str(cache_dir),
        )
        logger.info(f"  CTSegDataset ({split}): {len(data_dicts)} volumes, cache={cache_dir}")

    def __len__(self):
        return len(self._persistent_ds)

    def __getitem__(self, idx):
        item = self._persistent_ds[idx]

        if self.augmentation is not None:
            item = self.augmentation(item)

        # item may be a list (from RandCropByPosNegLabeld num_samples > 1)
        if isinstance(item, list):
            return [{"pixel_values": d["image"], "masks": d["label"]} for d in item]

        return {"pixel_values": item["image"], "masks": item["label"]}


def load_datasets(cfg, train=True, inference=False, image_processor=None):
    datasets = {}

    fold = cfg.get("fold", 0)
    spacing = tuple(cfg.get("spacing", [1.5, 1.5, 1.5]))
    orientation = cfg.get("orientation", "RAS")
    patch_size = tuple(cfg.get("patch_size", [96, 96, 96]))
    num_samples = cfg.get("num_samples", 2)
    a_min = cfg.get("a_min", -175.0)
    a_max = cfg.get("a_max", 250.0)
    b_min = cfg.get("b_min", 0.0)
    b_max = cfg.get("b_max", 1.0)
    data_root = cfg.get("data_root")
    cache_root = Path(resolve_path(cfg.get("cache_root", "monai_cache/ct_seg"), data_root))

    logger.info(f"Loading datasets: fold={fold}, spacing={spacing}, orientation={orientation}, patch_size={patch_size}")

    # must match the encoder's pretraining preprocessing
    cache_transform = build_cache_transforms(
        spacing=spacing, orientation=orientation, a_min=a_min, a_max=a_max, b_min=b_min, b_max=b_max
    )

    augment = cfg.get("augment", True)
    train_augmentation = build_train_augmentation(
        patch_size=patch_size,
        num_samples=num_samples,
        augment=augment,
    )
    eval_augmentation = build_eval_transform(patch_size=patch_size)

    dataset_configs = cfg.get("datasets", [])
    if not dataset_configs:
        raise ValueError("No datasets configured. Set dataset.datasets in config.")

    train_list, eval_list, test_list = [], [], []

    spacing_str = f"{spacing[0]}x{spacing[1]}x{spacing[2]}_{orientation}_HU{a_min}to{a_max}"

    for ds_cfg in dataset_configs:
        datalist_json = resolve_path(ds_cfg.get("datalist_json"), data_root)

        if datalist_json:
            ds_name = ds_cfg.get("name", Path(datalist_json).stem)
            split_json = None  # datalist carries its own train/val/test
            build_dicts = lambda split: _build_data_dicts_from_datalist(  # noqa: E731
                datalist_json, split, data_root
            )
            has_test = "test" in _load_datalist_splits(datalist_json)
        else:
            raw_dir = resolve_path(ds_cfg["raw_dir"], data_root)
            ds_name = ds_cfg.get("name", Path(raw_dir).name)
            split_json = _load_split_json(resolve_path(ds_cfg.get("split_json"), data_root), fold=fold)
            build_dicts = lambda split: _build_data_dicts(raw_dir, split, split_json)  # noqa: E731
            has_test = split_json is not None and "test" in split_json

        # Single cache dir per (preprocessing × dataset) — no fold duplication
        ds_cache_dir = cache_root / spacing_str / ds_name

        if train:
            train_dicts = build_dicts("train")
            if train_dicts:
                train_list.append(
                    CTSegDataset(
                        data_dicts=train_dicts,
                        cache_dir=ds_cache_dir,
                        cache_transform=cache_transform,
                        split="train",
                        augmentation=train_augmentation,
                    )
                )

            eval_dicts = build_dicts("eval")
            if eval_dicts:
                eval_list.append(
                    CTSegDataset(
                        data_dicts=eval_dicts,
                        cache_dir=ds_cache_dir,
                        cache_transform=cache_transform,
                        split="eval",
                        augmentation=eval_augmentation,
                    )
                )

        if inference:
            # 2-way / 5-fold splits have no "test" key, so fall back to "val"
            test_dicts = build_dicts("test") if has_test else build_dicts("eval")
            if test_dicts:
                test_list.append(
                    CTSegDataset(
                        data_dicts=test_dicts,
                        cache_dir=ds_cache_dir,
                        cache_transform=cache_transform,
                        split="test",
                        augmentation=None,
                    )
                )

    if train_list:
        datasets["train"] = ConcatDataset(train_list) if len(train_list) > 1 else train_list[0]
    if eval_list:
        datasets["eval"] = ConcatDataset(eval_list) if len(eval_list) > 1 else eval_list[0]
    if test_list:
        datasets["test"] = ConcatDataset(test_list) if len(test_list) > 1 else test_list[0]

    return datasets
