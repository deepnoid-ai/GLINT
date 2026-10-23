# ruff: noqa: E402
"""Warm the MONAI PersistentDataset cache for CT segmentation, once per (preprocessing × dataset)."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch

# Fix MONAI PersistentDataset cache compatibility with PyTorch 2.6+
torch.serialization.add_safe_globals([torch.Tensor])
try:
    from monai.data import MetaTensor

    torch.serialization.add_safe_globals([MetaTensor])
except ImportError:
    pass

from monai.data import PersistentDataset
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from ft.chest_ct.seg.dataset import _load_dataset_json
from ft.chest_ct.seg.dataset.transforms import build_cache_transforms

CONFIGS_DIR = Path(__file__).parent.parent / "configs"

_RESERVED_KEYS = {"data_root", "preprocessing"}


def load_data_cfg(name: str) -> OmegaConf:
    cfg_path = CONFIGS_DIR / f"{name}.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Data config not found: {cfg_path}")
    return OmegaConf.load(cfg_path)


def warmup(
    data_dicts: list[dict],
    cache_transform,
    cache_dir: Path,
    num_workers: int = 16,
):
    cache_dir.mkdir(parents=True, exist_ok=True)

    ds = PersistentDataset(
        data=data_dicts,
        transform=cache_transform,
        cache_dir=str(cache_dir),
    )

    total = len(ds)
    if total == 0:
        print("  No samples, skipping.")
        return

    loader = DataLoader(
        ds,
        batch_size=1,
        num_workers=num_workers,
        shuffle=False,
        prefetch_factor=2,
        persistent_workers=False,
    )

    t0 = time.time()
    for i, _ in enumerate(loader):
        if (i + 1) % 20 == 0 or i == total - 1:
            elapsed = time.time() - t0
            speed = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (total - i - 1) / speed if speed > 0 else 0
            print(f"  [{i + 1}/{total}] {speed:.1f} samples/s, ETA {eta / 60:.1f}min")

    elapsed = time.time() - t0
    print(f"  Done: {total} samples in {elapsed / 60:.1f}min")
    print(f"  Cache: {cache_dir}\n")


def build_all_data_dicts(raw_dir: Path) -> list[dict]:
    ds_json = _load_dataset_json(raw_dir)
    file_ending = ds_json.get("file_ending", ".nii.gz")

    labels_dir = raw_dir / "labelsTr"
    if labels_dir.exists():
        case_ids = sorted(f.name.replace(file_ending, "") for f in labels_dir.iterdir() if f.name.endswith(file_ending))
    else:
        num_training = ds_json.get("numTraining", 0)
        case_ids = [f"case_{i:04d}" for i in range(num_training)]

    data_dicts = []
    for case_id in case_ids:
        img_path = raw_dir / "imagesTr" / f"{case_id}_0000{file_ending}"
        if not img_path.exists():
            img_path = raw_dir / "imagesTr" / f"{case_id}{file_ending}"
        lbl_path = raw_dir / "labelsTr" / f"{case_id}{file_ending}"

        if not img_path.exists() or not lbl_path.exists():
            continue

        data_dicts.append({"image": str(img_path), "label": str(lbl_path)})

    return data_dicts


def build_datalist_data_dicts(datalist_json: Path, data_root: Path) -> list[dict]:
    """Collect train+val+test of a datalist entry so every split is cached up-front."""
    import json

    with open(datalist_json) as f:
        payload = json.load(f)

    data_dicts: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for split in ("train", "val", "test"):
        for entry in payload.get(split, []):
            key = (str(data_root / entry["image"]), str(data_root / entry["label"]))
            if key in seen:
                continue
            if not Path(key[0]).exists() or not Path(key[1]).exists():
                continue
            data_dicts.append({"image": key[0], "label": key[1]})
            seen.add(key)
    return data_dicts


def main():
    parser = argparse.ArgumentParser(description="MONAI cache warm-up for CT segmentation")
    parser.add_argument(
        "--data_cfg",
        type=str,
        required=True,
        help="Data config name (e.g. data_config_glint_ct). "
        "Reads preprocessing + dataset paths from configs/{name}.yaml.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=os.environ.get("DATA_ROOT"),
        help="Dataset root that relative paths in the data config are joined with (default: $DATA_ROOT)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Specific datasets to process (default: all datasets in the config)",
    )
    parser.add_argument("--num_workers", type=int, default=16)
    args = parser.parse_args()

    cfg = load_data_cfg(args.data_cfg)
    ds_cfg = cfg.dataset

    if not args.data_root:
        parser.error("--data_root (or $DATA_ROOT) is required")
    data_root = Path(args.data_root)
    prep = ds_cfg.preprocessing
    spacing = tuple(prep.spacing)
    orientation = prep.orientation
    a_min = float(prep.a_min)
    a_max = float(prep.a_max)
    b_min = float(prep.b_min)
    b_max = float(prep.b_max)
    cache_root = data_root / prep.cache_root

    cache_key = f"{spacing[0]}x{spacing[1]}x{spacing[2]}_{orientation}_HU{a_min}to{a_max}"

    cache_transform = build_cache_transforms(
        spacing=spacing, orientation=orientation, a_min=a_min, a_max=a_max, b_min=b_min, b_max=b_max
    )

    all_datasets = [k for k in ds_cfg if k not in _RESERVED_KEYS]

    if args.datasets:
        datasets_to_process = [d for d in args.datasets if d in all_datasets]
        skipped = [d for d in args.datasets if d not in all_datasets]
        if skipped:
            print(f"WARNING: datasets not in config, skipping: {skipped}")
    else:
        datasets_to_process = all_datasets

    print(f"Data config: {args.data_cfg}")
    print(f"Data root: {data_root}")
    print(f"Cache root: {cache_root}")
    print(f"Cache key: {cache_key}")
    print(f"Spacing: {spacing}, Orientation: {orientation}")
    print(f"HU: [{a_min}, {a_max}] → [{b_min}, {b_max}]")
    print(f"Datasets: {datasets_to_process}")
    print()

    for ds_name in datasets_to_process:
        ds_entry = ds_cfg[ds_name]

        print(f"{'=' * 60}")
        print(f"  {ds_name} — all volumes")
        print(f"{'=' * 60}")

        if "datalist_json" in ds_entry:
            datalist_path = data_root / ds_entry.datalist_json
            if not datalist_path.exists():
                print(f"SKIP: datalist {datalist_path} not found")
                continue
            data_dicts = build_datalist_data_dicts(datalist_path, data_root)
        else:
            raw_dir = data_root / ds_entry.raw_dir
            if not raw_dir.exists():
                print(f"SKIP: {raw_dir} not found")
                continue
            data_dicts = build_all_data_dicts(raw_dir)

        if not data_dicts:
            print("  No data found, skipping")
            continue

        print(f"  Found {len(data_dicts)} volumes")
        cache_dir = cache_root / cache_key / ds_name
        warmup(data_dicts, cache_transform, cache_dir, num_workers=args.num_workers)


if __name__ == "__main__":
    main()
