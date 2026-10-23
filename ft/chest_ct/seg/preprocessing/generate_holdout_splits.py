"""Generate flat 7:1:2 (train/val/test) holdout split JSONs for segmentation datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def get_case_ids(dataset_dir: Path) -> list[str]:
    """Extract case IDs from labelsTr (one label per case, no channel suffix)."""
    labels_dir = dataset_dir / "labelsTr"
    if not labels_dir.exists():
        raise FileNotFoundError(f"labelsTr not found: {labels_dir}")

    ds_json_path = dataset_dir / "dataset.json"
    file_ending = ".nii.gz"
    if ds_json_path.exists():
        with open(ds_json_path) as f:
            ds_json = json.load(f)
        file_ending = ds_json.get("file_ending", ".nii.gz")

    case_ids = sorted(f.name.replace(file_ending, "") for f in labels_dir.iterdir() if f.name.endswith(file_ending))
    return case_ids


def generate_holdout(
    case_ids: list[str],
    seed: int = 42,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
) -> dict:
    rng = np.random.RandomState(seed)
    indices = np.arange(len(case_ids))
    rng.shuffle(indices)

    n = len(case_ids)
    n_val = max(1, int(round(n * val_ratio)))
    n_test = max(1, int(round(n * test_ratio))) if test_ratio > 0 else 0

    val_idx = indices[:n_val]
    test_idx = indices[n_val : n_val + n_test]
    train_idx = indices[n_val + n_test :]

    out = {
        "train": sorted(case_ids[i] for i in train_idx),
        "val": sorted(case_ids[i] for i in val_idx),
    }
    if n_test > 0:
        out["test"] = sorted(case_ids[i] for i in test_idx)
    return out


def process_dataset(
    ds_dir: Path,
    seed: int = 42,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    out_name: str = "splits_holdout712.json",
):
    case_ids = get_case_ids(ds_dir)
    print(f"{ds_dir.name}: {len(case_ids)} cases")

    split = generate_holdout(case_ids, seed=seed, val_ratio=val_ratio, test_ratio=test_ratio)
    counts = ", ".join(f"{k}={len(v)}" for k, v in split.items())
    print(f"  {counts}")

    out_path = ds_dir / out_name
    with open(out_path, "w") as f:
        json.dump(split, f, indent=2)
    print(f"  Saved: {out_path}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate flat 7:1:2 (train/val/test) holdout splits for segmentation datasets"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root directory containing MSD Dataset* folders",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Dataset names under data_root (default: Dataset006_Lung)",
    )
    parser.add_argument(
        "--extra_dirs",
        nargs="+",
        default=None,
        help="Additional dataset directories (absolute paths)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.2,
        help="Test holdout ratio. Set 0 to produce a 2-way train/val split.",
    )
    parser.add_argument("--out_name", type=str, default="splits_holdout712.json")
    args = parser.parse_args()

    data_root = Path(args.data_root)

    if args.datasets is None:
        args.datasets = ["Dataset006_Lung"]

    for ds_name in args.datasets:
        ds_dir = data_root / ds_name
        if not ds_dir.exists():
            print(f"SKIP: {ds_name} not found at {ds_dir}")
            continue
        process_dataset(
            ds_dir,
            seed=args.seed,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            out_name=args.out_name,
        )

    if args.extra_dirs:
        for extra_dir in args.extra_dirs:
            ds_dir = Path(extra_dir)
            if not ds_dir.exists():
                print(f"SKIP: {extra_dir} not found")
                continue
            process_dataset(
                ds_dir,
                seed=args.seed,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                out_name=args.out_name,
            )


if __name__ == "__main__":
    main()
