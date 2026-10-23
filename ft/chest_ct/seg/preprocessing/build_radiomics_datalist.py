"""Write a 7:1:2 train/val/test datalist pairing each NSCLC-Radiomics CT with its mask_2class label."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
from pathlib import Path

import numpy as np

DEFAULT_LABEL_NAME = "mask_2class.nii.gz"
DEFAULT_OUT = "nsclc_radiomics_datalist_splits.json"

logger = logging.getLogger("build_radiomics_datalist")


def collect_entries(root: Path, label_name: str) -> list[dict]:
    """One entry per ``lung_XXX/`` dir that has both CT.nii.gz and the requested label."""
    entries: list[dict] = []
    for pdir in sorted(p for p in root.glob("lung_*") if p.is_dir()):
        ct_path = pdir / "CT.nii.gz"
        lbl_path = pdir / label_name
        if not ct_path.exists() or not lbl_path.exists():
            logger.debug("skip %s: missing CT or label", pdir.name)
            continue
        entries.append(
            {
                "pid": pdir.name,
                "image": str(ct_path),
                "label": str(lbl_path),
            }
        )
    logger.info("Collected %d entries from %s", len(entries), root)
    return entries


def split_712(entries: list[dict], seed: int) -> dict[str, list[dict]]:
    rng = np.random.RandomState(seed)
    idx = np.arange(len(entries))
    rng.shuffle(idx)

    n = len(entries)
    n_val = max(1, int(round(n * 0.1)))
    n_test = max(1, int(round(n * 0.2)))

    val_idx = idx[:n_val]
    test_idx = idx[n_val : n_val + n_test]
    train_idx = idx[n_val + n_test :]

    by_pid = lambda ii: sorted((entries[i] for i in ii), key=lambda e: e["pid"])  # noqa: E731
    return {
        "train": by_pid(train_idx),
        "val": by_pid(val_idx),
        "test": by_pid(test_idx),
    }


def assign_case_ids(splits: dict[str, list[dict]]) -> None:
    cid = 0
    for key in ("train", "val", "test"):
        for entry in splits[key]:
            entry["case_id"] = f"nsclc_{cid:04d}"
            cid += 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True, help="output dir of convert_radiomics.py")
    parser.add_argument(
        "--label_name",
        default=DEFAULT_LABEL_NAME,
        help="Label filename inside each lung_XXX dir (default: mask_2class.nii.gz)",
    )
    parser.add_argument("--out_name", default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level,
        format="[%(asctime)s %(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    entries = collect_entries(args.root, args.label_name)
    if not entries:
        logger.error("No entries collected. Exiting.")
        return 1

    splits = split_712(entries, seed=args.seed)
    assign_case_ids(splits)

    out_path = args.root / args.out_name
    payload = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "n_scans": len(entries),
        "label_name": args.label_name,
        "seed": args.seed,
        "splits": {k: len(v) for k, v in splits.items()},
        "train": splits["train"],
        "val": splits["val"],
        "test": splits["test"],
    }
    out_path.write_text(json.dumps(payload, indent=2))

    logger.info(
        "Wrote %s  n=%d  train=%d val=%d test=%d",
        out_path,
        len(entries),
        len(splits["train"]),
        len(splits["val"]),
        len(splits["test"]),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
