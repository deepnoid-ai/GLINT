import os

from common import load_json


def build_ct_rate_raw_path(data_root_dir, image_id, raw_root="CT-RATE/dataset"):
    spl, acc_idx, alphb, vol_idx = image_id.split("_")
    return os.path.join(
        data_root_dir,
        raw_root,
        f"{spl}_fixed",
        f"{spl}_{acc_idx}",
        f"{spl}_{acc_idx}_{alphb}",
        f"{image_id}.nii.gz",
    )


def load_rexgrounding_dataset(data_root_dir, cfg, split="val"):
    rex_cfg = cfg["dataset"]["ReXGroundingCT"]
    json_path = os.path.join(data_root_dir, rex_cfg[split])
    raw_data = load_json(json_path)

    if isinstance(raw_data, dict) and split in raw_data:
        raw_data = raw_data[split]

    seg_mask_root = os.path.join(data_root_dir, rex_cfg["segmentation_root"])
    raw_root = rex_cfg.get("ct_rate_raw_root", "CT-RATE/dataset")

    data_list = []
    for entry in raw_data:
        image_id = entry["name"].replace(".nii.gz", "")
        raw_path = build_ct_rate_raw_path(data_root_dir, image_id, raw_root=raw_root)
        mask_path = os.path.join(seg_mask_root, entry["name"])

        data_list.append(
            {
                "image_id": image_id,
                "raw_path": raw_path,
                "mask_path": mask_path,
                "findings": entry["findings"],
                "shape": tuple(entry["shape"]),
            }
        )

    return data_list
