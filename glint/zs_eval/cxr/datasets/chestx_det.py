import os

from common import load_json

CHESTX_DET_LABELS = [
    "Atelectasis",
    "Calcification",
    "Cardiomegaly",
    "Consolidation",
    "Diffuse Nodule",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Fracture",
    "Mass",
    "Nodule",
    "Pleural Thickening",
    "Pneumothorax",
]

CHESTX_DET_EVAL_FINDINGS = CHESTX_DET_LABELS


def load_chestx_det_json(data_dir, image_dir=None):
    base_image_dir = image_dir if image_dir is not None else data_dir

    splits = {
        "train": os.path.join(data_dir, "ChestX_Det_train.json"),
        "test": os.path.join(data_dir, "ChestX_Det_test.json"),
    }

    out = []
    for split, json_path in splits.items():
        data_list = load_json(json_path)
        for data in data_list:
            syms = data["syms"]
            polygons = data["polygons"]
            masks_by_finding = {label: [] for label in CHESTX_DET_LABELS}
            for sym, poly in zip(syms, polygons):
                if sym and sym in masks_by_finding:
                    masks_by_finding[sym].append(poly)
            out.append(
                {
                    "image_path": os.path.join(base_image_dir, split, data["file_name"]),
                    "masks": masks_by_finding,
                    "label": syms,
                }
            )
    return out
