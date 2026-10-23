import os

import cv2
import numpy as np

from common import load_json, logger

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


def polygon2mask(polygons, height, width):
    mask = np.zeros([height, width])

    for polygon in polygons:
        pts = np.asarray(polygon, dtype=np.int32)

        if pts.ndim != 2 or pts.shape[1] != 2:
            pts = pts.reshape(-1, 2).astype(np.int32)
        mask = cv2.fillPoly(mask, [pts], 1)

    return mask


def load_chestx_det_dataset(name, split, cfg):
    data_root = cfg["data_root"]
    json_path = os.path.join(data_root, cfg[name][split])
    logger.info(f"json path: {json_path}")
    input_json = load_json(json_path)

    data_list = list()
    for data in input_json:
        data_instance = dict()
        file_name = data["image_id"]
        file_path = os.path.join(data_root, cfg[name]["image_root"], file_name)

        data_instance["image"] = file_path
        data_instance["mask"] = data["masks"]
        data_list.append(data_instance)

    logger.info(f"{name}, {split}, number of loaded data: {len(data_list)}")
    return data_list
