import os

import pandas as pd
from tqdm import tqdm

from common import load_json, logger
from ft.chest_ct.processed_volumes import processed_volume_path

LIDC_LABELS = [
    "nodules",
]


def _build_lidc_dicom_path_map(data_root, metadata_csv):
    """Build SeriesUID -> DICOM directory path mapping from metadata CSV."""
    metadata_csv_path = os.path.join(data_root, metadata_csv)
    metadata_csv_dir = os.path.dirname(metadata_csv_path)

    df = pd.read_csv(metadata_csv_path, usecols=["Series UID", "File Location"])
    path_map = {}
    for _, row in df.iterrows():
        series_uid = row["Series UID"]
        file_loc = row["File Location"]
        if file_loc.startswith("./"):
            file_loc = file_loc[2:]
        path_map[series_uid] = os.path.join(metadata_csv_dir, file_loc)
    return path_map


def load_lidc_pe_dataset(name, split, cfg):
    data_root = cfg["data_root"]
    json_path = os.path.join(data_root, cfg[name][split])
    logger.info(f"json path: {json_path}")

    input_json = load_json(json_path)

    dicom_path_map = _build_lidc_dicom_path_map(data_root, cfg[name]["metadata_csv"])

    data_list = list()
    for data in tqdm(input_json):
        data_instance = dict()

        metadata = data.get("metadata", {})
        series_uid = metadata.get("SeriesInstanceUID", "")
        if series_uid and series_uid in dicom_path_map:
            data_instance["3d_raw_image_path"] = dicom_path_map[series_uid]
        processed_path = processed_volume_path(
            cfg, name, os.path.join(metadata.get("PatientID", ""), metadata.get("StudyInstanceUID", ""), series_uid)
        )
        if processed_path:
            data_instance["3d_image_path"] = processed_path

        data_instance["labels"] = [int(data["labels"][key]) for key in LIDC_LABELS]
        data_list.append(data_instance)

    logger.info(f"{name}, {split}, number of loaded data: {len(data_list)}")
    return data_list
