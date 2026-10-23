import os

from tqdm import tqdm

from common import load_json, logger
from ft.chest_ct.processed_volumes import processed_volume_path

RSNA_PE_LABELS = [
    "pulmonary_embolism",
]


def load_rsna_pe_dataset(name, split, cfg):
    data_root = cfg["data_root"]
    json_path = os.path.join(data_root, cfg[name][split])
    logger.info(f"json path: {json_path}")

    input_json = load_json(json_path)

    data_list = list()
    for data in tqdm(input_json):
        data_instance = dict()

        # image_id = "{study}/{series}"; all splits live under the competition train/ DICOM dir
        data_instance["3d_raw_image_path"] = os.path.join(data_root, cfg[name]["raw_root"], data["image_id"])
        processed_path = processed_volume_path(cfg, name, data["image_id"])
        if processed_path:
            data_instance["3d_image_path"] = processed_path

        data_instance["labels"] = [int(data["labels"][key]) for key in RSNA_PE_LABELS]
        data_list.append(data_instance)

    logger.info(f"{name}, {split}, number of loaded data: {len(data_list)}")
    return data_list
