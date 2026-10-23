import os

from common import load_json, logger

RSNA_LABELS = ["Pneumonia"]


def load_rsna_dataset(name, split, cfg):
    data_root = cfg["data_root"]
    json_path = os.path.join(data_root, cfg[name][split])
    logger.info(f"json path: {json_path}")
    input_json = load_json(json_path)

    data_list = list()
    for data in input_json:
        data_instance = dict()
        file_name = data["image_id"]
        file_path = os.path.join(data_root, cfg[name]["image_root"], file_name + ".jpg")

        data_instance["image"] = file_path
        data_instance["labels"] = [int(data["labels"][key]) for key in RSNA_LABELS]
        data_list.append(data_instance)

    logger.info(f"{name}, {split}, number of loaded data: {len(data_list)}")
    return data_list
