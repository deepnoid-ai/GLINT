import os

from common import logger

SIIM_LABELS = ["Pneumothorax"]


def load_siim_dataset(name, split, cfg):
    data_root = cfg["data_root"]

    txt_path = os.path.join(data_root, cfg[name][split])
    logger.info(f"txt path: {txt_path}")
    with open(txt_path, "r") as f:
        file_list = f.readlines()

    data_list = []
    for file_name in file_list:
        data_instance = dict()
        file_name = file_name.replace("\n", "")

        image_path = os.path.join(data_root, cfg[name]["image_root"], file_name + ".png")
        mask_path = os.path.join(data_root, cfg[name]["mask_root"], file_name + ".png")

        data_instance["image"] = image_path
        data_instance["mask"] = mask_path
        data_list.append(data_instance)

    logger.info(f"{name}, {split}, number of loaded data: {len(data_list)}")
    return data_list
