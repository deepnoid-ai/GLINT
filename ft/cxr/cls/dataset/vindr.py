import os

from common import load_json, logger

VINDR_LABELS = [
    "Aortic enlargement",
    "Atelectasis",
    "Calcification",
    "Cardiomegaly",
    "Clavicle fracture",
    "Consolidation",
    # "Edema" is excluded because it is absent from the test split
    "Emphysema",
    "Enlarged PA",
    "ILD",
    "Infiltration",
    "Lung Opacity",
    "Lung cavity",
    "Lung cyst",
    "Mediastinal shift",
    "Nodule/Mass",
    "Pleural effusion",
    "Pleural thickening",
    "Pneumothorax",
    "Pulmonary fibrosis",
    "Rib fracture",
    "Other lesion",
    "COPD",
    "Lung tumor",
    "Pneumonia",
    "Tuberculosis",
    "Other diseases",
    "No finding",
]


def load_vindr_dataset(name, split, cfg):
    data_root = cfg["data_root"]
    json_path = os.path.join(data_root, cfg[name][split])
    logger.info(f"json path: {json_path}")
    input_json = load_json(json_path)

    data_list = list()
    for data in input_json:
        data_instance = dict()
        file_name = data["image_id"]
        file_path = os.path.join(data_root, cfg[name]["image_root"], f"{file_name}.png")

        data_instance["image"] = file_path
        data_instance["labels"] = [int(data["labels"][key]) for key in VINDR_LABELS]
        data_list.append(data_instance)

    logger.info(f"{name}, {split}, number of loaded data: {len(data_list)}")
    return data_list
