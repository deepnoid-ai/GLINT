import os

from tqdm import tqdm

from common import load_json, logger
from ft.chest_ct.processed_volumes import processed_volume_path

except_files = [
    "train_1267_a_4",
    "train_11755_a_3",
    "train_11755_a_4",
    "train_8451_a_3",
    "train_12477_a_1",
    "train_13442_a_1",
    "train_15637_a_2",
]

CT_RATE_LABELS = [
    "Medical material",
    "Arterial wall calcification",
    "Cardiomegaly",
    "Pericardial effusion",
    "Coronary artery wall calcification",
    "Hiatal hernia",
    "Lymphadenopathy",
    "Emphysema",
    "Atelectasis",
    "Lung nodule",
    "Lung opacity",
    "Pulmonary fibrotic sequela",
    "Pleural effusion",
    "Mosaic attenuation pattern",
    "Peribronchial thickening",
    "Consolidation",
    "Bronchiectasis",
    "Interlobular septal thickening",
]


def load_ct_rate_dataset(name, split, cfg):
    data_root = cfg["data_root"]
    json_path = os.path.join(data_root, cfg[name][split])
    logger.info(f"json path: {json_path}")

    input_json = load_json(json_path)

    raw_root = cfg[name].get("raw_root", "CT-RATE/dataset")

    data_list = list()
    for data in tqdm(input_json):
        data_instance = dict()
        image_id = data["image_id"]
        spl, patientID, scanID, reconID = image_id.replace(".nii.gz", "").split("_")

        # CT-RATE layout: <raw_root>/<spl>_fixed/<spl>_<pid>/<spl>_<pid>_<scan>/<image_id>
        raw_file_path = os.path.join(
            data_root,
            raw_root,
            f"{spl}_fixed",
            f"{spl}_{patientID}",
            f"{spl}_{patientID}_{scanID}",
            image_id,
        )

        # except_files entries are full IDs with the recon suffix (e.g. ``train_12477_a_1``)
        if image_id.replace(".nii.gz", "") in except_files:
            continue

        if "report_decomposition" not in data.keys():
            continue

        data_instance["3d_raw_image_path"] = raw_file_path
        scan_stem = f"{spl}/{spl}_{patientID}/{spl}_{patientID}_{scanID}/{image_id.replace('.nii.gz', '')}"
        processed_path = processed_volume_path(cfg, name, scan_stem)
        if processed_path:
            data_instance["3d_image_path"] = processed_path
        data_instance["labels"] = [int(data["labels"][key]) for key in CT_RATE_LABELS]
        data_instance["report"] = data["Findings_EN"] + data["Impressions_EN"]

        data_list.append(data_instance)

    logger.info(f"{name}, {split}, number of loaded data: {len(data_list)}")
    return data_list
