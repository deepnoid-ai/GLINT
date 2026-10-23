# ruff: noqa: E731
import os
import re

from tqdm import tqdm

from common import load_json, logger


# Adapted from https://github.com/cuhksz-nlp/R2Gen/blob/main/modules/tokenizers.py
# (no upstream license; rights remain with the original authors)
def clean_report_mimic_cxr(report):
    report_cleaner = (
        lambda t: t.replace("\n", " ")
        .replace("__", "_")
        .replace("__", "_")
        .replace("__", "_")
        .replace("__", "_")
        .replace("__", "_")
        .replace("__", "_")
        .replace("__", "_")
        .replace("  ", " ")
        .replace("  ", " ")
        .replace("  ", " ")
        .replace("  ", " ")
        .replace("  ", " ")
        .replace("  ", " ")
        .replace("..", ".")
        .replace("..", ".")
        .replace("..", ".")
        .replace("..", ".")
        .replace("..", ".")
        .replace("..", ".")
        .replace("..", ".")
        .replace("..", ".")
        .replace("1. ", "")
        .replace(". 2. ", ". ")
        .replace(". 3. ", ". ")
        .replace(". 4. ", ". ")
        .replace(". 5. ", ". ")
        .replace(" 2. ", ". ")
        .replace(" 3. ", ". ")
        .replace(" 4. ", ". ")
        .replace(" 5. ", ". ")
        .strip()
        .lower()
        .split(". ")
    )
    sent_cleaner = lambda t: re.sub(
        r"[.,?;*!%^&_+():-\[\]{}]",
        "",
        t.replace('"', "").replace("/", "").replace("\\", "").replace("'", "").strip().lower(),
    )
    tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
    report = " . ".join(tokens) + " ."
    return report


def load_mimic_cxr_dataset(name, split, cfg):
    data_root = cfg["data_root"]
    json_path = os.path.join(data_root, cfg[name][split])
    logger.info(f"json path: {json_path}")

    input_json = load_json(json_path)

    data_list = list()

    for data in tqdm(input_json):
        if not data["findings"]:
            continue

        data_instance = dict()
        image_path = os.path.join(data_root, cfg[name]["image_root"], data["dicom_id"])
        data_instance["image"] = image_path
        data_instance["report"] = clean_report_mimic_cxr(data["findings"])

        data_list.append(data_instance)

    logger.info(f"{name}, {split}, number of loaded data: {len(data_list)}")
    return data_list
