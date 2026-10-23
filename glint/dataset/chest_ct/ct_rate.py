import os

from tqdm import tqdm

from common import load_json, logger


def load_ct_rate_dataset(name, split, cfg):
    data_root = cfg["data_root"]
    json_path = os.path.join(data_root, cfg[name][split])
    logger.info(f"json path: {json_path}")
    input_json = load_json(json_path)

    data_list = []
    for data in tqdm(input_json):
        if not data.get("report_decomposition"):
            continue

        image_path = data["image_path"]
        if not os.path.isabs(image_path):
            image_path = os.path.join(data_root, image_path)

        data_list.append(
            {
                "image_id": data["image_id"],
                "3d_raw_image_path": image_path,
                "sentences": [s for s in dict.fromkeys(data["report_decomposition"]) if s],
            }
        )

    logger.info(f"{name} / {split}: {len(data_list)} samples")
    return data_list
