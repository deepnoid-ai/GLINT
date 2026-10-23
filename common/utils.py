import argparse
import json
import os

from dotenv import load_dotenv


def update_nested_dict(original, updates):
    for key, value in updates.items():
        if key in original:
            if isinstance(value, dict) and isinstance(original[key], dict):
                update_nested_dict(original[key], value)
            else:
                original[key] = value
        else:
            original[key] = value


def load_json(file_path, encoding="utf-8"):
    if file_path.endswith(".json"):
        with open(file_path, "r", encoding=encoding) as f:
            return json.load(f)
    elif file_path.endswith(".jsonl"):
        with open(file_path, "r", encoding=encoding) as f:
            return [json.loads(line) for line in f if line.strip()]


def save_json(data, file_path, encoding="utf-8", indent=2):
    with open(file_path, "w", encoding=encoding) as f:
        json.dump(data, f, indent=indent)


def output_directory_setting(cfg, logger):
    if not cfg["experiment"]["output_root_dir"]:
        raise ValueError("experiment.output_root_dir is not set. Set the OUTPUT_ROOT environment variable.")
    if "dataset" in cfg and not cfg["dataset"].get("data_root"):
        raise ValueError("dataset.data_root is not set. Set the DATA_ROOT environment variable.")
    cfg["train"]["output_dir"] = os.path.join(
        cfg["experiment"]["output_root_dir"],
        cfg["experiment"]["project"],
        cfg["experiment"]["user"],
        cfg["experiment"]["name"],
    )
    logger.info(f"experiment output directory : {cfg['train']['output_dir']}")

    if cfg["args"]["no_report"] or cfg["experiment"]["user"] == "debug":
        logger.info("skip report to wandb")
        cfg["train"]["report_to"] = "none"
    elif cfg["train"]["report_to"] == "wandb":
        cfg["train"]["run_name"] = os.path.join(cfg["experiment"]["user"], cfg["experiment"]["name"])
        load_dotenv()
        os.environ["WANDB_PROJECT"] = cfg["experiment"]["project"]
        os.environ["WANDB_DIR"] = cfg["train"]["output_dir"]


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")
