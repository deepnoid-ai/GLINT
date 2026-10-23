import json
import os
import shutil

from omegaconf import OmegaConf


def update_nested_dict(original, updates):
    for key, value in updates.items():
        if key in original and isinstance(value, dict) and isinstance(original[key], dict):
            update_nested_dict(original[key], value)
        else:
            original[key] = value


def load_json(file_path):
    with open(file_path, encoding="utf-8") as f:
        return json.load(f)


def save_json(data, file_path, indent=2):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)


def load_config(config_list, config_dir=None):
    if config_dir is None:
        config_dir = os.path.join(os.path.dirname(__file__), "configs")

    config = OmegaConf.load(os.path.join(config_dir, "common.yaml"))
    config = OmegaConf.to_container(config, resolve=True)

    cfg_files = []

    common_path = os.path.join(config_dir, "common.yaml")
    if os.path.exists(common_path):
        cfg_files.append(common_path)

    for config_name in config_list:
        config_path = os.path.join(config_dir, config_name)
        if not config_name.endswith(".yaml"):
            config_path += ".yaml"

        if os.path.exists(config_path):
            add_config = OmegaConf.load(config_path)
            add_config = OmegaConf.to_container(add_config, resolve=True)
            update_nested_dict(config, add_config)
            cfg_files.append(config_path)
        else:
            raise FileNotFoundError(f"Config file {config_name} not found")

    config["args"] = {
        "cfg_files": cfg_files,
    }

    return config


def save_snapshot_prompt(save_dir, config):
    snapshot_dir = os.path.join(save_dir, "snapshot")
    os.makedirs(snapshot_dir, exist_ok=True)
    OmegaConf.save(OmegaConf.create(config), os.path.join(snapshot_dir, "config.yaml"))
    shutil.copytree(
        os.path.join(os.path.dirname(__file__), "prompts"), os.path.join(snapshot_dir, "prompts"), dirs_exist_ok=True
    )
    print(f"Saved snapshot to: {snapshot_dir}")
    return snapshot_dir


def get_step_file_path(intermediate_save_dir, step_idx, step_name, suffix):
    return os.path.join(intermediate_save_dir, f"{step_idx:02d}_{step_name}_{suffix}.json")
