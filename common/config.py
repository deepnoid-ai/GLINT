import os

from omegaconf import OmegaConf

from common.utils import update_nested_dict


class Config:
    def __init__(self, args):
        self.config = OmegaConf.load(args.cfg_path)
        self.config = OmegaConf.to_container(self.config, resolve=True)

        configs_dir = os.path.join(os.path.dirname(args.cfg_path), "configs")

        cfg_files = list()
        for cfg_name in args.add_cfg_list:
            if os.path.isabs(cfg_name):
                cfg_path = cfg_name
            else:
                cfg_path = os.path.join(configs_dir, cfg_name)
            if not cfg_path.endswith(".yaml"):
                cfg_path += ".yaml"

            if not os.path.exists(cfg_path):
                raise FileNotFoundError(f"No config file '{cfg_name}' under {configs_dir}")
            cfg_files.append(cfg_path)

            add_cfg = OmegaConf.load(cfg_path)
            add_cfg = OmegaConf.to_container(add_cfg, resolve=True)

            update_nested_dict(self.config, add_cfg)

        self.config.update({"args": vars(args)})
        self.config["args"]["cfg_files"] = cfg_files

        if self.config["args"].get("user"):
            self.config["experiment"]["user"] = self.config["args"].get("user")

        if self.config["args"].get("name"):
            self.config["experiment"]["name"] = self.config["args"].get("name")
