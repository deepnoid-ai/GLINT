import datetime
import json
import os
import shutil

from git import Repo
from omegaconf import OmegaConf


def _create_snapshot(save_dir, cfg):
    now = datetime.datetime.now()
    now_dir = "run-%s" % now.strftime("%m%d-%H%M%S")
    snapshot_dir = os.path.join(save_dir, now_dir)
    os.makedirs(snapshot_dir, exist_ok=True)

    try:
        repo = Repo(os.getcwd())
        diff = repo.git.diff()
        with open(os.path.join(snapshot_dir, "diff.diff"), "w") as f:
            f.write(diff)
            f.write("\n")

        last_commit = list(repo.iter_commits())[0]
        last_commit_info = {
            "id": last_commit.hexsha,
            "committed_datetime": str(last_commit.committed_datetime),
            "summary": last_commit.summary,
            "name_rev": last_commit.name_rev,
        }
        with open(os.path.join(snapshot_dir, "last_commit.json"), "w") as f:
            json.dump(last_commit_info, f, indent=2)
    except Exception:  # git metadata is best-effort; run without it outside a git checkout
        pass

    if cfg:
        OmegaConf.save(cfg, f=os.path.join(snapshot_dir, "config.yaml"))
        cfg_files = cfg["args"].get("cfg_files")
        if cfg_files:
            for cfg_file in cfg_files:
                filename = os.path.join(snapshot_dir, "cfg_files", "/".join(cfg_file.split("/")[-2:]))
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                shutil.copy(cfg_file, filename)

    return snapshot_dir


def code_snapshot(save_dir, cfg=None, distributed=False):
    if not distributed:
        return _create_snapshot(save_dir, cfg)

    import torch.distributed as dist

    snapshot_dir = None
    if dist.get_rank() == 0:
        snapshot_dir = _create_snapshot(save_dir, cfg)

    obj_list = [snapshot_dir]
    dist.broadcast_object_list(obj_list, src=0)
    snapshot_dir = obj_list[0]
    dist.barrier()

    return snapshot_dir
