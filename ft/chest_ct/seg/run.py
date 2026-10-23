import argparse
import copy
import os
from functools import partial

import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record
from transformers import EarlyStoppingCallback, TrainingArguments, set_seed
from transformers.integrations import WandbCallback

from common import (
    Config,
    code_snapshot,
    logger,
    logger_file_setting,
    output_directory_setting,
    str2bool,
)
from ft.chest_ct.seg.dataset import load_datasets, load_raw_test_cases
from ft.chest_ct.seg.dataset.data_collator import collate_fn
from ft.chest_ct.seg.inference import (
    aggregate_dice,
    run_ddp_sliding_window_inference,
)
from ft.model import load_model
from ft.trainer import BaseTrainer
from glint.zs_eval.distribution_utils import is_main_process


def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument(
        "--cfg-path",
        default=os.path.join(os.path.dirname(__file__), "config.yaml"),
        help="path to configuration file.",
    )
    parser.add_argument(
        "--add_cfg_list",
        default=[],
        type=str,
        nargs="+",
        help="List of YAML files. The cfg will be overwritten in the given order.",
    )
    parser.add_argument(
        "--user",
        type=str,
        default=None,
        help="User. This argument, when provided, overrides any previous settings.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Name of experiment. This argument, when provided, overrides any previous settings.",
    )
    parser.add_argument(
        "--train",
        default=True,
        type=str2bool,
        choices=[True, False],
        help="Set the train mode to True or False",
    )
    parser.add_argument(
        "--inference",
        default=True,
        type=str2bool,
        choices=[True, False],
        help="Set the inference mode to True or False",
    )
    parser.add_argument("--no_report", action="store_true", help="Skip report to wandb")
    args = parser.parse_args()

    return args


def run_single(cfg):
    output_directory_setting(cfg, logger)

    deepspeed = dict(cfg["deepspeed"]) if cfg.get("deepspeed") else None
    training_args = TrainingArguments(**cfg["train"], deepspeed=deepspeed)

    snapshot_dir = None
    if not dist.is_initialized() or dist.get_rank() == 0:
        snapshot_dir = code_snapshot(save_dir=os.path.join(cfg["train"]["output_dir"], "snapshot"), cfg=cfg)
    if dist.is_available() and dist.is_initialized():
        obj_list = [snapshot_dir]
        dist.broadcast_object_list(obj_list, src=0)
        snapshot_dir = obj_list[0]

    rank = dist.get_rank() if dist.is_initialized() else 0
    logger_file_setting(logger, os.path.join(snapshot_dir, f"output_{rank}.log"))

    models = load_model(cfg["model"], output_dir=cfg["train"]["output_dir"])

    dataset = load_datasets(
        cfg["dataset"],
        train=cfg["args"]["train"],
        inference=cfg["args"]["inference"],
    )

    callbacks = []
    if cfg["train"].get("eval_strategy") != "no":
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=cfg["experiment"]["early_stopping_patience"]))
    if cfg["train"]["report_to"] == "wandb":
        callbacks.append(WandbCallback())

    trainer = BaseTrainer(
        model=models["model"],
        args=training_args,
        train_dataset=dataset.get("train"),
        eval_dataset=dataset.get("eval"),
        data_collator=partial(collate_fn),
        cfg=cfg,
        callbacks=callbacks,
    )

    if cfg["args"]["train"]:
        logger.info("run train")
        trainer.train(resume_from_checkpoint=cfg["experiment"]["resume_from_checkpoint"])

    if cfg["args"]["inference"]:
        if trainer.state.best_model_checkpoint is not None:
            checkpoint = trainer.state.best_model_checkpoint
        else:
            checkpoint = cfg["experiment"]["resume_from_checkpoint"]

        models["model"].return_loss = False
        trainer.inference_load_from_checkpoint(resume_from_checkpoint=checkpoint)

        # roi_size = each encoder's patch_size; Invertd maps predictions to raw NIfTI space for a fair Dice
        ds_cfg = cfg["dataset"]["datasets"][0]
        ds_name = cfg["dataset"].get("matrix_entry_name") or ds_cfg.get("name", "dataset")
        num_classes = cfg["model"]["model_config"]["decoder_config"]["num_classes"]
        test_cases = load_raw_test_cases(ds_cfg, data_root=cfg["dataset"].get("data_root"))

        ds_conf = cfg["dataset"]
        viz_cfg = cfg.get("visualization", {})
        per_case = run_ddp_sliding_window_inference(
            model=models["model"],
            data_dicts=test_cases,
            num_classes=num_classes,
            spacing=tuple(ds_conf["spacing"]),
            roi_size=tuple(ds_conf["patch_size"]),
            orientation=ds_conf.get("orientation", "RAS"),
            a_min=ds_conf.get("a_min", -175.0),
            a_max=ds_conf.get("a_max", 250.0),
            b_min=ds_conf.get("b_min", 0.0),
            b_max=ds_conf.get("b_max", 1.0),
            sw_batch_size=ds_conf.get("sw_batch_size", 4),
            overlap=ds_conf.get("overlap", 0.5),
            save_predictions_dir=os.path.join(cfg["train"]["output_dir"], "predictions", ds_name),
            save_panels_dir=os.path.join(cfg["train"]["output_dir"], "panels", ds_name),
            panel_max_slices=viz_cfg.get("panel_max_slices"),
            panel_window_center=viz_cfg.get("window_center", 40.0),
            panel_window_width=viz_cfg.get("window_width", 400.0),
            regions=ds_conf.get("regions"),
        )

        if is_main_process() and per_case:
            metrics = aggregate_dice(per_case, dataset_name=ds_name)
            if cfg["train"]["report_to"] == "wandb":
                try:
                    import wandb

                    if wandb.run is not None:
                        wandb.log({f"test/{k}": v for k, v in metrics.items()})
                except ImportError:
                    pass

    # Close the wandb run so the next matrix iteration starts a fresh one.
    if cfg["train"]["report_to"] == "wandb":
        try:
            import wandb

            if wandb.run is not None:
                wandb.finish()
        except ImportError:
            pass


def _apply_matrix_entry(base_cfg: dict, entry: dict, base_name: str) -> dict:
    """Sub-config for one datasets_matrix entry (name, num_classes, dataset_entry, *_overrides, regions)."""
    sub = copy.deepcopy(base_cfg)
    sub["experiment"]["name"] = f"{base_name}_{entry['name']}"
    sub["model"]["model_config"]["decoder_config"]["num_classes"] = entry["num_classes"]
    sub["dataset"]["datasets"] = [entry["dataset_entry"]]
    # stash the short name so run_single uses it for metric keys
    sub["dataset"]["matrix_entry_name"] = entry["name"]

    for k, v in entry.get("train_overrides", {}).items():
        sub["train"][k] = v
    for k, v in entry.get("dataset_overrides", {}).items():
        sub["dataset"][k] = v
    for k, v in entry.get("experiment_overrides", {}).items():
        sub["experiment"][k] = v
    if entry.get("viz_overrides"):
        sub.setdefault("visualization", {}).update(entry["viz_overrides"])
    if entry.get("regions"):
        sub["dataset"]["regions"] = entry["regions"]

    return sub


@record
def main():
    args = parse_args()
    cfg = Config(args).config
    set_seed(cfg["train"]["seed"])

    matrix = cfg.pop("datasets_matrix", None)

    if not matrix:
        run_single(cfg)
        return

    base_name = cfg["experiment"]["name"]
    logger.info(f"datasets_matrix detected ({len(matrix)} entries) — running sequentially")
    for i, entry in enumerate(matrix):
        ds_label = entry["dataset_entry"].get("name", entry["name"])
        logger.info(f"[matrix {i + 1}/{len(matrix)}] {base_name}_{entry['name']} — dataset={ds_label}")
        sub_cfg = _apply_matrix_entry(cfg, entry, base_name)
        run_single(sub_cfg)


if __name__ == "__main__":
    main()
