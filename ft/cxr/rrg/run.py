import argparse
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
from ft.cxr.rrg.dataset import collate_fn, load_datasets
from ft.cxr.rrg.metrics import compute_metrics
from ft.model import load_model
from ft.trainer import BaseTrainer


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


@record
def main():
    args = parse_args()
    cfg = Config(args).config
    set_seed(cfg["train"]["seed"])

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
    logger_file_setting(
        logger,
        os.path.join(snapshot_dir, f"output_{rank}.log"),
    )

    models = load_model(cfg["model"], output_dir=cfg["train"]["output_dir"])

    dataset = load_datasets(
        cfg["dataset"],
        train=True,
        inference=cfg["args"]["inference"],
        image_processor=models["image_processor"],
    )

    if "vocabulary" in dataset:
        real_vocab_size = len(dataset["vocabulary"]["token2idx"])
        cfg["model"]["model_config"]["report_generator_config"]["vocab_size"] = real_vocab_size
        logger.info(f"Config updated with vocab_size: {real_vocab_size}")

    if "vocabulary" in dataset:
        models["tokenizer"].token2idx = dataset["vocabulary"]["token2idx"]
        models["tokenizer"].idx2token = dataset["vocabulary"]["idx2token"]

    training_args = TrainingArguments(**cfg["train"], deepspeed=deepspeed)

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
        data_collator=partial(collate_fn, tokenizer=models["tokenizer"]),
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

        models["model"].return_generated_sequence = True

        trainer.inference_load_from_checkpoint(resume_from_checkpoint=checkpoint)
        trainer.control.should_determine_best_metric = False
        trainer.control.should_save = False

        trainer.compute_metrics = partial(
            compute_metrics,
            tokenizer=models["tokenizer"],
            output_dir=snapshot_dir.replace("snapshot", "inference"),
        )
        trainer.predict(test_dataset=dataset.get("test"))


if __name__ == "__main__":
    main()
