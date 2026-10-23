import argparse
import os
from functools import partial

import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record
from transformers import EarlyStoppingCallback, TrainingArguments, set_seed

from common import (
    Config,
    code_snapshot,
    logger,
    logger_file_setting,
    output_directory_setting,
    str2bool,
)
from glint.dataset import collate_fn, load_datasets
from glint.model import load_model
from glint.trainer import GradientAccumulationTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="GLINT training / zero-shot inference")

    parser.add_argument(
        "--cfg-path",
        default=os.path.join(os.path.dirname(__file__), "config.yaml"),
        help="path to base configuration file.",
    )
    parser.add_argument(
        "--add_cfg_list",
        default=[],
        type=str,
        nargs="+",
        help="List of YAML overlays. Applied left-to-right; later configs win on conflict.",
    )
    parser.add_argument("--user", type=str, default=None, help="Override experiment.user.")
    parser.add_argument("--name", type=str, default=None, help="Override experiment.name.")
    parser.add_argument("--train", default=True, type=str2bool, choices=[True, False])
    parser.add_argument("--inference", default=True, type=str2bool, choices=[True, False])
    parser.add_argument("--no_report", action="store_true", help="Skip wandb reporting.")
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Checkpoint for inference: a local checkpoint directory or a Hugging Face repo id "
        "(e.g. Deepnoid/GLINT-CXR).",
    )
    return parser.parse_args()


def resolve_checkpoint(ckpt):
    if not isinstance(ckpt, str) or os.path.exists(ckpt):
        return ckpt
    from huggingface_hub import snapshot_download

    logger.info(f"Downloading checkpoint from Hugging Face Hub: {ckpt}")
    return snapshot_download(repo_id=ckpt, allow_patterns=["model.safetensors", "*.json", "vocab.txt"])


@record
def main():
    args = parse_args()
    cfg = Config(args).config
    set_seed(cfg["train"]["seed"])

    output_directory_setting(cfg, logger)

    deepspeed = dict(cfg["deepspeed"]) if cfg.get("deepspeed") else None

    if not cfg["args"]["train"]:
        cfg["train"]["eval_strategy"] = "no"
        cfg["train"]["save_strategy"] = "no"
        cfg["train"]["load_best_model_at_end"] = False
        cfg["model"]["model_config"].setdefault("dfr_config", {})["use_cache"] = True
        if cfg["args"].get("ckpt"):
            # build straight from the checkpoint's config.json: no base-encoder downloads
            cfg["model"]["pretrained_ckpt"] = resolve_checkpoint(cfg["args"]["ckpt"])

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
        train=cfg["args"]["train"],
        inference=False,
        image_processor=models.get("image_processor"),
    )

    callbacks = []
    if cfg["train"].get("eval_strategy") != "no":
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=cfg["experiment"]["early_stopping_patience"]))

    trainer = GradientAccumulationTrainer(
        model=models["model"],
        args=training_args,
        train_dataset=dataset.get("train"),
        eval_dataset=dataset.get("eval"),
        data_collator=partial(collate_fn, tokenizer=models["tokenizer"]),
        cfg=cfg,
        callbacks=callbacks,
    )

    trainer.can_return_loss = trainer.model.config.can_return_loss

    if cfg["args"]["train"]:
        logger.info("run train")
        trainer.train(resume_from_checkpoint=cfg["experiment"]["resume_from_checkpoint"])

    if cfg["args"]["inference"]:
        ckpt_src = (
            trainer.state.best_model_checkpoint
            or cfg["args"].get("ckpt")
            or cfg["experiment"]["resume_from_checkpoint"]
        )
        checkpoint = cfg["model"].get("pretrained_ckpt") or resolve_checkpoint(ckpt_src)
        resume_from_checkpoint = trainer.inference_load_from_checkpoint(resume_from_checkpoint=checkpoint)

        if resume_from_checkpoint is None:
            if cfg["args"]["train"]:
                logger.info("Inference: no checkpoint to load (training produced none); skipping inference.")
                return
            raise ValueError(
                "Inference requires a checkpoint. Pass --ckpt <local checkpoint dir or HF repo id>, "
                "e.g. --ckpt Deepnoid/GLINT-CXR."
            )

        models["model"].to(trainer.args.device)
        dtype_str = cfg["model"]["dtype"]
        dtype = getattr(torch, dtype_str.removeprefix("torch.")) if isinstance(dtype_str, str) else dtype_str
        models["model"].to(dtype)
        models["model"].eval()

        data_root_dir = cfg["dataset"]["data_root"]
        save_root_dir = os.path.join(
            cfg["train"]["output_dir"],
            "inference",
            os.path.basename(snapshot_dir),
            os.path.basename(resume_from_checkpoint),
        )

        logger.info(
            f"Inference: modality={cfg['dataset']['dataset_type']}, ckpt={resume_from_checkpoint}, "
            f"dtype={dtype}, world_size={trainer.args.world_size}, save_root={save_root_dir}"
        )

        if trainer.is_world_process_zero():
            save_path = os.path.join(cfg["train"]["output_dir"], "vision_model")
            models["model"].vision_model.save_pretrained(save_path)
            if "image_processor" in models:
                models["image_processor"].save_pretrained(save_path)

        if cfg["dataset"]["dataset_type"] == "chest_ct":
            from glint.zs_eval.chest_ct.inference import Inference as CTInference

            inference = CTInference(
                cls_dataset=cfg["dataset"]["test"],
                det_dataset=None,
                seg_dataset=cfg["dataset"].get("seg_dataset", []),
                data_root_dir=data_root_dir,
                cfg=cfg,
            )
            logger.info("Zero-shot CT classification...")
            for optimize_thrsh in [None, "test"]:
                inference.classification(
                    model=models["model"],
                    tokenizer=models["tokenizer"],
                    save_root_dir=save_root_dir,
                    binary=False,
                    optimize_thrsh=optimize_thrsh,
                )
            if inference.seg_dataset:
                logger.info("Zero-shot CT segmentation...")
                inference.segmentation(
                    model=models["model"],
                    tokenizer=models["tokenizer"],
                    save_root_dir=save_root_dir,
                    threshold_step=cfg["dataset"].get("threshold_step", 0.01),
                )

        elif cfg["dataset"]["dataset_type"] == "cxr":
            from glint.zs_eval.cxr.inference import Inference as CXRInference
            from glint.zs_eval.cxr.inference import resolve_dataset_paths

            dataset_paths = resolve_dataset_paths(cfg["inference"]["dataset_paths"], data_root_dir)
            inference = CXRInference(
                cls_dataset=cfg["inference"]["cls_dataset"],
                det_dataset=cfg["inference"]["det_dataset"],
                seg_dataset=cfg["inference"]["seg_dataset"],
                dataset_paths=dataset_paths,
                batch_size=cfg["inference"]["batch_size"],
                num_workers=cfg["inference"]["num_workers"],
            )
            inference.classification(**models, save_root_dir=os.path.join(save_root_dir, "classification"))
            inference.grounding(**models, save_root_dir=os.path.join(save_root_dir, "grounding"))
            inference.segmentation(
                **models,
                save_root_dir=os.path.join(save_root_dir, "segmentation"),
                compute_pixel_level_auroc=cfg["inference"].get("compute_pixel_level_auroc", False),
                threshold_step=cfg["inference"].get("threshold_step", 0.01),
            )
            vis_cfg = cfg["inference"].get("visualization", {}) or {}
            if vis_cfg.get("datasets"):
                inference.visualization(
                    **models,
                    save_root_dir=os.path.join(save_root_dir, "visualization"),
                    vis_dataset=vis_cfg["datasets"],
                    max_samples_per_finding=vis_cfg.get("max_samples_per_finding"),
                )


if __name__ == "__main__":
    main()
