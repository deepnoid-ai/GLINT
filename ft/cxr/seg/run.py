import argparse
import hashlib
import os
from functools import partial

import torch.distributed as dist
from dotenv import load_dotenv
from torch.distributed.elastic.multiprocessing.errors import record
from transformers import EarlyStoppingCallback, TrainingArguments, set_seed
from transformers.integrations import WandbCallback

from common import Config, code_snapshot, logger, logger_file_setting, str2bool
from ft.cxr.seg.dataset import collate_fn, load_datasets
from ft.model import load_model
from ft.seg_metrics import compute_metrics
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
    parser.add_argument(
        "--inference_run_name",
        type=str,
        default=None,
        help="Wandb run name to resume for inference. If provided, the run id is derived from this name "
        "and inference results are logged to that run.",
    )

    parser.add_argument("--no_report", action="store_true", help="Skip report to wandb")
    args = parser.parse_args()

    return args


def generate_run_id_from_name(run_name):
    hash_obj = hashlib.sha256(run_name.encode("utf-8"))
    hash_hex = hash_obj.hexdigest()
    return hash_hex[:8]


def setup_cxr_seg_experiment(cfg, logger):
    if not cfg["dataset"].get("data_root"):
        raise ValueError("dataset.data_root is not set. Set the DATA_ROOT environment variable.")
    cfg["train"]["output_dir"] = os.path.join(
        cfg["experiment"]["output_root_dir"],
        cfg["experiment"]["project"],
        cfg["experiment"]["user"],
        cfg["experiment"]["name"],
    )

    logger.info(f"experiment output directory : {cfg['train']['output_dir']}")

    if (cfg["args"]["no_report"]) or cfg["experiment"]["user"] == "debug":
        logger.info("skip report to wandb")
        cfg["train"]["report_to"] = "none"
    else:
        if cfg["train"]["report_to"] == "wandb":
            cfg["train"]["run_name"] = os.path.join(cfg["experiment"]["user"], cfg["experiment"]["name"])
            load_dotenv()
            os.environ["WANDB_PROJECT"] = cfg["experiment"]["project"]
            os.environ["WANDB_DIR"] = cfg["train"]["output_dir"]

            wandb_run_id = None

            # inference-only runs reuse inference_run_name's run id; train runs switch to it after training
            inference_run_name = cfg["args"].get("inference_run_name") or cfg["experiment"].get("inference_run_name")
            if inference_run_name and cfg["args"]["inference"] and not cfg["args"]["train"]:
                wandb_run_id = generate_run_id_from_name(inference_run_name)
                logger.info(f"Generated wandb run id {wandb_run_id} from run name: {inference_run_name}")

            if not wandb_run_id and cfg["experiment"].get("resume_from_checkpoint"):
                wandb_run_id = cfg["experiment"].get("wandb_run_id")
                if wandb_run_id:
                    os.environ["WANDB_RESUME"] = "allow"
                    os.environ["WANDB_RUN_ID"] = wandb_run_id
                    logger.info(f"Resuming wandb run with id: {wandb_run_id}")
                else:
                    logger.warning(
                        "Resume from checkpoint enabled but wandb_run_id not set. Wandb logging may not work properly."
                    )

            elif wandb_run_id:
                os.environ["WANDB_RESUME"] = "allow"
                os.environ["WANDB_RUN_ID"] = wandb_run_id
                cfg["experiment"]["tags"].append("inference")
                logger.info(f"Using generated wandb run id for inference: {wandb_run_id}")

            if cfg["experiment"].get("tags"):
                os.environ["WANDB_TAGS"] = ",".join(cfg["experiment"]["tags"])


@record
def main():
    args = parse_args()
    cfg = Config(args).config
    set_seed(cfg["train"]["seed"])

    setup_cxr_seg_experiment(cfg, logger)

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
        train=cfg["args"]["train"],
        inference=cfg["args"]["inference"],
        image_processor=models.get("image_processor"),
    )

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
        data_collator=partial(collate_fn),
        cfg=cfg,
        callbacks=callbacks,
    )

    if cfg["args"]["train"]:
        logger.info("run train")
        trainer.train(resume_from_checkpoint=cfg["experiment"]["resume_from_checkpoint"])

    if cfg["args"]["inference"]:
        if cfg["args"]["train"]:
            inference_run_name = cfg["args"].get("inference_run_name") or cfg["experiment"].get("inference_run_name")
            if inference_run_name and cfg["train"]["report_to"] == "wandb":
                import wandb

                inference_run_id = generate_run_id_from_name(inference_run_name)
                if wandb.run is not None:
                    wandb.finish()
                load_dotenv()
                os.environ["WANDB_RESUME"] = "allow"
                os.environ["WANDB_RUN_ID"] = inference_run_id
                wandb.init(
                    project=cfg["experiment"]["project"],
                    id=inference_run_id,
                    resume="allow",
                    name=inference_run_name,
                    dir=cfg["train"]["output_dir"],
                )
                logger.info(f"Switched to inference wandb run id: {inference_run_id}")

        if trainer.state.best_model_checkpoint is not None:
            checkpoint = trainer.state.best_model_checkpoint
        else:
            checkpoint = cfg["experiment"]["resume_from_checkpoint"]

        models["model"].return_loss = False

        trainer.inference_load_from_checkpoint(resume_from_checkpoint=checkpoint)
        trainer.control.should_determine_best_metric = False
        trainer.control.should_save = False

        trainer.compute_metrics = partial(compute_metrics, dataset_name=cfg["dataset"]["test"][0], threshold=0.5)

        dataset_name = cfg["dataset"]["test"][0]
        predictions = trainer.predict(
            test_dataset=dataset.get("test"),
            metric_key_prefix=dataset_name.lower(),
        )

        if cfg["train"]["report_to"] == "wandb":
            import wandb

            if wandb.run is not None and trainer.is_world_process_zero():
                metrics = predictions.metrics
                wandb.log(metrics)
                logger.info(f"Logged inference metrics to wandb: {metrics}")


if __name__ == "__main__":
    main()
