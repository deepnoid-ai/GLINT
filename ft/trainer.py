"""Adapted from Hugging Face Transformers v4.53.0, src/transformers/trainer.py
(https://github.com/huggingface/transformers/blob/v4.53.0/src/transformers/trainer.py).
Copyright 2020-present the HuggingFace Inc. team. Apache License 2.0 (LICENSES/Apache-2.0.txt)
"""

from typing import Any, Optional, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from packaging import version
from transformers import Trainer
from transformers.trainer import (
    get_last_checkpoint,
    is_sagemaker_mp_enabled,
    logger,
    nested_detach,
)
from transformers.utils import XLA_FSDPV2_MIN_VERSION, is_torch_xla_available

if is_sagemaker_mp_enabled():
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from transformers.trainer_pt_utils import (
        smp_forward_only,
        smp_nested_concat,
    )
else:
    IS_SAGEMAKER_MP_POST_1_10 = False

if is_torch_xla_available():
    from torch_xla import __version__ as XLA_VERSION

    IS_XLA_FSDPV2_POST_2_2 = version.parse(XLA_VERSION) >= version.parse(XLA_FSDPV2_MIN_VERSION)
    if IS_XLA_FSDPV2_POST_2_2:
        pass
else:
    IS_XLA_FSDPV2_POST_2_2 = False


class BaseTrainer(Trainer):
    def __init__(
        self,
        model=None,
        args=None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        processing_class=None,
        model_init=None,
        compute_loss_func=None,
        compute_metrics=None,
        callbacks=None,
        optimizers=(None, None),
        optimizer_cls_and_kwargs=None,
        preprocess_logits_for_metrics=None,
        **kwargs,
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            model_init=model_init,
            compute_loss_func=compute_loss_func,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            optimizer_cls_and_kwargs=optimizer_cls_and_kwargs,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        self.cfg = kwargs["cfg"]
        # eval component-loss sums, reset and averaged in evaluate()
        self._eval_loss_component_sums: dict[str, float] = {}
        self._eval_loss_component_count: int = 0

    # short wandb keys, e.g. "dice_loss" rather than "diceloss_loss"
    _LOSS_NAME_ALIASES = {
        "DiceLoss": "dice",
        "CrossEntropyLoss": "ce",
        "BCEwithLogitsLoss": "bce",
    }

    @classmethod
    def _loss_short_name(cls, name: str) -> str:
        """Class-name → short key. Falls back to stripping a trailing ``Loss``."""
        if name in cls._LOSS_NAME_ALIASES:
            return cls._LOSS_NAME_ALIASES[name]
        short = name[:-4] if name.endswith("Loss") else name
        return short.lower()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Also logs the model's per-component `losses` (train: every logging_steps; eval: averaged in evaluate())."""
        outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        losses_dict = outputs["losses"] if isinstance(outputs, dict) and "losses" in outputs else None

        if losses_dict is not None:
            if model.training:
                if (
                    self.args.logging_steps
                    and self.state.global_step > 0
                    and self.state.global_step % int(self.args.logging_steps) == 0
                ):
                    # no "train/" prefix: WandbCallback's rewrite_logs adds it
                    payload = {f"{self._loss_short_name(k)}_loss": float(v.item()) for k, v in losses_dict.items()}
                    if payload:
                        self.log(payload)
            else:
                for k, v in losses_dict.items():
                    self._eval_loss_component_sums[k] = self._eval_loss_component_sums.get(k, 0.0) + float(v.item())
                self._eval_loss_component_count += 1

        return (loss, outputs) if return_outputs else loss

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        self._eval_loss_component_sums = {}
        self._eval_loss_component_count = 0

        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        if self._eval_loss_component_count > 0:
            # underscore, not slash: rewrite_logs maps "eval_x" to "eval/x" (a slash would nest under "train/")
            payload = {
                f"{metric_key_prefix}_{self._loss_short_name(k)}_loss": total / self._eval_loss_component_count
                for k, total in self._eval_loss_component_sums.items()
            }
            if payload:
                self.log(payload)
                metrics.update(payload)

        return metrics

    def _save_checkpoint(self, model, trial):
        super()._save_checkpoint(model, trial)
        # other ranks may not see rank 0's new checkpoint dir yet (network FS) and would skip the final barrier
        if dist.is_initialized():
            best = [self.state.best_model_checkpoint]
            dist.broadcast_object_list(best, src=0)
            self.state.best_model_checkpoint = best[0]

    def inference_load_from_checkpoint(self, resume_from_checkpoint=None):
        # skip resume_from_checkpoint after train (self._load_best_model())
        if self.args.load_best_model_at_end and (resume_from_checkpoint == self.state.best_model_checkpoint):
            logger.info("Since best model has already been loaded, skip resume from checkpoint")
            pass

        else:
            if resume_from_checkpoint is False:
                resume_from_checkpoint = None

            if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
                resume_from_checkpoint = get_last_checkpoint(self.args.output_dir)
                if resume_from_checkpoint is None:
                    raise ValueError(f"No valid checkpoint found in output directory ({self.args.output_dir})")
            if resume_from_checkpoint is not None:
                logger.info(f"Loading from {resume_from_checkpoint}")
                self._load_from_checkpoint(resume_from_checkpoint)

        return resume_from_checkpoint

    def prediction_step(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        has_labels = False if len(self.label_names) == 0 else all(inputs.get(k) is not None for k in self.label_names)
        return_loss = inputs.get("return_loss")
        if return_loss is None:
            return_loss = self.can_return_loss
        loss_without_labels = len(self.label_names) == 0 and return_loss

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", ["past_key_values"])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels or loss_without_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if is_sagemaker_mp_enabled():
                raw_outputs = smp_forward_only(model, inputs)
                if has_labels or loss_without_labels:
                    if isinstance(raw_outputs, dict):
                        loss_mb = raw_outputs["loss"]
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        loss_mb = raw_outputs[0]
                        logits_mb = raw_outputs[1:]

                    loss = loss_mb.reduce_mean().detach().cpu()
                    logits = smp_nested_concat(logits_mb)
                else:
                    loss = None
                    if isinstance(raw_outputs, dict):
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys)
                    else:
                        logits_mb = raw_outputs
                    logits = smp_nested_concat(logits_mb)
            else:
                if has_labels or loss_without_labels:
                    with self.compute_loss_context_manager():
                        loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                    loss = loss.detach().mean() if loss else None

                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        logits = outputs[1:]
                else:
                    loss = None
                    with self.compute_loss_context_manager():
                        outputs = model(**inputs)
                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                    else:
                        logits = outputs
                    if self.args.past_index >= 0:
                        self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels)
