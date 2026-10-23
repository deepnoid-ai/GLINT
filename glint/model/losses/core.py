import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

from common import logger

from ..dfr import DFRLoss  # noqa: F401
from .gate_confidence_shift import GatedConfidenceShift
from .mpnce import multi_positive_nce_local_loss, multi_positive_nce_loss
from .utils import pad_and_gather, pad_and_gather_2nd_dim


class LossFunctions(nn.Module):
    def __init__(self, loss_cfg, **kwargs):
        super().__init__()
        self.loss_ratio = dict()
        self.loss_dict = nn.ModuleDict()
        for loss_type, ratio in zip(loss_cfg["apply"], loss_cfg["ratio"]):
            logger.info(f"Build {loss_type} loss function ...")
            if loss_cfg[loss_type] is None:
                loss_cfg[loss_type] = dict()
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                loss_cfg[loss_type]["rank"] = torch.distributed.get_rank()
                loss_cfg[loss_type]["world_size"] = torch.distributed.get_world_size()
            self.loss_dict[loss_type] = LOSS_REGISTRY[loss_type](**loss_cfg[loss_type])
            self.loss_ratio[loss_type] = ratio


class AlignmentLoss(nn.Module):
    def __init__(
        self,
        vl_logit_type="gate_confidence_shift",
        vl_logit_kwargs={},
        loss_type="mpnce",
        loss_kwargs={},
        local_loss=False,
        **kwargs,
    ):
        super().__init__()

        self.vl_logit_type = vl_logit_type
        self.init_vl_logit_kwargs(vl_logit_kwargs)

        self.loss_type = loss_type
        self.init_loss_kwargs(loss_kwargs)

        self.local_loss = local_loss

    def forward(
        self,
        enc_sents,
        vision_tokens,
        forward_text_model=None,
        ddp_gather=True,
        need_attn_weights=False,
        compute_loss=True,
        text_feats_list=None,
        neg_enc_sents=None,
        neg_text_feats_list=None,
        **kwargs,
    ):
        outputs = {}

        B_local = len(vision_tokens)
        local_rank = dist.get_rank() if (ddp_gather and dist.is_initialized()) else 0

        text_features, group_map, local_text_features, local_group_map = self.get_text_features(
            enc_sents, forward_text_model, ddp_gather, text_feats_list
        )

        local_vision_tokens = vision_tokens
        if ddp_gather and dist.is_initialized():
            vision_tokens = torch.cat(dist.nn.all_gather(vision_tokens), dim=0)

        if not compute_loss:
            logits_outputs = self.compute_logits(text_features, vision_tokens, need_attn_weights)

            logits = logits_outputs["logits"]

            if self.loss_type == "mpnce":
                logits = logits / self.loss_temperature.exp()

            logits_outputs["logits"] = logits
            outputs.update(logits_outputs)
            return outputs

        if neg_enc_sents or neg_text_feats_list:
            hidden_dim = vision_tokens.shape[-1]

            (
                neg_text_features,
                neg_text_mask,
                local_neg_text_features,
                local_neg_text_mask,
            ) = self.get_neg_text_features(
                neg_enc_sents,
                forward_text_model,
                hidden_dim,
                ddp_gather,
                neg_text_feats_list,
                device=vision_tokens.device,
            )
        else:
            (
                neg_text_features,
                neg_text_mask,
                local_neg_text_features,
                local_neg_text_mask,
            ) = (None, None, None, None)

        if not (self.local_loss and self.training):
            logits_outputs = self.compute_logits(text_features, vision_tokens, need_attn_weights)
            logits = logits_outputs["logits"]

            if neg_text_features is not None:
                neg_logits_outputs = self.compute_logits(
                    neg_text_features, vision_tokens, need_attn_weights, repeat=False
                )

                logits = torch.cat([logits, neg_logits_outputs["logits"]], dim=0)

            outputs["logits"] = logits

            losses = self.compute_loss(
                logits,
                group_map,
                additional_neg_mask=neg_text_mask,
                text_features=text_features,
            )

        else:
            logits_outputs_per_text = self.compute_logits(local_text_features, vision_tokens, need_attn_weights)
            logits_per_text = logits_outputs_per_text["logits"]
            outputs["logits_per_text"] = logits_per_text

            logits_outputs_per_image = self.compute_logits(text_features, local_vision_tokens, need_attn_weights)
            logits_per_image = logits_outputs_per_image["logits"]

            if local_neg_text_features is not None:
                neg_logits_outputs = self.compute_logits(
                    local_neg_text_features,
                    local_vision_tokens,
                    need_attn_weights,
                    repeat=False,
                )

                logits_per_image = torch.cat([logits_per_image, neg_logits_outputs["logits"]], dim=0)

            outputs["logits_per_image"] = logits_per_image
            local_group_map_per_image = local_group_map - local_rank * B_local

            losses = self.compute_local_loss(
                logits_per_text,
                logits_per_image,
                group_map,
                local_group_map,
                local_group_map_per_image,
                local_rank,
                B_local,
                additional_neg_mask=local_neg_text_mask,
                text_features=text_features,
            )

        outputs["losses"] = losses
        return outputs

    def get_feats_list(self, text_feats_list, ddp_gather=True):
        text_features = list()
        group_list = list()

        B_local = sum([len(i) for i in text_feats_list])
        local_rank = dist.get_rank() if (ddp_gather and dist.is_initialized()) else 0

        local_index = 0
        for i, features in enumerate(text_feats_list):
            for j, feats in enumerate(features):
                text_features.append(feats)
                group_list.extend([local_index + local_rank * B_local] * feats.size(0))
                local_index += 1

        return text_features, group_list

    def get_text_features(
        self,
        enc_sents,
        forward_text_model,
        ddp_gather=True,
        text_feats_list=None,
    ):
        if text_feats_list is not None:
            text_features, group_list = self.get_feats_list(text_feats_list, ddp_gather)
        else:
            text_features = list()
            group_list = list()

            B_local = len(enc_sents)
            local_rank = dist.get_rank() if (ddp_gather and dist.is_initialized()) else 0

            for i, kp in enumerate(enc_sents):
                feats = forward_text_model(kp)

                feat = feats["text_features"]
                text_features.append(feat)

                global_index = i + local_rank * B_local
                group_list.extend([global_index] * feat.size(0))

        text_features = torch.cat(text_features, dim=0)
        group_map = torch.tensor(group_list, device=text_features.device)

        local_text_features = text_features
        local_group_map = group_map

        if ddp_gather and dist.is_initialized():
            text_features = pad_and_gather(text_features)

            group_map = pad_and_gather(group_map)
            group_map = group_map.long()

        return text_features, group_map, local_text_features, local_group_map

    def get_neg_feats_list(self, neg_text_feats_list, hidden_dim=None, device=None):
        text_features = list()
        for i, features in enumerate(neg_text_feats_list):
            for j, feats in enumerate(features):
                text_features.append(feats)

        return text_features

    def get_neg_text_features(
        self,
        neg_enc_sents,
        forward_text_model,
        hidden_dim,
        ddp_gather=True,
        neg_text_feats_list=None,
        device=None,
    ):
        if neg_text_feats_list is not None:
            neg_feats_list = self.get_neg_feats_list(neg_text_feats_list, hidden_dim, device)
        else:
            neg_feats_list = list()

            for i, kp in enumerate(neg_enc_sents):
                if isinstance(kp["input_ids"], torch.Tensor):
                    feats = forward_text_model(kp)
                    feat = feats["text_features"]
                else:
                    feat = torch.zeros((0, hidden_dim)).to(device)

                neg_feats_list.append(feat)

        neg_text_features = torch.nn.utils.rnn.pad_sequence(neg_feats_list, batch_first=True, padding_value=0)

        neg_text_mask = torch.zeros(
            len(neg_feats_list),
            neg_text_features.size(1),
            dtype=torch.bool,
            device=neg_text_features.device,
        )
        for i, feat in enumerate(neg_feats_list):
            neg_text_mask[i, : feat.size(0)] = True

        local_neg_text_features = neg_text_features
        local_neg_text_mask = neg_text_mask

        if ddp_gather and dist.is_initialized():
            neg_text_features, neg_text_mask = pad_and_gather_2nd_dim(neg_text_features, neg_text_mask)

        local_neg_text_mask = local_neg_text_mask.T
        neg_text_mask = neg_text_mask.T

        return (
            neg_text_features,
            neg_text_mask,
            local_neg_text_features,
            local_neg_text_mask,
        )

    def init_vl_logit_kwargs(self, vl_logit_kwargs):
        self.vl_logit_kwargs = {}
        if self.vl_logit_type != "gate_confidence_shift":
            raise NotImplementedError(f"Unsupported vl_logit_type: {self.vl_logit_type}")
        vl_logit_kwargs.pop("attn_temperature", None)
        self.sga = GatedConfidenceShift(**vl_logit_kwargs)

    def compute_logits(self, text_features, vision_tokens, need_attn_weights, repeat=True):
        outputs = {}

        sim_scale_kwargs = {}
        if hasattr(self, "loss_temperature"):
            sim_scale_kwargs["loss_temperature"] = self.loss_temperature.exp()

        logits, attn_weights_list, sim_components = self.sga(
            text_features,
            vision_tokens,
            need_attn_weights,
            repeat=repeat,
            **sim_scale_kwargs,
            **self.vl_logit_kwargs,
        )
        outputs["logits"] = logits
        outputs["attn_weights"] = attn_weights_list
        if sim_components is not None:
            outputs["sim_components"] = sim_components
        return outputs

    def init_loss_kwargs(self, loss_kwargs):
        self.loss_kwargs = {}
        if self.loss_type != "mpnce":
            raise NotImplementedError(f"Unsupported loss_type: {self.loss_type}")
        self.loss_temperature = nn.Parameter(torch.FloatTensor([np.log(loss_kwargs["temperature"])]))

    def compute_loss(self, logits, group_map, additional_neg_mask=None, **kwargs):
        return multi_positive_nce_loss(
            logits,
            group_map,
            additional_neg_mask=additional_neg_mask,
            temperature=self.loss_temperature.exp(),
            **self.loss_kwargs,
        )

    def compute_local_loss(
        self,
        logits_per_text,
        logits_per_image,
        group_map,
        local_group_map,
        local_group_map_per_image,
        local_rank,
        B_local,
        additional_neg_mask=None,
        **kwargs,
    ):
        return multi_positive_nce_local_loss(
            logits_per_text,
            logits_per_image,
            group_map,
            local_group_map,
            local_group_map_per_image,
            temperature=self.loss_temperature.exp(),
            local_rank=local_rank,
            B_local=B_local,
            additional_neg_mask=additional_neg_mask,
            **self.loss_kwargs,
        )


LOSS_REGISTRY = {"AlignmentLoss": AlignmentLoss, "DFRLoss": DFRLoss}
