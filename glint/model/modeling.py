import torch
import torch.nn as nn
import torch.nn.functional as F

from common import logger

from .align_layers import build_align_layer
from .common_layers import BasePreTrainedModel
from .configuration import GLINTConfig
from .losses import LossFunctions
from .text_encoders import build_text_encoder
from .vision_encoders import build_vision_encoder


class GLINTModel(BasePreTrainedModel):
    config_class = GLINTConfig
    # only relevant when vision_only=True drops these submodules; in VLM mode every key loads
    _keys_to_ignore_on_load_unexpected = [
        r"^text_model\.",
        r"^align_layer\.",
        r"^loss_fns\.",
        r"^dfr_module\.",
        r"^vision_layer_norm\.",
        r"^text_layer_norm\.",
    ]

    def build_vision_model(self, config):
        vision_config = config.vision_config
        vision_model = build_vision_encoder(vision_config)

        return vision_model

    def build_text_model(self, config):
        text_config = config.text_config
        text_model = build_text_encoder(text_config)

        return text_model

    def build_align_layer_model(self, config):
        align_layer_config = config.align_layer_config
        align_layer = build_align_layer(align_layer_config)

        return align_layer

    def build_dfr_module(self, config):
        from .dfr import DFRModule

        dfr_module = DFRModule(config.dfr_config)
        return dfr_module

    def __init__(self, config: GLINTConfig):
        super().__init__(config)

        logger.info("Build vision model ...")
        self.vision_model = self.build_vision_model(config)

        if getattr(config, "vision_only", False):
            logger.info("vision_only=True: skipping text model, align layer, losses and DFR module.")
            return

        logger.info("Build text model ...")
        self.text_model = self.build_text_model(config)

        logger.info("Build align layer model ...")
        self.align_layer = self.build_align_layer_model(config)

        logger.info("Build loss functions ...")
        loss_cfg = config.kwargs["loss"]

        self.loss_fns = LossFunctions(loss_cfg)

        if hasattr(config, "dfr_config") and config.dfr_config is not None:
            logger.info("Build dfr module ...")
            self.dfr_module = self.build_dfr_module(config)

        self.vision_layer_norm = (
            nn.LayerNorm(config.align_layer_config.hidden_size) if config.use_vision_layer_norm else None
        )
        self.text_layer_norm = nn.LayerNorm(config.text_config.hidden_size) if config.use_text_layer_norm else None

        self.compute_logits_type = config.kwargs.get("compute_logits_type")
        self.module_to_update = config.kwargs.get("module_to_update")

    def forward_vision_model(self, pixel_values, output_hidden_states=False):
        vision_model_type = self.config.vision_config.model_type

        vision_hidden_states = None

        use_cls_token = getattr(self.config.vision_config, "use_cls_token", False)

        if vision_model_type == "dinov2":
            vision_outputs = self.vision_model(pixel_values, output_hidden_states=output_hidden_states)
            vision_tokens = vision_outputs["last_hidden_state"]
            if not use_cls_token:
                vision_tokens = vision_tokens[:, 1:, :]
            vision_hidden_states = vision_outputs.get("hidden_states", None)

        elif vision_model_type == "dinov3_vit":
            vision_outputs = self.vision_model(pixel_values, output_hidden_states=output_hidden_states)
            original_output = vision_outputs["last_hidden_state"]
            patch_tokens = original_output[:, 1 + self.config.vision_config.num_register_tokens :, :]
            if use_cls_token:
                cls_token = original_output[:, :1, :]
                vision_tokens = torch.cat([cls_token, patch_tokens], dim=1)
            else:
                vision_tokens = patch_tokens

            vision_hidden_states = vision_outputs.get("hidden_states", None)

        elif vision_model_type == "vjepa2":
            vision_outputs = self.vision_model(
                pixel_values, skip_predictor=True, output_hidden_states=output_hidden_states
            )
            vision_tokens = vision_outputs["last_hidden_state"]
            vision_hidden_states = vision_outputs.get("hidden_states", None)
        elif vision_model_type == "vjepa2.1":
            vision_outputs = self.vision_model(pixel_values, output_hidden_states=output_hidden_states)
            vision_tokens = vision_outputs["last_hidden_state"]
            vision_hidden_states = vision_outputs.get("hidden_states", None)
        else:
            raise NotImplementedError

        outputs = {}

        vision_tokens = self.align_layer(vision_tokens)

        if self.vision_layer_norm is not None:
            vision_tokens = self.vision_layer_norm(vision_tokens)

        outputs["vision_tokens"] = vision_tokens
        outputs["vision_hidden_states"] = vision_hidden_states

        return outputs

    def forward_text_model(self, encoded_input):
        text_outputs = {}

        text_model_type = self.config.text_config.model_type
        if text_model_type == "mpnet":
            model_output = self.text_model(encoded_input["input_ids"], attention_mask=encoded_input["attention_mask"])
            token_embeddings = model_output[0]

        else:
            raise NotImplementedError

        if self.config.text_config.use_cls_token:
            text_features = token_embeddings[:, 0, :]
        else:
            input_mask_expanded = (
                encoded_input["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).to(token_embeddings.dtype)
            )
            text_features = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
                input_mask_expanded.sum(1), min=1e-9
            )

        if self.text_layer_norm is not None:
            text_features = self.text_layer_norm(text_features)

        text_outputs["text_features"] = text_features
        text_outputs["text_features_norm"] = F.normalize(text_features, p=2, dim=1)

        return text_outputs

    def forward(
        self,
        pixel_values=None,
        enc_sents=None,
        neg_enc_sents=None,
        return_loss=True,
        forward_accum_outputs=False,
        forward_accum_loss=False,
        **kwargs,
    ):
        if forward_accum_outputs:
            return self.forward_accum_outputs(
                pixel_values=pixel_values,
                enc_sents=enc_sents,
                neg_enc_sents=neg_enc_sents,
                **kwargs,
            )

        if forward_accum_loss:
            return self.forward_accum_loss(**kwargs)

        if getattr(self.config, "vision_only", False):
            return self.vision_model(pixel_values)

        outputs = {}
        use_dfr_module = hasattr(self, "dfr_module") and self.dfr_module is not None
        vision_outputs = self.forward_vision_model(pixel_values, output_hidden_states=use_dfr_module)

        if use_dfr_module:
            dfr_outputs = self.dfr_module(
                pixel_values=pixel_values,
                student_hidden_states=vision_outputs["vision_hidden_states"],
                cached_teacher_features=kwargs.get("cached_teacher_features"),
            )
            outputs.update(dfr_outputs)

        outputs.update(vision_outputs)

        if return_loss:
            loss = 0
            losses = {}

            for loss_type, loss_fn in self.loss_fns.loss_dict.items():
                loss_ratio = self.loss_fns.loss_ratio[loss_type]

                if loss_fn.__class__.__name__ == "AlignmentLoss":
                    loss_outputs = loss_fn(
                        enc_sents=enc_sents,
                        neg_enc_sents=neg_enc_sents,
                        vision_tokens=outputs["vision_tokens"],
                        forward_text_model=self.forward_text_model,
                    )
                    _losses = loss_outputs["losses"]
                    loop_loss = _losses.pop("loss")
                    losses.update(_losses)

                elif loss_fn.__class__.__name__ == "DFRLoss":
                    loss_outputs = loss_fn(
                        proj_student_features=outputs["proj_student_features"],
                        teacher_features=outputs["teacher_features"],
                    )
                    _losses = loss_outputs["losses"]
                    loop_loss = _losses.pop("loss")
                    losses.update(_losses)

                else:
                    raise NotImplementedError

                loss += loop_loss * loss_ratio

            losses["loss"] = loss

            outputs["loss"] = loss
            outputs["losses"] = losses

        return outputs

    def compute_logits(
        self,
        pixel_values,
        enc_sents,
        **kwargs,
    ):
        vision_outputs = self.forward_vision_model(pixel_values)

        outputs = {}

        if self.compute_logits_type == "vl_alignment":
            loss_fn = self.loss_fns.loss_dict["AlignmentLoss"]

            splited_sentences = [
                {
                    "input_ids": enc_sents[0]["input_ids"][i : i + 1],
                    "attention_mask": enc_sents[0]["attention_mask"][i : i + 1],
                }
                for i in range(enc_sents[0]["input_ids"].size(0))
            ]

            loss_outputs = loss_fn(
                splited_sentences,
                vision_outputs["vision_tokens"],
                self.forward_text_model,
                ddp_gather=False,
                need_attn_weights=True,
                compute_loss=False,
            )
            outputs.update(loss_outputs)

            similarity_scores = loss_outputs["attn_weights"][0]
            if getattr(self.config.vision_config, "use_cls_token", False):
                similarity_scores = similarity_scores[:, :, 1:]
            outputs["similarity_scores"] = similarity_scores

            if "sim_components" in loss_outputs:
                comp = loss_outputs["sim_components"]
                if getattr(self.config.vision_config, "use_cls_token", False):
                    comp = {k: v[:, :, 1:] for k, v in comp.items()}
                outputs["sim_components"] = comp

            logits = loss_outputs["logits"]
            logits = logits.T

        else:
            raise NotImplementedError

        outputs["logits"] = logits
        return outputs

    def forward_accum_outputs(
        self,
        pixel_values,
        enc_sents=None,
        neg_enc_sents=None,
        **kwargs,
    ):
        use_dfr_module = hasattr(self, "dfr_module") and self.dfr_module is not None

        outputs = {}

        vision_outputs = self.forward_vision_model(pixel_values, output_hidden_states=use_dfr_module)
        outputs.update(vision_outputs)

        if use_dfr_module:
            dfr_outputs = self.dfr_module(
                pixel_values=pixel_values,
                student_hidden_states=vision_outputs["vision_hidden_states"],
                cached_teacher_features=kwargs.get("cached_teacher_features"),
            )
            outputs.update(dfr_outputs)

        text_outputs = []
        for kp in enc_sents:
            text_outputs.append(self.forward_text_model(kp)["text_features"])

        text_outputs = {"text_features": text_outputs}
        outputs.update(text_outputs)

        if neg_enc_sents is not None:
            negative_text_outputs = []
            for kp in neg_enc_sents:
                if isinstance(kp["input_ids"], torch.Tensor):
                    negative_text_outputs.append(self.forward_text_model(kp)["text_features"])
                else:
                    negative_text_outputs.append(
                        torch.zeros((0, self.config.text_config.hidden_size)).to(pixel_values.device)
                    )

            negative_text_outputs = {"negative_text_features": negative_text_outputs}
            outputs.update(negative_text_outputs)

        if "ids_mask" in kwargs:
            outputs["ids_mask"] = kwargs["ids_mask"]
            outputs["mask"] = kwargs["mask"]

        return outputs

    def forward_accum_loss(
        self,
        vision_tokens,
        text_features,
        negative_text_features=None,
        **kwargs,
    ):
        outputs = {}
        losses = {}
        loss = 0

        for loss_type, loss_fn in self.loss_fns.loss_dict.items():
            loss_ratio = self.loss_fns.loss_ratio[loss_type]

            if loss_fn.__class__.__name__ == "AlignmentLoss":
                loss_outputs = loss_fn(
                    enc_sents=None,
                    neg_enc_sents=None,
                    vision_tokens=vision_tokens,
                    forward_text_model=None,
                    text_feats_list=text_features,
                    neg_text_feats_list=negative_text_features,
                )
                _losses = loss_outputs["losses"]
                loop_loss = _losses.pop("loss")
                losses.update(_losses)
            elif loss_fn.__class__.__name__ == "DFRLoss":
                loss_outputs = loss_fn(
                    proj_student_features=kwargs.get("proj_student_features"),
                    teacher_features=kwargs.get("teacher_features"),
                )
                _losses = loss_outputs["losses"]
                loop_loss = _losses.pop("loss")
                losses.update(_losses)
            else:
                raise NotImplementedError

            loss += loop_loss * loss_ratio

        losses["loss"] = loss

        outputs["loss"] = loss
        outputs["losses"] = losses

        return outputs
