import contextlib

import torch
from torch import nn

from common import logger

from .configuration import FineTuningConfig
from .vision_encoders import build_vision_encoder


class AttentivePooling(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, start_token_index=0):
        super().__init__()
        self.start_token_index = start_token_index

        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        nn.init.trunc_normal_(self.query, std=0.02)

        self.ln_q = nn.LayerNorm(hidden_dim)
        self.ln_k = nn.LayerNorm(hidden_dim)
        self.ln_v = nn.LayerNorm(hidden_dim)
        self.ln_post = nn.LayerNorm(hidden_dim)

    def forward(self, features):
        # features: (batch_size, seq_len, hidden_dim)
        if self.start_token_index > 0:
            features = features[:, self.start_token_index :, :]

        batch_size = features.size(0)

        query = self.query.expand(batch_size, -1, -1)
        q = self.ln_q(query)
        k = self.ln_k(features)
        v = self.ln_v(features)

        attn_output, _ = self.attn(q, k, v)

        out = query + attn_output
        out = self.ln_post(out)

        return out.squeeze(1)


class ClassificationFineTuningModel(nn.Module):
    supports_gradient_checkpointing = True

    def build_vision_model(self, config):
        vision_config = config.vision_config
        vision_model = build_vision_encoder(vision_config)
        return vision_model

    def build_pooling_layer(self, config):
        if config.pooling_config["model_type"] == "cls_token":
            return lambda x: x[:, 0, :]
        elif config.pooling_config["model_type"] == "attentive_pooling":
            return AttentivePooling(
                hidden_dim=config.vision_config.hidden_size,
                num_heads=config.pooling_config["num_pooling_heads"],
                start_token_index=config.pooling_config.get("start_token_index", 0),
            )
        else:
            raise NotImplementedError

    def build_classifier(self, config):
        if config.classifier_config["model_type"] == "linear":
            return nn.Linear(
                config.vision_config.hidden_size,
                config.classifier_config["num_classes"],
            )
        else:
            raise NotImplementedError

    def build_loss_fn(self, config):
        if config.loss_config["model_type"] == "bce":
            return nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError

    def __init__(self, config: FineTuningConfig):
        super().__init__()

        logger.info("Build vision model ...")
        self.vision_model = self.build_vision_model(config)

        logger.info("Build pooling layer...")
        self.pooling_layer = self.build_pooling_layer(config)

        logger.info("Build classifier ...")
        self.classifier = self.build_classifier(config)

        logger.info("Build loss functions ...")
        self.loss_fn = self.build_loss_fn(config)

        self.module_to_update = config.module_to_update

        self.config = config
        self.return_loss = True

    def forward_vision_model(self, pixel_values):
        vision_model_type = self.config.vision_config.model_type

        # no autograd through a fully frozen encoder; checked per call so unfreeze variants still backprop
        vision_frozen = not any(p.requires_grad for p in self.vision_model.parameters())
        vision_ctx = torch.no_grad() if vision_frozen else contextlib.nullcontext()

        with vision_ctx:
            if vision_model_type == "vjepa2":
                vision_tokens = self.vision_model(pixel_values, skip_predictor=True)["last_hidden_state"]
            elif vision_model_type == "vjepa2.1":
                # input is already (B, T, C=3, H, W) from normalize_pixel_values_videos
                vision_tokens = self.vision_model(pixel_values)["last_hidden_state"]

            elif vision_model_type == "dinov2":
                vision_tokens = self.vision_model(pixel_values)["last_hidden_state"]

            elif vision_model_type == "dinov3_vit":
                vision_outputs = self.vision_model(pixel_values)["last_hidden_state"]
                cls_token = vision_outputs[:, :1, :]
                patch_tokens = vision_outputs[:, 1 + self.config.vision_config.num_register_tokens :, :]
                vision_tokens = torch.cat([cls_token, patch_tokens], dim=1)

            elif vision_model_type in ("mgca", "kad", "medklip", "mrm", "mavl", "chexworld"):
                vision_tokens = self.vision_model.forward_features(pixel_values)

            elif vision_model_type == "ct_clip":
                # (B, t, h, w, d) token grid -> (B, t*h*w, d)
                vision_tokens = self.vision_model(pixel_values).flatten(1, 3)

            elif vision_model_type in ("fvlm", "visd_boost"):
                vision_tokens = self.vision_model(pixel_values)

            elif vision_model_type == "colipri":
                vision_tokens = self.vision_model(pixel_values)["last_hidden_state"]

            else:
                raise NotImplementedError

        return vision_tokens

    def forward(self, pixel_values, labels=None):
        vision_features = self.forward_vision_model(pixel_values)

        pooled_token = self.pooling_layer(vision_features)

        logits = self.classifier(pooled_token)

        outputs = {}
        outputs["output"] = logits
        outputs["loss"] = None

        if self.return_loss and labels is not None:
            loss = self.loss_fn(logits, labels.float())
            outputs["loss"] = loss

        return outputs
