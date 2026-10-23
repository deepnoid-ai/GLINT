import torch
from einops import rearrange
from monai.losses.dice import DiceLoss
from torch import nn

from common import logger

from .configuration import FineTuningConfig
from .segmentation_decoders import build_segmentation_decoder
from .vision_encoders import build_vision_encoder


class AdaptiveAveragePooling(nn.Module):
    def __init__(self, target_patch_size):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d((target_patch_size, target_patch_size))

    def forward(self, x):
        x = self.avg_pool(x)
        return x


class ProjectionLayer(nn.Module):
    def __init__(self, vision_hidden_dim, projection_hidden_dim):
        super().__init__()

        self.proj = nn.Conv2d(
            in_channels=vision_hidden_dim,
            out_channels=projection_hidden_dim,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.vision_hidden_dim = vision_hidden_dim

    def forward(self, vision_features):
        b, n, c = vision_features.shape
        assert (n**0.5).is_integer(), "n must be a square"
        p = int(n**0.5)

        x = rearrange(
            vision_features,
            "b (h w) c -> b c h w",
            h=p,
            w=p,
            c=self.vision_hidden_dim,
        )

        x = self.proj(x)
        return x


class SegmentationFineTuningModel(nn.Module):
    supports_gradient_checkpointing = True

    def build_vision_model(self, config):
        vision_config = config.vision_config
        vision_encoder = build_vision_encoder(vision_config)
        return vision_encoder

    def build_segmentation_decoder_model(self, config):
        decoder_config = config.decoder_config
        segmentation_decoder = build_segmentation_decoder(decoder_config)
        return segmentation_decoder

    def build_projection_layer(self, config):
        if config.projection_config["model_type"] == "linear":
            projection_layer = ProjectionLayer(
                vision_hidden_dim=config.projection_config["vision_hidden_dim"],
                projection_hidden_dim=config.projection_config["projection_hidden_dim"],
            )
            return projection_layer
        else:
            raise NotImplementedError

    def build_pooling_layer(self, config):
        pooling_type = config.pooling_config["model_type"]
        if pooling_type == "average_pooling":
            pooling_layer = AdaptiveAveragePooling(config.pooling_config["target_patch_size"])
            return pooling_layer
        else:
            raise NotImplementedError(f"Unknown pooling type: {pooling_type}")

    def build_loss_fn(self, config):
        loss_config = config.kwargs["loss"]
        return self.load_loss(loss_config)

    def load_loss(self, loss_config):
        apply_list = loss_config.get("apply", [])
        ratio_list = loss_config.get("ratio", [1.0] * len(apply_list))

        if len(apply_list) == 0:
            raise ValueError("'apply' list cannot be empty.")

        if len(apply_list) != len(ratio_list):
            raise ValueError(f"Length of 'apply' ({len(apply_list)}) and 'ratio' ({len(ratio_list)}) must match.")

        active_losses = []
        for loss_name, weight in zip(apply_list, ratio_list):
            loss_cfg = loss_config.get(loss_name, {})

            if loss_name == "BCEwithLogitsLoss":
                loss_fn = nn.BCEWithLogitsLoss(**loss_cfg)
                active_losses.append((weight, loss_fn, loss_name))

            elif loss_name == "DiceLoss":
                loss_fn = DiceLoss(**loss_cfg)
                active_losses.append((weight, loss_fn, loss_name))

            else:
                raise ValueError(f"Undefined Loss: {loss_name}. Only BCEwithLogitsLoss and DiceLoss are supported.")

        logger.info("=" * 50)
        logger.info("***** Loss Configuration *****")
        if len(active_losses) > 0:
            config_str = ", ".join(f"({w}, {name})" for w, _, name in active_losses)
            logger.info(f"(ratio, loss)  = {config_str}\n")
        else:
            logger.info("No active loss found in config file!\n")

        def total_loss_fn(logits, targets):
            total = 0.0
            loss_dict = {}
            for w, fn, name in active_losses:
                loss_val = fn(logits, targets)
                weighted_loss = w * loss_val
                total += weighted_loss

                loss_dict[name] = loss_val.detach()

            return total, loss_dict

        return total_loss_fn

    def __init__(self, config: FineTuningConfig):
        super().__init__()

        logger.info("Build vision model ...")
        self.vision_model = self.build_vision_model(config)

        logger.info("Build segmentation decoder ...")
        self.segmentation_decoder = self.build_segmentation_decoder_model(config)

        logger.info("Build projection layer...")
        self.projection_layer = self.build_projection_layer(config)

        logger.info("Build pooling layer...")
        self.pooling_layer = self.build_pooling_layer(config)

        logger.info("Build loss functions ...")
        self.loss_fn = self.build_loss_fn(config)
        self.module_to_update = config.module_to_update

        self.config = config
        self.return_loss = True

    def forward_vision_model(self, pixel_values):
        vision_model_type = self.config.vision_config.model_type

        if vision_model_type == "dinov2":
            vision_tokens = self.vision_model(pixel_values)["last_hidden_state"]

        elif vision_model_type == "dinov3_vit":
            vision_tokens = self.vision_model(pixel_values)["last_hidden_state"]

        elif vision_model_type in ("mgca", "kad", "medklip", "mrm", "mavl", "chexworld"):
            vision_tokens = self.vision_model.forward_features(pixel_values)

        else:
            raise NotImplementedError

        if hasattr(self.config.vision_config, "patch_token_start_index"):
            patch_token_start_index = self.config.vision_config.patch_token_start_index
            vision_tokens = vision_tokens[:, patch_token_start_index:, :]

        return vision_tokens

    def forward(self, pixel_values, masks=None):
        vision_features = self.forward_vision_model(pixel_values)

        projected_token = self.projection_layer(vision_features)

        pooled_token = self.pooling_layer(projected_token)

        logits = self.segmentation_decoder(pooled_token)

        outputs = {}
        outputs["output"] = logits
        outputs["loss"] = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        outputs["losses"] = {"loss": torch.tensor(0.0, device=logits.device, dtype=logits.dtype)}

        if self.return_loss and masks is not None:
            loss, loss_dict = self.loss_fn(logits, masks.float())
            outputs["loss"] = loss
            outputs["losses"] = loss_dict

        return outputs
