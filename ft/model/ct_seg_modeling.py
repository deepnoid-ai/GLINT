"""3D CT segmentation fine-tuning model: vision encoder -> channel adapter -> segmentation decoder."""

import contextlib

import torch
from einops import rearrange
from monai.losses.dice import DiceLoss
from torch import nn

from common import logger

from .configuration import FineTuningConfig
from .segmentation_decoders import build_segmentation_decoder
from .vision_encoders import build_vision_encoder


class CTSegmentationFineTuningModel(nn.Module):
    supports_gradient_checkpointing = True

    def __init__(self, config: FineTuningConfig):
        super().__init__()

        logger.info("Build vision model ...")
        self.vision_model = build_vision_encoder(config.vision_config)

        logger.info("Build segmentation decoder ...")
        self.segmentation_decoder = build_segmentation_decoder(config.decoder_config)

        vision_model_type = config.vision_config.model_type
        in_chans = getattr(config.vision_config, "in_chans", 3)
        input_channels = config.kwargs.get("input_channels", 1)
        # vjepa2.1 stacks 3 consecutive slices as channels in forward_vision_model; CT-CLIP / COLIPRI take 1 channel
        if vision_model_type in ("vjepa2.1", "ct_clip", "colipri"):
            self.channel_adapter = nn.Identity()
        elif input_channels != in_chans:
            self.channel_adapter = nn.Conv3d(input_channels, in_chans, kernel_size=1, stride=1, padding=0, bias=False)
        else:
            self.channel_adapter = nn.Identity()

        input_shape = config.kwargs.get("input_shape", [64, 256, 256])
        D, H, W = input_shape
        self.input_shape = tuple(input_shape)
        if vision_model_type == "colipri":
            self.grid_size = (D // 8, H // 8, W // 8)
        elif vision_model_type == "ct_clip":
            # CTViT tokens per (temporal_patch, patch, patch) block, mirrored by the decoder's patch_size_3d
            ps = list(getattr(config.decoder_config, "patch_size_3d", [10, 20, 20]))
            self.grid_size = (D // ps[0], H // ps[1], W // ps[2])
        elif vision_model_type == "vjepa2.1":
            input_modality = getattr(config.vision_config, "input_modality", 3)
            tubelet_size = getattr(config.vision_config, "tubelet_size", 2)
            patch_size = getattr(config.vision_config, "patch_size", 16)
            assert (
                D % input_modality == 0
            ), f"vjepa2.1 input_slices ({D}) must be divisible by input_modality ({input_modality})"
            t_frames = D // input_modality
            self.grid_size = (t_frames // tubelet_size, H // patch_size, W // patch_size)
        else:
            patch_size = getattr(config.vision_config, "patch_size", 16)
            tubelet_size = getattr(config.vision_config, "tubelet_size", 2)
            self.grid_size = (
                D // tubelet_size,
                H // patch_size,
                W // patch_size,
            )

        # V-JEPA 2.1 was pretrained with ImageNet stats; they are reapplied to CT rescaled from [-1, 1]
        self.register_buffer(
            "_imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_imagenet_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1),
            persistent=False,
        )
        self._vjepa_input_modality = getattr(config.vision_config, "input_modality", 3)

        logger.info("Build loss functions ...")
        self.loss_fn = self.build_loss_fn(config)
        self.module_to_update = config.module_to_update

        self.config = config
        self.return_loss = True

    def build_loss_fn(self, config):
        loss_config = config.kwargs.get("loss", {})
        apply_list = loss_config.get("apply", ["DiceLoss"])
        ratio_list = loss_config.get("ratio", [1.0] * len(apply_list))

        active_losses = []
        for loss_name, weight in zip(apply_list, ratio_list):
            loss_cfg = loss_config.get(loss_name, {})

            if loss_name == "BCEwithLogitsLoss":
                loss_fn = nn.BCEWithLogitsLoss(**loss_cfg)
            elif loss_name == "DiceLoss":
                loss_fn = DiceLoss(**loss_cfg)
            elif loss_name == "CrossEntropyLoss":
                loss_fn = nn.CrossEntropyLoss(**loss_cfg)
            else:
                raise ValueError(f"Undefined Loss: {loss_name}")

            active_losses.append((weight, loss_fn, loss_name))

        config_str = ", ".join(f"({w}, {name})" for w, _, name in active_losses)
        logger.info(f"Loss config: {config_str}")

        def total_loss_fn(logits, targets):
            # targets are (N, 1, D, H, W) class ids: DiceLoss(to_onehot_y) takes them as-is, CE needs (N, D, H, W)
            total = 0.0
            loss_dict = {}
            for w, fn, name in active_losses:
                if name == "CrossEntropyLoss":
                    targets_i = targets.squeeze(1).long() if targets.dim() == 5 else targets.long()
                    loss_val = fn(logits, targets_i)
                else:
                    loss_val = fn(logits, targets)
                total += w * loss_val
                loss_dict[name] = loss_val.detach()
            return total, loss_dict

        return total_loss_fn

    def forward_vision_model(self, pixel_values):
        """(B, C_in, D, H, W) CT volume -> (B, N, embed_dim) patch tokens."""
        vision_model_type = self.config.vision_config.model_type

        x = self.channel_adapter(pixel_values)

        # no autograd through a fully frozen encoder; checked per call so unfreeze variants still backprop
        vision_frozen = not any(p.requires_grad for p in self.vision_model.parameters())
        vision_ctx = torch.no_grad() if vision_frozen else contextlib.nullcontext()

        with vision_ctx:
            if vision_model_type == "vjepa2.1":
                # (B, 1, D, H, W) in [-1, 1] -> ImageNet-normalized (B, D/m, m, H, W), the pretraining layout
                m = self._vjepa_input_modality
                B, _, D, H, W = x.shape
                x = x.view(B, 1, D // m, m, H, W).squeeze(1)  # (B, T, m, H, W)
                x = (x + 1.0) / 2.0
                x = (x - self._imagenet_mean.to(x.dtype)) / self._imagenet_std.to(x.dtype)
                vision_output = self.vision_model(x)
                tokens = vision_output["last_hidden_state"]
            elif vision_model_type == "ct_clip":
                tokens = rearrange(self.vision_model(x), "b t h w d -> b (t h w) d")
            elif vision_model_type == "colipri":
                tokens = self.vision_model(x)["last_hidden_state"]
            else:
                raise NotImplementedError(f"CT seg not implemented for: {vision_model_type}")

        return tokens

    def forward(self, pixel_values, masks=None):
        tokens = self.forward_vision_model(pixel_values)

        d, h, w = self.grid_size
        features = rearrange(tokens, "b (d h w) c -> b c d h w", d=d, h=h, w=w)

        logits = self.segmentation_decoder(features)

        # the decoder upsamples by powers of 2 (D: grid 40 -> 80, not 240 slices), so resize to the input shape
        if tuple(logits.shape[-3:]) != self.input_shape:
            logits = torch.nn.functional.interpolate(
                logits, size=self.input_shape, mode="trilinear", align_corners=False
            )

        outputs = {}
        outputs["output"] = logits
        outputs["loss"] = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        outputs["losses"] = {}

        if self.return_loss and masks is not None:
            loss, loss_dict = self.loss_fn(logits, masks.float())
            outputs["loss"] = loss
            outputs["losses"] = loss_dict

        return outputs
