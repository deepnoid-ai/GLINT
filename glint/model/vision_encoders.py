import torch.nn as nn
from transformers import Dinov2Config, Dinov2Model, DINOv3ViTConfig, DINOv3ViTModel, VJEPA2Config, VJEPA2Model


def _hf_config(config_class, config):
    fields = config.to_dict()
    fields.pop("model_type", None)
    return config_class(**fields)


def build_vision_encoder(config):
    from_config_only = getattr(config, "from_config_only", False) or not getattr(
        config, "pretrained_name_or_path", None
    )

    if config.model_type == "dinov2":
        model = (
            Dinov2Model(_hf_config(Dinov2Config, config))
            if from_config_only
            else Dinov2Model.from_pretrained(config.pretrained_name_or_path)
        )
        nn.init.trunc_normal_(model.embeddings.mask_token, std=0.02)

    elif config.model_type == "dinov3_vit":
        model = (
            DINOv3ViTModel(_hf_config(DINOv3ViTConfig, config))
            if from_config_only
            else DINOv3ViTModel.from_pretrained(config.pretrained_name_or_path)
        )
        nn.init.trunc_normal_(model.embeddings.mask_token, std=0.02)

    elif config.model_type == "vjepa2":
        model = (
            VJEPA2Model(_hf_config(VJEPA2Config, config))
            if from_config_only
            else VJEPA2Model.from_pretrained(config.pretrained_name_or_path)
        )

    elif config.model_type == "vjepa2.1":
        from .vjepa21_encoder import VJEPA21Encoder

        model = VJEPA21Encoder(config)

    elif config.model_type == "m3ae":
        from .carzero_encoder import CARZeroM3AEEncoder

        model = CARZeroM3AEEncoder(config)

    else:
        raise NotImplementedError()

    return model
