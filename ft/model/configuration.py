from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig


def resolve_glint_vision_config(vision_config: dict) -> dict:
    """Expand a ``model_type: glint`` entry into the backbone config stored in the GLINT checkpoint."""
    if vision_config.get("model_type") != "glint":
        return vision_config

    glint_name_or_path = vision_config["pretrained_name_or_path"]
    glint_config, _ = PretrainedConfig.get_config_dict(glint_name_or_path)

    resolved = dict(glint_config["vision_config"])
    resolved.update({k: v for k, v in vision_config.items() if k not in ("model_type", "pretrained_name_or_path")})
    resolved["glint_name_or_path"] = glint_name_or_path
    return resolved


# encoders without an HF config: what the heads need (e.g. hidden_size) is set in the experiment YAML
EXTERNAL_VISION_MODEL_TYPES = (
    "vjepa2.1",
    "mgca",
    "kad",
    "medklip",
    "mrm",
    "mavl",
    "chexworld",
    "ct_clip",
    "fvlm",
    "visd_boost",
    "colipri",
)


class VisionConfig(PretrainedConfig):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    @staticmethod
    def from_exp_config(vision_config: dict):
        vision_config = resolve_glint_vision_config(vision_config)
        model_type = vision_config["model_type"]

        if model_type in ["vjepa2", "dinov2", "dinov3_vit"]:
            if not vision_config.get("glint_name_or_path"):
                config = AutoConfig.from_pretrained(vision_config["pretrained_name_or_path"])
                config = config.to_dict()
                vision_config.update(config)

        elif model_type in EXTERNAL_VISION_MODEL_TYPES:
            pass

        else:
            raise NotImplementedError(f"Vision model type {model_type} not supported")

        vision_config = VisionConfig(**vision_config)
        return vision_config


class SegmentationDecoderConfig(PretrainedConfig):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    @staticmethod
    def from_exp_config(decoder_config: dict):
        decoder_config = SegmentationDecoderConfig(**decoder_config)
        return decoder_config


class FineTuningConfig(PretrainedConfig):
    def __init__(
        self,
        vision_config: dict = None,
        decoder_config: dict = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if kwargs.get("pretrained_dir"):
            vision_config["pretrained_dir"] = kwargs.get("pretrained_dir")

        self.vision_config = VisionConfig.from_exp_config(vision_config)

        if decoder_config is not None:
            self.decoder_config = SegmentationDecoderConfig.from_exp_config(decoder_config)

        self.kwargs = kwargs
