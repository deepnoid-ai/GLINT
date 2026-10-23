from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig


class SubConfig(PretrainedConfig):
    # PretrainedConfig.to_dict() emits the class-level model_type (""), breaking save/load round-trips
    def to_dict(self):
        output = super().to_dict()
        output["model_type"] = self.model_type
        return output


class VisionConfig(SubConfig):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    @staticmethod
    def from_exp_config(vision_config: dict):
        model_type = vision_config["model_type"]

        if model_type in ["vjepa2", "dinov2", "dinov3_vit"]:
            # checkpoint configs already embed the architecture fields; only experiment configs need the hub fetch
            if vision_config.get("pretrained_name_or_path"):
                config = AutoConfig.from_pretrained(vision_config["pretrained_name_or_path"])
                config = config.to_dict()
                vision_config.update(config)
        elif model_type == "m3ae":
            vision_config.setdefault("hidden_size", 768)
        elif model_type == "vjepa2.1":
            from .vjepa21_encoder import ARCH_MAP, VIT_EMBED_DIMS

            arch = vision_config.get("arch", "vjepa2_1_vit_large_384")
            factory_name = ARCH_MAP[arch][0].__name__
            embed_dim = VIT_EMBED_DIMS.get(factory_name, 1024)
            use_hierarchical = vision_config.get("use_hierarchical", False)
            n_levels = ARCH_MAP[arch][1]
            vision_config["hidden_size"] = embed_dim * n_levels if use_hierarchical else embed_dim
        else:
            raise NotImplementedError()

        vision_config = VisionConfig(**vision_config)

        return vision_config


class TextConfig(SubConfig):
    def __init__(
        self,
        model_type,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_type = model_type

    @staticmethod
    def from_exp_config(
        text_config: dict,
    ):
        model_type = text_config["model_type"]

        if model_type != "mpnet":
            raise NotImplementedError()
        if text_config.get("pretrained_name_or_path"):
            text_config.update(AutoConfig.from_pretrained(text_config["pretrained_name_or_path"]).to_dict())
        text_config = TextConfig(**text_config)

        return text_config


class AlignLayerConfig(SubConfig):
    def __init__(
        self,
        model_type: str = "align_transformer",
        projector_config=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_type = model_type
        self.projector_config = projector_config

    @staticmethod
    def from_exp_config(
        align_layer_config: dict,
    ):
        if align_layer_config["model_type"] == "align_transformer":
            from transformers.models.dinov2.configuration_dinov2 import Dinov2Config

            config = Dinov2Config(**align_layer_config)
            config = config.to_dict()
            align_layer_config.update(config)
            align_layer_config["model_type"] = "align_transformer"

        align_layer_config = AlignLayerConfig(
            **align_layer_config,
        )

        return align_layer_config


class ProjectorConfig(SubConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class DFRConfig(SubConfig):
    def __init__(
        self,
        vfm_teacher_config: dict = None,
        projector_config: dict = None,
        layer_mapping: dict = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vfm_teacher_config = vfm_teacher_config
        self.projector_config = projector_config
        self.layer_mapping = layer_mapping

    @staticmethod
    def from_exp_config(dfr_config: dict):
        if "vfm_teacher_config" in dfr_config and dfr_config["vfm_teacher_config"] is not None:
            dfr_config["vfm_teacher_config"] = VisionConfig.from_exp_config(dfr_config["vfm_teacher_config"])

        if "projector_config" in dfr_config and dfr_config["projector_config"] is not None:
            dfr_config["projector_config"] = ProjectorConfig(**dfr_config["projector_config"])

        return DFRConfig(**dfr_config)


class GLINTConfig(PretrainedConfig):
    is_composition = True

    def __init__(
        self,
        vision_config: dict = None,
        text_config: dict = None,
        align_layer_config: dict = None,
        predictor_config: dict = None,
        prototype_head_config: dict = None,
        dfr_config: dict = None,
        vision_only: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vision_only = vision_only

        if vision_config is not None:
            pretrained_dir = kwargs.get("pretrained_dir")
            if pretrained_dir and "pretrained_dir" not in vision_config:
                vision_config["pretrained_dir"] = pretrained_dir
            self.vision_config = VisionConfig.from_exp_config(vision_config)

        if text_config is not None:
            self.text_config = TextConfig.from_exp_config(text_config)

        if align_layer_config is not None:
            self.align_layer_config = AlignLayerConfig.from_exp_config(align_layer_config)

        if dfr_config is not None:
            pretrained_dir = kwargs.get("pretrained_dir")
            teacher = dfr_config.get("vfm_teacher_config")
            if pretrained_dir and teacher is not None:
                teacher.setdefault("pretrained_dir", pretrained_dir)
            self.dfr_config = DFRConfig.from_exp_config(dfr_config)

        self.kwargs = kwargs
