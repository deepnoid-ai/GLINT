import os

from transformers import Dinov2Model, DINOv3ViTModel, VJEPA2Model

from glint.model import GLINTModel
from glint.model.vjepa21_encoder import VJEPA21Encoder


def build_glint_vision_encoder(config):
    """Load the vision tower of a released GLINT checkpoint (HF repo id or local checkpoint dir)."""
    model = GLINTModel.from_pretrained(config.glint_name_or_path, vision_only=True).vision_model

    use_activation_checkpointing = getattr(config, "use_activation_checkpointing", None)
    if use_activation_checkpointing is not None and hasattr(model, "encoder"):
        model.encoder.use_activation_checkpointing = use_activation_checkpointing

    return model


def resolve_pretrained_path(path, pretrained_dir):
    """Prefer ``pretrained_dir/path`` when it exists; otherwise keep ``path`` (absolute path or HF repo id)."""
    if isinstance(path, (list, tuple)):
        return [resolve_pretrained_path(p, pretrained_dir) for p in path]
    if path and pretrained_dir and not os.path.isabs(path):
        local_path = os.path.join(pretrained_dir, path)
        if os.path.exists(local_path):
            return local_path
        if path.endswith((".pt", ".pth", ".ckpt", ".tar", ".safetensors")):
            raise FileNotFoundError(
                f"{local_path} not found: download the weights listed in ft/README.md to $PRETRAINED_DIR"
            )
    return path


def build_vision_encoder(config):
    if getattr(config, "glint_name_or_path", None):
        return build_glint_vision_encoder(config)

    # vjepa2.1 reads pretrained_checkpoint instead of pretrained_name_or_path
    pretrained_path = resolve_pretrained_path(
        getattr(config, "pretrained_name_or_path", None), getattr(config, "pretrained_dir", None)
    )

    if config.model_type == "dinov2":
        model = Dinov2Model.from_pretrained(pretrained_path)

    elif config.model_type == "dinov3_vit":
        model = DINOv3ViTModel.from_pretrained(pretrained_path)

    elif config.model_type == "vjepa2":
        model = VJEPA2Model.from_pretrained(pretrained_path)

    elif config.model_type == "vjepa2.1":
        model = VJEPA21Encoder(config)

    elif config.model_type == "mgca":
        from .vision_encoders_modeling.mgca import build_mgca_vit_b16

        model = build_mgca_vit_b16(pretrained_path)

    elif config.model_type == "kad":
        from .vision_encoders_modeling.kad import build_kad_resnet

        model = build_kad_resnet(pretrained_path)

    elif config.model_type == "medklip":
        from .vision_encoders_modeling.medklip import build_medklip_resnet

        model = build_medklip_resnet(pretrained_path)

    elif config.model_type == "mrm":
        from .vision_encoders_modeling.mrm import build_mrm_vit_b16

        model = build_mrm_vit_b16(pretrained_path)

    elif config.model_type == "mavl":
        from .vision_encoders_modeling.mavl import build_mavl_resnet

        model = build_mavl_resnet(pretrained_path)

    elif config.model_type == "chexworld":
        from .vision_encoders_modeling.chexworld import build_chexworld_vit_b16

        model = build_chexworld_vit_b16(pretrained_path)

    elif config.model_type == "ct_clip":
        from .vision_encoders_modeling.ct_clip import build_ct_clip_encoder

        model = build_ct_clip_encoder(pretrained_path)

    elif config.model_type == "fvlm":
        from .vision_encoders_modeling.fvlm import build_fvlm_encoder

        model = build_fvlm_encoder(pretrained_path)

    elif config.model_type == "visd_boost":
        from .vision_encoders_modeling.visd_boost import build_visd_boost_encoder

        model = build_visd_boost_encoder(pretrained_path)

    elif config.model_type == "colipri":
        from .vision_encoders_modeling.colipri import CoLiPriEncoder

        model = CoLiPriEncoder(
            pretrained_path,
            embed_dim=getattr(config, "embed_dim", 864),
            img_size=tuple(getattr(config, "img_size", [160, 160, 160])),
        )

    else:
        raise NotImplementedError(f"Vision model type {config.model_type} not supported")

    return model
