from transformers import AutoModel, MPNetConfig, MPNetModel


def build_text_encoder(config):
    if config.model_type == "mpnet":
        if getattr(config, "from_config_only", False) or not getattr(config, "pretrained_name_or_path", None):
            fields = config.to_dict()
            fields.pop("model_type", None)
            return MPNetModel(MPNetConfig(**fields))
        return AutoModel.from_pretrained(config.pretrained_name_or_path)
    else:
        raise NotImplementedError(f"Unsupported text model_type: {config.model_type}")
