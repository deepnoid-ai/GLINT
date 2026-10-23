from .segmentation_decoders_modeling.unetr_noskip.modeling import UNETRNoSkipDecoder
from .segmentation_decoders_modeling.UperNet.modeling import (
    SimpleUperNetDecoder,
)


def build_segmentation_decoder(config):
    if config.model_type == "simple_upernet":
        vision_decoder_model = SimpleUperNetDecoder(
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            ppm_pool_sizes=config.ppm_pool_sizes,
            ppm_out_channels=config.ppm_out_channels,
            fpn_out_channels=config.fpn_out_channels,
        )
    elif config.model_type == "unetr_noskip":
        vision_decoder_model = UNETRNoSkipDecoder(
            patch_size_3d=tuple(config.patch_size_3d),
            embed_dim=config.embed_dim,
            num_classes=config.num_classes,
            base_channels=getattr(config, "base_channels", 32),
        )
    else:
        raise NotImplementedError(f"Unknown decoder model_type: {config.model_type}")

    return vision_decoder_model
