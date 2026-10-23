from torch import nn
from transformers import PreTrainedModel
from transformers.models.dinov2.modeling_dinov2 import Dinov2Encoder

from .configuration import AlignLayerConfig


def build_align_layer(config):
    if config.model_type == "align_transformer":
        model = AlignTransformer(config)
    elif config.model_type == "identity":
        model = Identity()
    elif config.model_type == "mlp":
        model = MLP(config)
    elif config.model_type == "linear":
        model = Linear(config)
    else:
        raise NotImplementedError()

    return model


class AlignTransformer(PreTrainedModel):
    def __init__(self, config: AlignLayerConfig):
        super().__init__(config)

        self.linear = nn.Linear(config.input_dim, config.hidden_size)

        if config.num_hidden_layers:
            self.transformer_layers = Dinov2Encoder(config)
        else:
            self.transformer_layers = None

        if config.use_layer_norm:
            self.layer_norm = nn.LayerNorm(config.hidden_size)
        else:
            self.layer_norm = None

    def forward(self, vision_tokens):
        if self.transformer_layers is not None:
            vision_tokens = self.linear(vision_tokens)
            vision_tokens = self.transformer_layers(vision_tokens)["last_hidden_state"]

        if self.layer_norm is not None:
            vision_tokens = self.layer_norm(vision_tokens)

        return vision_tokens


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Linear(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.linear = nn.Linear(config.input_dim, config.hidden_size)

    def forward(self, x):
        return self.linear(x)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mlp_layer = nn.Sequential(
            nn.Linear(config.input_dim, config.input_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(config.input_dim, config.input_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(config.input_dim, config.hidden_size),
        )

    def forward(self, x):
        return self.mlp_layer(x)
