import torch.nn as nn


def build_projector(config):
    if config.model_type == "mlp":
        model = MLP(config)
    else:
        raise NotImplementedError()

    return model


def get_activation(activation_name):
    activation_map = {"silu": nn.SiLU(), "relu": nn.ReLU(), "gelu": nn.GELU(), "tanh": nn.Tanh()}
    activation_name = activation_name.lower()
    if activation_name not in activation_map:
        raise ValueError(f"Unknown activation: {activation_name}. Available: {list(activation_map.keys())}")
    return activation_map[activation_name]


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        activation = get_activation(getattr(config, "activation", "silu"))
        dropout = getattr(config, "dropout", 0.0)
        num_layers = getattr(config, "num_layers", 3)

        layers = []
        layers.append(nn.Linear(config.input_dim, config.hidden_dim))
        layers.append(activation)
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        for _ in range(num_layers - 2):
            layers.append(nn.Linear(config.hidden_dim, config.hidden_dim))
            layers.append(activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(config.hidden_dim, config.out_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
