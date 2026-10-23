from torch import nn
from transformers.modeling_utils import PreTrainedModel


class BasePreTrainedModel(PreTrainedModel):
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        if (
            isinstance(module, nn.Conv2d)  # noqa: SIM101
            or isinstance(module, nn.Embedding)
            or isinstance(module, nn.Linear)
        ):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Parameter):
            raise ValueError()
