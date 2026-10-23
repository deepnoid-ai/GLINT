"""Adapted from R2Gen (https://github.com/cuhksz-nlp/R2Gen) via BenchX (https://github.com/yangzhou12/BenchX).
No upstream license; rights remain with the original authors.
"""

import torch.nn as nn

from ..configuration import FineTuningConfig
from ..vision_encoders import build_vision_encoder
from .custom import EncoderDecoder
from .loss import R2GenLMLoss
from .utils import get_n_params


class Tokenizer(object):
    def __init__(self, max_token=512, threshold=3):
        self.threshold = threshold
        self.token2idx, self.idx2token = None, None
        self.max_token = max_token

    def get_token_by_id(self, id):
        return self.idx2token[id]

    def get_id_by_token(self, token):
        if token not in self.token2idx:
            return self.token2idx["<unk>"]
        return self.token2idx[token]

    def get_vocab_size(self):
        return len(self.token2idx)

    def __call__(self, report):
        tokens = report.split()

        if len(tokens) > self.max_token - 2:
            tokens = tokens[: self.max_token - 2]

        ids = []

        for token in tokens:
            ids.append(self.get_id_by_token(token))
        ids = [0] + ids + [0]
        return ids

    def decode(self, ids):
        txt = ""
        for i, idx in enumerate(ids):
            if idx > 0:
                if i >= 1:
                    txt += " "
                txt += self.idx2token[idx]
            else:
                break
        return txt

    def batch_decode(self, ids_batch):
        out = []
        for ids in ids_batch:
            out.append(self.decode(ids))
        return out


class ReportGenerationFineTuningModel(nn.Module):
    def build_vision_model(self, config):
        vision_config = config.vision_config
        vision_model = build_vision_encoder(vision_config)
        return vision_model

    def build_loss_fn(self, config):
        if config.loss_config["model_type"] == "R2GenLMLoss":
            return R2GenLMLoss().cuda()
        else:
            raise NotImplementedError

    def __init__(self, config: FineTuningConfig):
        super(ReportGenerationFineTuningModel, self).__init__()

        self.config = config
        self.tokenizer = Tokenizer(max_token=config.report_generator_config["max_seq_length"])

        self.vision_model = self.build_vision_model(config)
        self.encoder_decoder = EncoderDecoder(config.report_generator_config)
        self.logit = nn.Linear(
            config.report_generator_config["d_model"],
            config.report_generator_config["vocab_size"] + 1,
        )

        self.loss_func = self.build_loss_fn(config)

        self.return_generated_sequence = False

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if hasattr(self.vision_model, "gradient_checkpointing_enable"):
            self.vision_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
        if hasattr(self.encoder_decoder, "gradient_checkpointing_enable"):
            self.encoder_decoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
            )

    def gradient_checkpointing_disable(self):
        if hasattr(self.vision_model, "gradient_checkpointing_disable"):
            self.vision_model.gradient_checkpointing_disable()
        if hasattr(self.encoder_decoder, "gradient_checkpointing_disable"):
            self.encoder_decoder.gradient_checkpointing_disable()

    def forward_vision_model(self, pixel_values):
        vision_model_type = self.config.vision_config.model_type

        if vision_model_type == "dinov2":
            vision_outputs = self.vision_model(pixel_values)
            patch_feats = vision_outputs["last_hidden_state"][:, 1:]

        elif vision_model_type == "dinov3_vit":
            vision_outputs = self.vision_model(pixel_values)
            patch_feats = vision_outputs["last_hidden_state"][:, 1 + self.config.vision_config.num_register_tokens :]

        elif vision_model_type == "vjepa2":
            vision_outputs = self.vision_model(
                pixel_values,
                skip_predictor=True,
            )
            patch_feats = vision_outputs["last_hidden_state"]

        elif vision_model_type == "vjepa2.1":
            vision_outputs = self.vision_model(pixel_values)
            patch_feats = vision_outputs["last_hidden_state"]

        elif vision_model_type in ("mgca", "mrm", "chexworld"):
            patch_feats = self.vision_model.forward_features(pixel_values)[:, 1:]

        elif vision_model_type in ("kad", "medklip", "mavl"):
            patch_feats = self.vision_model.forward_features(pixel_values)

        elif vision_model_type == "ct_clip":
            patch_feats = self.vision_model(pixel_values).flatten(1, 3)

        elif vision_model_type in ("fvlm", "visd_boost"):
            patch_feats = self.vision_model(pixel_values)

        elif vision_model_type == "colipri":
            patch_feats = self.vision_model(pixel_values)["last_hidden_state"]

        else:
            raise NotImplementedError(f"patch features for {vision_model_type} is not implemented")

        avg_feats = patch_feats.mean(1)

        return patch_feats, avg_feats

    def forward(self, pixel_values, reports=None, reports_masks=None, **kwargs):
        att_feats, fc_feats = self.forward_vision_model(pixel_values)

        if not self.return_generated_sequence:
            log_probs = self.encoder_decoder(fc_feats, att_feats, reports, mode="forward")
            loss = self.loss_func(log_probs, reports[:, 1:], reports_masks[:, 1:], **kwargs)

            return {
                "loss": loss,
                "output": log_probs,
            }

        else:
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode="sample")

            return {
                "loss": None,
                "output": output if self.return_generated_sequence else log_probs,
            }

    def __repr__(self):
        s = super().__repr__() + "\n"
        s += "{}\n".format(get_n_params(self))
        return s
