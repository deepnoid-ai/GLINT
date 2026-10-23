import torch
import torch.nn as nn
import torch.nn.functional as F
from entmax import entmax_bisect


def _build_projection(embed_dim, hidden_dim, mode="mlp"):
    if mode == "identity":
        return nn.Identity()
    if mode == "linear":
        return nn.Linear(embed_dim, hidden_dim)
    if mode == "mlp":
        return nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
    raise ValueError(f"Unknown projection mode: {mode}")


class GatedConfidenceShift(nn.Module):
    def __init__(
        self,
        embed_dim=768,
        gate_hidden_dim=768,
        sem_hidden_dim=768,
        gate_proj_mode="mlp",
        sem_proj_mode="identity",
        gate_log_temperature=2.302585,
        shift_w_init=0.0,
        shift_w_no_weight_decay=True,
    ):
        super().__init__()

        self.text_gate_proj = _build_projection(embed_dim, gate_hidden_dim, gate_proj_mode)
        self.vision_gate_proj = _build_projection(embed_dim, gate_hidden_dim, gate_proj_mode)
        self.gate_log_temperature = nn.Parameter(torch.FloatTensor([gate_log_temperature]))

        self.text_sem_proj = _build_projection(embed_dim, sem_hidden_dim, sem_proj_mode)
        self.vision_sem_proj = _build_projection(embed_dim, sem_hidden_dim, sem_proj_mode)

        self.shift_w = nn.Parameter(torch.tensor(float(shift_w_init)))
        if shift_w_no_weight_decay:
            self.shift_w._no_weight_decay = True

    def _compute_gate_evidence(self, gate):
        w = entmax_bisect(gate, alpha=2.0, n_iter=50, dim=-1)
        return (w * gate).sum(dim=-1)

    def forward(self, text_features, vision_tokens, need_attn_weights=False, repeat=True, **kwargs):
        if repeat:
            query = text_features.unsqueeze(0).expand(vision_tokens.shape[0], text_features.shape[0], -1)
        else:
            assert text_features.dim() == 3
            query = text_features

        text_gate = F.normalize(self.text_gate_proj(query), p=2, dim=-1)
        vision_gate = F.normalize(self.vision_gate_proj(vision_tokens), p=2, dim=-1)
        gate_scores = torch.bmm(text_gate, vision_gate.permute(0, 2, 1))

        gate = torch.sigmoid(gate_scores * self.gate_log_temperature.exp())
        weight = gate

        gate_evidence = self._compute_gate_evidence(gate)
        shift = self.shift_w * gate_evidence

        sem_text_norm = F.normalize(self.text_sem_proj(query), p=2, dim=-1)
        sem_patches_norm = F.normalize(self.vision_sem_proj(vision_tokens), p=2, dim=-1)

        aggregated = torch.bmm(weight, sem_patches_norm)
        aggregated_norm = F.normalize(aggregated, p=2, dim=-1)
        cos_sim = (sem_text_norm * aggregated_norm).sum(dim=-1)
        logits = (cos_sim + shift).T

        if need_attn_weights:
            sem_scores = torch.bmm(sem_text_norm, sem_patches_norm.permute(0, 2, 1))
            if "loss_temperature" in kwargs:
                scaled_sem = (sem_scores / kwargs["loss_temperature"]).sigmoid()
            elif "logit_scale" in kwargs:
                scaled_sem = (sem_scores * kwargs["logit_scale"] + kwargs.get("logit_bias", 0.0)).sigmoid()
            else:
                scaled_sem = sem_scores.sigmoid()

            similarity_scores = [scaled_sem * weight]
            _comp = {"gate": gate, "semantic": scaled_sem, "weight": weight}
        else:
            similarity_scores = None
            _comp = None

        return logits, similarity_scores, _comp
