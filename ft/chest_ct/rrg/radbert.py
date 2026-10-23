"""CT-RATE report labeler used for the RadBERT F1 and CRG metrics.

Adapted from CT-CLIP (https://github.com/ibrahimethemhamamci/CT-CLIP), ct_rate_evaluation/RadBertClassifier.
CT-CLIP is released under CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/); this file is shared
under the same license. The classifier checkpoint (``models/RadBertClassifier.pth``) ships with the CT-RATE dataset
(https://huggingface.co/datasets/ibrahimhamamci/CT-RATE); its RoBERTa backbone is zzxslp/RadBERT-RoBERTa-4m.
"""

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

BACKBONE = "zzxslp/RadBERT-RoBERTa-4m"


class RadBertClassifier(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.config = AutoConfig.from_pretrained(BACKBONE)
        self.model = AutoModel.from_pretrained(BACKBONE, config=self.config)
        self.classifier = nn.Linear(self.model.config.hidden_size, n_classes)

    def forward(self, input_ids, attn_mask):
        output = self.model(input_ids=input_ids, attention_mask=attn_mask)
        return self.classifier(output.pooler_output)


def load_radbert_classifier(checkpoint_path, n_classes, device="cuda"):
    model = RadBertClassifier(n_classes)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    # position_ids is a non-persistent buffer in newer transformers
    unexpected = [k for k in unexpected if not k.endswith("position_ids")]
    if missing or unexpected:
        raise RuntimeError(f"RadBERT checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    tokenizer = AutoTokenizer.from_pretrained(BACKBONE)
    return {"RadBERTLabeler": model.eval().to(device), "tokenizer": tokenizer}
