import torch
import torch.nn as nn
import torch.nn.functional as F


class DFRLoss(nn.Module):
    def __init__(
        self,
        use_token_wise=True,
        **kwargs,
    ):
        super().__init__()
        self.use_token_wise = use_token_wise

    def forward(
        self,
        proj_student_features,
        teacher_features,
    ):
        outputs = dict()

        if isinstance(proj_student_features, list):
            total_loss = 0.0
            n = 0
            for ps, ts in zip(proj_student_features, teacher_features):
                if isinstance(ps, dict):
                    for k in ps.keys():
                        total_loss += self._compute_loss(ps[k], ts[k])
                        n += 1
                else:
                    total_loss += self._compute_loss(ps, ts)
                    n += 1
            loss = total_loss / max(n, 1)

        elif isinstance(proj_student_features, dict):
            total_loss = 0
            num_pairs = len(proj_student_features)
            for key in proj_student_features.keys():
                proj_student = proj_student_features[key]
                teacher = teacher_features[key]
                total_loss += self._compute_loss(proj_student, teacher)
            loss = total_loss / num_pairs
        else:
            loss = self._compute_loss(proj_student_features, teacher_features)

        losses = {
            "dfr_loss": loss,
            "loss": loss,
        }
        outputs["losses"] = losses

        return outputs

    def _compute_loss(self, proj_student, teacher):
        proj_student = F.normalize(proj_student, p=2, dim=-1)
        teacher = F.normalize(teacher, p=2, dim=-1)

        if self.use_token_wise:
            similarity = torch.sum(proj_student * teacher, dim=-1)
            loss = (1.0 - similarity).mean()
        else:
            proj_student_global = proj_student.mean(dim=1)
            teacher_global = teacher.mean(dim=1)
            similarity = torch.sum(proj_student_global * teacher_global, dim=-1)
            loss = (1.0 - similarity).mean()

        loss = loss.mean()
        return loss
