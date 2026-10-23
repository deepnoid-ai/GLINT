import torch
import torch.nn as nn

from common import logger
from glint.model.vision_encoders import build_vision_encoder

from .projector import build_projector


class DFRModule(nn.Module):
    def __init__(self, dfr_config):
        super().__init__()
        self.config = dfr_config
        self.use_cache = getattr(dfr_config, "use_cache", False)

        if not self.use_cache:
            self.vision_teacher_model = build_vision_encoder(dfr_config.vfm_teacher_config)
        else:
            logger.info("DFRModule: use_cache=True, skipping teacher model loading")

        self.layer_mapping = dfr_config.layer_mapping
        student_layers = self.layer_mapping["student_layers"]
        teacher_layers = self.layer_mapping["teacher_layers"]

        self._validate_layer_mapping(student_layers, teacher_layers)
        self.num_layer_pairs = len(student_layers)

        self._build_projectors(dfr_config.projector_config, student_layers, teacher_layers)

    def _validate_layer_mapping(self, student_layers, teacher_layers):
        if len(student_layers) != len(teacher_layers):
            raise ValueError(
                f"student_layer and teacher_layer must have same length. "
                f"Got {len(student_layers)} and {len(teacher_layers)}"
            )

    def _build_projectors(self, projector_config, student_layers, teacher_layers):
        use_multiple_projectors = self.layer_mapping.get("use_multiple_projectors", False)

        if use_multiple_projectors and self.num_layer_pairs > 1:
            self.projector = nn.ModuleDict()
            for s_layer, t_layer in zip(student_layers, teacher_layers):
                projector_key = f"projector_{s_layer}_{t_layer}"
                self.projector[projector_key] = build_projector(projector_config)
            self.use_multiple_projectors = True
        else:
            self.projector = build_projector(projector_config)
            self.use_multiple_projectors = False

    def forward(self, pixel_values, student_hidden_states, cached_teacher_features=None):
        student_layers = self.layer_mapping["student_layers"]
        teacher_layers = self.layer_mapping["teacher_layers"]

        if cached_teacher_features is not None:
            teacher_hidden_states = cached_teacher_features
        else:
            with torch.no_grad():
                teacher_outputs = self.vision_teacher_model(pixel_values, output_hidden_states=True)

        results = {
            "teacher_features": {},
            "proj_student_features": {},
        }

        for s_layer, t_layer in zip(student_layers, teacher_layers):
            student_feat = student_hidden_states[s_layer]

            if cached_teacher_features is not None:
                teacher_feat = teacher_hidden_states[t_layer]
                teacher_feat = teacher_feat.to(dtype=student_feat.dtype)
            else:
                teacher_feat = teacher_outputs.hidden_states[t_layer]

            proj_student_feat = self._project_student_features(student_feat, s_layer, t_layer)

            results["teacher_features"][f"layer_{s_layer}_{t_layer}"] = teacher_feat
            results["proj_student_features"][f"layer_{s_layer}_{t_layer}"] = proj_student_feat

        return results

    def _project_student_features(self, student_feat, s_layer, t_layer):
        if self.use_multiple_projectors:
            projector_key = f"projector_{s_layer}_{t_layer}"
            return self.projector[projector_key](student_feat)
        else:
            return self.projector(student_feat)
