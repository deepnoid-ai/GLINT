"""Adapted from ViSD-Boost (https://github.com/alibaba-damo-academy/ViSD-Boost; no upstream license, rights remain
with the original authors), lavis/utils/{vit,resnet}.py. The 3D ResNet comes from MedicalNet
(https://github.com/Tencent/MedicalNet): Copyright (C) 2019 THL A29 Limited, a Tencent company;
Copyright (c) 2017 Kensho Hara. MIT License (LICENSES/MIT.txt)

The ViSD-Boost image encoder is a 3D ResNet-18 (shortcut A, dilated layer3/4) whose last stage is projected to 256
channels; only that last-stage token map is kept (the organ-token bookkeeping used in pretraining is dropped).
"""

from functools import partial

import torch
import torch.nn.functional as F
from torch import nn

from .utils import load_encoder_weights


def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    return nn.Conv3d(
        in_planes, out_planes, kernel_size=3, dilation=dilation, stride=stride, padding=dilation, bias=False
    )


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.zeros(out.size(0), planes - out.size(1), *out.shape[2:], dtype=out.dtype, device=out.device)
    # as upstream, the type-A shortcut is detached (``Variable(torch.cat([out.data, zero_pads]))``)
    return torch.cat([out.detach(), zero_pads], dim=1)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


class ResNet3D(nn.Module):
    def __init__(self, layers=(2, 2, 2, 2)):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2, dilation=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2, dilation=4)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * BasicBlock.expansion:
            downsample = partial(downsample_basic_block, planes=planes * BasicBlock.expansion, stride=stride)

        layers = [BasicBlock(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample)]
        self.inplanes = planes * BasicBlock.expansion
        layers += [BasicBlock(self.inplanes, planes, dilation=dilation) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        return self.layer4(self.layer3(self.layer2(self.layer1(x))))


class ViSDBoostEncoder(nn.Module):
    """(B, 1, 112, 256, 352) volume -> (B, 7 * 8 * 11, 256) last-stage tokens."""

    def __init__(self):
        super().__init__()
        self.res_model = ResNet3D()
        self.proj4 = nn.Conv3d(512, 256, kernel_size=1)

    def forward(self, x):
        return self.proj4(self.res_model(x)).flatten(2).transpose(1, 2)


def build_visd_boost_encoder(checkpoint_paths):
    """ViSD-Boost image encoder (``visual_encoder``) from the official ``checkpoint_vqvae.pth`` + ``checkpoint.pth``."""
    model = ViSDBoostEncoder()
    prefix = "visual_encoder."
    state = {}
    # same order as the official eval.py: the main checkpoint overrides the BN statistics of the VQ-VAE stage
    for path in checkpoint_paths:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)["model"]
        state.update({k.removeprefix(prefix): v for k, v in ckpt.items() if k.startswith(prefix)})
    load_encoder_weights(model, state, "ViSD-Boost")
    return model
