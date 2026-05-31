# -*- coding: utf-8 -*-
"""
torchvision 版 modern backbones (避开 timm 的 huggingface.co 依赖).

注册:
    convnext_tiny_tv     — ConvNeXt-Tiny (torchvision cached)
    swin_v2_tiny_tv      — Swin-V2-Tiny
    efficientnet_v2_s_tv — EfficientNetV2-S

权重均使用 torchvision IMAGENET1K_V1, 缓存在 ~/.cache/torch/hub/checkpoints/.
"""

from __future__ import annotations

import torch.nn as nn
import torchvision.models as models

from ..base_classifier import BaseClassifier, ClassificationHead
from .registry import register_backbone


def _freeze_first_fraction(model: nn.Module, freeze_fraction: float) -> None:
    if freeze_fraction <= 0.0:
        return
    params = list(model.parameters())
    total = sum(p.numel() for p in params)
    target = int(total * freeze_fraction)
    cum = 0
    for p in params:
        if cum >= target:
            break
        p.requires_grad = False
        cum += p.numel()


@register_backbone('convnext_tiny_tv')
class ConvNeXtTinyTVClassifier(BaseClassifier):
    """ConvNeXt-Tiny (torchvision)."""

    def build_model(self):
        backbone = models.convnext_tiny(weights="IMAGENET1K_V1")
        print("ConvNeXt-Tiny pre-trained weights loaded (torchvision)")

        ft_ratio = float(self.config.model.fine_tune_ratio)
        _freeze_first_fraction(backbone, freeze_fraction=1.0 - ft_ratio)

        # backbone.classifier = LayerNorm2d + Flatten + Linear(768, 1000)
        # 替换最后 Linear 为 Identity, 拿 768-D 特征
        feature_dim = 768  # convnext_tiny.classifier[2].in_features
        backbone.classifier[2] = nn.Identity()

        self.model = nn.Sequential(
            backbone,
            ClassificationHead(
                in_features=feature_dim,
                num_classes=self.num_classes,
                fc_units=self.config.model.fc_units,
                dropout1=self.config.model.dropout1,
                dropout2=self.config.model.dropout2,
            ),
        )
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"ConvNeXt-Tiny TV: total {total:,}, trainable {trainable:,}")
        return self.model


@register_backbone('swin_v2_t_tv')
class SwinV2TinyTVClassifier(BaseClassifier):
    """Swin-V2-Tiny (torchvision)."""

    def build_model(self):
        backbone = models.swin_v2_t(weights="IMAGENET1K_V1")
        print("Swin-V2-Tiny pre-trained weights loaded (torchvision)")

        ft_ratio = float(self.config.model.fine_tune_ratio)
        _freeze_first_fraction(backbone, freeze_fraction=1.0 - ft_ratio)

        # swin_v2_t.head = Linear(768, 1000)
        feature_dim = 768
        backbone.head = nn.Identity()

        self.model = nn.Sequential(
            backbone,
            ClassificationHead(
                in_features=feature_dim,
                num_classes=self.num_classes,
                fc_units=self.config.model.fc_units,
                dropout1=self.config.model.dropout1,
                dropout2=self.config.model.dropout2,
            ),
        )
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"Swin-V2-Tiny TV: total {total:,}, trainable {trainable:,}")
        return self.model


@register_backbone('efficientnet_v2_s_tv')
class EfficientNetV2STVClassifier(BaseClassifier):
    """EfficientNetV2-S (torchvision)."""

    def build_model(self):
        backbone = models.efficientnet_v2_s(weights="IMAGENET1K_V1")
        print("EfficientNetV2-S pre-trained weights loaded (torchvision)")

        ft_ratio = float(self.config.model.fine_tune_ratio)
        _freeze_first_fraction(backbone, freeze_fraction=1.0 - ft_ratio)

        # backbone.classifier = (Dropout, Linear(1280, 1000))
        feature_dim = 1280
        backbone.classifier = nn.Identity()

        self.model = nn.Sequential(
            backbone,
            ClassificationHead(
                in_features=feature_dim,
                num_classes=self.num_classes,
                fc_units=self.config.model.fc_units,
                dropout1=self.config.model.dropout1,
                dropout2=self.config.model.dropout2,
            ),
        )
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"EfficientNetV2-S TV: total {total:,}, trainable {trainable:,}")
        return self.model
