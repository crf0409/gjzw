# -*- coding: utf-8 -*-
"""
timm 通用 SOTA backbone 工厂

注册:
    convnext_tiny       — ConvNeXt-Tiny (Liu 2022)
    swin_tiny           — Swin-Tiny (Liu 2021)
    maxvit_tiny         — MaxViT-Tiny (Tu 2022)
    efficientnetv2_s    — EfficientNetV2-S (Tan 2021)
    regnety_032         — RegNetY-032 (Radosavovic 2020)

每个 backbone 都接 ClassificationHead, 与原版风格一致.
冻结策略: fine_tune_ratio (config.model.fine_tune_ratio) 控制比例;
默认 0.8 表示后 80% 参数可训练 (前 20% 冻).
"""

from __future__ import annotations

import timm
import torch.nn as nn

from ..base_classifier import BaseClassifier, ClassificationHead
from .registry import register_backbone


def _freeze_first_fraction(model: nn.Module, freeze_fraction: float) -> None:
    """按参数顺序冻结前 freeze_fraction 比例的参数."""
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


def _make_timm_classifier(register_name: str, timm_name: str,
                           default_input: int = 224):
    """
    为一个 timm 模型生成并注册 BaseClassifier 子类.

    Args:
        register_name: 在 BACKBONE_REGISTRY 里的名字
        timm_name:     timm.create_model 用的完整名 (含预训练 tag)
        default_input: 默认输入分辨率 (供 train_all_paper 使用)
    """

    @register_backbone(register_name)
    class _Classifier(BaseClassifier):
        TIMM_NAME = timm_name
        DEFAULT_INPUT = default_input

        def build_model(self):
            backbone = timm.create_model(
                self.TIMM_NAME,
                pretrained=True,
                num_classes=0,        # 去掉自带分类头, 我们用 ClassificationHead
                global_pool="avg",    # 强制返回 [B, C] 池化向量
            )
            print(f"{register_name} pre-trained weights loaded "
                  f"(timm: {self.TIMM_NAME})")

            # 冻结策略
            ft_ratio = float(self.config.model.fine_tune_ratio)
            _freeze_first_fraction(backbone, freeze_fraction=1.0 - ft_ratio)

            feature_dim = backbone.num_features
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

            trainable = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
            total = sum(p.numel() for p in self.model.parameters())
            print(f"{register_name}: total {total:,}, trainable {trainable:,}")
            return self.model

    _Classifier.__name__ = f"{register_name.title().replace('_','')}Classifier"
    return _Classifier


# 注册 5 个新 SOTA backbone
ConvNeXtTiny      = _make_timm_classifier("convnext_tiny",
                                           "convnext_tiny.fb_in22k_ft_in1k", 224)
SwinTiny          = _make_timm_classifier("swin_tiny",
                                           "swin_tiny_patch4_window7_224.ms_in1k", 224)
MaxViTTiny        = _make_timm_classifier("maxvit_tiny",
                                           "maxvit_tiny_tf_224.in1k", 224)
EfficientNetV2S   = _make_timm_classifier("efficientnetv2_s",
                                           "tf_efficientnetv2_s.in21k_ft_in1k", 224)
RegNetY032        = _make_timm_classifier("regnety_032",
                                           "regnety_032.tv2_in1k", 224)
