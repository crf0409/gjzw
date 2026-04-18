# -*- coding: utf-8 -*-
"""Vision Transformer 分类器 (PyTorch)"""

import torch.nn as nn
import torchvision.models as models

from ..base_classifier import BaseClassifier, ClassificationHead
from .registry import register_backbone


@register_backbone('vit_b16')
class ViTB16Classifier(BaseClassifier):
    """Vision Transformer B/16 图像分类器（使用预训练权重）"""

    def build_model(self):
        """构建 ViT-B/16 模型架构"""
        backbone = models.vit_b_16(weights='IMAGENET1K_V1')
        print("ViT-B/16 pre-trained weights loaded (ImageNet)")

        # 微调策略：冻结前 80% 的参数
        all_params = list(backbone.named_parameters())
        fine_tune_at = int(len(all_params) * self.config.model.fine_tune_ratio)
        for name, param in all_params[:fine_tune_at]:
            param.requires_grad = False

        trainable = sum(1 for p in backbone.parameters() if p.requires_grad)
        total = sum(1 for _ in backbone.parameters())
        print(f"ViT-B/16: Total params {total}, Trainable {trainable}")

        # 替换 heads
        feature_dim = backbone.heads.head.in_features  # 768
        backbone.heads = nn.Identity()

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

        print(f"\n=== ViT-B/16 Model Architecture ===")
        print(self.model)

        return self.model
