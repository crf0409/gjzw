# -*- coding: utf-8 -*-
"""ResNet50 分类器 (PyTorch)"""

import torch.nn as nn
import torchvision.models as models

from ..base_classifier import BaseClassifier, ClassificationHead
from .registry import register_backbone


@register_backbone('resnet50')
class ResNet50Classifier(BaseClassifier):
    """ResNet50 图像分类器"""

    def build_model(self):
        """构建 ResNet50 模型架构"""
        backbone = models.resnet50(weights='IMAGENET1K_V1')
        print("ResNet50 pre-trained weights loaded (ImageNet)")

        # 冻结策略：冻结除 layer3, layer4 之外的所有层
        for name, param in backbone.named_parameters():
            if 'layer3' not in name and 'layer4' not in name:
                param.requires_grad = False

        trainable = sum(1 for p in backbone.parameters() if p.requires_grad)
        total = sum(1 for _ in backbone.parameters())
        print(f"ResNet50: Total params {total}, Trainable {trainable}")

        # 替换分类头
        feature_dim = backbone.fc.in_features  # 2048
        backbone.fc = nn.Identity()

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

        print(f"\n=== ResNet50 Model Architecture ===")
        print(self.model)

        return self.model
