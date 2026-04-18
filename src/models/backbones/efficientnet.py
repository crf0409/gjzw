# -*- coding: utf-8 -*-
"""EfficientNet 分类器 (PyTorch)"""

import torch.nn as nn
import torchvision.models as models

from ..base_classifier import BaseClassifier, ClassificationHead
from .registry import register_backbone


@register_backbone('efficientnet_b3')
class EfficientNetB3Classifier(BaseClassifier):
    """EfficientNetB3 图像分类器"""

    def build_model(self):
        """构建 EfficientNetB3 模型架构"""
        backbone = models.efficientnet_b3(weights='IMAGENET1K_V1')
        print("EfficientNetB3 pre-trained weights loaded (ImageNet)")

        # 微调策略：冻结前 80% 的参数
        all_params = list(backbone.named_parameters())
        fine_tune_at = int(len(all_params) * self.config.model.fine_tune_ratio)
        for name, param in all_params[:fine_tune_at]:
            param.requires_grad = False

        trainable = sum(1 for p in backbone.parameters() if p.requires_grad)
        total = sum(1 for _ in backbone.parameters())
        print(f"EfficientNetB3: Total params {total}, Trainable {trainable}")

        # 替换分类头
        feature_dim = backbone.classifier[1].in_features  # 1536
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

        print(f"\n=== EfficientNetB3 Model Architecture ===")
        print(self.model)

        return self.model
