# -*- coding: utf-8 -*-
"""VGG16/VGG19 分类器 (PyTorch)"""

import torch.nn as nn
import torchvision.models as models

from ..base_classifier import BaseClassifier, ClassificationHead
from .registry import register_backbone


class _VGGBase(BaseClassifier):
    """VGG 系列公共基类"""

    def _build_vgg(self, backbone, model_name, freeze_until):
        """
        构建 VGG 模型

        Args:
            backbone: torchvision VGG 模型实例
            model_name: 模型名称
            freeze_until: features 中冻结的层索引上限
        """
        print(f"{model_name} pre-trained weights loaded (ImageNet)")

        # 冻结 features 前半部分
        for i, layer in enumerate(backbone.features):
            if i < freeze_until:
                for param in layer.parameters():
                    param.requires_grad = False

        trainable = sum(1 for p in backbone.parameters() if p.requires_grad)
        total = sum(1 for _ in backbone.parameters())
        print(f"{model_name}: Total params {total}, Trainable {trainable}")

        # 去掉原始 classifier，改用 AdaptiveAvgPool2d + ClassificationHead
        features = backbone.features
        pool = nn.AdaptiveAvgPool2d(1)
        flatten = nn.Flatten()

        # VGG features 最后一层输出 512 通道
        feature_dim = 512

        head = ClassificationHead(
            in_features=feature_dim,
            num_classes=self.num_classes,
            fc_units=self.config.model.fc_units,
            dropout1=self.config.model.dropout1,
            dropout2=self.config.model.dropout2,
        )

        self.model = nn.Sequential(features, pool, flatten, head)

        print(f"\n=== {model_name} Model Architecture ===")
        print(self.model)

        return self.model


@register_backbone('vgg16')
class VGG16Classifier(_VGGBase):
    """VGG16 图像分类器"""

    def build_model(self):
        backbone = models.vgg16(weights='IMAGENET1K_V1')
        # 冻结 features[:17] (block1-3)
        return self._build_vgg(backbone, 'VGG16', freeze_until=17)


@register_backbone('vgg19')
class VGG19Classifier(_VGGBase):
    """VGG19 图像分类器"""

    def build_model(self):
        backbone = models.vgg19(weights='IMAGENET1K_V1')
        # 冻结 features[:19] (block1-3)
        return self._build_vgg(backbone, 'VGG19', freeze_until=19)
