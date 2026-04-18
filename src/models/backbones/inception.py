# -*- coding: utf-8 -*-
"""Inception 系列分类器 (PyTorch)"""

import torch
import torch.nn as nn
import torchvision.models as models

from ..base_classifier import BaseClassifier, ClassificationHead
from .registry import register_backbone


class _InceptionV3Wrapper(nn.Module):
    """InceptionV3 包装器，处理 aux_logits 输出问题"""

    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        # InceptionV3 训练时返回 InceptionOutputs(logits, aux_logits)
        # 我们只取主输出
        out = self.backbone(x)
        if isinstance(out, tuple):
            out = out[0]
        return self.head(out)


@register_backbone('inception_v3')
class InceptionV3Classifier(BaseClassifier):
    """InceptionV3 图像分类器"""

    def build_model(self):
        """构建 InceptionV3 模型架构"""
        # 先用 aux_logits=True 加载权重，再禁用
        backbone = models.inception_v3(weights='IMAGENET1K_V1')
        backbone.aux_logits = False
        backbone.AuxLogits = None
        print("InceptionV3 pre-trained weights loaded (ImageNet)")

        # 微调策略：冻结前 80% 的参数
        all_params = list(backbone.named_parameters())
        fine_tune_at = int(len(all_params) * self.config.model.fine_tune_ratio)
        for name, param in all_params[:fine_tune_at]:
            param.requires_grad = False

        trainable = sum(1 for p in backbone.parameters() if p.requires_grad)
        total = sum(1 for _ in backbone.parameters())
        print(f"InceptionV3: Total params {total}, Trainable {trainable}")

        # 替换分类头
        feature_dim = backbone.fc.in_features  # 2048
        backbone.fc = nn.Identity()

        head = ClassificationHead(
            in_features=feature_dim,
            num_classes=self.num_classes,
            fc_units=self.config.model.fc_units,
            dropout1=self.config.model.dropout1,
            dropout2=self.config.model.dropout2,
        )

        self.model = nn.Sequential(backbone, head)

        print(f"\n=== InceptionV3 Model Architecture ===")
        print(self.model)

        return self.model


@register_backbone('inception_resnet_v2')
class InceptionResNetV2Classifier(BaseClassifier):
    """Inception-ResNet-V2 图像分类器"""

    def build_model(self):
        """构建 Inception-ResNet-V2 模型架构"""
        import timm

        backbone = timm.create_model('inception_resnet_v2', pretrained=True)
        print("InceptionResNetV2 pre-trained weights loaded (timm)")

        # 微调策略：冻结前 80% 的参数
        all_params = list(backbone.named_parameters())
        fine_tune_at = int(len(all_params) * self.config.model.fine_tune_ratio)
        for name, param in all_params[:fine_tune_at]:
            param.requires_grad = False

        trainable = sum(1 for p in backbone.parameters() if p.requires_grad)
        total = sum(1 for _ in backbone.parameters())
        print(f"InceptionResNetV2: Total params {total}, Trainable {trainable}")

        # 获取分类器维度并移除分类头
        feature_dim = backbone.get_classifier().in_features
        backbone.reset_classifier(0)

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

        print(f"\n=== InceptionResNetV2 Model Architecture ===")
        print(self.model)

        return self.model
