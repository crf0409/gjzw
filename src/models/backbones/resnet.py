# -*- coding: utf-8 -*-
"""ResNet50 分类器 (PyTorch) — 支持原版 + AAFNet MSSA 多尺度模式"""

import torch.nn as nn
import torchvision.models as models

from ..base_classifier import BaseClassifier, ClassificationHead
from ..modules import MSSABackbone, build_mssa_backbone
from .registry import register_backbone


# ─────────────────────────────────────────────────────────────────
# ResNet-50 多尺度特征抽取器: 返回 [layer2_out, layer3_out, layer4_out]
# ─────────────────────────────────────────────────────────────────

class _ResNet50MultiScale(nn.Module):
    """暴露 layer2/3/4 三尺度特征图的 ResNet-50 抽取器."""

    # layer2/3/4 输出通道数
    channels = [512, 1024, 2048]

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = "IMAGENET1K_V2" if pretrained else None
        backbone = models.resnet50(weights=weights)
        # 冻结策略: layer3/4 之外全冻 (与原版一致)
        for name, p in backbone.named_parameters():
            if "layer3" not in name and "layer4" not in name:
                p.requires_grad = False
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        f2 = self.layer2(x)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return [f2, f3, f4]


# ─────────────────────────────────────────────────────────────────
# 主分类器
# ─────────────────────────────────────────────────────────────────

@register_backbone('resnet50')
class ResNet50Classifier(BaseClassifier):
    """ResNet50 图像分类器 (支持 AAFNet MSSA 模式)"""

    def build_model(self):
        """
        当 config.aafnet.msa.enabled == True 时构建 MSSA 多尺度模型,
        否则构建原版 nn.Sequential(backbone, head).
        """
        aafnet = getattr(self.config, "aafnet", None)
        use_msa = bool(aafnet and getattr(aafnet.msa, "enabled", False))

        if use_msa:
            extractor = _ResNet50MultiScale(pretrained=True)
            self.model = build_mssa_backbone(
                feature_extractor=extractor,
                channels=extractor.channels,
                num_classes=self.num_classes,
                aafnet_cfg=aafnet,
            )
            print(f"ResNet50 + MSSA pre-trained weights loaded (ImageNet V2)")
            print(f"  multi-scale channels = {extractor.channels}")
            print(f"  fused_dim = {self.model.fused_dim}, "
                  f"style_dim = {self.model.style_dim}, "
                  f"proj_dim = {self.model.proj_dim}")
        else:
            backbone = models.resnet50(weights='IMAGENET1K_V2')
            print("ResNet50 pre-trained weights loaded (ImageNet V2)")
            for name, param in backbone.named_parameters():
                if 'layer3' not in name and 'layer4' not in name:
                    param.requires_grad = False
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

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"ResNet50: total params {total:,}, trainable {trainable:,}")

        return self.model
