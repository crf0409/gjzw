# -*- coding: utf-8 -*-
"""自定义MLP分类器 (PyTorch)"""

import torch.nn as nn

from ..base_classifier import BaseClassifier
from .registry import register_backbone


@register_backbone('custom_mlp')
class CustomMLPClassifier(BaseClassifier):
    """
    自定义MLP图像分类器

    配置：
    - 输入: 224x224 灰度 (1通道，不转RGB)
    - 隐藏层: fc_units 神经元
    - Dropout / L2 正则化通过 config 控制
    """

    def __init__(self, config):
        # 强制设置图像尺寸为 224x224
        config.data.img_height = 224
        config.data.img_width = 224
        super().__init__(config)
        self._to_rgb = False  # MLP 使用 1 通道灰度输入

    def build_model(self):
        """构建自定义MLP模型"""
        fc_units = self.config.model.get('fc_units', 300)
        dropout_rate = self.config.model.get('dropout1', 0.2)

        print(f"\n=== Custom MLP Configuration ===")
        print(f"Input size: {self.img_height}x{self.img_width}")
        print(f"Hidden units: {fc_units}")
        print(f"Dropout rate: {dropout_rate}")
        print("=" * 40)

        input_dim = self.img_height * self.img_width  # 1通道

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, fc_units),
            nn.BatchNorm1d(fc_units),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(fc_units, fc_units // 2),
            nn.BatchNorm1d(fc_units // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(fc_units // 2, self.num_classes),
        )

        print(f"\n=== Custom MLP Model Architecture ===")
        print(self.model)

        return self.model
