# -*- coding: utf-8 -*-
"""
INR-based 分类器: 把每张图像的 SIREN 权重 (作为压缩神经表示) 输入分类头.

提供两种分类头:
    INRMLPClassifier     — 朴素 flatten + MLP, 简单 baseline
    INRDeepSetsClassifier — 把 SIREN 权重按层分组, 用 DeepSets 风格聚合
                            (W_i 视为一组神经元, 用 permutation-invariant pool)

架构呼应 paper_texture_analysis.py 的"显式纹理特征"分析: 我们的 INR
表示是"隐式纹理特征"——在论文 Discussion 里可以做对应.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..base_classifier import BaseClassifier, ClassificationHead
from .registry_helpers import register_inr_backbone


class INRMLPClassifier(nn.Module):
    """
    Flatten SIREN 权重 -> Layer-norm -> MLP -> logits

    Args:
        in_dim:      SIREN 权重总数 (params_per_image)
        num_classes: 类别数
        hidden:      隐层宽度
        dropout:     dropout 率
    """

    def __init__(self, in_dim: int, num_classes: int,
                 hidden: int = 512, dropout: float = 0.3):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_classes),
        )

    def forward(self, weights: torch.Tensor) -> torch.Tensor:
        """
        Args:
            weights: [B, in_dim] flatten 的 SIREN 权重
        Returns:
            logits: [B, num_classes]
        """
        x = self.norm(weights)
        return self.head(x)


class INRDeepSetsClassifier(nn.Module):
    """
    把每层的 W (out_features 维) 视为一组神经元, 在神经元维度上做 DeepSets pool.
    这种做法 permutation-invariant 于神经元排列, 比纯 flatten 更鲁棒.

    Args:
        layer_specs:  list of (W_shape, b_shape), e.g. [((2, 256), (256,)),
                                                          ((256, 256), (256,)),
                                                          ((256, 3), (3,))]
        num_classes:  类别数
        embed_dim:    每个神经元的嵌入维度
        hidden:       聚合后 MLP 宽度
    """

    def __init__(self, layer_specs: list, num_classes: int,
                 embed_dim: int = 64, hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.layer_specs = layer_specs
        # 每层一个神经元嵌入: 每个神经元向量 = (W[:, k], b[k]) 拼接 -> embed
        self.neuron_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(W_shape[0] + 1, embed_dim),  # +1 for bias
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
            ) for W_shape, _ in layer_specs
        ])
        self.layer_pool_dim = embed_dim * 2  # mean + max
        total_dim = self.layer_pool_dim * len(layer_specs)
        self.head = nn.Sequential(
            nn.LayerNorm(total_dim),
            nn.Linear(total_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, layer_weights: list) -> torch.Tensor:
        """
        Args:
            layer_weights: list of dict, 每元素 {"W": [B, d_in, d_out],
                                                  "b": [B, d_out]}
        Returns:
            logits: [B, num_classes]
        """
        layer_pools = []
        for spec, encoder, lw in zip(self.layer_specs, self.neuron_encoders,
                                       layer_weights):
            W = lw["W"]                                         # [B, d_in, d_out]
            b = lw["b"]                                         # [B, d_out]
            B, d_in, d_out = W.shape
            # 每个神经元的输入向量 (W[:, k], b[k])
            neurons = torch.cat(
                [W.transpose(1, 2),                              # [B, d_out, d_in]
                 b.unsqueeze(-1)],                                # [B, d_out, 1]
                dim=-1,
            )                                                   # [B, d_out, d_in+1]
            embed = encoder(neurons)                            # [B, d_out, embed_dim]
            mean_pool = embed.mean(dim=1)                       # [B, embed_dim]
            max_pool, _ = embed.max(dim=1)                      # [B, embed_dim]
            layer_pools.append(torch.cat([mean_pool, max_pool], dim=1))
        x = torch.cat(layer_pools, dim=1)                       # [B, total_dim]
        return self.head(x)
