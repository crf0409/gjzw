# -*- coding: utf-8 -*-
"""
WIRE: Wavelet Implicit Neural Representation (Saragadam et al., CVPR 2023)

激活: ψ(x) = sin(ω₀ x) · exp(-(s₀ x)²)  (复 Gabor wavelet 的实部)

相比 SIREN 在边缘/纹理细节上更鲁棒, 适合古建筑细粒度纹理 (雕刻/瓦片).
本实现仅提供单图像版本; 批量版本可类比 BatchedSIREN 推广.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class GaborLayer(nn.Module):
    """y = sin(ω₀ · (Wx + b)) · exp(-(s₀ · (Wx + b))²)"""

    def __init__(self, in_features: int, out_features: int,
                 omega_0: float = 10.0, sigma_0: float = 10.0,
                 trainable_omega: bool = False):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.omega_0 = (nn.Parameter(torch.tensor(omega_0))
                        if trainable_omega
                        else torch.tensor(omega_0))
        self.sigma_0 = (nn.Parameter(torch.tensor(sigma_0))
                        if trainable_omega
                        else torch.tensor(sigma_0))
        self._init_weights(in_features)

    def _init_weights(self, in_features: int):
        # WIRE 用普通 Kaiming 初始化即可
        with torch.no_grad():
            bound = 1.0 / math.sqrt(in_features)
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.zero_()

    def forward(self, x):
        z = self.linear(x)
        omega = self.omega_0.to(z.device)
        sigma = self.sigma_0.to(z.device)
        return torch.sin(omega * z) * torch.exp(-(sigma * z) ** 2)


class WIRE(nn.Module):
    """单图像 WIRE 网络."""

    def __init__(self, in_dim: int = 2, out_dim: int = 3,
                 hidden_dim: int = 256, num_layers: int = 4,
                 omega_0: float = 10.0, sigma_0: float = 10.0,
                 final_linear: bool = True):
        super().__init__()
        layers = []
        layers.append(GaborLayer(in_dim, hidden_dim,
                                  omega_0=omega_0, sigma_0=sigma_0))
        for _ in range(num_layers - 2):
            layers.append(GaborLayer(hidden_dim, hidden_dim,
                                      omega_0=omega_0, sigma_0=sigma_0))
        if final_linear:
            layers.append(nn.Linear(hidden_dim, out_dim))
        else:
            layers.append(GaborLayer(hidden_dim, out_dim,
                                      omega_0=omega_0, sigma_0=sigma_0))
        self.net = nn.Sequential(*layers)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.net(coords)
