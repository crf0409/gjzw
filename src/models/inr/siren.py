# -*- coding: utf-8 -*-
"""
SIREN: Sinusoidal Implicit Neural Representation (Sitzmann et al., NeurIPS 2020)

把单张图像表达为 f_θ: (x, y) ∈ [-1,1]² -> RGB ∈ [-1,1]³ 的小型 MLP。
权重 θ 即图像在 INR 空间下的紧凑神经编码, 可作为下游分类的特征。

提供两种实现:
    SIREN          — 标准单图像版本, nn.Module 形式
    BatchedSIREN   — 一次性容纳 B 张独立 SIREN 的批量版本, 用 einsum 在 GPU
                     上并行拟合 B 张图像, 极大加速 pretext task

激活: x -> sin(ω₀ · x)
初始化 (论文要求, 关键!):
    第一层: U(-1/d_in, 1/d_in)
    后续层: U(-sqrt(6/d_in)/ω₀, sqrt(6/d_in)/ω₀)
ω₀ 默认 30.0
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────
# 标准单图像 SIREN
# ─────────────────────────────────────────────────────────────────

class SineLayer(nn.Module):
    """单层 sine: y = sin(ω₀ · (W x + b))"""

    def __init__(self, in_features: int, out_features: int,
                 is_first: bool = False, omega_0: float = 30.0):
        super().__init__()
        self.in_features = in_features
        self.is_first = is_first
        self.omega_0 = float(omega_0)
        self.linear = nn.Linear(in_features, out_features)
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / self.in_features
            else:
                bound = math.sqrt(6.0 / self.in_features) / self.omega_0
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.zero_()

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class SIREN(nn.Module):
    """
    标准 SIREN 单图像实现.

    Args:
        in_dim:       输入坐标维度, 默认 2 (x, y)
        out_dim:      输出维度, 默认 3 (RGB)
        hidden_dim:   隐藏层宽度
        num_layers:   总层数 (含输出层)
        omega_0:      首层角频率 (论文推荐 30)
        omega_hidden: 后续层角频率 (一般 = omega_0)
        final_linear: 末层是否用普通 Linear (不加 sin), 这样输出可任意值域
    """

    def __init__(self, in_dim: int = 2, out_dim: int = 3,
                 hidden_dim: int = 256, num_layers: int = 4,
                 omega_0: float = 30.0,
                 omega_hidden: Optional[float] = None,
                 final_linear: bool = True):
        super().__init__()
        omega_hidden = float(omega_hidden) if omega_hidden is not None else float(omega_0)
        layers = []
        layers.append(SineLayer(in_dim, hidden_dim,
                                 is_first=True, omega_0=omega_0))
        for _ in range(num_layers - 2):
            layers.append(SineLayer(hidden_dim, hidden_dim,
                                     is_first=False, omega_0=omega_hidden))
        if final_linear:
            final = nn.Linear(hidden_dim, out_dim)
            with torch.no_grad():
                bound = math.sqrt(6.0 / hidden_dim) / omega_hidden
                final.weight.uniform_(-bound, bound)
                final.bias.zero_()
            layers.append(final)
        else:
            layers.append(SineLayer(hidden_dim, out_dim,
                                     is_first=False, omega_0=omega_hidden))
        self.net = nn.Sequential(*layers)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """coords: [N, in_dim] -> output: [N, out_dim]"""
        return self.net(coords)


# ─────────────────────────────────────────────────────────────────
# 批量并行 SIREN: 一次容纳 B 个独立 SIREN, 共享坐标输入
# ─────────────────────────────────────────────────────────────────

class BatchedSIREN(nn.Module):
    """
    把 B 个独立 SIREN 网络的权重打包成一个张量, 用 einsum 在 GPU 上并行 forward.
    每个图像有独立的 (W, b), 不共享.

    Args:
        batch_size:   并行图像数 B
        in_dim, out_dim, hidden_dim, num_layers, omega_0, final_linear: 同 SIREN
    """

    def __init__(self, batch_size: int,
                 in_dim: int = 2, out_dim: int = 3,
                 hidden_dim: int = 256, num_layers: int = 4,
                 omega_0: float = 30.0,
                 omega_hidden: Optional[float] = None,
                 final_linear: bool = True):
        super().__init__()
        self.batch_size = int(batch_size)
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.omega_0 = float(omega_0)
        self.omega_hidden = float(omega_hidden) if omega_hidden is not None else float(omega_0)
        self.final_linear = bool(final_linear)

        # 维度序列
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.dims = dims

        # 参数: 每层一个 (W [B, d_in, d_out], b [B, d_out])
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        for i in range(num_layers):
            d_in, d_out = dims[i], dims[i + 1]
            w = torch.empty(batch_size, d_in, d_out)
            b = torch.empty(batch_size, d_out)
            self.weights.append(nn.Parameter(w))
            self.biases.append(nn.Parameter(b))

        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            for i, (w, b) in enumerate(zip(self.weights, self.biases)):
                d_in = self.dims[i]
                if i == 0:
                    bound = 1.0 / d_in
                else:
                    omega = (self.omega_hidden if i < self.num_layers - 1
                             else (self.omega_hidden if not self.final_linear
                                   else self.omega_hidden))
                    bound = math.sqrt(6.0 / d_in) / omega
                w.uniform_(-bound, bound)
                b.zero_()

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: [N, in_dim]   共享坐标网格 (所有 B 张图像同坐标)
        Returns:
            output: [B, N, out_dim]
        """
        x = coords                                            # [N, d_in]
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            if i == 0:
                # [N, d_in] × [B, d_in, d_out] -> [B, N, d_out]
                out = torch.einsum("nd,bdh->bnh", x, w) + b.unsqueeze(1)
            else:
                # [B, N, d_in] × [B, d_in, d_out] -> [B, N, d_out]
                out = torch.einsum("bnd,bdh->bnh", x, w) + b.unsqueeze(1)

            # 激活
            if i < self.num_layers - 1:
                out = torch.sin(self.omega_0 * out)
            elif not self.final_linear:
                out = torch.sin(self.omega_hidden * out)
            x = out
        return x

    def export_weights(self) -> torch.Tensor:
        """
        把每张图像的 SIREN 参数 flatten 成一个向量, 便于做下游分类.

        Returns:
            tensor [B, total_params_per_siren]
        """
        flat_list = []
        for w, b in zip(self.weights, self.biases):
            # w: [B, d_in, d_out], b: [B, d_out]
            flat_list.append(w.detach().reshape(self.batch_size, -1))
            flat_list.append(b.detach().reshape(self.batch_size, -1))
        return torch.cat(flat_list, dim=1)                    # [B, total]

    def export_per_layer(self) -> list[dict]:
        """逐层导出 weight matrix (供 Set Transformer / DeepSets 等)"""
        out = []
        for w, b in zip(self.weights, self.biases):
            out.append({
                "W": w.detach().clone(),
                "b": b.detach().clone(),
            })
        return out

    @property
    def params_per_image(self) -> int:
        """每张图像的 SIREN 参数总数."""
        n = 0
        for w, b in zip(self.weights, self.biases):
            n += w.shape[1] * w.shape[2] + b.shape[1]
        return n


def build_coord_grid(h: int, w: int, device=None) -> torch.Tensor:
    """生成 [-1, 1] 归一化坐标网格 [H*W, 2] (y, x)."""
    ys = torch.linspace(-1.0, 1.0, h, device=device)
    xs = torch.linspace(-1.0, 1.0, w, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    coords = torch.stack([xx, yy], dim=-1).reshape(-1, 2)     # [H*W, 2]
    return coords
