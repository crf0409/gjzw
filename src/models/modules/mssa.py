# -*- coding: utf-8 -*-
"""
多尺度风格注意力 (MSSA) + 跨尺度门控融合 (CSGF) — AAFNet 核心模块

动机:
    古建筑诊断特征跨三个尺度——微观纹理 (砖瓦木纹)、中观结构 (斗拱、檐口、
    窗格)、宏观轮廓 (屋顶形制、层叠立面)。单一 GAP 把这些坍缩了。
    本模块在每个尺度上提取风格 token, 做空间-通道双路注意力, 再用三尺度
    softmax 门控融合, 强化建筑层次结构感知。

接口:
    feature_extractor: 任何返回 list[Tensor] 的特征抽取器, 每个 Tensor 是
                       [B, C_i, H_i, W_i] 形状的中间特征图。
    MSSABackbone(extractor, channels=[C2,C3,C4], num_classes=K)
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────
# 模块 1: 单尺度风格注意力 MSSABlock
# ─────────────────────────────────────────────────────────────────

class MSSABlock(nn.Module):
    """
    在某个尺度的特征图 F ∈ R^{B×C×H×W} 上做风格-空间双路注意力:
        z = GAP(F)                                # [B, C]
        s = MLP(z)                                # [B, style_dim]  风格 token
        spatial_attn = sigmoid(Conv1x1(F))        # [B, 1, H, W]
        channel_gate = sigmoid(MLP_inv(s))        # [B, C, 1, 1]
        F' = F * spatial_attn * channel_gate
    再返回向量化的 GAP(F').

    Returns:
        feat_vec: [B, C]   GAP(F') 的向量
        F_attn:   [B, C, H, W]  注意力调制后的特征图（供可视化）
    """

    def __init__(self, in_channels: int, style_dim: int = 128,
                 reduction: int = 4):
        super().__init__()
        self.in_channels = in_channels
        self.style_dim = style_dim

        hidden = max(in_channels // reduction, 16)
        # 风格 token: GAP -> MLP -> style_dim
        self.style_mlp = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, style_dim),
        )
        # 通道门: style_dim -> C
        self.channel_gate = nn.Sequential(
            nn.Linear(style_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, in_channels),
            nn.Sigmoid(),
        )
        # 空间注意力: 1x1 conv -> 1 channel sigmoid
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, F_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # F_map: [B, C, H, W]
        b, c, h, w = F_map.shape
        z = F.adaptive_avg_pool2d(F_map, 1).flatten(1)        # [B, C]
        s = self.style_mlp(z)                                  # [B, style_dim]

        ch_gate = self.channel_gate(s).view(b, c, 1, 1)        # [B, C, 1, 1]
        sp_attn = self.spatial_attn(F_map)                     # [B, 1, H, W]

        F_attn = F_map * sp_attn * ch_gate                     # [B, C, H, W]
        feat_vec = F.adaptive_avg_pool2d(F_attn, 1).flatten(1) # [B, C]

        return feat_vec, F_attn, s


# ─────────────────────────────────────────────────────────────────
# 模块 2: 跨尺度门控融合 CSGFBlock
# ─────────────────────────────────────────────────────────────────

class CSGFBlock(nn.Module):
    """
    多尺度特征向量列表 [v_2, v_3, v_4] (各 [B, C_i]) -> 融合向量 [B, fused_dim]

    1. 每个尺度过一个独立的 1xC_i -> fused_dim 投影 (Linear + LN)
    2. softmax 门控: g = softmax(W_g · concat(v_2, v_3, v_4))  ∈ [B, 3]
    3. fused = Σ_i g_i * proj_i(v_i)

    Returns:
        fused: [B, fused_dim]
        gate:  [B, 3]  各尺度权重，可用于可视化每类对哪个尺度依赖
    """

    def __init__(self, channels: Sequence[int], fused_dim: int = 512):
        super().__init__()
        self.n_scales = len(channels)
        self.projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(c, fused_dim),
                nn.LayerNorm(fused_dim),
                nn.ReLU(inplace=True),
            ) for c in channels
        ])
        # 门控: 输入 concat 的原始向量(求和后维度 = sum(channels))
        total = sum(channels)
        self.gate = nn.Sequential(
            nn.Linear(total, total // 4),
            nn.ReLU(inplace=True),
            nn.Linear(total // 4, self.n_scales),
        )

    def forward(self, vecs: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        assert len(vecs) == self.n_scales, \
            f"expected {self.n_scales} scale vectors, got {len(vecs)}"
        concat = torch.cat(vecs, dim=1)                  # [B, sum(C_i)]
        gate_logits = self.gate(concat)                  # [B, n_scales]
        gate = F.softmax(gate_logits, dim=1)             # [B, n_scales]
        projected = [p(v) for p, v in zip(self.projs, vecs)]  # n × [B, fused_dim]
        stacked = torch.stack(projected, dim=1)          # [B, n_scales, fused_dim]
        fused = (gate.unsqueeze(-1) * stacked).sum(dim=1) # [B, fused_dim]
        return fused, gate


# ─────────────────────────────────────────────────────────────────
# 模块 3: MSSABackbone — 把 backbone + MSSA + CSGF 包装到一起
# ─────────────────────────────────────────────────────────────────

class MSSABackbone(nn.Module):
    """
    包装一个返回多尺度特征图列表的特征抽取器, 再套一层 MSSA + CSGF 融合,
    最终通过 head 得到分类 logits 与（可选）SupCon 投影向量。

    Args:
        feature_extractor: 一个 nn.Module, forward(x) 返回 list[Tensor] (各
            尺度的特征图, e.g. [F2, F3, F4])
        channels: 与 feature_extractor 输出对应的通道数列表 (e.g. [512, 1024, 2048])
        num_classes: 分类类别数
        fused_dim: 融合后特征维度
        style_dim: 风格 token 维度
        head: 分类头模块 (从外部传入, 通常为 ClassificationHead)。若为 None,
            内部构造一个简单的 BN→Dropout→FC→ReLU→Dropout→FC 头。
        proj_dim: SupCon 投影头维度 (>0 时启用), 0 表示不返回投影向量
        head_dropout1, head_dropout2, head_fc_units: 默认 head 的超参数

    Forward Returns:
        若 proj_dim == 0:   logits, gate
        若 proj_dim > 0:    (logits, proj), gate
    """

    def __init__(
        self,
        feature_extractor: nn.Module,
        channels: Sequence[int],
        num_classes: int,
        fused_dim: int = 512,
        style_dim: int = 128,
        head: nn.Module | None = None,
        proj_dim: int = 0,
        head_dropout1: float = 0.3,
        head_dropout2: float = 0.2,
        head_fc_units: int = 256,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.channels = list(channels)
        self.num_classes = num_classes
        self.fused_dim = fused_dim
        self.style_dim = style_dim
        self.proj_dim = proj_dim

        # 每尺度一个 MSSA
        self.mssa_blocks = nn.ModuleList([
            MSSABlock(c, style_dim=style_dim) for c in self.channels
        ])
        # 跨尺度门控融合
        self.csgf = CSGFBlock(self.channels, fused_dim=fused_dim)

        # 分类头 (默认走 ClassificationHead 风格)
        if head is None:
            self.head = nn.Sequential(
                nn.BatchNorm1d(fused_dim),
                nn.Dropout(head_dropout1),
                nn.Linear(fused_dim, head_fc_units),
                nn.ReLU(inplace=True),
                nn.Dropout(head_dropout2),
                nn.Linear(head_fc_units, num_classes),
            )
        else:
            self.head = head

        # SupCon 投影头 (可选)
        if proj_dim > 0:
            self.proj_head = nn.Sequential(
                nn.Linear(fused_dim, fused_dim),
                nn.ReLU(inplace=True),
                nn.Linear(fused_dim, proj_dim),
            )
        else:
            self.proj_head = None

    def forward(self, x: torch.Tensor):
        """前向. 返回:
            - 默认: logits
            - 若 supcon 启用 (proj_head is not None): (logits, proj)
        gate 与中间特征缓存到 self.last_* 供可视化, 不出现在返回里, 保持接口
        与现有 nn.Sequential(backbone, head) 一致.
        """
        feats = self.feature_extractor(x)
        if not isinstance(feats, (list, tuple)):
            raise TypeError(
                "feature_extractor must return list/tuple of feature maps; "
                f"got {type(feats)}"
            )
        if len(feats) != len(self.channels):
            raise ValueError(
                f"feature_extractor returned {len(feats)} maps, "
                f"but channels has {len(self.channels)}"
            )
        # 每尺度过 MSSA
        scale_vecs = []
        style_tokens = []
        for fmap, mssa in zip(feats, self.mssa_blocks):
            v, _, s = mssa(fmap)
            scale_vecs.append(v)
            style_tokens.append(s)

        fused, gate = self.csgf(scale_vecs)               # [B, fused_dim], [B, n_scales]
        # 缓存供可视化 (detach 防止梯度泄漏到下次 forward)
        self.last_gate = gate.detach()
        self.last_fused = fused.detach()
        self.last_style_tokens = [s.detach() for s in style_tokens]

        logits = self.head(fused)                          # [B, num_classes]

        if self.proj_head is not None:
            proj = self.proj_head(fused)
            proj = F.normalize(proj, dim=1)               # 单位化便于 SupCon
            return logits, proj
        return logits


# ─────────────────────────────────────────────────────────────────
# 工具: 从一个普通 backbone 构造 MSSABackbone 的助手
# ─────────────────────────────────────────────────────────────────

def build_mssa_backbone(
    feature_extractor: nn.Module,
    channels: Sequence[int],
    num_classes: int,
    aafnet_cfg,
) -> MSSABackbone:
    """
    根据 aafnet 配置 (config.aafnet) 构造 MSSABackbone.

    Args:
        feature_extractor: 多尺度特征抽取器
        channels: 通道列表
        num_classes: 类别数
        aafnet_cfg: config.aafnet 子树 (DictConfig)
    """
    msa_cfg = aafnet_cfg.msa
    loss_cfg = aafnet_cfg.loss

    fused_dim = getattr(msa_cfg, "fused_dim", 512)
    style_dim = getattr(msa_cfg, "style_dim", 128)
    proj_dim = int(getattr(loss_cfg, "proj_dim", 0)) if \
        float(getattr(loss_cfg, "supcon_weight", 0.0)) > 0 else 0

    return MSSABackbone(
        feature_extractor=feature_extractor,
        channels=channels,
        num_classes=num_classes,
        fused_dim=fused_dim,
        style_dim=style_dim,
        proj_dim=proj_dim,
    )
