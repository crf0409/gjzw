# -*- coding: utf-8 -*-
"""
DW-MoE: 多 backbone 集成

实现三种集成模式:
    soft_vote           — 等权 softmax 平均
    diversity_weighted  — 权重 ∝ val_acc · (1 − mean_pearson_corr_to_others)
    moe                 — 学习门控: 在 val 集上用 MLP 把 [B, K, ...] 学成
                          alpha [B, K], 加权求和 logits

输入数据格式 (collect_predictions.py 的输出):
    members.npz {
        'logits':   [N, K, C]   # K 个成员各自的 logits
        'labels':   [N]
        'val_acc':  [K]         # 每个成员在 val 上的精度
    }
其中 N=验证或测试样本数, K=成员数, C=类别数.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────
# 1. Soft-vote
# ─────────────────────────────────────────────────────────────────

def soft_vote(logits_KNC: np.ndarray) -> np.ndarray:
    """[K, N, C] -> [N, C] 等权 softmax 平均."""
    probs = _softmax(logits_KNC, axis=2)
    return probs.mean(axis=0)


def _softmax(x: np.ndarray, axis: int) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / ex.sum(axis=axis, keepdims=True)


# ─────────────────────────────────────────────────────────────────
# 2. Diversity-weighted
# ─────────────────────────────────────────────────────────────────

def diversity_weighted(logits_KNC: np.ndarray,
                        val_acc_K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        probs: [N, C]
        weights: [K]
    """
    K = logits_KNC.shape[0]
    probs_KNC = _softmax(logits_KNC, axis=2)
    # 计算成员两两 Pearson 相关 (按预测向量 flatten 后)
    flat = probs_KNC.reshape(K, -1)
    corr = np.corrcoef(flat)                                # [K, K]
    np.fill_diagonal(corr, 0)
    diversity = 1.0 - np.abs(corr).mean(axis=1)             # [K]
    # 权重 = val_acc × diversity, 归一化
    raw = np.asarray(val_acc_K) * diversity
    raw = np.clip(raw, 1e-6, None)
    weights = raw / raw.sum()
    fused = (weights[:, None, None] * probs_KNC).sum(axis=0)
    return fused, weights


# ─────────────────────────────────────────────────────────────────
# 3. MoE 学习门控 (可训练)
# ─────────────────────────────────────────────────────────────────

class DWMoE(nn.Module):
    """
    输入每个成员的 logits [B, K, C], 输出加权后的 logits [B, C].

    门控网络: MLP 接受 概率分布的 'meta features':
        - 每成员 max prob, top-2 间距, entropy, val_acc 常数
    输入维度: K * 4
    """

    def __init__(self, num_members: int, num_classes: int,
                 val_acc: torch.Tensor,
                 hidden: int = 32):
        super().__init__()
        self.K = num_members
        self.C = num_classes
        # 学习门控
        self.gate = nn.Sequential(
            nn.Linear(num_members * 4, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_members),
        )
        self.register_buffer("val_acc",
                              torch.as_tensor(val_acc, dtype=torch.float32))

    def _meta(self, logits_BKC: torch.Tensor) -> torch.Tensor:
        """从 [B, K, C] 提取每成员 4 个 meta 特征."""
        probs = F.softmax(logits_BKC, dim=2)
        max_p = probs.max(dim=2).values                      # [B, K]
        top2_gap = torch.topk(probs, 2, dim=2).values
        gap = top2_gap[..., 0] - top2_gap[..., 1]            # [B, K]
        entropy = -(probs * (probs.clamp_min(1e-12).log())).sum(dim=2)  # [B, K]
        acc = self.val_acc.unsqueeze(0).expand_as(max_p)     # [B, K]
        feat = torch.stack([max_p, gap, entropy, acc], dim=-1)  # [B, K, 4]
        return feat.flatten(1)                                # [B, K*4]

    def forward(self, logits_BKC: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits_BKC: [B, K, C]
        Returns:
            logits: [B, C]   (加权 softmax 后再 log -> 等价 log-mixture)
        """
        meta = self._meta(logits_BKC)
        alpha = F.softmax(self.gate(meta), dim=1)            # [B, K]
        probs = F.softmax(logits_BKC, dim=2)                 # [B, K, C]
        fused = (alpha.unsqueeze(-1) * probs).sum(dim=1)     # [B, C]
        # 返回 log 概率作为 'logits' 等价
        return torch.log(fused.clamp_min(1e-12))

    @torch.no_grad()
    def gating_weights(self, logits_BKC: torch.Tensor) -> torch.Tensor:
        """诊断用: 返回每个样本的门控权重 [B, K]."""
        meta = self._meta(logits_BKC)
        return F.softmax(self.gate(meta), dim=1)


def fit_moe_gate(train_logits: torch.Tensor,
                  train_labels: torch.Tensor,
                  val_acc: torch.Tensor,
                  num_classes: int,
                  epochs: int = 5, lr: float = 1e-2,
                  device: torch.device = torch.device("cpu")
                  ) -> DWMoE:
    """在 val 集上训练 MoE 门控."""
    K = train_logits.shape[1]
    moe = DWMoE(num_members=K, num_classes=num_classes,
                  val_acc=val_acc).to(device)
    opt = torch.optim.Adam(moe.parameters(), lr=lr)
    crit = nn.NLLLoss()
    for ep in range(epochs):
        moe.train()
        perm = torch.randperm(len(train_logits))
        for i in range(0, len(perm), 32):
            idx = perm[i:i + 32]
            x = train_logits[idx].to(device)
            y = train_labels[idx].to(device)
            log_p = moe(x)
            loss = crit(log_p, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return moe
