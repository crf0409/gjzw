# -*- coding: utf-8 -*-
"""
SASC-KD 损失体系 — AAFNet 训练目标

提供:
    FocalLabelSmoothCE — Focal Loss + Label Smoothing 复合, 处理类别不均与过自信
    SupConLoss          — 监督对比损失 (Khosla et al., 2020)
    KDLoss              — KL 散度知识蒸馏 (Hinton, 2015)
    CompositeLoss       — 按 config.aafnet.loss.* 自动组装上述损失

接口:
    L = build_loss(num_classes, class_weights, loss_cfg, device, teacher_model=None)
    L(logits, labels)              # type=ce/focal/focalls 时
    L(logits, labels, proj=...)    # type=focalls_supcon 时
    L(logits, labels, proj=..., teacher_logits=...)  # type=focalls_supcon_kd 时
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────
# 1. Focal + Label Smoothing 复合
# ─────────────────────────────────────────────────────────────────

class FocalLabelSmoothCE(nn.Module):
    """
    Focal Loss with Label Smoothing.

        L = -Σ_k (1 - p_k)^γ * ŷ_k * log p_k
        ŷ = (1 - ε) y_onehot + ε / K

    Args:
        gamma: focal 聚焦系数 (>=0). 0 退化为 weighted CE
        smoothing: label smoothing ε ∈ [0, 1)
        weight: 类别权重 [K]
        reduction: 'mean' | 'sum' | 'none'
    """

    def __init__(self, gamma: float = 2.0, smoothing: float = 0.0,
                 weight: Optional[torch.Tensor] = None,
                 reduction: str = "mean"):
        super().__init__()
        self.gamma = float(gamma)
        self.smoothing = float(smoothing)
        self.register_buffer("weight",
                              weight if weight is not None else torch.tensor([]),
                              persistent=False)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, K]
            target: [B] long
        """
        K = logits.size(1)
        log_p = F.log_softmax(logits, dim=1)              # [B, K]
        p = log_p.exp()

        # label smoothing 分布
        with torch.no_grad():
            target_dist = torch.full_like(log_p, self.smoothing / K)
            target_dist.scatter_(
                1, target.unsqueeze(1), 1.0 - self.smoothing + self.smoothing / K
            )

        # focal modulation: (1 - p_k)^γ 作用在每个类别
        if self.gamma > 0:
            focal = (1.0 - p).pow(self.gamma)
        else:
            focal = 1.0

        # 类别权重
        if self.weight.numel() > 0:
            w = self.weight.to(logits.device).unsqueeze(0)  # [1, K]
        else:
            w = 1.0

        # 逐元素 element-wise CE
        per_class_loss = -focal * w * target_dist * log_p   # [B, K]
        loss_per_sample = per_class_loss.sum(dim=1)         # [B]

        if self.reduction == "mean":
            return loss_per_sample.mean()
        if self.reduction == "sum":
            return loss_per_sample.sum()
        return loss_per_sample


# ─────────────────────────────────────────────────────────────────
# 2. 监督对比损失 SupCon (Khosla 2020)
# ─────────────────────────────────────────────────────────────────

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss.

        L_supcon = -1/|P(i)| Σ_p∈P(i) log [exp(z_i·z_p/τ) / Σ_a∈A(i) exp(z_i·z_a/τ)]

    其中 P(i) 是 batch 中和 i 同类的 indices, A(i) 是除 i 外所有 indices.

    Args:
        temperature: τ
        base_temperature: 用于规范化
    """

    def __init__(self, temperature: float = 0.07,
                 base_temperature: float = 0.07):
        super().__init__()
        self.temperature = float(temperature)
        self.base_temperature = float(base_temperature)

    def forward(self, features: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, D] 已 L2-normalized
            labels:   [B] long
        """
        device = features.device
        batch_size = features.size(0)

        # mask[i, j] = 1 if labels[i] == labels[j]
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)   # [B, B]

        # 余弦相似度 (features 已 normalized) / τ
        anchor_dot = torch.matmul(features, features.T) / self.temperature  # [B, B]

        # 数值稳定: 减最大
        anchor_dot_max, _ = torch.max(anchor_dot, dim=1, keepdim=True)
        logits = anchor_dot - anchor_dot_max.detach()

        # mask 掉对角 (自身)
        logits_mask = torch.ones_like(mask, device=device)
        logits_mask.fill_diagonal_(0)
        mask = mask * logits_mask                              # 同类且非自身

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # 对每个 anchor 在其正样本上平均
        pos_count = mask.sum(1)
        # 防止某些样本在 batch 中没有同类 -> 跳过 (loss=0)
        pos_count_safe = pos_count.clamp(min=1.0)
        mean_log_prob_pos = (mask * log_prob).sum(1) / pos_count_safe

        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        # 对所有有效 anchor 取平均
        valid = (pos_count > 0).float()
        if valid.sum() == 0:
            return torch.zeros((), device=device, requires_grad=True)
        return (loss * valid).sum() / valid.sum()


# ─────────────────────────────────────────────────────────────────
# 3. 知识蒸馏损失 (Hinton 2015)
# ─────────────────────────────────────────────────────────────────

class KDLoss(nn.Module):
    """
    Soft-target KL distillation.

        L_KD = T² · KL(softmax(s/T) ‖ softmax(t/T))

    Args:
        temperature: T
    """

    def __init__(self, temperature: float = 4.0):
        super().__init__()
        self.T = float(temperature)

    def forward(self, student_logits: torch.Tensor,
                teacher_logits: torch.Tensor) -> torch.Tensor:
        T = self.T
        s = F.log_softmax(student_logits / T, dim=1)
        t = F.softmax(teacher_logits / T, dim=1)
        return F.kl_div(s, t, reduction="batchmean") * (T * T)


# ─────────────────────────────────────────────────────────────────
# 4. 复合损失组装
# ─────────────────────────────────────────────────────────────────

class CompositeLoss(nn.Module):
    """
    根据 loss_cfg 动态组装 CE + Focal/LS + SupCon + KD.

    支持的 loss_cfg.type:
        ce              — 普通 weighted CE (兼容现有训练)
        focal           — Focal Loss
        focalls         — Focal + Label Smoothing
        focalls_supcon  — focalls + λ1·SupCon  (需要 forward 提供 proj)
        focalls_supcon_kd — focalls + λ1·SupCon + λ2·KD (需要 proj 和 teacher_logits)
    """

    def __init__(self, num_classes: int,
                 class_weights: Optional[torch.Tensor],
                 loss_cfg, device: torch.device):
        super().__init__()
        self.type = str(getattr(loss_cfg, "type", "ce"))
        self.gamma = float(getattr(loss_cfg, "focal_gamma", 2.0))
        self.smoothing = float(getattr(loss_cfg, "label_smoothing", 0.0))
        self.supcon_weight = float(getattr(loss_cfg, "supcon_weight", 0.0))
        self.supcon_temp = float(getattr(loss_cfg, "supcon_temp", 0.07))
        self.kd_weight = float(getattr(loss_cfg, "kd_weight", 0.0))
        self.kd_temp = float(getattr(loss_cfg, "kd_temp", 4.0))
        self.num_classes = int(num_classes)

        # 构建主 CE 分支
        if self.type == "ce":
            if class_weights is not None:
                self.ce = nn.CrossEntropyLoss(weight=class_weights.to(device))
            else:
                self.ce = nn.CrossEntropyLoss()
            self.use_focal_ls = False
        else:
            # focal / focalls / focalls_supcon[_kd]
            gamma = self.gamma if self.type != "ce" else 0.0
            smoothing = self.smoothing if "ls" in self.type else 0.0
            self.ce = FocalLabelSmoothCE(
                gamma=gamma, smoothing=smoothing, weight=class_weights,
            )
            self.use_focal_ls = True

        # SupCon 分支
        self.supcon = (
            SupConLoss(temperature=self.supcon_temp)
            if "supcon" in self.type and self.supcon_weight > 0
            else None
        )

        # KD 分支
        self.kd = (
            KDLoss(temperature=self.kd_temp)
            if "kd" in self.type and self.kd_weight > 0
            else None
        )

        self.last_components: dict[str, float] = {}

    def forward(self, logits: torch.Tensor, target: torch.Tensor,
                proj: Optional[torch.Tensor] = None,
                teacher_logits: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 主 CE
        loss_ce = self.ce(logits, target)
        loss = loss_ce
        self.last_components = {"ce": float(loss_ce.detach().item())}

        if self.supcon is not None and proj is not None:
            loss_supcon = self.supcon(proj, target)
            loss = loss + self.supcon_weight * loss_supcon
            self.last_components["supcon"] = float(loss_supcon.detach().item())

        if self.kd is not None and teacher_logits is not None:
            loss_kd = self.kd(logits, teacher_logits)
            loss = loss + self.kd_weight * loss_kd
            self.last_components["kd"] = float(loss_kd.detach().item())

        return loss


def build_loss(num_classes: int,
               class_weights: Optional[torch.Tensor],
               loss_cfg,
               device: torch.device) -> CompositeLoss:
    """工厂函数."""
    return CompositeLoss(num_classes, class_weights, loss_cfg, device)
