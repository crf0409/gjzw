# -*- coding: utf-8 -*-
"""
Capsule Network (Sabour, Frosst, Hinton — "Dynamic Routing Between Capsules",
NeurIPS 2017) adapted for 224×224 architectural images.

Architecture:
  stem (3 → 64 → 128 → 256 → 256 channels, downsample 224 → 14)
  PrimaryCaps   (Conv9×9 stride=2, 32 capsule channels × 8 dim → 288 capsules × 8 dim)
  ClassCaps     (dynamic routing 3 iters → K capsules × 16 dim)
  output logits = ||class_caps|| × scale  (compatible with CE / FocalLS)

The squash + dynamic routing remain faithful to the original paper.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base_classifier import BaseClassifier
from .registry import register_backbone


def squash(s: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """Squash activation: short vectors → 0, long vectors → unit length."""
    sq = (s ** 2).sum(dim=dim, keepdim=True)
    norm = torch.sqrt(sq + eps)
    scale = sq / (1.0 + sq) / norm
    return scale * s


class PrimaryCaps(nn.Module):
    """One conv layer reshaped into capsule vectors."""

    def __init__(self, in_channels: int, out_caps_channels: int,
                 n_caps_dim: int, kernel_size: int, stride: int):
        super().__init__()
        self.out_caps_channels = out_caps_channels
        self.n_caps_dim = n_caps_dim
        self.conv = nn.Conv2d(in_channels,
                              out_caps_channels * n_caps_dim,
                              kernel_size=kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)                                          # [B, OC*D, H', W']
        B, _, H, W = out.shape
        out = out.view(B, self.out_caps_channels, self.n_caps_dim, H, W)
        out = out.permute(0, 1, 3, 4, 2).contiguous()               # [B, OC, H, W, D]
        out = out.view(B, -1, self.n_caps_dim)                      # [B, n_caps, D]
        return squash(out, dim=-1)


class DigitCaps(nn.Module):
    """Class-capsule layer with dynamic routing-by-agreement."""

    def __init__(self, n_caps_in: int, n_caps_in_dim: int,
                 n_caps_out: int, n_caps_out_dim: int, n_routing: int = 3):
        super().__init__()
        # W: [1, n_caps_in, n_caps_out, n_caps_out_dim, n_caps_in_dim]
        self.W = nn.Parameter(
            0.01 * torch.randn(1, n_caps_in, n_caps_out, n_caps_out_dim, n_caps_in_dim))
        self.n_routing = n_routing
        self.n_caps_in = n_caps_in
        self.n_caps_out = n_caps_out

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        # u: [B, n_caps_in, n_caps_in_dim]
        B = u.shape[0]
        # broadcast W (1,Ci,Co,Do,Di) × u (B,Ci,1,Di,1) → u_hat (B,Ci,Co,Do)
        u_in = u.unsqueeze(2).unsqueeze(-1)        # [B, Ci, 1, Di, 1]
        u_hat = (self.W @ u_in).squeeze(-1)        # [B, Ci, Co, Do]

        # Routing: don't backprop through routing weights (only through last iter's W·u)
        u_hat_d = u_hat.detach()
        b = torch.zeros(B, self.n_caps_in, self.n_caps_out, 1, device=u.device, dtype=u.dtype)
        v = None
        for r in range(self.n_routing):
            c = F.softmax(b, dim=2)                # [B, Ci, Co, 1]
            if r == self.n_routing - 1:
                s = (c * u_hat).sum(dim=1)         # last iter uses non-detached u_hat for grad
            else:
                s = (c * u_hat_d).sum(dim=1)
            v = squash(s, dim=-1)                  # [B, Co, Do]
            if r < self.n_routing - 1:
                # b_ij += u_hat_ij · v_j
                v_exp = v.unsqueeze(1)             # [B, 1, Co, Do]
                b = b + (u_hat_d * v_exp).sum(dim=-1, keepdim=True)
        return v                                   # [B, n_caps_out, n_caps_out_dim]


class CapsuleNetwork(nn.Module):
    """Stem + PrimaryCaps + ClassCaps. Returns logits = ||class_caps|| × scale."""

    def __init__(self, num_classes: int, primary_caps_n: int = 32,
                 primary_dim: int = 8, class_dim: int = 16,
                 routing_iters: int = 3, output_scale: float = 10.0):
        super().__init__()
        # Stem: 224 → 14 with progressive stride
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                 # 56
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),            # 28
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),            # 14
            nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),            # 14
        )
        # PrimaryCaps: kernel=9 stride=2 on 14×14 → output H' = floor((14-9)/2)+1 = 3 → 3×3
        # n_caps = primary_caps_n × 3 × 3 = 32 × 9 = 288
        self.primary = PrimaryCaps(
            in_channels=256, out_caps_channels=primary_caps_n,
            n_caps_dim=primary_dim, kernel_size=9, stride=2)
        n_caps_in = primary_caps_n * 3 * 3
        self.digit = DigitCaps(
            n_caps_in=n_caps_in, n_caps_in_dim=primary_dim,
            n_caps_out=num_classes, n_caps_out_dim=class_dim,
            n_routing=routing_iters)
        self.output_scale = output_scale
        self.class_dim = class_dim
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor):
        f = self.stem(x)
        u = self.primary(f)
        v = self.digit(u)                          # [B, K, class_dim]
        # logits ∝ ||v_k||  (scale up so softmax has reasonable range; raw ||v|| ≤ 1 from squash)
        logits = torch.norm(v, dim=-1) * self.output_scale  # [B, K]
        return logits


@register_backbone('capsnet')
class CapsNetClassifier(BaseClassifier):
    """Capsule Network classifier registered as 'capsnet' backbone."""

    def build_model(self):
        # Read optional config knobs (defaults match Sabour 2017 + adaptation for 224)
        capsnet_cfg = getattr(self.config, "capsnet", None)
        primary_caps_n = 32
        primary_dim = 8
        class_dim = 16
        routing_iters = 3
        output_scale = 10.0
        if capsnet_cfg is not None:
            primary_caps_n = int(getattr(capsnet_cfg, "primary_caps_n", primary_caps_n))
            primary_dim    = int(getattr(capsnet_cfg, "primary_dim",    primary_dim))
            class_dim      = int(getattr(capsnet_cfg, "class_dim",      class_dim))
            routing_iters  = int(getattr(capsnet_cfg, "routing_iters",  routing_iters))
            output_scale   = float(getattr(capsnet_cfg, "output_scale", output_scale))

        self.model = CapsuleNetwork(
            num_classes=self.num_classes,
            primary_caps_n=primary_caps_n,
            primary_dim=primary_dim,
            class_dim=class_dim,
            routing_iters=routing_iters,
            output_scale=output_scale,
        )
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"CapsNet: total {total:,}, trainable {trainable:,} | "
              f"primary {primary_caps_n}×{primary_dim}d, class {class_dim}d, routing {routing_iters}")
        return self.model
