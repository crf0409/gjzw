# -*- coding: utf-8 -*-
"""
鲁棒性测试套件 — 5 类扰动 × 3-4 严重度 评测训完模型的稳健性.

扰动:
    gauss_noise   : 高斯噪声 σ ∈ {0.05, 0.10, 0.20}
    motion_blur   : 运动模糊 kernel ∈ {5, 11, 17}
    jpeg_compress : JPEG 压缩 quality ∈ {20, 40, 60}
    brightness    : 亮度漂移 Δ ∈ {±0.2, ±0.4}
    occlusion     : 随机遮挡 ratio ∈ {5%, 15%, 30%}

输入:  PT 缓存 (uint8 [N, C, H, W]) + 训完模型权重
输出:  per-(perturbation, severity) accuracy + 6 子图准确率-严重度曲线
"""

from __future__ import annotations

import io
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# ─────────────────────────────────────────────────────────────────
# 单图扰动函数 (作用于 uint8 [C, H, W], 输出 uint8 [C, H, W])
# ─────────────────────────────────────────────────────────────────

def perturb_gauss_noise(img: torch.Tensor, sigma: float) -> torch.Tensor:
    """像素级 σ 是相对 [0, 1] 的标准差, 实际加在 [0, 255] 范围."""
    x = img.float()
    noise = torch.randn_like(x) * sigma * 255.0
    return (x + noise).clamp(0, 255).to(torch.uint8)


def perturb_motion_blur(img: torch.Tensor, kernel: int) -> torch.Tensor:
    """简化运动模糊: 1xK 横向均值核."""
    if kernel < 3:
        return img
    x = img.float().unsqueeze(0)                                # [1, C, H, W]
    C = x.shape[1]
    k = torch.zeros((C, 1, 1, kernel), device=x.device)
    k[:, 0, 0, :] = 1.0 / kernel
    pad = kernel // 2
    blurred = F.conv2d(x, k, padding=(0, pad), groups=C)
    return blurred.squeeze(0).clamp(0, 255).to(torch.uint8)


def perturb_jpeg(img: torch.Tensor, quality: int) -> torch.Tensor:
    """JPEG 编解码 (用 PIL, CPU 上做)."""
    arr = img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    if arr.shape[-1] == 1:
        pil = Image.fromarray(arr[..., 0], mode="L")
    else:
        pil = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    decoded = Image.open(buf).convert(pil.mode)
    out = np.asarray(decoded, dtype=np.uint8)
    if out.ndim == 2:
        out = out[..., None]
    return torch.from_numpy(out).permute(2, 0, 1)


def perturb_brightness(img: torch.Tensor, delta: float) -> torch.Tensor:
    """亮度漂移 Δ ∈ [-1, 1] 对应 ±255."""
    x = img.float() + delta * 255.0
    return x.clamp(0, 255).to(torch.uint8)


def perturb_occlusion(img: torch.Tensor, ratio: float,
                       seed: int = 0) -> torch.Tensor:
    """中心矩形遮挡, ratio = 遮挡面积比例."""
    g = torch.Generator()
    g.manual_seed(seed)
    c, h, w = img.shape
    side = int(round((ratio * h * w) ** 0.5))
    if side < 1:
        return img
    y0 = int(torch.randint(0, max(1, h - side), (1,), generator=g).item())
    x0 = int(torch.randint(0, max(1, w - side), (1,), generator=g).item())
    out = img.clone()
    out[..., y0:y0 + side, x0:x0 + side] = 0
    return out


# ─────────────────────────────────────────────────────────────────
# 扰动套件
# ─────────────────────────────────────────────────────────────────

PERTURBATION_GRID = {
    "gauss_noise":   [0.05, 0.10, 0.20],
    "motion_blur":   [5, 11, 17],
    "jpeg_compress": [60, 40, 20],     # quality 越低越严重
    "brightness":    [0.2, 0.3, 0.4],
    "occlusion":     [0.05, 0.15, 0.30],
}


def apply_perturbation(images: torch.Tensor, kind: str,
                        severity: float) -> torch.Tensor:
    """
    对 batch [B, C, H, W] uint8 应用扰动.
    motion_blur / jpeg / occlusion 是逐样本的循环, 其他可矢量化.
    """
    if kind == "gauss_noise":
        out = []
        for i in range(images.shape[0]):
            out.append(perturb_gauss_noise(images[i], sigma=float(severity)))
        return torch.stack(out)
    if kind == "motion_blur":
        out = []
        for i in range(images.shape[0]):
            out.append(perturb_motion_blur(images[i], kernel=int(severity)))
        return torch.stack(out)
    if kind == "jpeg_compress":
        out = []
        for i in range(images.shape[0]):
            out.append(perturb_jpeg(images[i], quality=int(severity)))
        return torch.stack(out)
    if kind == "brightness":
        # 随机正负
        deltas = (torch.bernoulli(torch.full((images.shape[0],), 0.5)) * 2 - 1) \
                 * float(severity)
        out = []
        for i in range(images.shape[0]):
            out.append(perturb_brightness(images[i], delta=float(deltas[i])))
        return torch.stack(out)
    if kind == "occlusion":
        out = []
        for i in range(images.shape[0]):
            out.append(perturb_occlusion(images[i], ratio=float(severity),
                                            seed=i))
        return torch.stack(out)
    raise ValueError(f"unknown perturbation: {kind}")


# ─────────────────────────────────────────────────────────────────
# 评估: 在扰动后的测试集上跑模型
# ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_under_perturbation(model: torch.nn.Module,
                                  images: torch.Tensor,
                                  labels: torch.Tensor,
                                  kind: str, severity,
                                  device: torch.device,
                                  batch_size: int = 64,
                                  normalize_fn: Callable | None = None
                                  ) -> dict:
    """
    Args:
        model:        已训完, 已 .eval()
        images:       uint8 [N, C, H, W]
        labels:       int64 [N]
        kind:         扰动类型
        severity:     扰动强度
        normalize_fn: 把 uint8 [N, C, H, W] -> normalized float -> 输入模型
    Returns:
        dict {accuracy, n}
    """
    if normalize_fn is None:
        # 默认 ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        def normalize_fn(x):
            x = x.float().to(device) / 255.0
            return (x - mean) / std

    model = model.to(device).eval()
    correct = 0
    total = 0
    for i in range(0, len(images), batch_size):
        batch_imgs = images[i:i + batch_size]
        batch_lbls = labels[i:i + batch_size].to(device)
        perturbed = apply_perturbation(batch_imgs, kind, severity)
        x = normalize_fn(perturbed)
        out = model(x)
        if isinstance(out, tuple):
            out = out[0]
        pred = out.argmax(1)
        correct += int((pred == batch_lbls).sum().item())
        total += int(batch_lbls.numel())
    return {"accuracy": correct / max(1, total), "n": total,
            "kind": kind, "severity": severity}


def evaluate_full_robustness(model: torch.nn.Module,
                              images: torch.Tensor,
                              labels: torch.Tensor,
                              device: torch.device,
                              batch_size: int = 64,
                              normalize_fn=None,
                              grid: dict = None) -> dict:
    """对全套扰动 × 严重度网格评估."""
    grid = grid or PERTURBATION_GRID
    out = {}
    for kind, severities in grid.items():
        out[kind] = []
        for sev in severities:
            r = evaluate_under_perturbation(
                model, images, labels, kind, sev,
                device, batch_size=batch_size, normalize_fn=normalize_fn,
            )
            out[kind].append(r)
            print(f"  {kind:>14} @ {sev}: acc = {r['accuracy']:.4f}")
    return out
