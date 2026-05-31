# -*- coding: utf-8 -*-
"""
INR Fitter — 把图像批量拟合成 SIREN 权重 (pretext task)

每张图像独立拟合, 但用 BatchedSIREN 在 GPU 上并行 B 张图像.
拟合完成后导出每张图像的权重向量, 作为下游分类的特征.

主接口:
    fit_batch(images, ...)        — 一次拟合 B 张图像, 返回 [B, total_params]
    fit_dataset(loader, ...)       — 遍历整个数据集逐 batch 拟合, 拼接所有权重

工作流:
    1. 准备 PT 缓存的图像数据 [N, C, H, W]
    2. 按 batch_size B 切片, 每片初始化 BatchedSIREN(B)
    3. 训练 N_steps 步 (Adam, MSE), 直到 PSNR > 阈值
    4. 导出 weights -> [B, total_params], 累积到全数据集 [N, total_params]
    5. 保存为 .pt 文件 供分类训练用

支持 SIREN / WIRE 切换 (--inr-arch).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .siren import BatchedSIREN, build_coord_grid


# ─────────────────────────────────────────────────────────────────
# 工具
# ─────────────────────────────────────────────────────────────────

def psnr(mse: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """[B] mse -> [B] psnr (dB)"""
    mse = mse.clamp_min(1e-10)
    return 10.0 * torch.log10(max_val ** 2 / mse)


@dataclass
class INRFitConfig:
    arch: str = "siren"          # siren | wire
    hidden_dim: int = 256
    num_layers: int = 4
    omega_0: float = 30.0
    n_steps: int = 500           # 每张图像拟合迭代步数
    lr: float = 1e-4
    batch_size: int = 32         # 每个 GPU 一次并行拟合 B 张图像
    target_psnr: float = 35.0    # 早停阈值 (达到则提前停)
    log_every: int = 50


# ─────────────────────────────────────────────────────────────────
# 单 batch 拟合
# ─────────────────────────────────────────────────────────────────

def fit_batch(images: torch.Tensor,
              cfg: INRFitConfig,
              device: torch.device,
              return_metrics: bool = False
              ) -> tuple[torch.Tensor, dict]:
    """
    并行拟合一个 batch 的图像.

    Args:
        images: [B, C, H, W] float in [-1, 1]   (论文常用 [-1, 1] 而不是 [0, 1])
        cfg:    INRFitConfig
        device: torch.device
        return_metrics: 是否返回训练指标

    Returns:
        weights:  [B, total_params_per_siren]
        metrics:  dict, 含 final_mse, final_psnr 等
    """
    B, C, H, W = images.shape
    coords = build_coord_grid(H, W, device=device)              # [H*W, 2]
    targets = images.permute(0, 2, 3, 1).reshape(B, H * W, C)   # [B, N, C]

    if cfg.arch == "siren":
        model = BatchedSIREN(
            batch_size=B, in_dim=2, out_dim=C,
            hidden_dim=cfg.hidden_dim, num_layers=cfg.num_layers,
            omega_0=cfg.omega_0,
        ).to(device)
    else:
        raise NotImplementedError(
            f"BatchedWIRE not yet implemented; use arch=siren for batch mode."
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    final_mse = None
    for step in range(cfg.n_steps):
        pred = model(coords)                                    # [B, N, C]
        # 每张图像独立 MSE
        mse_per_img = ((pred - targets) ** 2).mean(dim=(1, 2))  # [B]
        loss = mse_per_img.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        final_mse = mse_per_img.detach()
        # 早停: 所有图像都达到 target PSNR
        if cfg.target_psnr > 0 and step % cfg.log_every == 0:
            current_psnr = psnr(final_mse, max_val=2.0)
            if current_psnr.min() >= cfg.target_psnr:
                break

    weights = model.export_weights()                            # [B, total]
    metrics = {}
    if return_metrics:
        metrics["final_mse_mean"] = float(final_mse.mean().item())
        metrics["final_psnr_mean"] = float(psnr(final_mse, max_val=2.0).mean().item())
        metrics["final_psnr_min"] = float(psnr(final_mse, max_val=2.0).min().item())
        metrics["params_per_image"] = int(model.params_per_image)
        metrics["steps_run"] = step + 1
    return weights, metrics


# ─────────────────────────────────────────────────────────────────
# 全数据集拟合
# ─────────────────────────────────────────────────────────────────

@torch.enable_grad()
def fit_dataset(images: torch.Tensor,
                labels: torch.Tensor,
                cfg: INRFitConfig,
                device: torch.device,
                progress: bool = True
                ) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """
    遍历整个数据集逐 batch 拟合 INR, 返回每张图像的权重向量.

    Args:
        images: [N, C, H, W] float in [-1, 1]
        labels: [N] long
        cfg:    INRFitConfig
        device: torch.device

    Returns:
        all_weights:  [N, total_params_per_siren]   每张图像的 INR 权重
        labels:       [N] long  (原样返回)
        metrics:      dict 全局统计
    """
    N = images.size(0)
    bs = cfg.batch_size
    total_psnr = []
    total_mse = []
    weights_list = []

    bar = range(0, N, bs)
    if progress:
        bar = tqdm(bar, desc=f"Fitting {cfg.arch.upper()}", total=(N + bs - 1) // bs)

    t0 = time.time()
    for start in bar:
        end = min(start + bs, N)
        batch = images[start:end].to(device, non_blocking=True)
        weights, m = fit_batch(batch, cfg, device, return_metrics=True)
        weights_list.append(weights.cpu())
        total_psnr.append(m["final_psnr_mean"])
        total_mse.append(m["final_mse_mean"])

    elapsed = time.time() - t0

    all_weights = torch.cat(weights_list, dim=0)                # [N, total]
    metrics = {
        "n_images": int(N),
        "params_per_image": int(all_weights.shape[1]),
        "total_storage_mb": float(all_weights.element_size()
                                   * all_weights.nelement() / 1024 / 1024),
        "mean_psnr": float(sum(total_psnr) / max(1, len(total_psnr))),
        "mean_mse": float(sum(total_mse) / max(1, len(total_mse))),
        "fit_seconds": float(elapsed),
        "fit_seconds_per_image": float(elapsed / max(1, N)),
        "config": cfg.__dict__,
    }
    return all_weights, labels, metrics
