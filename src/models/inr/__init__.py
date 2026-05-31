# -*- coding: utf-8 -*-
"""
INR-AncientArch — 古建筑图像隐式神经表示分类

本模块灵感来源:
    [1] Sitzmann et al. SIREN: Implicit Neural Representations with Periodic
        Activation Functions. NeurIPS 2020.
    [2] Saragadam et al. WIRE: Wavelet Implicit Neural Representations. CVPR 2023.
    [3] Vaidyanathan et al. Random-Access Neural Compression of Material
        Textures. SIGGRAPH 2023 / NVIDIA RTX Neural Texture Compression (NTC),
        2024–2025. NVIDIA RTXNTC SDK 实测可将 6.5 GB BCn 纹理压缩至 970 MB
        (~85 % VRAM 缩减), 同时保持视觉等价.

叙事呼应 (论文 Introduction / Discussion 写法):
    NVIDIA NTC 把"材质纹理"压缩成 (网络权重 + latent tensor), 在 GPU shader
    中按需解压, 将渲染所需 VRAM 缩减约 8 倍. 我们把这一 implicit-neural-
    representation 范式从图形渲染拓展到古建筑分类:
        - NTC: 像素 → 神经压缩 → 解压回像素 → 渲染
        - INR-AncientArch: 像素 → SIREN 拟合 → 直接用权重做分类 (免解压)
    相比传统模型剪枝 (magnitude / structured pruning) 仍需保留全部输入像素
    + 大型推理网络, INR-AncientArch 提供:
        (1) 端到端更小的存储 (~30K float / image vs ResNet-50 25M+ params)
        (2) 更快的推理 latency (small classifier on weight-feat
            vs full backbone forward)
        (3) 由连续坐标表示带来的对几何变换 / 噪声 / 低分辨率的天然鲁棒性

本模块对应原项目 paper_texture_analysis.py 的"显式纹理特征"分析
(边缘 / 梯度 / 频率), 形成 显式 → 隐式 的递进叙事.
"""

from .siren import SIREN, BatchedSIREN, SineLayer, build_coord_grid
from .wire import WIRE, GaborLayer
from .inr_fitter import INRFitConfig, fit_batch, fit_dataset, psnr
from .inr_classifier import INRMLPClassifier, INRDeepSetsClassifier

__all__ = [
    "SIREN", "BatchedSIREN", "SineLayer", "build_coord_grid",
    "WIRE", "GaborLayer",
    "INRFitConfig", "fit_batch", "fit_dataset", "psnr",
    "INRMLPClassifier", "INRDeepSetsClassifier",
]
