# -*- coding: utf-8 -*-
"""
倒数第二层特征可视化: t-SNE / UMAP

输入: 训完的模型 + 测试集
输出: t-SNE 散点 + UMAP 散点 (按真实类着色)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


@torch.no_grad()
def extract_features(model: nn.Module, images: torch.Tensor,
                       labels: torch.Tensor,
                       device: torch.device,
                       batch_size: int = 64,
                       feature_layer_idx: int = 0) -> tuple:
    """
    抽取模型倒数第二层的特征向量.

    对 nn.Sequential(backbone, head): backbone 输出即特征 (head 之前)
    feature_layer_idx 指定取哪个子模块的输出. 默认 0 = backbone.

    Args:
        model:        nn.Module
        images:       uint8 [N, C, H, W]
        labels:       int64 [N]
        device:       torch.device
        batch_size:   batch
        feature_layer_idx: 用 Sequential 的第几个子模块输出做特征

    Returns:
        feats: float32 [N, D]
        labs:  int64 [N]
    """
    model = model.to(device).eval()
    if isinstance(model, nn.Sequential) and len(model) > feature_layer_idx + 1:
        feature_module = model[feature_layer_idx]
    else:
        feature_module = model

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    feats_list = []
    for i in range(0, len(images), batch_size):
        x = images[i:i + batch_size].float().to(device) / 255.0
        x = (x - mean) / std
        out = feature_module(x)
        if isinstance(out, (list, tuple)):
            # 某些 wrappers 返回多尺度, 取最后一个池化
            out = out[-1]
            if out.dim() == 4:
                out = out.mean(dim=(2, 3))
        if out.dim() == 4:
            out = out.mean(dim=(2, 3))                              # GAP
        feats_list.append(out.detach().cpu().numpy())
    feats = np.concatenate(feats_list, axis=0)
    return feats.astype(np.float32), labels.cpu().numpy().astype(np.int64)


def reduce_tsne(feats: np.ndarray, perplexity: float = 30.0,
                 random_state: int = 42) -> np.ndarray:
    from sklearn.manifold import TSNE
    import inspect
    n = feats.shape[0]
    perp = min(perplexity, max(5, (n - 1) / 3))
    # sklearn 新版用 max_iter, 旧版用 n_iter
    sig = inspect.signature(TSNE.__init__).parameters
    iter_kw = {"max_iter": 1000} if "max_iter" in sig else {"n_iter": 1000}
    return TSNE(n_components=2, perplexity=perp, init="pca",
                  random_state=random_state, **iter_kw).fit_transform(feats)


def reduce_umap(feats: np.ndarray, n_neighbors: int = 15,
                 random_state: int = 42) -> np.ndarray:
    try:
        import umap
    except ImportError:
        return None
    return umap.UMAP(n_components=2, n_neighbors=n_neighbors,
                      random_state=random_state).fit_transform(feats)


def render_scatter(coords: np.ndarray, labels: np.ndarray,
                    title: str, save_path: Path,
                    class_names: list = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 7))
    K = int(labels.max()) + 1
    cmap = plt.get_cmap("tab20", K)
    for k in range(K):
        m = labels == k
        name = class_names[k] if class_names and k < len(class_names) else f"C{k}"
        plt.scatter(coords[m, 0], coords[m, 1], s=12, alpha=0.7,
                     color=cmap(k), label=name)
    if K <= 12:
        plt.legend(fontsize=8, ncol=2, loc="best")
    plt.title(title)
    plt.xlabel("dim 1")
    plt.ylabel("dim 2")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
