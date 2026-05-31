# -*- coding: utf-8 -*-
"""
错误分析: confusion pairs / 失败案例网格 / 类间相似度
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def confusion_pair_topk(cm: np.ndarray, k: int = 5) -> list:
    """从混淆矩阵找出最容易混淆的 K 个 (true, pred) 对 (off-diagonal)."""
    K = cm.shape[0]
    pairs = []
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            pairs.append((i, j, int(cm[i, j])))
    pairs.sort(key=lambda x: -x[2])
    return pairs[:k]


@torch.no_grad()
def lowest_confidence_examples(model, images, labels, device,
                                  per_class: int = 8) -> dict:
    """每个真实类下找置信度最低的 per_class 个样本 (按预测错误优先)."""
    import torch.nn as nn
    model = model.to(device).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    confidences = []
    preds = []
    for i in range(0, len(images), 64):
        x = images[i:i+64].float().to(device) / 255.0
        x = (x - mean) / std
        out = model(x)
        if isinstance(out, tuple):
            out = out[0]
        prob = out.softmax(dim=1)
        c, p = prob.max(dim=1)
        confidences.append(c.cpu().numpy())
        preds.append(p.cpu().numpy())
    conf = np.concatenate(confidences)
    pred = np.concatenate(preds)
    lab = labels.cpu().numpy()

    out = {}
    K = int(lab.max()) + 1
    for k in range(K):
        idx = np.where(lab == k)[0]
        if len(idx) == 0:
            continue
        wrong = idx[pred[idx] != k]
        # 优先错误的 + 按置信度低的
        if len(wrong) >= per_class:
            chosen = wrong[np.argsort(conf[wrong])[:per_class]]
        else:
            # 不够错误的, 用所有错误 + 置信度最低的正确
            need = per_class - len(wrong)
            correct = idx[pred[idx] == k]
            extra = correct[np.argsort(conf[correct])[:need]]
            chosen = np.concatenate([wrong, extra])
        out[int(k)] = [
            {"index": int(i), "pred": int(pred[i]),
             "confidence": float(conf[i]),
             "is_correct": bool(pred[i] == k)}
            for i in chosen
        ]
    return out


def class_mean_feature_similarity(feats: np.ndarray,
                                     labels: np.ndarray) -> np.ndarray:
    """[N, D] features + [N] labels -> [K, K] cosine 相似度矩阵."""
    K = int(labels.max()) + 1
    centers = np.zeros((K, feats.shape[1]), dtype=np.float32)
    for k in range(K):
        m = labels == k
        if m.sum() == 0:
            continue
        centers[k] = feats[m].mean(axis=0)
    norm = np.linalg.norm(centers, axis=1, keepdims=True).clip(min=1e-12)
    centers_n = centers / norm
    return centers_n @ centers_n.T
