#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DW-MoE 集成训练 — 用 collect_member_predictions.py 输出的 members.npz, 跑三种集成对比.

用法:
    python scripts/train_ensemble.py --inputs outputs/ensemble_inputs/<id>/members.npz

输出:
    outputs/ensemble/<run_id>/
        results.json      (三种方法的 test 指标 + diversity matrix)
        diversity_heatmap.png
        gate_weights.png
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (classification_report, confusion_matrix,
                              precision_recall_fscore_support)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.ensemble import (
    soft_vote, diversity_weighted, DWMoE, fit_moe_gate,
)


def metrics(y_true, y_pred):
    acc = float((y_true == y_pred).mean())
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0,
    )
    wt_p, wt_r, wt_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0,
    )
    return {
        "test_accuracy": acc,
        "macro_f1": float(macro_f1),
        "weighted_f1": float(wt_f1),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", required=True, help="members.npz 路径")
    p.add_argument("--moe-epochs", type=int, default=5)
    p.add_argument("--moe-lr", type=float, default=1e-2)
    p.add_argument("--output-subdir", default="ensemble")
    args = p.parse_args()

    data = np.load(args.inputs, allow_pickle=True)
    val_logits = data["val_logits"]                              # [K, N_val, C]
    test_logits = data["test_logits"]                             # [K, N_test, C]
    val_labels = data["val_labels"]
    test_labels = data["test_labels"]
    val_acc = data["val_acc"]
    members = list(data["members"])
    K, _, C = val_logits.shape

    print(f"\n=== Ensemble Eval ===")
    print(f"  members: {members}")
    print(f"  val_acc: {dict(zip(members, [float(v) for v in val_acc]))}")

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "outputs" / args.output_subdir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {"members": members, "val_acc": val_acc.tolist()}

    # ── 1) 单成员 baseline ──
    results["per_member"] = {}
    for k, name in enumerate(members):
        m = metrics(test_labels, test_logits[k].argmax(axis=1))
        results["per_member"][name] = m
        print(f"  [{name:>20}] test_acc = {m['test_accuracy']:.4f}")

    # ── 2) Soft-vote ──
    fused_test = soft_vote(test_logits)
    m_sv = metrics(test_labels, fused_test.argmax(axis=1))
    results["soft_vote"] = m_sv
    print(f"\n  [soft_vote]            test_acc = {m_sv['test_accuracy']:.4f}")

    # ── 3) Diversity-weighted ──
    fused_test_dw, weights_dw = diversity_weighted(test_logits, val_acc)
    m_dw = metrics(test_labels, fused_test_dw.argmax(axis=1))
    results["diversity_weighted"] = {
        **m_dw,
        "weights": weights_dw.tolist(),
    }
    print(f"  [diversity_weighted]   test_acc = {m_dw['test_accuracy']:.4f}")
    print(f"    weights: {dict(zip(members, [float(w) for w in weights_dw]))}")

    # ── 4) MoE 学习门控 ──
    val_logits_t = torch.from_numpy(val_logits.transpose(1, 0, 2)).float()  # [N,K,C]
    val_labels_t = torch.from_numpy(val_labels).long()
    test_logits_t = torch.from_numpy(test_logits.transpose(1, 0, 2)).float()
    val_acc_t = torch.from_numpy(val_acc).float()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    moe = fit_moe_gate(
        val_logits_t, val_labels_t, val_acc_t, C,
        epochs=args.moe_epochs, lr=args.moe_lr, device=device,
    )
    moe.eval()
    with torch.no_grad():
        log_p = moe(test_logits_t.to(device))
        moe_preds = log_p.argmax(dim=1).cpu().numpy()
        gates_test = moe.gating_weights(test_logits_t.to(device)).cpu().numpy()
    m_moe = metrics(test_labels, moe_preds)
    results["moe"] = {**m_moe,
                       "mean_gates": gates_test.mean(axis=0).tolist()}
    print(f"  [moe (learned gate)]   test_acc = {m_moe['test_accuracy']:.4f}")

    # ── 5) Diversity matrix (论文 Discussion 用) ──
    flat_t = test_logits.transpose(1, 0, 2).reshape(test_logits.shape[1], -1)
    # 重新算: 用 test 上的预测概率做相关
    K = test_logits.shape[0]
    probs = np.exp(test_logits - test_logits.max(axis=2, keepdims=True))
    probs = probs / probs.sum(axis=2, keepdims=True)
    flat = probs.reshape(K, -1)
    div_matrix = 1.0 - np.abs(np.corrcoef(flat))
    np.fill_diagonal(div_matrix, 0)
    results["diversity_matrix"] = div_matrix.tolist()

    # ── 保存 ──
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 图: diversity heatmap + gate weights
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(8, 6))
        sns.heatmap(div_matrix, annot=True, fmt=".2f", cmap="rocket_r",
                     xticklabels=members, yticklabels=members)
        plt.title("Member Diversity (1 - |Pearson corr|)")
        plt.tight_layout()
        plt.savefig(out_dir / "diversity_heatmap.png", dpi=300)
        plt.close()

        plt.figure(figsize=(10, 5))
        x = np.arange(len(members))
        plt.bar(x - 0.2, weights_dw, width=0.4, label="diversity_weighted")
        plt.bar(x + 0.2, gates_test.mean(axis=0), width=0.4,
                 label="MoE mean gate")
        plt.xticks(x, members, rotation=30, ha="right")
        plt.ylabel("weight")
        plt.title("Member weights")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "gate_weights.png", dpi=300)
        plt.close()
    except Exception as e:
        print(f"  warn: plot failed: {e}")

    print(f"\n=== Summary ===")
    print(f"  best single:   "
          f"{max(results['per_member'].items(), key=lambda x: x[1]['test_accuracy'])}")
    print(f"  soft_vote:     {m_sv['test_accuracy']:.4f}")
    print(f"  diversity_wt:  {m_dw['test_accuracy']:.4f}")
    print(f"  moe:           {m_moe['test_accuracy']:.4f}")
    print(f"\n  saved: {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
