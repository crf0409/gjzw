#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练 INR 权重分类器 — 把 SIREN 权重当作图像的"神经压缩特征"做分类.

输入:  data/cache_inr/<dataset>_<arch>_h<H>_L<L>_<HxW>_<split>.pt
        含 weights [N, total_params], labels [N]
输出:  outputs/inr_clf/<run_id>/<head>/
        含 best_<head>.pth, training_log.json, test_metrics.json,
        confusion_matrix.png, training_history.png

用法:
    python scripts/train_inr_classifier.py --dataset AL6 \\
        --arch siren --hidden 256 --layers 4 --img-size 224 224 \\
        --head mlp --epochs 100 --batch-size 64

支持 head: mlp (flatten + MLP) | deepsets (per-layer 神经元 DeepSets pool)
此脚本不用 DDP — 输入是 INR 权重向量, 数据量小, 单 GPU 够快.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (classification_report, confusion_matrix,
                              precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.inr import INRMLPClassifier, INRDeepSetsClassifier


def parse_layer_specs(in_dim: int, hidden: int, layers: int,
                      out_dim: int = 3) -> list:
    """重建 SIREN 拟合时使用的层规格 (用于 deepsets head)."""
    dims = [2] + [hidden] * (layers - 1) + [out_dim]
    return [((dims[i], dims[i + 1]), (dims[i + 1],)) for i in range(layers)]


def split_into_layers(weights: torch.Tensor, layer_specs: list,
                      batch_size: int) -> list:
    """
    把 flattened weights 切回 per-layer dict { 'W': ..., 'b': ... }.

    Args:
        weights:    [B, total]
        layer_specs: list of ((d_in, d_out), (d_out,))
    Returns:
        list of dict, 每元素 W [B, d_in, d_out] / b [B, d_out]
    """
    out = []
    cursor = 0
    for (W_shape, b_shape) in layer_specs:
        d_in, d_out = W_shape
        n_w = d_in * d_out
        n_b = d_out
        W = weights[:, cursor:cursor + n_w].view(batch_size, d_in, d_out)
        cursor += n_w
        b = weights[:, cursor:cursor + n_b].view(batch_size, d_out)
        cursor += n_b
        out.append({"W": W, "b": b})
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="AL6")
    p.add_argument("--arch", default="siren")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--img-size", type=int, nargs=2, default=[224, 224],
                   metavar=("H", "W"))
    p.add_argument("--head", choices=["mlp", "deepsets"], default="mlp")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--head-hidden", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--output-subdir", default="inr_clf")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── 加载 INR 权重数据集 ──
    H, W = args.img_size
    in_dir = ROOT / "data" / "cache_inr"
    train_pt = in_dir / (
        f"{args.dataset}_{args.arch}_h{args.hidden}_L{args.layers}_"
        f"{H}x{W}_train.pt"
    )
    test_pt = in_dir / (
        f"{args.dataset}_{args.arch}_h{args.hidden}_L{args.layers}_"
        f"{H}x{W}_test.pt"
    )
    if not train_pt.exists():
        sys.exit(
            f"missing INR cache: {train_pt}\n"
            f"  run: python scripts/fit_inr_dataset.py --dataset {args.dataset} "
            f"--split train --arch {args.arch} --hidden {args.hidden} "
            f"--layers {args.layers}"
        )

    train_data = torch.load(train_pt, map_location="cpu", weights_only=False)
    test_data = (torch.load(test_pt, map_location="cpu", weights_only=False)
                  if test_pt.exists() else None)

    X_all = train_data["weights"].float()                          # [N, total]
    y_all = train_data["labels"].long()
    in_dim = X_all.shape[1]
    num_classes = int(y_all.max().item()) + 1

    print(f"\n=== INR Classifier ===")
    print(f"  train INR cache:  {train_pt}")
    print(f"  test  INR cache:  {test_pt if test_pt.exists() else '(none)'}")
    print(f"  N train+val:      {len(X_all)}")
    print(f"  in_dim:           {in_dim}")
    print(f"  num_classes:      {num_classes}")
    print(f"  head:             {args.head}")
    print(f"  fit metrics:      mean PSNR "
          f"{train_data['metrics']['mean_psnr']:.2f} dB")

    # train/val 切分
    idx = np.arange(len(X_all))
    tr, va = train_test_split(
        idx, test_size=args.val_split, random_state=args.seed,
        stratify=y_all.numpy(),
    )
    X_tr, y_tr = X_all[tr], y_all[tr]
    X_va, y_va = X_all[va], y_all[va]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 模型 ──
    if args.head == "mlp":
        model = INRMLPClassifier(
            in_dim=in_dim, num_classes=num_classes,
            hidden=args.head_hidden, dropout=args.dropout,
        ).to(device)
        forward_fn = lambda batch_x: model(batch_x)
    else:
        layer_specs = parse_layer_specs(in_dim, args.hidden, args.layers)
        model = INRDeepSetsClassifier(
            layer_specs=layer_specs, num_classes=num_classes,
            embed_dim=64, hidden=args.head_hidden, dropout=args.dropout,
        ).to(device)
        def forward_fn(batch_x):
            B = batch_x.size(0)
            layer_w = split_into_layers(batch_x, layer_specs, B)
            return model(layer_w)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  classifier params: {n_params:,}")

    # ── 数据加载 ──
    train_loader = DataLoader(
        TensorDataset(X_tr, y_tr), batch_size=args.batch_size,
        shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        TensorDataset(X_va, y_va), batch_size=args.batch_size,
        shuffle=False, num_workers=0,
    )

    # ── 优化器 ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                    weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01,
    )
    criterion = nn.CrossEntropyLoss()

    # ── 输出目录 ──
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "outputs" / args.output_subdir / run_id / args.head
    out_dir.mkdir(parents=True, exist_ok=True)

    history = {"loss": [], "accuracy": [],
               "val_loss": [], "val_accuracy": [],
               "lr": [], "epoch_seconds": []}
    best_val = 0.0
    patience = 0
    ckpt_path = out_dir / f"best_{args.head}.pth"

    train_start = time.time()
    for epoch in range(args.epochs):
        t0 = time.time()
        # train
        model.train()
        sum_loss = 0.0; sum_correct = 0; sum_total = 0
        for x, y in train_loader:
            x = x.to(device); y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = forward_fn(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            sum_loss += float(loss.detach().item()) * y.numel()
            sum_correct += int((logits.argmax(1) == y).sum().item())
            sum_total += int(y.numel())
        train_loss = sum_loss / max(1, sum_total)
        train_acc = sum_correct / max(1, sum_total)

        # val
        model.eval()
        v_loss = 0.0; v_correct = 0; v_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device); y = y.to(device)
                logits = forward_fn(x)
                loss = criterion(logits, y)
                v_loss += float(loss.detach().item()) * y.numel()
                v_correct += int((logits.argmax(1) == y).sum().item())
                v_total += int(y.numel())
        val_loss = v_loss / max(1, v_total)
        val_acc = v_correct / max(1, v_total)

        scheduler.step()
        elapsed = time.time() - t0
        cur_lr = optimizer.param_groups[0]["lr"]
        history["loss"].append(train_loss)
        history["accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        history["lr"].append(cur_lr)
        history["epoch_seconds"].append(round(elapsed, 3))

        print(f"Epoch {epoch+1:>3}/{args.epochs} ({elapsed:.1f}s) - "
              f"loss {train_loss:.4f} acc {train_acc:.4f} - "
              f"val_loss {val_loss:.4f} val_acc {val_acc:.4f} - "
              f"lr {cur_lr:.2e}")

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), ckpt_path)
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                print(f"Early stop at epoch {epoch+1}")
                break

    # ── 测试 ──
    if test_data is not None:
        model.load_state_dict(torch.load(ckpt_path, weights_only=True,
                                          map_location=device))
        model.eval()
        X_te = test_data["weights"].float()
        y_te = test_data["labels"].long()
        all_pred = []
        with torch.no_grad():
            for i in range(0, len(X_te), args.batch_size):
                xb = X_te[i:i + args.batch_size].to(device)
                logits = forward_fn(xb)
                all_pred.append(logits.argmax(1).cpu())
        y_pred = torch.cat(all_pred).numpy()
        y_true = y_te.numpy()
        acc = float((y_true == y_pred).mean())
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0,
        )
        wt_p, wt_r, wt_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0,
        )
        cm = confusion_matrix(y_true, y_pred)
        test_metrics = {
            "test_accuracy": acc,
            "macro_precision": float(macro_p),
            "macro_recall": float(macro_r),
            "macro_f1": float(macro_f1),
            "weighted_precision": float(wt_p),
            "weighted_recall": float(wt_r),
            "weighted_f1": float(wt_f1),
            "confusion_matrix": cm.tolist(),
            "classification_report_text": classification_report(
                y_true, y_pred, zero_division=0,
            ),
            "n_test": int(len(y_te)),
        }
        with open(out_dir / "test_metrics.json", "w") as f:
            json.dump(test_metrics, f, indent=2, ensure_ascii=False)
        print(f"\n[Test] acc={acc:.4f}  macro-F1={macro_f1:.4f}  "
              f"weighted-F1={wt_f1:.4f}")

        # 混淆矩阵 PNG
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import seaborn as sns
            plt.figure(figsize=(7, 5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.title(f"INR-{args.arch} h{args.hidden}L{args.layers} "
                      f"+ {args.head} head — test acc {acc*100:.2f}%")
            plt.xlabel("Predicted"); plt.ylabel("True")
            plt.tight_layout()
            plt.savefig(out_dir / "confusion_matrix.png", dpi=300)
            plt.close()
        except Exception as e:
            print(f"  warn: cm png failed: {e}")
    else:
        test_metrics = None

    # ── 训练日志 ──
    log = {
        "run_id": run_id,
        "model": f"INR-{args.arch}-{args.head}",
        "dataset": args.dataset,
        "in_dim": int(in_dim),
        "num_classes": int(num_classes),
        "epochs_completed": len(history["loss"]),
        "best_val_accuracy": float(best_val),
        "total_seconds": round(time.time() - train_start, 2),
        "history": history,
        "args": vars(args),
        "fit_metrics": train_data["metrics"],
        "test_metrics": test_metrics,
    }
    with open(out_dir / "training_log.json", "w") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)

    # 训练曲线 PNG
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ep = list(range(1, len(history["loss"]) + 1))
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(ep, history["loss"], label="train")
        ax[0].plot(ep, history["val_loss"], label="val")
        ax[0].set_title(f"INR Classifier Loss"); ax[0].legend()
        ax[0].grid(alpha=0.3)
        ax[1].plot(ep, history["accuracy"], label="train")
        ax[1].plot(ep, history["val_accuracy"], label="val")
        ax[1].set_title(f"INR Classifier Accuracy"); ax[1].legend()
        ax[1].grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "training_history.png", dpi=300)
        plt.close()
    except Exception:
        pass

    print(f"\nartifacts -> {out_dir}")


if __name__ == "__main__":
    main()
