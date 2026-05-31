#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
重新计算所有训完 ckpt 的 test_metrics.json — 修复早期 DDP all_gather bug
导致 test_accuracy 不一致问题.

流程:
    1. 扫描 outputs/{ddp*,cv,data_eff,ablations}/<run>/<model>/ 找
       best_*.pth + training_log.json
    2. 用单卡 path 重新算 test 指标 (无 DDP gather, 直接顺序)
    3. 备份原 test_metrics.json -> test_metrics.json.bak
    4. 覆盖写入新 test_metrics.json
    5. 写一份全局修复 summary

用法:
    python scripts/recompute_test_metrics.py
    python scripts/recompute_test_metrics.py --root outputs/ddp_baseline --apply
    python scripts/recompute_test_metrics.py --dry-run     # 只查不写
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (classification_report, confusion_matrix,
                              precision_recall_fscore_support)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.backbones import get_backbone
from src.utils.config import load_config


@torch.no_grad()
def evaluate_ckpt(model_name: str, dataset: str, h: int, w: int,
                   ckpt_path: Path, log_path: Path | None,
                   device: torch.device, batch_size: int = 64) -> dict:
    """单卡重新算 test metrics."""
    cache = ROOT / "data" / "cache" / f"{dataset}_{h}x{w}_rgb_test.pt"
    if not cache.exists():
        raise FileNotFoundError(f"missing: {cache}")
    test = torch.load(cache, map_location="cpu", weights_only=False)
    images = test["images"]
    labels = test["labels"]
    num_classes = int(labels.max().item()) + 1

    # 还原训练配置 (重要!)
    overrides = {
        "model": {"name": model_name},
        "data": {"dataset": dataset, "img_height": h, "img_width": w},
    }
    if log_path and log_path.exists():
        with open(log_path) as f:
            log = json.load(f)
        snap = log.get("config_snapshot", {})
        if "aafnet" in snap:
            overrides["aafnet"] = snap["aafnet"]

    config = load_config(overrides=overrides)
    Cls = get_backbone(model_name)
    instance = Cls.__new__(Cls)
    instance.config = config
    instance.num_classes = num_classes
    instance.device = device
    instance._to_rgb = (model_name != "custom_mlp")
    model = instance.build_model()

    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    model = model.to(device).eval()

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    criterion = nn.CrossEntropyLoss()

    all_pred = []
    sum_loss = 0.0; total = 0
    for i in range(0, len(images), batch_size):
        x = images[i:i+batch_size].float().to(device) / 255.0
        x = (x - mean) / std
        y = labels[i:i+batch_size].to(device)
        out = model(x)
        if isinstance(out, tuple):
            out = out[0]
        loss = criterion(out, y)
        sum_loss += float(loss.item()) * y.numel()
        all_pred.append(out.argmax(1).cpu().numpy())
        total += y.numel()
    y_pred = np.concatenate(all_pred)
    y_true = labels.numpy()

    acc = float((y_true == y_pred).mean())
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    wt_p, wt_r, wt_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred,
                            labels=list(range(num_classes)))
    return {
        "test_loss": float(sum_loss / max(1, total)),
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
        "n_test": int(len(y_true)),
        "model": model_name,
        "dataset": dataset,
        "_recomputed": True,
        "_ckpt_path": str(ckpt_path),
    }


def find_ckpts(root: Path) -> list:
    """找所有 best_*.pth + training_log.json 配对."""
    out = []
    for ckpt in root.rglob("best_*.pth"):
        log = ckpt.parent / "training_log.json"
        out.append((ckpt, log if log.exists() else None))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=str(ROOT / "outputs"),
                   help="搜索根目录, 递归找 best_*.pth")
    p.add_argument("--dry-run", action="store_true",
                   help="只查不写")
    p.add_argument("--apply", action="store_true",
                   help="覆盖写入 (默认不写, 仅打印 diff)")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    root = Path(args.root)
    ckpts = find_ckpts(root)
    print(f"\n=== Found {len(ckpts)} checkpoints under {root} ===\n")

    summary = []
    for ckpt, log in ckpts:
        # 推断 model name 从 ckpt name (best_<model>.pth)
        model_name = ckpt.stem.replace("best_", "")
        # 推断 dataset/img_size 从 training_log
        dataset, h, w = "AL6", 224, 224
        if log:
            with open(log) as f:
                lg = json.load(f)
            args_log = lg.get("args", {})
            dataset = args_log.get("dataset", "AL6")
            h = args_log.get("img_height", 224)
            w = args_log.get("img_width", 224)

        # 旧 metrics
        old_metrics_path = ckpt.parent / "test_metrics.json"
        old_acc = None
        if old_metrics_path.exists():
            with open(old_metrics_path) as f:
                old_acc = json.load(f).get("test_accuracy")

        try:
            t0 = time.time()
            new = evaluate_ckpt(model_name, dataset, h, w, ckpt, log,
                                  device)
            elapsed = time.time() - t0
        except Exception as e:
            print(f"[FAIL] {ckpt}: {e}")
            summary.append({"ckpt": str(ckpt), "status": "failed",
                            "error": str(e)})
            continue

        new_acc = new["test_accuracy"]
        diff = (new_acc - old_acc) if old_acc is not None else None
        flag = ""
        if old_acc is not None and abs(new_acc - old_acc) > 0.01:
            flag = "  [⚠️  diverges from old]"
        print(f"  {ckpt.relative_to(root)}")
        print(f"    model={model_name} dataset={dataset} {h}x{w}  "
              f"({elapsed:.1f}s)")
        print(f"    old test_acc: {old_acc}  ->  new: {new_acc:.4f}{flag}")

        summary.append({
            "ckpt": str(ckpt.relative_to(root)),
            "model": model_name, "dataset": dataset,
            "old_test_accuracy": old_acc,
            "new_test_accuracy": new_acc,
            "macro_f1": new["macro_f1"],
            "weighted_f1": new["weighted_f1"],
            "elapsed_seconds": round(elapsed, 2),
            "status": "ok",
        })

        if args.apply:
            # 备份原文件
            if old_metrics_path.exists():
                bak = old_metrics_path.with_suffix(".json.bak")
                if not bak.exists():
                    old_metrics_path.rename(bak)
            with open(old_metrics_path, "w") as f:
                json.dump(new, f, indent=2, ensure_ascii=False)
            # 也回写 training_log.json 的 best_val 不动, 但加 _recomputed_test_acc
            if log and log.exists():
                with open(log) as f:
                    lg = json.load(f)
                lg["recomputed_test_accuracy"] = new_acc
                lg["recomputed_macro_f1"] = new["macro_f1"]
                lg["recomputed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                with open(log, "w") as f:
                    json.dump(lg, f, indent=2, ensure_ascii=False)

    out_dir = ROOT / "outputs" / "data_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "recompute_summary.json", "w") as f:
        json.dump({
            "applied": bool(args.apply),
            "n_ckpts": len(ckpts),
            "results": summary,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n  saved: {out_dir / 'recompute_summary.json'}")

    if not args.apply:
        print("\n  (dry-run mode; pass --apply to actually overwrite "
              "test_metrics.json files)")


if __name__ == "__main__":
    main()
