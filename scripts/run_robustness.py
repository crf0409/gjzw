#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
鲁棒性测试 driver — 加载训完模型, 跑 5 类扰动 × 3 严重度评测.

用法:
    python scripts/run_robustness.py \\
        --model resnet50 --dataset AL6 --img-size 224 224 \\
        --ckpt outputs/ddp/<run_id>/resnet50/best_resnet50.pth \\
        --output-subdir robustness

输出:
    outputs/robustness/<run_id>/<model>/
        results.json
        curve_<perturbation>.png
        summary_table.md
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.backbones import get_backbone
from src.utils.config import load_config
from src.evaluation.robustness import (
    evaluate_full_robustness, PERTURBATION_GRID,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", default="AL6")
    p.add_argument("--img-size", type=int, nargs=2, default=[224, 224])
    p.add_argument("--ckpt", required=True, help="训好的 .pth 权重")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-subdir", default="robustness")
    p.add_argument("--run-id", default=None, help="Optional deterministic output run id")
    args = p.parse_args()

    H, W = args.img_size
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 加载测试 PT 缓存
    cache_path = ROOT / "data" / "cache" / f"{args.dataset}_{H}x{W}_rgb_test.pt"
    if not cache_path.exists():
        sys.exit(f"missing test cache: {cache_path}")
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    images = cache["images"]                                    # uint8 [N,C,H,W]
    labels = cache["labels"]                                    # int64 [N]
    num_classes = int(labels.max().item()) + 1

    # 自动读 ckpt 同目录 training_log.json 还原训练时的 config (含 aafnet)
    log_path = Path(args.ckpt).parent / "training_log.json"
    overrides = {
        "model": {"name": args.model},
        "data": {"dataset": args.dataset, "img_height": H, "img_width": W},
    }
    if log_path.exists():
        log = json.load(open(log_path))
        snap = log.get("config_snapshot", {})
        # 把 aafnet 子树搬过来
        if "aafnet" in snap:
            overrides["aafnet"] = snap["aafnet"]
            print(f"  [config] restored aafnet from {log_path.name}: "
                  f"msa.enabled={snap['aafnet']['msa']['enabled']}, "
                  f"loss.type={snap['aafnet']['loss']['type']}")
    config = load_config(overrides=overrides)
    Cls = get_backbone(args.model)
    instance = Cls.__new__(Cls)
    instance.config = config
    instance.num_classes = num_classes
    instance.device = device
    instance._to_rgb = True
    model = instance.build_model()

    sd = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(sd, strict=False)
    model = model.to(device).eval()

    print(f"\n=== Robustness Eval ===")
    print(f"  model:   {args.model}")
    print(f"  dataset: {args.dataset}  N_test={len(labels)}")
    print(f"  ckpt:    {args.ckpt}")
    print(f"  device:  {device}")

    # 先跑干净测试
    print("\n[clean]")
    from src.evaluation.robustness import evaluate_under_perturbation
    clean_imgs = images.clone()
    correct = 0; total = 0
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    with torch.no_grad():
        for i in range(0, len(images), args.batch_size):
            batch = images[i:i+args.batch_size].float().to(device) / 255.0
            batch = (batch - mean) / std
            lbls = labels[i:i+args.batch_size].to(device)
            out = model(batch)
            if isinstance(out, tuple): out = out[0]
            correct += int((out.argmax(1) == lbls).sum().item())
            total += lbls.numel()
    clean_acc = correct / max(1, total)
    print(f"  clean accuracy: {clean_acc:.4f}")

    # 跑扰动套件
    print("\n[robustness suite]")
    rob = evaluate_full_robustness(
        model, images, labels, device,
        batch_size=args.batch_size,
    )

    # ── 保存 ──
    run_id = args.run_id or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "outputs" / args.output_subdir / run_id / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id, "model": args.model, "dataset": args.dataset,
        "img_size": [H, W], "ckpt": args.ckpt,
        "n_test": len(labels), "clean_accuracy": clean_acc,
        "robustness": rob, "grid": PERTURBATION_GRID,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    # ── 6 子图曲线 (5 扰动 + 1 总览) ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        kinds = list(rob.keys())
        n = len(kinds)
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        axes = axes.ravel()
        for i, kind in enumerate(kinds):
            sevs = [r["severity"] for r in rob[kind]]
            accs = [r["accuracy"] for r in rob[kind]]
            ax = axes[i]
            ax.plot(sevs, accs, marker="o", linewidth=2, color="tab:blue")
            ax.axhline(clean_acc, ls="--", color="gray",
                        label=f"clean={clean_acc:.3f}")
            ax.set_title(kind)
            ax.set_xlabel("severity")
            ax.set_ylabel("accuracy")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        # 总览: 平均 robust acc vs severity rank
        avg_per_rank = {0: [], 1: [], 2: []}
        for kind in kinds:
            for rank, r in enumerate(rob[kind]):
                avg_per_rank[rank].append(r["accuracy"])
        avg = [np.mean(v) for v in avg_per_rank.values() if v]
        axes[-1].bar(["mild", "medium", "severe"][:len(avg)], avg,
                      color="tab:orange")
        axes[-1].axhline(clean_acc, ls="--", color="gray")
        axes[-1].set_title("Avg robust acc by severity")
        axes[-1].set_ylim(0, 1)
        plt.tight_layout()
        plt.savefig(out_dir / "robustness_curves.png", dpi=300)
        plt.close()
    except Exception as e:
        print(f"  warn: plot failed: {e}")

    # ── markdown 表 ──
    md = [f"# Robustness — {args.model} on {args.dataset}",
           f"\n_run_id: {run_id}_  ·  ckpt: `{Path(args.ckpt).name}`",
           f"\nClean test accuracy: **{clean_acc:.4f}**\n",
           "| Perturbation | Severity | Accuracy | Δ vs clean |",
           "|--------------|----------|----------|-----------|"]
    for kind in rob:
        for r in rob[kind]:
            delta = r["accuracy"] - clean_acc
            md.append(f"| {kind} | {r['severity']} | {r['accuracy']:.4f} | "
                      f"{delta:+.4f} |")
    (out_dir / "summary_table.md").write_text("\n".join(md), encoding="utf-8")

    print(f"\n=== Done ===")
    print(f"  saved -> {out_dir}")


if __name__ == "__main__":
    main()
