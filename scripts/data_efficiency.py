#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据效率扫描 — 在不同训练数据比例下训练同一模型, 输出 (fraction, accuracy) 曲线.

为论文返修中"AAFNet 数据效率"那张关键图提供数据.

用法:
    python scripts/data_efficiency.py --model resnet50 --dataset AL6 \\
        --fractions 0.10 0.25 0.50 0.75 1.00 \\
        --seeds 42 1337 2024 --epochs 50 --img-size 224

输出:
    outputs/data_eff/<run_id>/<model>/
        results.json       # 每 (fraction, seed) 的 test acc
        curve.png          # 折线图
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", default="AL6")
    p.add_argument("--img-size", type=int, nargs=2, default=[224, 224])
    p.add_argument("--fractions", type=float, nargs="+",
                   default=[0.10, 0.25, 0.50, 0.75, 1.00])
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 1337, 2024])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--nproc", type=int, default=4)
    p.add_argument("--output-subdir", default="data_eff")
    p.add_argument("--extra-args", default="")
    args = p.parse_args()

    H, W = args.img_size
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = ROOT / "outputs" / args.output_subdir / run_id / args.model
    out_root.mkdir(parents=True, exist_ok=True)

    results = []

    for frac in args.fractions:
        for seed in args.seeds:
            tag = f"frac{frac:.2f}_seed{seed}"
            print(f"\n=== {tag} ===")
            cmd = [
                "torchrun", "--standalone",
                f"--nproc_per_node={args.nproc}",
                "--master_port=" + str(20000 + (int(frac*100)*100 + seed) % 30000),
                str(ROOT / "scripts" / "train_ddp.py"),
                "--model", args.model,
                "--dataset", args.dataset,
                "--img-size", str(H), str(W),
                "--epochs", str(args.epochs),
                "--batch-size", str(args.batch_size),
                "--seed", str(seed),
                "--train-fraction", f"{frac:.4f}",
                "--output-subdir",
                f"{args.output_subdir}/{run_id}/{args.model}/{tag}",
            ]
            if args.extra_args:
                cmd.extend(args.extra_args.split())

            env = os.environ.copy()
            env["RUN_ID"] = ""
            t = subprocess.run(cmd, env=env, cwd=str(ROOT))
            entry = {"fraction": frac, "seed": seed,
                     "status": "ok" if t.returncode == 0 else "failed"}
            if t.returncode == 0:
                # 找 test_metrics.json
                inner = (ROOT / "outputs" / args.output_subdir / run_id /
                         args.model / tag)
                sub = sorted([d for d in inner.iterdir() if d.is_dir()],
                              key=lambda d: d.stat().st_mtime, reverse=True)
                if sub:
                    log_dir = sub[0] / args.model
                    tm_path = log_dir / "test_metrics.json"
                    log_path = log_dir / "training_log.json"
                    if tm_path.exists():
                        with open(tm_path) as f:
                            tm = json.load(f)
                        entry["test_accuracy"] = tm["test_accuracy"]
                        entry["macro_f1"] = tm["macro_f1"]
                    if log_path.exists():
                        with open(log_path) as f:
                            log = json.load(f)
                        entry["best_val_accuracy"] = log["best_val_accuracy"]
                        entry["epochs_completed"] = log["epochs_completed"]
                        entry["log_dir"] = str(log_dir)
            results.append(entry)
            print(f"  -> frac={frac:.2f} seed={seed}: "
                  f"test_acc={entry.get('test_accuracy', 0):.4f}")

    # ── 聚合 ──
    summary = {"run_id": run_id, "model": args.model,
                "dataset": args.dataset, "results": results}

    # 按 fraction 算 mean/std
    by_frac = {}
    for r in results:
        if "test_accuracy" not in r:
            continue
        by_frac.setdefault(r["fraction"], []).append(r["test_accuracy"])
    summary["by_fraction"] = {
        f"{f:.2f}": {
            "mean": float(np.mean(v)), "std": float(np.std(v)),
            "min": float(np.min(v)), "max": float(np.max(v)),
            "n": len(v),
        } for f, v in sorted(by_frac.items())
    }

    with open(out_root / "results.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ── 曲线图 ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fracs = sorted(by_frac.keys())
        means = [np.mean(by_frac[f]) for f in fracs]
        stds = [np.std(by_frac[f]) for f in fracs]
        plt.figure(figsize=(8, 5))
        plt.errorbar(fracs, means, yerr=stds, marker="o", capsize=4,
                      linewidth=2, label=args.model)
        plt.xlabel("Training data fraction")
        plt.ylabel("Test accuracy")
        plt.title(f"Data Efficiency — {args.model} on {args.dataset}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_root / "curve.png", dpi=300)
        plt.close()
    except Exception as e:
        print(f"  warn: plot failed: {e}")

    print(f"\n=== Done ===")
    for f, v in summary["by_fraction"].items():
        print(f"  frac {f}: {v['mean']:.4f} ± {v['std']:.4f}  (n={v['n']})")
    print(f"  saved: {out_root / 'results.json'}")


if __name__ == "__main__":
    main()
