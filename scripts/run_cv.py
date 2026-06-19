#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
5-fold × N-seed 交叉验证驱动器

为每个 (seed, fold) 生成 indices.npy, 然后调用 train_ddp.sh 进行 DDP 训练.
每个 fold 完成后聚合 metrics 到 outputs/cv/<run_id>/<model>/cv_summary.json.

用法:
    python scripts/run_cv.py --model resnet50 --dataset AL6 \\
        --folds 5 --seeds 42 1337 2024 --epochs 60 --img-size 224

输出:
    outputs/cv/<run_id>/<model>/
        seed42_fold0/
            train_idx.npy, val_idx.npy
            (training_log.json + test_metrics.json by train_ddp)
        seed42_fold1/...
        ...
        cv_summary.json   (mean ± std + 每 fold 详情)
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]


def collect_fold_entry(args, run_id: str, seed: int, fold: int) -> dict | None:
    """Return an existing completed fold entry, if the fold already finished."""
    inner_dir = (
        ROOT
        / "outputs"
        / f"{args.output_subdir}"
        / run_id
        / args.model
        / f"seed{seed}_fold{fold}_train"
    )
    if not inner_dir.exists():
        return None
    sub_runs = sorted([d for d in inner_dir.iterdir() if d.is_dir()], key=lambda d: d.stat().st_mtime, reverse=True)
    for sub_run in sub_runs:
        log_dir = sub_run / args.model
        log_path = log_dir / "training_log.json"
        test_path = log_dir / "test_metrics.json"
        if not test_path.exists():
            continue
        entry = {"seed": seed, "fold": fold, "log_dir": str(log_dir), "status": "ok", "resumed": True}
        if log_path.exists():
            with open(log_path) as f:
                log = json.load(f)
            entry["best_val_accuracy"] = log.get("best_val_accuracy")
            entry["epochs_completed"] = log.get("epochs_completed")
            entry["total_training_seconds"] = log.get("total_training_seconds")
        with open(test_path) as f:
            tm = json.load(f)
        entry["test_accuracy"] = tm.get("test_accuracy", tm.get("accuracy"))
        entry["macro_f1"] = tm.get("macro_f1", tm.get("test_macro_f1"))
        entry["weighted_f1"] = tm.get("weighted_f1", tm.get("test_weighted_f1"))
        return entry
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", default="AL6")
    p.add_argument("--img-size", type=int, nargs=2, default=[224, 224])
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 1337, 2024])
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--nproc", type=int, default=4)
    p.add_argument("--output-subdir", default="cv")
    p.add_argument("--run-id", default=None,
                   help="可选固定 run_id; 配合 --resume 可续跑同一 CV 目录")
    p.add_argument("--resume", action="store_true",
                   help="跳过已有 test_metrics.json 的 seed/fold")
    p.add_argument("--extra-args", default="",
                   help="额外传给 train_ddp.py 的参数, 整串引号包起来")
    args = p.parse_args()

    H, W = args.img_size

    # 加载 train cache 拿到全量 labels (CV 仅在 train 内切, test 集独立)
    cache_path = ROOT / "data" / "cache" / \
        f"{args.dataset}_{H}x{W}_rgb_train.pt"
    if not cache_path.exists():
        sys.exit(f"missing cache: {cache_path}")
    print(f"loading labels from {cache_path}")
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    labels = cache["labels"].numpy()
    N = len(labels)
    all_idx = np.arange(N, dtype=np.int64)
    print(f"  N train+val = {N}")

    run_id = args.run_id or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = ROOT / "outputs" / args.output_subdir / run_id / args.model
    out_root.mkdir(parents=True, exist_ok=True)

    # 配置:
    cv_config = {
        "run_id": run_id, "model": args.model, "dataset": args.dataset,
        "folds": args.folds, "seeds": args.seeds, "epochs": args.epochs,
        "n_total": int(N), "resume": bool(args.resume),
    }
    with open(out_root / "cv_config.json", "w") as f:
        json.dump(cv_config, f, indent=2)

    # ── 主循环 ──
    fold_results = []
    for seed in args.seeds:
        kf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=seed)
        for k, (tr, va) in enumerate(kf.split(all_idx, labels)):
            fold_dir = out_root / f"seed{seed}_fold{k}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            tr_path = fold_dir / "train_idx.npy"
            va_path = fold_dir / "val_idx.npy"
            np.save(tr_path, tr.astype(np.int64))
            np.save(va_path, va.astype(np.int64))
            print(f"\n=== seed={seed} fold={k}  train={len(tr)} val={len(va)} ===")

            if args.resume:
                existing = collect_fold_entry(args, run_id, seed, k)
                if existing is not None:
                    fold_results.append(existing)
                    print(
                        f"  -> skip existing seed{seed} fold{k}: "
                        f"val_acc={existing.get('best_val_accuracy', 0) or 0:.4f} "
                        f"test_acc={existing.get('test_accuracy', 0) or 0:.4f}"
                    )
                    continue

            cmd = [
                "torchrun", "--standalone",
                f"--nproc_per_node={args.nproc}",
                "--master_port=" + str(20000 + (seed * args.folds + k) % 30000),
                str(ROOT / "scripts" / "train_ddp.py"),
                "--model", args.model,
                "--dataset", args.dataset,
                "--img-size", str(H), str(W),
                "--epochs", str(args.epochs),
                "--batch-size", str(args.batch_size),
                "--seed", str(seed),
                "--output-subdir", f"{args.output_subdir}/{run_id}/{args.model}/seed{seed}_fold{k}_train",
                "--train-indices-path", str(tr_path),
                "--val-indices-path", str(va_path),
            ]
            if args.extra_args:
                cmd.extend(shlex.split(args.extra_args))

            print("[cmd]", " ".join(cmd))
            env = os.environ.copy()
            env["RUN_ID"] = ""  # 让每 fold 用自己的 run_id
            t0 = subprocess.run(cmd, env=env, cwd=str(ROOT))
            if t0.returncode != 0:
                print(f"[!!] seed={seed} fold={k} failed (rc={t0.returncode})")
                fold_results.append({
                    "seed": seed, "fold": k, "status": "failed",
                    "returncode": t0.returncode,
                })
                continue

            # 找最新的 training_log.json (该 fold 训练产物)
            inner_dir = (ROOT / "outputs" / f"{args.output_subdir}" / run_id /
                          args.model / f"seed{seed}_fold{k}_train")
            entry = collect_fold_entry(args, run_id, seed, k)
            if entry is None:
                print(f"[!!] no complete test_metrics found in {inner_dir}")
                fold_results.append({
                    "seed": seed, "fold": k, "status": "no_metrics",
                })
                continue
            entry["resumed"] = False
            fold_results.append(entry)
            print(f"  -> seed{seed} fold{k}: "
                  f"val_acc={entry.get('best_val_accuracy', 0):.4f} "
                  f"test_acc={entry.get('test_accuracy', 0):.4f}")

    # ── 聚合 ──
    summary = {"run_id": run_id, "model": args.model,
                "dataset": args.dataset, "folds": fold_results}

    test_accs = [f["test_accuracy"] for f in fold_results if f.get("test_accuracy") is not None]
    val_accs = [f["best_val_accuracy"] for f in fold_results if f.get("best_val_accuracy") is not None]
    macro_f1s = [f["macro_f1"] for f in fold_results if f.get("macro_f1") is not None]
    if test_accs:
        summary["test_accuracy_mean"] = float(np.mean(test_accs))
        summary["test_accuracy_std"] = float(np.std(test_accs))
        summary["test_accuracy_min"] = float(np.min(test_accs))
        summary["test_accuracy_max"] = float(np.max(test_accs))
    if val_accs:
        summary["val_accuracy_mean"] = float(np.mean(val_accs))
        summary["val_accuracy_std"] = float(np.std(val_accs))
    if macro_f1s:
        summary["macro_f1_mean"] = float(np.mean(macro_f1s))
        summary["macro_f1_std"] = float(np.std(macro_f1s))
    summary["n_folds_completed"] = len(test_accs)
    summary["n_folds_total"] = len(args.seeds) * args.folds

    with open(out_root / "cv_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n=== CV Summary ===")
    if test_accs:
        print(f"  Test acc:    {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}"
              f"  (min {np.min(test_accs):.4f}, max {np.max(test_accs):.4f})")
        print(f"  Val acc:     {np.mean(val_accs):.4f} ± {np.std(val_accs):.4f}")
        print(f"  Macro-F1:    {np.mean(macro_f1s):.4f} ± {np.std(macro_f1s):.4f}")
    print(f"  Completed:   {len(test_accs)} / {len(args.seeds) * args.folds} folds")
    print(f"  saved:       {out_root / 'cv_summary.json'}")


if __name__ == "__main__":
    main()
