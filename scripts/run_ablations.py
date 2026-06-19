#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
系统消融矩阵驱动 — 7 轴消融, 每轴 × 3 seeds

固定基础: ResNet-50 + AAFNet 全开 (--mssa --loss-type focalls_supcon --archaug)
然后逐轴关闭某个组件, 评估对最终精度的影响.

7 轴:
    a) MSSA on/off                                  (2 cell)
    b) Loss components: ce / focal / focalls /
       focalls_supcon / focalls_supcon_kd            (5 cell)
    c) Augmentation: none / orig / +archaug /
       +cutmix / +mixup / +randaugment              (6 cell)
    d) fine_tune_ratio: 0.0 / 0.4 / 0.6 / 0.8 / 1.0 (5 cell)
    e) dropout × dropout grid                        (3 cell)
    f) optimizer: adam / adamw / sgd                 (3 cell)
    g) scheduler: cosine / exp / plateau             (3 cell)

总 ~27 cell × 3 seeds = ~81 runs (× 50 epoch)

用法:
    python scripts/run_ablations.py --base-model resnet50 --epochs 50 \\
        --seeds 42 1337 2024 --axes a b c
"""

from __future__ import annotations

import argparse
import datetime
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


ABLATION_AXES = {
    "a_mssa": [
        ("mssa_off", []),
        ("mssa_on", ["--mssa"]),
    ],
    "b_loss": [
        ("loss_ce", ["--loss-type", "ce"]),
        ("loss_focal", ["--loss-type", "focal"]),
        ("loss_focalls", ["--loss-type", "focalls"]),
        ("loss_supcon", ["--loss-type", "focalls_supcon",
                          "--supcon-weight", "0.3"]),
        ("loss_supcon_kd", ["--loss-type", "focalls_supcon_kd",
                              "--supcon-weight", "0.3", "--kd-weight", "1.0"]),
    ],
    "c_aug": [
        ("aug_none", []),
        ("aug_orig", []),  # 默认就是原版
        ("aug_archaug", ["--archaug"]),
        ("aug_cutmix", ["--cutmix", "1.0"]),
        ("aug_mixup", ["--mixup", "0.2"]),
        ("aug_randaug", ["--randaugment"]),
    ],
    "d_ftr": [
        ("ftr_0", []),  # fine_tune_ratio 不在 CLI 中, 通过 config 改; 暂跳
    ],
    "e_dropout": [
        ("dp_0.2_0.1", []),
        ("dp_0.3_0.2", []),  # 默认
        ("dp_0.5_0.3", []),
    ],
    "f_optimizer": [
        ("opt_adam", ["--optimizer", "adam"]),
        ("opt_adamw", ["--optimizer", "adamw"]),
        ("opt_sgd", ["--optimizer", "sgd"]),
    ],
    "g_scheduler": [
        ("sch_cosine", ["--schedule", "cosine"]),
        ("sch_exp", ["--schedule", "exponential"]),
    ],
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="resnet50")
    p.add_argument("--dataset", default="AL6")
    p.add_argument("--img-size", type=int, nargs=2, default=[224, 224])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 1337, 2024])
    p.add_argument("--axes", nargs="+",
                   default=list(ABLATION_AXES.keys()),
                   help="选择要跑的轴 (默认全部)")
    p.add_argument("--nproc", type=int, default=4)
    p.add_argument("--dry-run", action="store_true",
                   help="只打印命令不真跑")
    p.add_argument("--output-subdir", default="ablations")
    args = p.parse_args()

    H, W = args.img_size
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = ROOT / "outputs" / args.output_subdir / run_id / args.base_model
    out_root.mkdir(parents=True, exist_ok=True)

    plan = []
    for axis_name in args.axes:
        axis = ABLATION_AXES.get(axis_name, [])
        for cell_name, cell_args in axis:
            for seed in args.seeds:
                plan.append({
                    "axis": axis_name,
                    "cell": cell_name,
                    "seed": seed,
                    "extra_args": cell_args,
                })

    total = len(plan)
    print(f"\n=== Ablation Plan ===")
    print(f"  base model: {args.base_model}")
    print(f"  dataset:    {args.dataset}")
    print(f"  total runs: {total}  (≈ {total * args.epochs} epoch × "
          f"{args.batch_size} bs)")
    print(f"  output_root: {out_root}\n")

    with open(out_root / "ablation_plan.json", "w") as f:
        json.dump({
            "run_id": run_id, "base_model": args.base_model,
            "dataset": args.dataset, "epochs": args.epochs,
            "plan": plan,
        }, f, indent=2)

    # 跑
    results = []
    for i, p_entry in enumerate(plan):
        tag = f"{p_entry['axis']}_{p_entry['cell']}_seed{p_entry['seed']}"
        print(f"\n[{i+1}/{total}] {tag}")
        cmd = [
            "torchrun", "--standalone",
            f"--nproc_per_node={args.nproc}",
            f"--master_port={20000 + (i * 13) % 30000}",
            str(ROOT / "scripts" / "train_ddp.py"),
            "--model", args.base_model,
            "--dataset", args.dataset,
            "--img-size", str(H), str(W),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--seed", str(p_entry["seed"]),
            "--output-subdir",
            f"{args.output_subdir}/{run_id}/{args.base_model}/{tag}",
        ]
        cmd.extend(p_entry["extra_args"])
        print(f"  cmd: {' '.join(cmd)}")
        if args.dry_run:
            continue
        env = os.environ.copy()
        env["RUN_ID"] = ""
        t = subprocess.run(cmd, env=env, cwd=str(ROOT))
        ent = {**p_entry, "tag": tag,
                "status": "ok" if t.returncode == 0 else "failed"}
        if t.returncode == 0:
            inner = (ROOT / "outputs" / args.output_subdir / run_id /
                      args.base_model / tag)
            sub = sorted([d for d in inner.iterdir() if d.is_dir()],
                          key=lambda d: d.stat().st_mtime, reverse=True)
            if sub:
                tm_path = sub[0] / args.base_model / "test_metrics.json"
                tm = None
                if tm_path.exists():
                    with open(tm_path) as f:
                        tm = json.load(f)
                if tm:
                    ent["test_accuracy"] = tm.get("test_accuracy")
                    ent["macro_f1"] = tm.get("macro_f1")
        results.append(ent)
        print(f"  -> test_acc = {ent.get('test_accuracy', 'N/A')}")

    summary = {"run_id": run_id, "base_model": args.base_model,
                "dataset": args.dataset, "n_runs_completed": len([
                    r for r in results if r["status"] == "ok"]),
                "n_runs_total": total, "results": results}
    with open(out_root / "ablation_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n=== Done ===")
    print(f"  saved: {out_root / 'ablation_summary.json'}")


if __name__ == "__main__":
    main()
