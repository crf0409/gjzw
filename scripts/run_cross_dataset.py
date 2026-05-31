#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
跨数据集实验 driver — 三类实验:

1. within     — 在每个数据集独立训练 + 测试
2. transfer   — 在 source 上预训练 -> 在 target 上 fine-tune (vs ImageNet 直接 fine-tune)
3. joint      — source ∪ target 联合训练 (类别空间合并)

目的: 论文返修 "数据广度 / 跨数据集泛化" 这条审稿点.
对比 baseline: ImageNet -> Y 直接 fine-tune.
对比 transfer: ImageNet -> X (pretrain) -> Y (fine-tune).

用法:
    # within (基本运行, 同 train_ddp 但跑多个数据集)
    python scripts/run_cross_dataset.py --mode within \\
        --model resnet50 --datasets AL6 ASP AS25 --img-size 224 --epochs 60

    # transfer X->Y
    python scripts/run_cross_dataset.py --mode transfer \\
        --model resnet50 --source ASP --target AL6 \\
        --img-size 224 --epochs 30
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_train(model: str, dataset: str, h: int, w: int,
              epochs: int, batch: int, seed: int, nproc: int,
              extra: list, output_subdir: str) -> tuple[int, Path | None]:
    """调用 train_ddp.py, 返回 (returncode, ckpt 路径 if found)."""
    cmd = [
        "torchrun", "--standalone",
        f"--nproc_per_node={nproc}",
        f"--master_port={20000 + (hash((model, dataset, seed)) & 0x7FFF)}",
        str(ROOT / "scripts" / "train_ddp.py"),
        "--model", model,
        "--dataset", dataset,
        "--img-size", str(h), str(w),
        "--epochs", str(epochs),
        "--batch-size", str(batch),
        "--seed", str(seed),
        "--output-subdir", output_subdir,
    ]
    cmd.extend(extra)
    print("[cmd]", " ".join(cmd))
    env = os.environ.copy()
    env["RUN_ID"] = ""
    t = subprocess.run(cmd, env=env, cwd=str(ROOT))
    # 找最新 ckpt
    out_root = ROOT / "outputs" / output_subdir
    if not out_root.exists():
        return t.returncode, None
    sub = sorted([d for d in out_root.iterdir() if d.is_dir()],
                  key=lambda d: d.stat().st_mtime, reverse=True)
    if sub:
        ckpt = sub[0] / model / f"best_{model}.pth"
        if ckpt.exists():
            return t.returncode, ckpt
    return t.returncode, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["within", "transfer", "joint"],
                   required=True)
    p.add_argument("--model", default="resnet50")
    p.add_argument("--datasets", nargs="+", default=["AL6", "ASP", "AS25"])
    p.add_argument("--source", help="transfer 模式的源数据集")
    p.add_argument("--target", help="transfer 模式的目标数据集")
    p.add_argument("--img-size", type=int, nargs=2, default=[224, 224])
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--epochs-source", type=int, default=None,
                   help="transfer 模式中 source 阶段 epoch (默认 = epochs)")
    p.add_argument("--epochs-target", type=int, default=None,
                   help="transfer 模式中 target fine-tune epoch")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--nproc", type=int, default=4)
    p.add_argument("--extra-args", default="")
    p.add_argument("--output-subdir", default="cross_dataset")
    args = p.parse_args()

    H, W = args.img_size
    extra = args.extra_args.split() if args.extra_args else []
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = ROOT / "outputs" / args.output_subdir / run_id
    out_root.mkdir(parents=True, exist_ok=True)

    summary = {"run_id": run_id, "mode": args.mode, "model": args.model,
                "results": {}}

    if args.mode == "within":
        for ds in args.datasets:
            print(f"\n=== within: {ds} ===")
            sub = f"{args.output_subdir}/{run_id}/within_{ds}"
            rc, ckpt = run_train(args.model, ds, H, W,
                                    args.epochs, args.batch_size,
                                    args.seed, args.nproc, extra, sub)
            summary["results"][ds] = {"status": "ok" if rc == 0 else "failed",
                                       "ckpt": str(ckpt) if ckpt else None}

    elif args.mode == "transfer":
        if not args.source or not args.target:
            sys.exit("transfer mode requires --source and --target")
        ep_src = args.epochs_source or args.epochs
        ep_tgt = args.epochs_target or args.epochs

        # Phase A: Imagenet -> source pretrain
        print(f"\n=== transfer phase A: ImageNet -> {args.source} ===")
        sub_a = f"{args.output_subdir}/{run_id}/A_imagenet2{args.source}"
        rc_a, ckpt_a = run_train(args.model, args.source, H, W,
                                    ep_src, args.batch_size,
                                    args.seed, args.nproc, extra, sub_a)
        summary["results"]["imagenet2source"] = {
            "status": "ok" if rc_a == 0 else "failed",
            "ckpt": str(ckpt_a) if ckpt_a else None,
        }
        if rc_a != 0 or ckpt_a is None:
            sys.exit(f"phase A failed: rc={rc_a}")

        # Phase B: source ckpt -> target fine-tune
        # 这里的 trick: train_ddp.py 的 weights 还是用 ImageNet, 我们 通过
        # config.model.weights_path 注入 source ckpt. 当前脚本未走 config 注入,
        # 所以我们用一个 hack: 先训完后把 ckpt 复制到 outputs/ddp/_xfer_init/
        # 然后用 --extra-args 传一个特殊标志? 直接修改不太干净.
        # 最简单: 用 train_all_paper 风格, 让 train_ddp 启动后以 weights_path
        # 加载. 我们要给 train_ddp 加一个 --init-ckpt 选项.
        print(f"\n=== transfer phase B: {args.source} -> {args.target} ===")
        sub_b = f"{args.output_subdir}/{run_id}/B_{args.source}2{args.target}"
        # 可以通过 env var 把 ckpt 路径传给训练 (需要 train_ddp 支持读取)
        env_ckpt = str(ckpt_a)
        os.environ["INIT_CKPT_PATH"] = env_ckpt
        rc_b, ckpt_b = run_train(args.model, args.target, H, W,
                                    ep_tgt, args.batch_size,
                                    args.seed, args.nproc,
                                    extra + ["--init-ckpt", env_ckpt],
                                    sub_b)
        summary["results"]["source2target"] = {
            "status": "ok" if rc_b == 0 else "failed",
            "ckpt": str(ckpt_b) if ckpt_b else None,
            "init_from": env_ckpt,
        }

        # Phase C (control): ImageNet -> target 直接 fine-tune
        print(f"\n=== transfer phase C (control): ImageNet -> {args.target} ===")
        sub_c = f"{args.output_subdir}/{run_id}/C_imagenet2{args.target}"
        # 清掉 INIT_CKPT_PATH
        os.environ.pop("INIT_CKPT_PATH", None)
        rc_c, ckpt_c = run_train(args.model, args.target, H, W,
                                    ep_tgt, args.batch_size,
                                    args.seed, args.nproc, extra, sub_c)
        summary["results"]["imagenet2target_control"] = {
            "status": "ok" if rc_c == 0 else "failed",
            "ckpt": str(ckpt_c) if ckpt_c else None,
        }

    elif args.mode == "joint":
        sys.exit("joint mode not yet implemented; use --mode within or transfer")

    with open(out_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n=== Done ===")
    print(f"  saved: {out_root / 'summary.json'}")


if __name__ == "__main__":
    main()
