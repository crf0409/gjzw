#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对整个数据集拟合 INR (SIREN) 权重 -> 保存为 PT 数据集供分类训练用.

输入:  data/cache/<dataset>_<H>x<W>_rgb_<split>.pt   (uint8 像素缓存)
输出:  data/cache_inr/<dataset>_<arch>_<hidden>x<layers>_<H>x<W>_<split>.pt
        包含 weights [N, total_params], labels [N], metrics

用法:
    python scripts/fit_inr_dataset.py --dataset AL6 --split train \\
        --arch siren --hidden 256 --layers 4 --steps 500 --batch 32

作为 INR-AncientArch 的 pretext task. 拟合质量 (PSNR) 越高,
权重作为分类特征的判别力越强 (论文消融的关联关系).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.inr import INRFitConfig, fit_dataset


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="AL6",
                   help="任意 data/processed/<name>/ 目录 (含 _clean / _strict 变体)")
    p.add_argument("--split", choices=["train", "test"], default="train")
    p.add_argument("--img-size", type=int, nargs=2, default=[224, 224],
                   metavar=("H", "W"))
    p.add_argument("--arch", choices=["siren"], default="siren",
                   help="目前仅支持 SIREN 的 batched 并行拟合")
    p.add_argument("--hidden", type=int, default=256, help="hidden_dim")
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--omega", type=float, default=30.0)
    p.add_argument("--steps", type=int, default=500,
                   help="每 batch 拟合迭代步数")
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--batch", type=int, default=32,
                   help="一次并行拟合的图像数 (BatchedSIREN B)")
    p.add_argument("--target-psnr", type=float, default=35.0,
                   help="达到此 PSNR 提前停 (0 表示不早停)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=None,
                   help="只拟合前 N 张 (调试用)")
    args = p.parse_args()

    H, W = args.img_size
    in_path = ROOT / "data" / "cache" / \
        f"{args.dataset}_{H}x{W}_rgb_{args.split}.pt"
    if not in_path.exists():
        sys.exit(
            f"missing input: {in_path}\n"
            f"  run: python scripts/build_pt_cache.py --dataset {args.dataset} "
            f"--size {H} {W}"
        )

    out_dir = ROOT / "data" / "cache_inr"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (
        f"{args.dataset}_{args.arch}_h{args.hidden}_L{args.layers}_"
        f"{H}x{W}_{args.split}.pt"
    )
    # 命中缓存则跳过 (limit 不同时仍重做)
    if out_path.exists() and args.limit is None:
        print(f"\n[skip] cache exists: {out_path}")
        return

    print(f"\n=== INR Fitting ===")
    print(f"  input:  {in_path}")
    print(f"  output: {out_path}")
    print(f"  arch:   {args.arch}  hidden={args.hidden}  layers={args.layers}")
    print(f"  steps:  {args.steps}  lr={args.lr}  batch={args.batch}")

    cache = torch.load(in_path, map_location="cpu", weights_only=False)
    images = cache["images"]                                       # uint8 [N,C,H,W]
    labels = cache["labels"]                                       # int64 [N]

    if args.limit is not None:
        images = images[:args.limit]
        labels = labels[:args.limit]

    # 归一化到 [-1, 1] (INR 论文常用)
    images_f = images.float() / 127.5 - 1.0

    cfg = INRFitConfig(
        arch=args.arch,
        hidden_dim=args.hidden,
        num_layers=args.layers,
        omega_0=args.omega,
        n_steps=args.steps,
        lr=args.lr,
        batch_size=args.batch,
        target_psnr=args.target_psnr,
    )

    device = torch.device(args.device)
    t0 = time.time()
    weights, labels_out, metrics = fit_dataset(
        images_f, labels, cfg, device, progress=True
    )
    elapsed = time.time() - t0

    payload = {
        "weights": weights,                  # [N, total_params]
        "labels": labels_out,                # [N]
        "class_names": cache.get("class_names", []),
        "metrics": metrics,
        "fit_args": vars(args),
    }
    torch.save(payload, out_path)

    size_mb = weights.element_size() * weights.nelement() / 1024 / 1024
    print(f"\n=== Done ({elapsed:.1f}s) ===")
    print(f"  weights shape: {tuple(weights.shape)}")
    print(f"  storage:       {size_mb:.1f} MB")
    print(f"  mean PSNR:     {metrics['mean_psnr']:.2f} dB")
    print(f"  mean MSE:      {metrics['mean_mse']:.5f}")
    print(f"  saved to:      {out_path}")

    # 同时输出可读 metadata json
    meta_json = out_path.with_suffix(".meta.json")
    with open(meta_json, "w") as f:
        json.dump({
            "metrics": metrics,
            "args": vars(args),
            "input_pt": str(in_path),
            "output_pt": str(out_path),
            "weight_storage_mb": float(size_mb),
            "elapsed_seconds": float(elapsed),
        }, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
