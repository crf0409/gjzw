#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
预处理数据集 -> 单文件 PT 缓存 (uint8, 驻留 RAM)

输出布局:
    data/cache/<dataset>_<H>x<W>_<split>.pt
    -> dict{
        'images': uint8 [N, C, H, W],
        'labels': int64 [N],
        'class_names': [str, ...],
        'meta': {...}
    }

用法:
    python scripts/build_pt_cache.py --dataset AL6 --size 224 224
    python scripts/build_pt_cache.py --dataset AL6 --size 299 299 --grayscale
    python scripts/build_pt_cache.py --dataset all --sizes "224x224,299x299,300x300"

为何 uint8:
    2270 × 3 × 224 × 224 × float32 ≈ 1.4 GB
    2270 × 3 × 224 × 224 × uint8   ≈ 0.34 GB
    GPU 端再做 /255 + normalize, 既省 RAM 又零 IO 延迟.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PROCESSED = ROOT / "data" / "processed"
CACHE_DIR = ROOT / "data" / "cache"


def parse_size(s: str) -> tuple[int, int]:
    """'224x224' -> (224, 224)"""
    parts = s.lower().replace(" ", "").split("x")
    if len(parts) != 2:
        raise ValueError(f"bad size: {s}; expected HxW like 224x224")
    return int(parts[0]), int(parts[1])


def load_split(
    dataset_root: Path,
    split: str,
    size: tuple[int, int],
    grayscale: bool = False,
    auto_orient: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """
    Returns:
        images: uint8 [N, C, H, W]   (C=1 if grayscale else 3)
        labels: int64 [N]            (0-indexed)
        class_names: list[str]
    """
    csv_path = dataset_root / f"{split}_mapping.csv"
    split_dir = dataset_root / split
    class_map_path = dataset_root / "class_map.json"

    df = pd.read_csv(csv_path)
    print(f"  [{split}] {len(df)} entries from {csv_path.name}")

    # 类别名
    if class_map_path.exists():
        cm = json.loads(class_map_path.read_text())
        class_names = [cm[str(k)] for k in sorted(cm, key=lambda x: int(x))]
    else:
        class_names = []

    H, W = size
    C = 1 if grayscale else 3

    # 决定目标朝向 (与 BaseClassifier 行为一致): 看 train 集众数
    target_landscape = None
    if auto_orient:
        sample_paths = [split_dir / f for f in df["文件名"].head(200)]
        ws, hs = [], []
        for p in sample_paths:
            if p.exists():
                with Image.open(p) as im:
                    w, h = im.size
                    ws.append(w)
                    hs.append(h)
        if ws and hs:
            from collections import Counter
            most = Counter(zip(ws, hs)).most_common(1)[0][0]
            mw, mh = most
            target_landscape = mw > mh

    labels: list[int] = []
    images_buf = np.empty((len(df), C, H, W), dtype=np.uint8)

    skipped = 0
    write_idx = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  {split}",
                        leave=False):
        fname = row["文件名"]
        label = int(row["标签"]) - 1  # 0-indexed
        img_path = split_dir / fname

        if not img_path.exists():
            skipped += 1
            continue

        try:
            img = Image.open(img_path).convert("L")
            if auto_orient and target_landscape is not None:
                w, h = img.size
                cur_landscape = w > h
                if cur_landscape != target_landscape:
                    img = img.rotate(90, expand=True)
            img = img.resize((W, H), Image.BILINEAR)
            if not grayscale:
                img = img.convert("RGB")
            arr = np.asarray(img, dtype=np.uint8)
            if grayscale:
                arr = arr[None, :, :]                       # [1, H, W]
            else:
                arr = arr.transpose(2, 0, 1)                # [3, H, W]
            images_buf[write_idx] = arr
            labels.append(label)
            write_idx += 1
        except Exception as e:
            print(f"  warn: skip {img_path}: {e}")
            skipped += 1

    if skipped:
        print(f"  [{split}] skipped {skipped} images")
    images_t = torch.from_numpy(images_buf[:write_idx]).contiguous()
    labels_t = torch.tensor(labels, dtype=torch.int64)
    return images_t, labels_t, class_names


def build_one(
    dataset: str,
    size: tuple[int, int],
    grayscale: bool,
    splits: list[str],
    overwrite: bool = False,
) -> None:
    H, W = size
    suffix = "gray" if grayscale else "rgb"
    dataset_root = PROCESSED / dataset
    if not dataset_root.exists():
        print(f"[{dataset}] skip: {dataset_root} not found "
              f"(run dataset_unifier first)")
        return

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    for split in splits:
        out_path = CACHE_DIR / f"{dataset}_{H}x{W}_{suffix}_{split}.pt"
        if out_path.exists() and not overwrite:
            print(f"[{dataset}/{split}/{H}x{W}/{suffix}] skip (exists): {out_path.name}")
            continue

        print(f"\n[{dataset}/{split}/{H}x{W}/{suffix}] building -> {out_path.name}")
        t0 = time.time()
        images, labels, class_names = load_split(
            dataset_root, split, size, grayscale=grayscale
        )
        size_mb = images.element_size() * images.nelement() / (1024 ** 2)
        print(f"  shape={tuple(images.shape)}  dtype={images.dtype}  "
              f"size={size_mb:.1f} MB")

        torch.save({
            "images": images,
            "labels": labels,
            "class_names": class_names,
            "meta": {
                "dataset": dataset,
                "split": split,
                "h": H, "w": W,
                "mode": "L" if grayscale else "RGB",
                "n_samples": int(images.shape[0]),
                "n_classes": int(labels.max().item()) + 1 if len(labels) > 0 else 0,
            },
        }, out_path)
        print(f"  saved in {time.time() - t0:.1f}s -> {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="AL6",
                   help="任意 data/processed/<name>/ 下的目录名 "
                        "(AL6/ASP/AS25 + _clean/_strict 变体, 或 'all')")
    p.add_argument("--size", nargs=2, type=int, metavar=("H", "W"),
                   default=None,
                   help="单一目标尺寸, e.g. --size 224 224")
    p.add_argument("--sizes", type=str, default=None,
                   help="多尺寸列表: '224x224,299x299,300x300'")
    p.add_argument("--grayscale", action="store_true",
                   help="构建灰度缓存 (C=1), 给 custom_mlp 用")
    p.add_argument("--splits", nargs="+", default=["train", "test"])
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    # 决定 size 列表
    if args.sizes:
        sizes = [parse_size(s) for s in args.sizes.split(",")]
    elif args.size:
        sizes = [tuple(args.size)]
    else:
        sizes = [(224, 224)]

    # 决定 dataset 列表
    if args.dataset == "all":
        datasets = ["AL6", "ASP", "AS25"]
    else:
        datasets = [args.dataset]
        # 验证目标目录存在
        if not (PROCESSED / args.dataset).exists():
            sys.exit(f"missing: {PROCESSED / args.dataset}")

    print(f"Building PT caches:")
    print(f"  datasets: {datasets}")
    print(f"  sizes:    {sizes}")
    print(f"  splits:   {args.splits}")
    print(f"  grayscale: {args.grayscale}")
    print()

    for ds in datasets:
        for sz in sizes:
            build_one(ds, sz, args.grayscale, args.splits,
                      overwrite=args.overwrite)

    print("\nAll requested caches built under:", CACHE_DIR)


if __name__ == "__main__":
    main()
