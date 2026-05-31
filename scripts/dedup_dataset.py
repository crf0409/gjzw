#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据集去重清洗 — 针对 AS25 / ASP 等 Kaggle 数据集本身存在的重复.

策略 (在审计基础上):
    1. 移除 train_internal / test_internal exact duplicate (保留每组第一个)
    2. 移除 train↔test 跨集 exact duplicate (从 test 里删掉)
    3. 可选: 移除 train↔test pHash near-duplicate (Hamming ≤ threshold)
       用于产生 "strict-clean" 版本

输出:
    data/processed/<dataset>_clean/   — 清洗后版本, 与原结构兼容
    outputs/data_audit/<dataset>/dedup_action.json  — 移除清单

用法:
    python scripts/dedup_dataset.py --dataset AS25
    python scripts/dedup_dataset.py --dataset AS25 --strict   # 也移除 phash 近似
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import imagehash
import pandas as pd
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            buf = f.read(65536)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True,
                   help="AL6 / ASP / AS25")
    p.add_argument("--strict", action="store_true",
                   help="也移除 phash 近似 (≤6 Hamming)")
    p.add_argument("--phash-threshold", type=int, default=6)
    p.add_argument("--suffix", default="_clean",
                   help="输出目录后缀 (e.g., _clean, _strict)")
    args = p.parse_args()

    src = PROCESSED / args.dataset
    if not src.exists():
        sys.exit(f"missing: {src}")

    suffix = "_strict" if args.strict else args.suffix
    dst = PROCESSED / f"{args.dataset}{suffix}"
    if dst.exists():
        print(f"[!] {dst} exists, removing ...")
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    print(f"\n=== Dedup {args.dataset} -> {dst.name} ===")

    train_df = pd.read_csv(src / "train_mapping.csv")
    test_df = pd.read_csv(src / "test_mapping.csv")
    print(f"  source: {len(train_df)} train, {len(test_df)} test")

    # 1) 计算 hashes
    print("  hashing...")
    train_md5, train_ph = [], []
    for _, r in tqdm(train_df.iterrows(), total=len(train_df),
                       desc="  train", leave=False):
        p = src / "train" / r["文件名"]
        train_md5.append(md5_file(p) if p.exists() else None)
        if args.strict:
            try:
                train_ph.append(imagehash.phash(Image.open(p).convert("RGB")))
            except Exception:
                train_ph.append(None)
    test_md5, test_ph = [], []
    for _, r in tqdm(test_df.iterrows(), total=len(test_df),
                       desc="  test", leave=False):
        p = src / "test" / r["文件名"]
        test_md5.append(md5_file(p) if p.exists() else None)
        if args.strict:
            try:
                test_ph.append(imagehash.phash(Image.open(p).convert("RGB")))
            except Exception:
                test_ph.append(None)

    # 2) 决定移除集合
    actions = {"train_drop": [], "test_drop": [], "reason": {}}

    # 2a) train internal dup: 保留第一个
    seen_train = {}
    for i, h in enumerate(train_md5):
        if h is None:
            continue
        if h not in seen_train:
            seen_train[h] = i
        else:
            actions["train_drop"].append(i)
            actions["reason"][f"train_{i}"] = f"dup of train_{seen_train[h]} (md5)"

    # 2b) test internal dup: 保留第一个
    seen_test = {}
    for i, h in enumerate(test_md5):
        if h is None:
            continue
        if h not in seen_test:
            seen_test[h] = i
        else:
            actions["test_drop"].append(i)
            actions["reason"][f"test_{i}"] = f"dup of test_{seen_test[h]} (md5)"

    # 2c) train↔test cross dup: 从 test 移除
    train_md5_set = set(seen_train)
    for i, h in enumerate(test_md5):
        if h is None:
            continue
        if i in actions["test_drop"]:
            continue
        if h in train_md5_set:
            actions["test_drop"].append(i)
            actions["reason"][f"test_{i}"] = "cross dup with train (md5)"

    # 2d) strict mode: phash near-dup train↔test 也从 test 移除
    if args.strict:
        ph_thresh = args.phash_threshold
        n_test_before = len(test_md5)
        for i, ph in enumerate(test_ph):
            if ph is None or i in actions["test_drop"]:
                continue
            for j, ph_tr in enumerate(train_ph):
                if ph_tr is None or j in actions["train_drop"]:
                    continue
                if (ph - ph_tr) <= ph_thresh:
                    actions["test_drop"].append(i)
                    actions["reason"][f"test_{i}"] = (
                        f"phash near-dup with train_{j} "
                        f"(Hamming={int(ph - ph_tr)})"
                    )
                    break

    train_drop_set = set(actions["train_drop"])
    test_drop_set = set(actions["test_drop"])
    print(f"\n  train: drop {len(train_drop_set)} / {len(train_df)} "
          f"-> keep {len(train_df) - len(train_drop_set)}")
    print(f"  test:  drop {len(test_drop_set)} / {len(test_df)} "
          f"-> keep {len(test_df) - len(test_drop_set)}")

    # 3) 物化 cleaned 数据集
    (dst / "train").mkdir()
    (dst / "test").mkdir()
    new_train_rows = []
    for i, row in tqdm(train_df.iterrows(), total=len(train_df),
                        desc="  copy train", leave=False):
        if i in train_drop_set:
            continue
        src_p = src / "train" / row["文件名"]
        cls_dir = dst / "train" / str(int(row["标签"]))
        cls_dir.mkdir(parents=True, exist_ok=True)
        dst_p = cls_dir / Path(row["文件名"]).name
        if not dst_p.exists():
            shutil.copy2(src_p, dst_p)
        new_train_rows.append({
            "序号": len(new_train_rows) + 1,
            "原始索引": int(row.get("原始索引", i + 1)),
            "标签": int(row["标签"]),
            "文件名": f"{int(row['标签'])}/{Path(row['文件名']).name}",
        })
    new_test_rows = []
    for i, row in tqdm(test_df.iterrows(), total=len(test_df),
                        desc="  copy test", leave=False):
        if i in test_drop_set:
            continue
        src_p = src / "test" / row["文件名"]
        cls_dir = dst / "test" / str(int(row["标签"]))
        cls_dir.mkdir(parents=True, exist_ok=True)
        dst_p = cls_dir / Path(row["文件名"]).name
        if not dst_p.exists():
            shutil.copy2(src_p, dst_p)
        new_test_rows.append({
            "序号": len(new_test_rows) + 1,
            "原始索引": int(row.get("原始索引", i + 1)),
            "标签": int(row["标签"]),
            "文件名": f"{int(row['标签'])}/{Path(row['文件名']).name}",
        })

    pd.DataFrame(new_train_rows).to_csv(
        dst / "train_mapping.csv", index=False, encoding="utf-8-sig",
    )
    pd.DataFrame(new_test_rows).to_csv(
        dst / "test_mapping.csv", index=False, encoding="utf-8-sig",
    )

    # 复制 class_map
    cm_src = src / "class_map.json"
    if cm_src.exists():
        shutil.copy2(cm_src, dst / "class_map.json")

    # 写 action log
    action_log = ROOT / "outputs" / "data_audit" / args.dataset / \
        f"dedup_action_{suffix.lstrip('_')}.json"
    action_log.parent.mkdir(parents=True, exist_ok=True)
    with open(action_log, "w") as f:
        json.dump({
            "source": args.dataset,
            "destination": f"{args.dataset}{suffix}",
            "strict_mode": args.strict,
            "n_train_dropped": len(train_drop_set),
            "n_test_dropped": len(test_drop_set),
            "train_dropped_indices": sorted(train_drop_set),
            "test_dropped_indices": sorted(test_drop_set),
            "reasons": actions["reason"],
        }, f, indent=2, ensure_ascii=False)

    print(f"\n  cleaned dataset: {dst}")
    print(f"  action log:      {action_log}")


if __name__ == "__main__":
    main()
