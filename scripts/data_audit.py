#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据审计: 防止 SCI 2 区审稿人质疑数据泄漏 / 近重复 / 任务过简单.

检查项:
    1. Exact duplicate (MD5 hash) — 全集内 + train↔test 跨集
    2. Near-duplicate (perceptual hash, pHash + dHash) — train↔test 跨集
    3. Original index intersection — 从 mapping CSV "原始索引" 列检查
    4. 类别样本可视化 — 每类 8 张代表样本网格
    5. 类别均衡 / 分辨率统计 / 文件大小统计

输出:
    outputs/data_audit/<dataset>/
        audit.json
        per_class_samples.png
        duplicate_report.md
        near_duplicate_pairs.png       (若有发现)
        size_distribution.png

用法:
    python scripts/data_audit.py --dataset AL6
    python scripts/data_audit.py --dataset all   # AL6 + ASP + AS25
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import imagehash

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"


# ────────────────────────────────────────────────────────────
# Hashing
# ────────────────────────────────────────────────────────────

def md5_of_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            buf = f.read(65536)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def perceptual_hashes(path: Path):
    """返回 (phash, dhash, ahash) 三种 perceptual hash."""
    img = Image.open(path).convert("RGB")
    return (
        imagehash.phash(img),
        imagehash.dhash(img),
        imagehash.average_hash(img),
    )


# ────────────────────────────────────────────────────────────
# Per-class sample grid
# ────────────────────────────────────────────────────────────

def render_per_class_grid(dataset_root: Path, split: str,
                            class_map: dict, save_path: Path,
                            n_samples: int = 8) -> None:
    """每类抽 n_samples 张, 渲染 K-row × n_samples-col 网格."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(dataset_root / f"{split}_mapping.csv")
    classes = sorted(set(df["标签"].astype(int)))
    K = len(classes)

    fig, axes = plt.subplots(K, n_samples,
                               figsize=(n_samples * 1.6, K * 1.6))
    if K == 1:
        axes = axes.reshape(1, -1)
    rng = np.random.RandomState(42)
    for r, lab in enumerate(classes):
        sub = df[df["标签"] == lab]
        chosen = sub.sample(n=min(n_samples, len(sub)), random_state=42)
        cname = class_map.get(str(lab), f"C{lab}")
        for c, (_, row) in enumerate(chosen.iterrows()):
            ax = axes[r, c]
            try:
                img = Image.open(dataset_root / split / row["文件名"])
                ax.imshow(img)
            except Exception:
                ax.text(0.5, 0.5, "missing", ha="center", va="center")
            ax.axis("off")
            if c == 0:
                ax.set_ylabel(cname[:20], rotation=0, ha="right",
                                va="center", fontsize=8)
        # 不够 n_samples 的留空
        for c in range(len(chosen), n_samples):
            axes[r, c].axis("off")
    plt.suptitle(f"{dataset_root.name} — {split} 每类 {n_samples} 个样本",
                  fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


# ────────────────────────────────────────────────────────────
# Stats
# ────────────────────────────────────────────────────────────

def collect_image_stats(dataset_root: Path, split: str
                          ) -> tuple[list, list, list]:
    """返回 (paths, sizes_bytes, dimensions_HxW)"""
    df = pd.read_csv(dataset_root / f"{split}_mapping.csv")
    split_dir = dataset_root / split
    paths, byts, dims, labs = [], [], [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  {split} stats",
                        leave=False):
        p = split_dir / row["文件名"]
        if not p.exists():
            continue
        paths.append(p)
        labs.append(int(row["标签"]))
        byts.append(p.stat().st_size)
        try:
            with Image.open(p) as im:
                dims.append(im.size)                           # (W, H)
        except Exception:
            dims.append((0, 0))
    return paths, byts, dims, labs


# ────────────────────────────────────────────────────────────
# 主审计
# ────────────────────────────────────────────────────────────

def audit_dataset(name: str, force: bool = False) -> dict:
    print(f"\n{'='*60}\n  AUDIT: {name}\n{'='*60}")
    dataset_root = PROCESSED / name
    if not dataset_root.exists():
        print(f"  [skip] {dataset_root} not found")
        return {"status": "missing"}

    out_dir = ROOT / "outputs" / "data_audit" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    audit_json = out_dir / "audit.json"
    if audit_json.exists() and not force:
        print(f"  [skip] already audited: {audit_json}")
        with open(audit_json) as f:
            return json.load(f)

    class_map_path = dataset_root / "class_map.json"
    class_map = (json.loads(class_map_path.read_text())
                  if class_map_path.exists() else {})

    splits = ["train", "test"]

    # 1) 收集 stats + hashes
    print("\n[1/5] 收集图像统计 + 计算 hash ...")
    all_paths = {}
    all_labs = {}
    all_md5 = {}
    all_phash = {}
    all_dhash = {}
    all_dims = {}
    all_bytes = {}

    for split in splits:
        paths, byts, dims, labs = collect_image_stats(dataset_root, split)
        all_paths[split] = paths
        all_labs[split] = labs
        all_dims[split] = dims
        all_bytes[split] = byts

        md5s, phashes, dhashes = [], [], []
        t0 = time.time()
        for p in tqdm(paths, desc=f"  {split} hashes", leave=False):
            md5s.append(md5_of_file(p))
            try:
                phs = perceptual_hashes(p)
                phashes.append(phs[0])
                dhashes.append(phs[1])
            except Exception:
                phashes.append(None)
                dhashes.append(None)
        all_md5[split] = md5s
        all_phash[split] = phashes
        all_dhash[split] = dhashes
        print(f"  [{split}] N={len(paths)} hashed in {time.time()-t0:.1f}s")

    # 2) Exact duplicate
    print("\n[2/5] Exact duplicate (MD5) 检查 ...")
    exact = {"train_internal": [], "test_internal": [],
             "train_test_cross": []}

    def find_dupes_internal(md5s, paths):
        d = defaultdict(list)
        for i, h in enumerate(md5s):
            d[h].append(i)
        return [(h, [str(paths[i].name) for i in idxs])
                for h, idxs in d.items() if len(idxs) > 1]

    exact["train_internal"] = find_dupes_internal(
        all_md5["train"], all_paths["train"])
    exact["test_internal"] = find_dupes_internal(
        all_md5["test"], all_paths["test"])

    train_md5_set = set(all_md5["train"])
    cross = []
    for i, h in enumerate(all_md5["test"]):
        if h in train_md5_set:
            j = all_md5["train"].index(h)
            cross.append({
                "test_file": all_paths["test"][i].name,
                "train_file": all_paths["train"][j].name,
                "md5": h,
            })
    exact["train_test_cross"] = cross
    print(f"  exact dup train_internal: {len(exact['train_internal'])}")
    print(f"  exact dup test_internal:  {len(exact['test_internal'])}")
    print(f"  exact dup train↔test:    {len(exact['train_test_cross'])}")

    # 3) Near-duplicate (perceptual hash, Hamming distance ≤ threshold)
    print("\n[3/5] Near-duplicate (perceptual hash) 检查 train↔test 跨集 ...")
    near_dups = []
    pthresh = 6  # phash 64-bit, ≤6 bits 通常视为近似 (imagehash 官方推荐)
    train_phs = all_phash["train"]
    test_phs = all_phash["test"]
    for i, ph_t in enumerate(tqdm(test_phs, desc="  near-dup",
                                     leave=False)):
        if ph_t is None:
            continue
        for j, ph_tr in enumerate(train_phs):
            if ph_tr is None:
                continue
            if (ph_t - ph_tr) <= pthresh:                       # Hamming distance
                near_dups.append({
                    "test_file": all_paths["test"][i].name,
                    "train_file": all_paths["train"][j].name,
                    "phash_distance": int(ph_t - ph_tr),
                    "test_label": all_labs["test"][i],
                    "train_label": all_labs["train"][j],
                })
                if len(near_dups) >= 200:
                    break
        if len(near_dups) >= 200:
            break

    # 按相似度排序
    near_dups.sort(key=lambda x: x["phash_distance"])
    print(f"  near-dup train↔test (phash≤{pthresh}): {len(near_dups)}")
    if near_dups:
        for d in near_dups[:5]:
            print(f"    [{d['phash_distance']}] test={d['test_file']} ↔ "
                  f"train={d['train_file']} (label {d['test_label']}↔"
                  f"{d['train_label']})")

    # 4) Original index intersection (从 mapping CSV "原始索引")
    print("\n[4/5] 原始索引交集检查 ...")
    df_train = pd.read_csv(dataset_root / "train_mapping.csv")
    df_test = pd.read_csv(dataset_root / "test_mapping.csv")
    if "原始索引" in df_train.columns and "原始索引" in df_test.columns:
        train_idx = set(df_train["原始索引"].astype(int).tolist())
        test_idx = set(df_test["原始索引"].astype(int).tolist())
        idx_intersect = sorted(train_idx & test_idx)
        print(f"  原始索引交集: {len(idx_intersect)} (示例: {idx_intersect[:10]})")
    else:
        idx_intersect = []
        print("  no 原始索引 column, skip")

    # 5) 类别均衡 + 分辨率
    print("\n[5/5] 类别均衡 + 分辨率统计 ...")
    class_counts = {split: defaultdict(int) for split in splits}
    for split in splits:
        for lab in all_labs[split]:
            class_counts[split][int(lab)] += 1
    class_balance = {
        split: dict(sorted(d.items())) for split, d in class_counts.items()
    }
    res_summary = {}
    for split in splits:
        ws = [d[0] for d in all_dims[split] if d[0] > 0]
        hs = [d[1] for d in all_dims[split] if d[1] > 0]
        res_summary[split] = {
            "n": len(ws),
            "w_mean": float(np.mean(ws)) if ws else 0,
            "w_std": float(np.std(ws)) if ws else 0,
            "h_mean": float(np.mean(hs)) if hs else 0,
            "h_std": float(np.std(hs)) if hs else 0,
            "w_min": int(min(ws)) if ws else 0,
            "w_max": int(max(ws)) if ws else 0,
            "h_min": int(min(hs)) if hs else 0,
            "h_max": int(max(hs)) if hs else 0,
        }
    bytes_summary = {
        split: {
            "total_mb": float(sum(all_bytes[split]) / 1024 / 1024),
            "mean_kb": float(np.mean(all_bytes[split]) / 1024),
        } for split in splits
    }

    # 6) 类别样本网格图
    for split in splits:
        grid_path = out_dir / f"per_class_samples_{split}.png"
        try:
            render_per_class_grid(dataset_root, split, class_map, grid_path)
            print(f"  saved grid: {grid_path}")
        except Exception as e:
            print(f"  warn: grid failed: {e}")

    # 7) 大小分布图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for split, c in zip(splits, ["tab:blue", "tab:orange"]):
            sizes_kb = np.array(all_bytes[split]) / 1024
            axes[0].hist(sizes_kb, bins=40, alpha=0.5, label=split, color=c)
            ws = [d[0] for d in all_dims[split] if d[0] > 0]
            axes[1].hist(ws, bins=40, alpha=0.5, label=split, color=c)
        axes[0].set_xlabel("size (KB)"); axes[0].set_ylabel("count")
        axes[0].set_title("File size distribution"); axes[0].legend()
        axes[1].set_xlabel("width (px)"); axes[1].set_ylabel("count")
        axes[1].set_title("Image width distribution"); axes[1].legend()
        plt.tight_layout()
        plt.savefig(out_dir / "size_distribution.png", dpi=150)
        plt.close()
    except Exception as e:
        print(f"  warn: size plot failed: {e}")

    # 8) 近重复对可视化 (top-12 最像的)
    if near_dups:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            top = near_dups[:12]
            fig, axes = plt.subplots(len(top), 2, figsize=(6, 2.4 * len(top)))
            if len(top) == 1:
                axes = axes.reshape(1, -1)
            for i, d in enumerate(top):
                test_p = dataset_root / "test" / d["test_file"]
                train_p = dataset_root / "train" / d["train_file"]
                try:
                    axes[i, 0].imshow(Image.open(test_p))
                    axes[i, 0].set_title(
                        f"test #{d['test_label']}", fontsize=8)
                    axes[i, 1].imshow(Image.open(train_p))
                    axes[i, 1].set_title(
                        f"train #{d['train_label']} (Hamming={d['phash_distance']})",
                        fontsize=8)
                except Exception:
                    pass
                axes[i, 0].axis("off"); axes[i, 1].axis("off")
            plt.tight_layout()
            plt.savefig(out_dir / "near_duplicate_pairs.png", dpi=150)
            plt.close()
        except Exception as e:
            print(f"  warn: near-dup viz failed: {e}")

    # ─── 总结 ───
    audit = {
        "dataset": name,
        "n_train": len(all_paths["train"]),
        "n_test": len(all_paths["test"]),
        "class_balance": class_balance,
        "image_resolution": res_summary,
        "file_size": bytes_summary,
        "exact_duplicate": {
            "train_internal_count": len(exact["train_internal"]),
            "test_internal_count": len(exact["test_internal"]),
            "train_test_cross_count": len(exact["train_test_cross"]),
            "train_internal_examples": exact["train_internal"][:5],
            "test_internal_examples": exact["test_internal"][:5],
            "train_test_cross_examples": exact["train_test_cross"][:10],
        },
        "near_duplicate_phash": {
            "threshold_hamming": pthresh,
            "n_pairs_found": len(near_dups),
            "examples_top10": near_dups[:10],
        },
        "original_index_intersection": {
            "count": len(idx_intersect),
            "indices_sample": idx_intersect[:20],
        },
    }
    with open(audit_json, "w") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)

    # markdown 简报
    md = [f"# Data Audit — {name}\n",
           f"- Train: {audit['n_train']}  Test: {audit['n_test']}",
           f"- Resolution train: "
           f"{res_summary['train']['w_mean']:.0f}±{res_summary['train']['w_std']:.0f} "
           f"× {res_summary['train']['h_mean']:.0f}±{res_summary['train']['h_std']:.0f}",
           f"\n## Duplicate findings\n",
           f"- Exact dup train_internal: **{len(exact['train_internal'])}**",
           f"- Exact dup test_internal: **{len(exact['test_internal'])}**",
           f"- Exact dup train↔test: **{len(exact['train_test_cross'])}**",
           f"- pHash near-dup train↔test (≤{pthresh}): **{len(near_dups)}**",
           f"- Original index 交集: **{len(idx_intersect)}**\n",
           "## Class balance\n",
           "| label | train | test |", "|---|---|---|"]
    for lab in sorted(set(class_balance["train"]) | set(class_balance["test"])):
        cn = class_map.get(str(lab), f"C{lab}")
        md.append(f"| {lab} ({cn}) | "
                  f"{class_balance['train'].get(lab, 0)} | "
                  f"{class_balance['test'].get(lab, 0)} |")
    (out_dir / "duplicate_report.md").write_text("\n".join(md), encoding="utf-8")

    print(f"\n  audit -> {audit_json}")
    print(f"  report -> {out_dir / 'duplicate_report.md'}")
    return audit


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="all",
                   help="AL6 / ASP / AS25 / all")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    if args.dataset == "all":
        targets = ["AL6", "ASP", "AS25"]
    else:
        targets = [args.dataset]

    summary_all = {}
    for ds in targets:
        summary_all[ds] = audit_dataset(ds, force=args.force)

    # 全局 summary
    out_global = ROOT / "outputs" / "data_audit" / "summary_all.json"
    with open(out_global, "w") as f:
        json.dump(summary_all, f, indent=2, ensure_ascii=False)
    print(f"\n=== ALL Done ===\n  global summary: {out_global}")


if __name__ == "__main__":
    main()
