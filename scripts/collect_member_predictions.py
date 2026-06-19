#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
为集成实验收集每个 backbone member 的预测 logits.

输入: 多个训完的 backbone checkpoint (.pth) + 它们的 backbone 名 + 输入尺寸
输出: outputs/ensemble_inputs/<run_id>/members.npz
       含 val_logits[K,N_val,C], test_logits[K,N_test,C], labels, val_acc[K]

用法:
    python scripts/collect_member_predictions.py --dataset AL6 --img-size 224 224 \\
        --members resnet50:outputs/ddp/<id>/resnet50/best_resnet50.pth:224 \\
                  efficientnet_b3:.../best_efficientnet_b3.pth:300 \\
                  ...
        # 格式: <model_name>:<ckpt_path>:<img_h> (假设方形, h=w)
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
from src.data.cached_dataset import RAMCachedDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


def parse_member_spec(s: str) -> tuple[str, str, int]:
    parts = s.split(":")
    if len(parts) != 3:
        raise ValueError(
            f"member spec should be <name>:<ckpt>:<img_size>, got: {s}"
        )
    name, ckpt, sz = parts
    return name, ckpt, int(sz)


@torch.no_grad()
def collect_logits(model: torch.nn.Module, dataset_pt: Path,
                    indices: np.ndarray | None,
                    device: torch.device, batch_size: int = 64,
                    img_size: int = 224) -> tuple[np.ndarray, np.ndarray]:
    """对 PT 缓存里指定 indices 的样本跑前向, 返回 logits + labels."""
    ds = RAMCachedDataset(dataset_pt, normalize=False, gpu_normalize=True,
                            in_memory_indices=indices)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                         num_workers=0)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    model = model.to(device).eval()
    all_logits, all_labels = [], []
    for x, y in loader:
        x = x.float().to(device) / 255.0
        x = (x - mean) / std
        out = model(x)
        if isinstance(out, tuple):
            out = out[0]
        all_logits.append(out.cpu().numpy())
        all_labels.append(y.numpy())
    return np.concatenate(all_logits, axis=0), np.concatenate(all_labels, axis=0)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="AL6")
    p.add_argument("--img-size", type=int, nargs=2, default=[224, 224])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-split", type=float, default=0.3)
    p.add_argument("--members", nargs="+", required=True,
                   help="<name>:<ckpt>:<img_size> 三元组列表")
    p.add_argument("--output-subdir", default="ensemble_inputs")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    H, W = args.img_size
    device = torch.device(args.device)
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "outputs" / args.output_subdir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    members = [parse_member_spec(m) for m in args.members]
    print(f"\n=== Collecting member predictions ===")
    print(f"  dataset: {args.dataset}  img: {H}x{W}")
    print(f"  members: {[m[0] for m in members]}")

    # 切分: 用相同 seed/split 重现 BaseClassifier 的 train/val 划分
    train_pt = ROOT / "data" / "cache" / f"{args.dataset}_{H}x{W}_rgb_train.pt"
    test_pt = ROOT / "data" / "cache" / f"{args.dataset}_{H}x{W}_rgb_test.pt"
    if not train_pt.exists():
        sys.exit(f"missing: {train_pt}")
    train_cache = torch.load(train_pt, map_location="cpu", weights_only=False)
    labels_all = train_cache["labels"].numpy()
    all_idx = np.arange(len(labels_all))
    _, val_idx = train_test_split(
        all_idx, test_size=args.val_split, random_state=args.seed,
        stratify=labels_all,
    )
    val_idx = val_idx.astype(np.int64)

    # 收集每个成员的 val/test logits
    val_logits_list, test_logits_list, val_accs = [], [], []
    val_labels_ref = None
    test_labels_ref = None
    num_classes = int(labels_all.max()) + 1

    for name, ckpt, msize in members:
        # 重新构建模型 (不依赖 .pth 文件中的 config, 用 default)
        config = load_config(overrides={
            "model": {"name": name},
            "data": {"dataset": args.dataset,
                      "img_height": msize, "img_width": msize},
        })
        Cls = get_backbone(name)
        instance = Cls.__new__(Cls)
        instance.config = config
        instance.num_classes = num_classes
        instance.device = device
        instance._to_rgb = (name != "custom_mlp")
        model = instance.build_model()
        sd = torch.load(ckpt, map_location="cpu", weights_only=True)
        model.load_state_dict(sd, strict=False)

        # 收集 val
        member_train_pt = ROOT / "data" / "cache" / \
            f"{args.dataset}_{msize}x{msize}_rgb_train.pt"
        member_test_pt = ROOT / "data" / "cache" / \
            f"{args.dataset}_{msize}x{msize}_rgb_test.pt"
        if not member_train_pt.exists() or not member_test_pt.exists():
            print(f"  [skip {name}] missing PT cache for {msize}x{msize}")
            continue

        # 对该 size 的 train cache 用相同 stratified split (val 索引一致)
        # 注意: 不同 size 的 PT 缓存样本顺序应该一致 (基于同一 train_mapping.csv)
        v_logits, v_labels = collect_logits(
            model, member_train_pt, val_idx, device,
            img_size=msize,
        )
        t_logits, t_labels = collect_logits(
            model, member_test_pt, None, device, img_size=msize,
        )
        val_acc = float((v_logits.argmax(1) == v_labels).mean())
        val_logits_list.append(v_logits)
        test_logits_list.append(t_logits)
        val_accs.append(val_acc)
        if val_labels_ref is None:
            val_labels_ref = v_labels
            test_labels_ref = t_labels
        print(f"  [{name}] val_acc = {val_acc:.4f}  "
              f"val_logits {v_logits.shape}, test_logits {t_logits.shape}")

    val_logits = np.stack(val_logits_list, axis=0)              # [K, N_val, C]
    test_logits = np.stack(test_logits_list, axis=0)            # [K, N_test, C]
    val_accs_arr = np.array(val_accs)

    payload = {
        "members": [m[0] for m in members],
        "val_logits": val_logits,
        "test_logits": test_logits,
        "val_labels": val_labels_ref,
        "test_labels": test_labels_ref,
        "val_acc": val_accs_arr,
        "dataset": args.dataset,
        "num_classes": num_classes,
    }
    np.savez_compressed(out_dir / "members.npz", **payload)
    with open(out_dir / "members_meta.json", "w") as f:
        json.dump({
            "members": [{"name": m[0], "ckpt": m[1], "size": m[2],
                          "val_acc": float(va)}
                         for m, va in zip(members, val_accs)],
            "dataset": args.dataset, "num_classes": num_classes,
            "val_logits_shape": list(val_logits.shape),
            "test_logits_shape": list(test_logits.shape),
        }, f, indent=2, ensure_ascii=False)

    print(f"\n=== Done ===")
    print(f"  members.npz: {out_dir / 'members.npz'}")
    print(f"  K x N_val x C = {val_logits.shape}")
    print(f"  K x N_test x C = {test_logits.shape}")


if __name__ == "__main__":
    main()
