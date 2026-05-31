#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
可解释性 driver — t-SNE / UMAP 嵌入 + 错例分析 + 类间相似度

用法:
    python scripts/run_interpretability.py \\
        --model resnet50 --dataset AL6 --img-size 224 224 \\
        --ckpt outputs/ddp/<id>/resnet50/best_resnet50.pth
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
from src.evaluation.interpret.embedding_viz import (
    extract_features, reduce_tsne, reduce_umap, render_scatter,
)
from src.evaluation.interpret.error_analysis import (
    confusion_pair_topk, lowest_confidence_examples,
    class_mean_feature_similarity,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", default="AL6")
    p.add_argument("--img-size", type=int, nargs=2, default=[224, 224])
    p.add_argument("--ckpt", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-subdir", default="interpret")
    p.add_argument("--per-class-failures", type=int, default=8)
    args = p.parse_args()

    H, W = args.img_size
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 加载模型
    cache_test = ROOT / "data" / "cache" / f"{args.dataset}_{H}x{W}_rgb_test.pt"
    if not cache_test.exists():
        sys.exit(f"missing: {cache_test}")
    cache = torch.load(cache_test, map_location="cpu", weights_only=False)
    images = cache["images"]
    labels = cache["labels"]
    class_names = cache.get("class_names", [])
    num_classes = int(labels.max().item()) + 1

    # 自动从 ckpt 同目录 training_log.json 还原 aafnet 配置
    log_path = Path(args.ckpt).parent / "training_log.json"
    overrides = {
        "model": {"name": args.model},
        "data": {"dataset": args.dataset, "img_height": H, "img_width": W},
    }
    if log_path.exists():
        log = json.load(open(log_path))
        snap = log.get("config_snapshot", {})
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
    instance._to_rgb = (args.model != "custom_mlp")
    model = instance.build_model()
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()

    print(f"\n=== Interpretability ===")
    print(f"  model: {args.model}  N_test: {len(labels)}")

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "outputs" / args.output_subdir / run_id / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 抽取特征
    print("\n[1/4] extracting features ...")
    feats, labs = extract_features(model, images, labels, device,
                                      batch_size=64, feature_layer_idx=0)
    print(f"  features: {feats.shape}")

    # 2) t-SNE
    print("\n[2/4] t-SNE projection ...")
    tsne = reduce_tsne(feats)
    render_scatter(tsne, labs, f"t-SNE — {args.model}",
                    out_dir / "tsne.png", class_names=class_names)

    # 3) UMAP (可选)
    print("\n[3/4] UMAP projection ...")
    umap_coords = reduce_umap(feats)
    if umap_coords is not None:
        render_scatter(umap_coords, labs, f"UMAP — {args.model}",
                        out_dir / "umap.png", class_names=class_names)
    else:
        print("  (umap-learn not installed, skip)")

    # 4) 错误分析
    print("\n[4/4] error analysis ...")
    # 跑混淆矩阵
    from sklearn.metrics import confusion_matrix
    import torch.nn.functional as F
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    preds = []
    with torch.no_grad():
        for i in range(0, len(images), 64):
            x = images[i:i+64].float().to(device) / 255.0
            x = (x - mean) / std
            out = model(x)
            if isinstance(out, tuple):
                out = out[0]
            preds.append(out.argmax(1).cpu().numpy())
    preds = np.concatenate(preds)
    cm = confusion_matrix(labs, preds, labels=list(range(num_classes)))
    top_pairs = confusion_pair_topk(cm, k=8)

    failures = lowest_confidence_examples(model, images, labels, device,
                                             per_class=args.per_class_failures)
    sim_matrix = class_mean_feature_similarity(feats, labs)

    # 类间相似度热图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.figure(figsize=(8, 6))
        sns.heatmap(sim_matrix, annot=True, fmt=".2f", cmap="viridis",
                     xticklabels=class_names if class_names else range(num_classes),
                     yticklabels=class_names if class_names else range(num_classes))
        plt.title(f"Class-mean Feature Cosine Similarity — {args.model}")
        plt.tight_layout()
        plt.savefig(out_dir / "class_similarity.png", dpi=300)
        plt.close()
    except Exception as e:
        print(f"  warn: similarity plot failed: {e}")

    # ── 保存 ──
    payload = {
        "run_id": run_id,
        "model": args.model,
        "dataset": args.dataset,
        "ckpt": args.ckpt,
        "n_test": int(len(labels)),
        "confusion_matrix": cm.tolist(),
        "top_confused_pairs": [
            {"true": int(t), "pred": int(p), "count": int(c),
              "true_name": class_names[t] if class_names and t < len(class_names) else f"C{t}",
              "pred_name": class_names[p] if class_names and p < len(class_names) else f"C{p}"}
            for t, p, c in top_pairs
        ],
        "low_confidence_examples": failures,
        "class_mean_similarity": sim_matrix.tolist(),
    }
    with open(out_dir / "interpret.json", "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\n=== Done ===")
    print(f"  saved: {out_dir}")


if __name__ == "__main__":
    main()
