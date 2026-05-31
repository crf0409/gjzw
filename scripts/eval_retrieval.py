"""
Retrieval / 1-shot prototype classification evaluation.

For each (model_ckpt, dataset, seed):
  1. Extract penultimate-layer features for all test images (FC head removed).
     - For AAFNet: feature is the 512-D fused vector after MSSA + CSGF.
     - For baseline: feature is the 2048-D GAP-pooled vector before head.
  2. For each class, sample 1 image as the prototype.
  3. For remaining test images, compute cosine similarity to all prototypes
     and predict the class of the nearest prototype.
  4. Report top-1 accuracy.

Aggregated across multiple prototype-sampling seeds → mean ± std.
We use the *full class set* of each test dataset (AL6=6, ASP_clean=9, AS25_clean=25).

Output: outputs/retrieval/<run>/results.json + paper/SUMMARY-retrieval.md
"""
from __future__ import annotations
import json
import sys
import argparse
import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import normalized_mutual_info_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.backbones import get_backbone
from src.utils.config import load_config


def latest(pat: str) -> Path | None:
    matches = sorted(ROOT.glob(pat))
    return matches[-1] if matches else None


def build_model_for_ckpt(model_name: str, ckpt_path: Path, dataset: str, num_classes: int):
    """Build model + load ckpt. Return (model, device, has_mssa)."""
    log_path = ckpt_path.parent / "training_log.json"
    overrides = {
        "model": {"name": model_name},
        "data":  {"dataset": dataset, "img_height": 224, "img_width": 224},
    }
    has_mssa = False
    if log_path.exists():
        snap = json.loads(log_path.read_text()).get("config_snapshot", {})
        if "aafnet" in snap:
            overrides["aafnet"] = snap["aafnet"]
            has_mssa = bool(snap["aafnet"].get("msa", {}).get("enabled", False))

    config = load_config(overrides=overrides)
    Cls = get_backbone(model_name)
    instance = Cls.__new__(Cls)
    instance.config = config
    instance.num_classes = num_classes
    instance.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    instance._to_rgb = True
    model = instance.build_model()

    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd, strict=False)

    return model.to(instance.device).eval(), instance.device, has_mssa


@torch.no_grad()
def extract_features(model: nn.Module, has_mssa: bool, images: torch.Tensor,
                     device, batch_size: int = 64) -> np.ndarray:
    """Extract penultimate features. For MSSA models: return fused 512-D.
    For baseline models: return backbone[0] output (pre-head GAP feature)."""
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    feats_all = []

    if has_mssa:
        # model is the MSSABackbone wrapped (or wrapped in Sequential? let's check)
        # AAFNet build_model: returns instance.model = MSSABackbone (not Sequential)
        # So we can call model.feature_extractor + mssa pipeline manually,
        # or call model(x) and use cached self.last_fused.
        # Use the cached approach - cleanest.
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size].float().to(device) / 255.0
            batch = (batch - mean) / std
            _ = model(batch)  # forward populates model.last_fused
            feats = model.last_fused          # [B, fused_dim]
            feats_all.append(feats.cpu().numpy())
    else:
        # Sequential(backbone, ClassificationHead). model[0] = backbone, model[1] = head
        # We need output of model[0] which is the GAP-pooled features
        # But ClassificationHead does BN/Drop/Linear → 256 → Linear → num_classes
        # We want pre-head features = output of model[0]
        backbone = model[0]
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size].float().to(device) / 255.0
            batch = (batch - mean) / std
            feats = backbone(batch)                  # [B, 2048]
            feats_all.append(feats.cpu().numpy())

    return np.concatenate(feats_all, axis=0)


def eval_retrieval(features: np.ndarray, labels: np.ndarray,
                   n_seeds: int = 10) -> dict:
    """For each seed: 1 prototype per class, predict via nearest cosine sim.
    Returns mean ± std accuracy + recall@k + NMI on the prototype subset."""
    # L2 normalize features for cosine similarity
    feats = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-12)
    classes = np.unique(labels)
    K = len(classes)

    seed_results = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(42 + seed)
        # 1 prototype per class
        proto_idx = []
        for c in classes:
            cls_idxs = np.where(labels == c)[0]
            chosen = rng.choice(cls_idxs)
            proto_idx.append(chosen)
        proto_idx = np.array(proto_idx)
        proto_set = set(proto_idx.tolist())

        # Query = all non-prototype images
        query_mask = np.array([i not in proto_set for i in range(len(labels))])
        query_idx = np.where(query_mask)[0]
        Q = feats[query_idx]                       # [Nq, D]
        Qy = labels[query_idx]                     # [Nq]
        proto_feats = feats[proto_idx]             # [K, D]
        proto_labels = labels[proto_idx]           # [K]

        # Cosine similarity (since L2 normalized → just dot product)
        sims = Q @ proto_feats.T                   # [Nq, K]
        # Sort descending
        order = np.argsort(-sims, axis=1)          # [Nq, K]
        # Predicted top-k
        top1 = proto_labels[order[:, 0]]
        accs = (top1 == Qy).astype(float)

        # Recall@k for k in {1, 3}
        recalls = {}
        for k in [1, min(3, K)]:
            topk_classes = proto_labels[order[:, :k]]
            hit = (topk_classes == Qy[:, None]).any(axis=1)
            recalls[f"recall@{k}"] = float(hit.mean())

        # NMI: cluster query → top1 prediction.
        nmi = normalized_mutual_info_score(Qy, top1)

        seed_results.append({
            "seed": seed,
            "n_classes": int(K),
            "n_query": int(len(query_idx)),
            "top1_acc": float(accs.mean()),
            "nmi": float(nmi),
            **recalls,
        })

    accs = [r["top1_acc"] for r in seed_results]
    nmis = [r["nmi"] for r in seed_results]
    return {
        "n_classes": int(K),
        "n_seeds": n_seeds,
        "top1_mean": float(np.mean(accs)),
        "top1_std":  float(np.std(accs)),
        "nmi_mean":  float(np.mean(nmis)),
        "nmi_std":   float(np.std(nmis)),
        "recall@1_mean": float(np.mean([r["recall@1"] for r in seed_results])),
        "recall@3_mean": float(np.mean([r.get("recall@3", r["recall@1"]) for r in seed_results])),
        "per_seed": seed_results,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-seeds", type=int, default=10)
    p.add_argument("--output-subdir", default="retrieval")
    args = p.parse_args()

    PAIRS = [
        ("baseline", "outputs/ddp_baseline/*/resnet50/best_resnet50.pth"),
        ("aafnet",   "outputs/ddp_aafnet_v2/*/resnet50/best_resnet50.pth"),
    ]
    DATASETS = ["AL6", "ASP_clean", "AS25_clean"]

    # Cache features: avoid extracting twice
    feature_cache: dict[tuple[str, str], np.ndarray] = {}
    label_cache: dict[str, np.ndarray] = {}

    results: dict = {}
    for model_label, glob_pat in PAIRS:
        ckpt = latest(glob_pat)
        if ckpt is None:
            print(f"[skip] {model_label}: no ckpt for {glob_pat}")
            continue

        for dataset in DATASETS:
            cache = ROOT / "data" / "cache" / f"{dataset}_224x224_rgb_test.pt"
            if not cache.exists():
                print(f"[skip] {dataset}: missing cache")
                continue

            data = torch.load(cache, map_location="cpu", weights_only=False)
            images = data["images"]
            labels = data["labels"].numpy()

            # Build model with the dataset's num_classes for head sizing
            # (we don't use the head; just need the feature path to load)
            ckpt_num_classes = 6  # the ckpt was trained on AL6 with 6 classes
            model, device, has_mssa = build_model_for_ckpt(
                "resnet50", ckpt, "AL6", ckpt_num_classes
            )

            print(f"\n=== {model_label} | {dataset} (has_mssa={has_mssa}) ===")
            print(f"  ckpt: {ckpt}")
            print(f"  test set: {len(labels)} images, {len(np.unique(labels))} classes")

            # Extract features
            feats = extract_features(model, has_mssa, images, device)
            print(f"  feature shape: {feats.shape}")

            # Eval retrieval
            r = eval_retrieval(feats, labels, n_seeds=args.n_seeds)
            print(f"  top-1 acc: {r['top1_mean']*100:.2f} ± {r['top1_std']*100:.2f} %  "
                  f"(NMI {r['nmi_mean']:.3f}, recall@1 {r['recall@1_mean']*100:.2f}, "
                  f"recall@3 {r['recall@3_mean']*100:.2f})")

            results.setdefault(model_label, {})[dataset] = r
            label_cache[dataset] = labels
            feature_cache[(model_label, dataset)] = feats

            del model
            torch.cuda.empty_cache()

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "outputs" / args.output_subdir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_dir / 'results.json'}")

    # Markdown summary
    md = ["# P3 — Retrieval (1-shot prototype) summary\n",
          f"Per-class 1-shot prototype matching with cosine similarity, "
          f"averaged over **{args.n_seeds} prototype-sampling seeds**.",
          "Penultimate-layer features extracted from each model after removing",
          "the FC classification head (AAFNet: 512-D fused vector after MSSA+CSGF;",
          "baseline: 2048-D GAP-pooled feature).\n",
          "| Model | Dataset | n_classes | Top-1 acc (mean ± std) | Recall@3 | NMI |",
          "|---|---|---|---|---|---|"]
    for m in results:
        for d, r in results[m].items():
            md.append(f"| {m} | {d} | {r['n_classes']} | "
                      f"{r['top1_mean']*100:.2f} ± {r['top1_std']*100:.2f} % | "
                      f"{r['recall@3_mean']*100:.2f} % | "
                      f"{r['nmi_mean']:.3f} |")

    md.append("\n## Δ AAFNet − baseline (top-1)")
    md.append("| Dataset | Baseline | AAFNet | Δ |")
    md.append("|---|---|---|---|")
    for d in DATASETS:
        b = results.get("baseline", {}).get(d)
        a = results.get("aafnet", {}).get(d)
        if not (b and a):
            continue
        delta = (a["top1_mean"] - b["top1_mean"]) * 100
        md.append(f"| {d} | {b['top1_mean']*100:.2f} % | {a['top1_mean']*100:.2f} % | "
                  f"**{delta:+.2f} pp** |")

    md_path = out_dir / "summary.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"Wrote {md_path}")
    # Also publish to outputs/p3_retrieval.md for top-level access
    with open(ROOT / "outputs" / "p3_retrieval.md", "w") as f:
        f.write("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
