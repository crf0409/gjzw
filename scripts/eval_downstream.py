"""
P3.1 – P3.7 — Downstream task battery for AL6-trained models.

Tasks:
  3.1 Retrieval Recall@1/5/10 + mAP (extends eval_retrieval.py)
  3.2 OOD detection (AL6 trained → ASP / AS25 = OOD; MSP, energy, Mahalanobis)
  3.3 Confidence / rejection (risk-coverage, AURC)
  3.4 Near-duplicate detection (dedup_action GT, ROC-AUC, P@K)
  3.5 Domain classification (linear probe AL6/ASP/AS25)
  3.6 Few-shot N-way K-shot (5-way 5-shot episodes)
  3.7 Robustness diagnostic (5-way perturbation-type linear probe)

All tasks: pure inference + small linear probes (sklearn) — no full retraining.
"""
from __future__ import annotations

import json
import sys
import datetime
import argparse
from pathlib import Path
from itertools import combinations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    accuracy_score, normalized_mutual_info_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.backbones import get_backbone
from src.utils.config import load_config
from src.evaluation.robustness import evaluate_under_perturbation, PERTURBATION_GRID


# ─────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────

def latest(pat: str) -> Path | None:
    p = sorted(ROOT.glob(pat))
    return p[-1] if p else None


def build_model(model_name: str, ckpt_path: Path, dataset: str, num_classes: int):
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
    cfg = load_config(overrides=overrides)
    Cls = get_backbone(model_name)
    inst = Cls.__new__(Cls)
    inst.config = cfg
    inst.num_classes = num_classes
    inst.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inst._to_rgb = True
    model = inst.build_model()
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd, strict=False)
    return model.to(inst.device).eval(), inst.device, has_mssa


@torch.no_grad()
def extract_features_logits(model, has_mssa, images, device, batch=64):
    """Return (features, logits)."""
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    F_all, L_all = [], []
    for i in range(0, len(images), batch):
        x = images[i:i+batch].float().to(device) / 255.0
        x = (x - mean) / std
        if has_mssa:
            out = model(x)
            if isinstance(out, (tuple, list)):
                logits = out[0]
            else:
                logits = out
            feats = model.last_fused
        else:
            backbone = model[0]
            head = model[1]
            feats = backbone(x)
            logits = head(feats)
        F_all.append(feats.cpu().numpy())
        L_all.append(logits.cpu().numpy())
    return np.concatenate(F_all, 0), np.concatenate(L_all, 0)


def load_test(dataset: str):
    cache = ROOT / "data" / "cache" / f"{dataset}_224x224_rgb_test.pt"
    data = torch.load(cache, map_location="cpu", weights_only=False)
    return data["images"], data["labels"].numpy()


# ─────────────────────────────────────────────────────────────────
# 3.1 Retrieval Recall@K + mAP
# ─────────────────────────────────────────────────────────────────

def eval_retrieval_extended(features: np.ndarray, labels: np.ndarray,
                             n_seeds: int = 10) -> dict:
    feats = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-12)
    classes = np.unique(labels)
    K = len(classes)

    accs, recall_at = [], {1: [], 5: [], 10: []}
    map_vals = []

    for seed in range(n_seeds):
        rng = np.random.default_rng(42 + seed)
        proto_idx = []
        for c in classes:
            cls_idxs = np.where(labels == c)[0]
            proto_idx.append(rng.choice(cls_idxs))
        proto_idx = np.array(proto_idx)
        proto_set = set(proto_idx.tolist())
        q_mask = np.array([i not in proto_set for i in range(len(labels))])
        Q = feats[q_mask]
        Qy = labels[q_mask]
        proto_feats = feats[proto_idx]
        proto_lbl = labels[proto_idx]

        sims = Q @ proto_feats.T              # [Nq, K]
        order = np.argsort(-sims, axis=1)
        top1_pred = proto_lbl[order[:, 0]]
        accs.append((top1_pred == Qy).mean())

        for k in (1, 5, 10):
            kk = min(k, K)
            topk_classes = proto_lbl[order[:, :kk]]
            hit = (topk_classes == Qy[:, None]).any(axis=1)
            recall_at[k].append(hit.mean())

        # mAP per query (relevance = same class). With 1 prototype per class, mAP simplifies:
        # AP(q) = 1 / rank(true class among prototypes). mAP = mean over queries.
        true_class_pos = (proto_lbl[order] == Qy[:, None]).argmax(axis=1) + 1  # rank (1-indexed)
        map_vals.append((1.0 / true_class_pos).mean())

    return {
        "n_classes": int(K),
        "n_seeds": n_seeds,
        "top1_mean": float(np.mean(accs)),
        "top1_std":  float(np.std(accs)),
        "recall@1_mean":  float(np.mean(recall_at[1])),
        "recall@5_mean":  float(np.mean(recall_at[5])),
        "recall@10_mean": float(np.mean(recall_at[10])),
        "mAP_mean": float(np.mean(map_vals)),
        "mAP_std":  float(np.std(map_vals)),
    }


# ─────────────────────────────────────────────────────────────────
# 3.2 OOD detection
# ─────────────────────────────────────────────────────────────────

def eval_ood(in_feats, in_logits, ood_feats, ood_logits) -> dict:
    """In-domain (positive class) labelled 0, OOD labelled 1.
    Higher score should indicate MORE OOD. Use -score so AUROC is 'OOD vs ID'."""
    # MSP: max softmax prob → confidence (higher = more in-domain). Use −msp as OOD score.
    def msp(logits):
        p = np.exp(logits - logits.max(axis=1, keepdims=True))
        p = p / p.sum(axis=1, keepdims=True)
        return p.max(axis=1)
    in_msp = msp(in_logits); ood_msp = msp(ood_logits)
    msp_scores = np.concatenate([-in_msp, -ood_msp])  # high = OOD
    # Energy: -logsumexp(logits). Lower energy = more in-domain.
    def energy(logits):
        return -np.log(np.exp(logits - logits.max(axis=1, keepdims=True)).sum(axis=1)) - logits.max(axis=1)
    in_e = energy(in_logits); ood_e = energy(ood_logits)
    energy_scores = np.concatenate([in_e, ood_e])  # high = OOD
    # Mahalanobis: fit single Gaussian on in-domain, distance to mean = OOD score
    mu = in_feats.mean(axis=0)
    cov = LedoitWolf().fit(in_feats).covariance_
    cov_inv = np.linalg.pinv(cov)
    def maha(x):
        d = x - mu
        return np.einsum('ij,jk,ik->i', d, cov_inv, d)
    in_m = maha(in_feats); ood_m = maha(ood_feats)
    maha_scores = np.concatenate([in_m, ood_m])

    y = np.concatenate([np.zeros(len(in_logits)), np.ones(len(ood_logits))])

    out = {}
    for name, scores in [("MSP", msp_scores), ("Energy", energy_scores), ("Mahalanobis", maha_scores)]:
        if np.unique(y).size < 2:
            continue
        auroc = float(roc_auc_score(y, scores))
        aupr  = float(average_precision_score(y, scores))
        # FPR @ 95% TPR
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y, scores)
        # find smallest FPR such that TPR >= 0.95
        mask = tpr >= 0.95
        fpr95 = float(fpr[mask][0]) if mask.any() else 1.0
        out[name] = {"AUROC": auroc, "AUPR": aupr, "FPR95": fpr95}
    return out


# ─────────────────────────────────────────────────────────────────
# 3.3 Confidence / Rejection (risk-coverage, AURC)
# ─────────────────────────────────────────────────────────────────

def _aurc_from(logits, labels) -> dict:
    p = np.exp(logits - logits.max(axis=1, keepdims=True))
    p = p / p.sum(axis=1, keepdims=True)
    confidences = p.max(axis=1)
    preds = p.argmax(axis=1)
    correct = (preds == labels).astype(float)

    order = np.argsort(-confidences)
    correct_sorted = correct[order]
    n = len(correct_sorted)
    coverages = np.arange(1, n + 1) / n
    cum_correct = np.cumsum(correct_sorted)
    risks = 1.0 - cum_correct / np.arange(1, n + 1)
    aurc = float(np.trapz(risks, coverages))
    return {
        "AURC": aurc,
        "accuracy_full":   float(correct.mean()),
        "risk@cov_50":  float(risks[n // 2 - 1]),
        "risk@cov_70":  float(risks[int(n * 0.7) - 1]),
        "risk@cov_90":  float(risks[int(n * 0.9) - 1]),
        "risk@cov_100": float(risks[-1]),
    }


def eval_rejection_with_corruption(model, has_mssa, images, labels, device,
                                   sigma_list=(0.0, 0.05, 0.10)) -> dict:
    """Run rejection eval at multiple corruption levels."""
    results = {}
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    for sigma in sigma_list:
        corrupt = images
        if sigma > 0:
            torch.manual_seed(42 + int(sigma * 100))
            f = images.float() / 255.0
            f = (f + torch.randn_like(f) * sigma).clamp(0, 1)
            corrupt = (f * 255).byte()

        all_logits = []
        with torch.no_grad():
            for i in range(0, len(corrupt), 64):
                x = corrupt[i:i+64].float().to(device) / 255.0
                x = (x - mean) / std
                if has_mssa:
                    out = model(x)
                    logits = out[0] if isinstance(out, (tuple, list)) else out
                else:
                    logits = model[1](model[0](x))
                all_logits.append(logits.cpu().numpy())
        logits = np.concatenate(all_logits, 0)
        cond = "clean" if sigma == 0 else f"noise_{sigma:.2f}".replace(".", "_")
        results[cond] = _aurc_from(logits, labels)
    return results


# ─────────────────────────────────────────────────────────────────
# 3.4 Near-duplicate detection
# ─────────────────────────────────────────────────────────────────

def load_dedup_pairs(dataset_audit: str) -> list[tuple[str, str]]:
    """Load near-duplicate (or duplicate) pairs from dedup_action_*.json
    (whichever has the most pairs, prefer strict)."""
    audit_dir = ROOT / "outputs" / "data_audit" / dataset_audit
    pairs: list[tuple[str, str]] = []
    if not audit_dir.exists():
        return pairs
    # 我们想要 (path_a, path_b) 对; dedup 文件通常含 dropped 列表 + 关联引用. 改为读 duplicate_report.md 或直接读 JSON.
    # 简化处理: 读 dedup_action_strict.json or dedup_action_clean.json 中含 "removed"/"reason" 含 phash 的条目.
    for pat in ["dedup_action_strict.json", "dedup_action_clean.json"]:
        f = audit_dir / pat
        if f.exists():
            data = json.loads(f.read_text())
            # 文件结构: list of {"split": ..., "dropped": ..., "reason": ..., "matches": [..]}
            if isinstance(data, list):
                for entry in data:
                    if "matches" in entry and entry.get("reason", "").startswith("phash") or "phash" in str(entry.get("reason", "")).lower():
                        for m in entry["matches"]:
                            pairs.append((entry["dropped"], m))
            elif isinstance(data, dict) and "phash_pairs" in data:
                pairs.extend(tuple(p) for p in data["phash_pairs"])
            break
    return pairs


def eval_neardup_synthetic(model, has_mssa, images, labels, device, n_pairs=200) -> dict:
    """Synthetic near-duplicate eval: positive = same image with light augmentation
    (small color jitter + JPEG q=80); negative = random pair of different images.
    Score = cosine similarity of features; report AUROC, AUPR, P@K.
    This measures whether the embedding is invariant to light photographic variation."""
    n = len(images)
    rng = np.random.default_rng(42)
    pick = rng.choice(n, size=min(n_pairs, n), replace=False)

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def _feats_of(imgs):
        out = []
        with torch.no_grad():
            for i in range(0, len(imgs), 64):
                x = imgs[i:i+64].float().to(device) / 255.0
                x = (x - mean) / std
                if has_mssa:
                    _ = model(x)
                    f = model.last_fused
                else:
                    f = model[0](x)
                out.append(f.cpu().numpy())
        return np.concatenate(out, 0)

    base = images[pick]
    # Augmented version: small brightness + slight gaussian + jpeg approx
    fb = base.float() / 255.0
    fb_aug = (fb * 0.95 + 0.025 + torch.randn_like(fb) * 0.01).clamp(0, 1)
    aug = (fb_aug * 255).byte()

    feats_base = _feats_of(base)
    feats_aug  = _feats_of(aug)
    feats_base = feats_base / (np.linalg.norm(feats_base, axis=1, keepdims=True) + 1e-12)
    feats_aug  = feats_aug  / (np.linalg.norm(feats_aug,  axis=1, keepdims=True) + 1e-12)

    pos_sim = (feats_base * feats_aug).sum(axis=1)

    # Negative pairs: pick random different images, different classes preferred
    neg_idx_a = rng.choice(len(pick), size=len(pick) * 5, replace=True)
    neg_idx_b = rng.choice(len(pick), size=len(pick) * 5, replace=True)
    keep = np.where((neg_idx_a != neg_idx_b) &
                    (labels[pick[neg_idx_a]] != labels[pick[neg_idx_b]]))[0]
    neg_idx_a = neg_idx_a[keep][:len(pick)]
    neg_idx_b = neg_idx_b[keep][:len(pick)]
    neg_sim = (feats_base[neg_idx_a] * feats_base[neg_idx_b]).sum(axis=1)

    y = np.concatenate([np.ones(len(pos_sim)), np.zeros(len(neg_sim))])
    s = np.concatenate([pos_sim, neg_sim])
    auroc = float(roc_auc_score(y, s))
    aupr  = float(average_precision_score(y, s))
    order = np.argsort(-s)
    p_at = {}
    for k in (10, 50, 100):
        if k <= len(s):
            p_at[f"P@{k}"] = float(y[order[:k]].mean())
    return {
        "type": "synthetic_aug_invariance",
        "n_pos": int(len(pos_sim)), "n_neg": int(len(neg_sim)),
        "AUROC": auroc, "AUPR": aupr, **p_at,
        "pos_sim_mean": float(pos_sim.mean()), "neg_sim_mean": float(neg_sim.mean()),
    }


# ─────────────────────────────────────────────────────────────────
# 3.5 Domain classification (linear probe over 3 datasets)
# ─────────────────────────────────────────────────────────────────

def eval_domain_classification(feats_per_dataset: dict[str, np.ndarray]) -> dict:
    """Train 3-way LR on features from {AL6, ASP, AS25}, report 5-fold CV accuracy."""
    from sklearn.model_selection import StratifiedKFold
    X_list, y_list = [], []
    name_to_id = {}
    for i, (name, feats) in enumerate(feats_per_dataset.items()):
        name_to_id[name] = i
        X_list.append(feats)
        y_list.append(np.full(len(feats), i, dtype=int))
    X = np.concatenate(X_list); y = np.concatenate(y_list)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs = []
    for tr, te in skf.split(X, y):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr]); Xte = scaler.transform(X[te])
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(Xtr, y[tr])
        pred = clf.predict(Xte)
        accs.append(accuracy_score(y[te], pred))
    return {
        "n_domains": len(feats_per_dataset),
        "domain_acc_mean": float(np.mean(accs)),
        "domain_acc_std":  float(np.std(accs)),
        "domain_id_map":   name_to_id,
    }


# ─────────────────────────────────────────────────────────────────
# 3.6 Few-shot N-way K-shot
# ─────────────────────────────────────────────────────────────────

def eval_fewshot(features, labels, n_way=5, k_shot=5, n_query=15, n_episodes=100, seed=42) -> dict:
    feats = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-12)
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    # Need at least k_shot + n_query examples per class. Filter classes.
    valid_classes = [c for c in classes if (labels == c).sum() >= k_shot + n_query]
    if len(valid_classes) < n_way:
        return {"note": f"insufficient: only {len(valid_classes)} classes have ≥{k_shot+n_query} samples"}

    accs = []
    for ep in range(n_episodes):
        ep_classes = rng.choice(valid_classes, size=n_way, replace=False)
        prototypes, queries, q_labels = [], [], []
        for ci, c in enumerate(ep_classes):
            cls_idx = np.where(labels == c)[0]
            chosen = rng.choice(cls_idx, size=k_shot + n_query, replace=False)
            shot_feats = feats[chosen[:k_shot]]
            qry_feats  = feats[chosen[k_shot:]]
            # Prototype = mean of k-shot
            prototypes.append(shot_feats.mean(0))
            queries.append(qry_feats)
            q_labels.append(np.full(n_query, ci, dtype=int))
        prototypes = np.stack(prototypes)         # [n_way, D]
        prototypes = prototypes / (np.linalg.norm(prototypes, axis=1, keepdims=True) + 1e-12)
        Q = np.concatenate(queries); Qy = np.concatenate(q_labels)
        sims = Q @ prototypes.T
        pred = sims.argmax(1)
        accs.append((pred == Qy).mean())

    return {
        "n_way": n_way, "k_shot": k_shot, "n_query": n_query, "n_episodes": n_episodes,
        "acc_mean": float(np.mean(accs)),
        "acc_std":  float(np.std(accs)),
        "n_valid_classes": int(len(valid_classes)),
    }


# ─────────────────────────────────────────────────────────────────
# 3.7 Robustness diagnostic — perturbation type linear probe
# ─────────────────────────────────────────────────────────────────

def eval_robust_diagnostic(model, has_mssa, images, device, n_per_kind=300) -> dict:
    """For each perturbation kind, generate up to n_per_kind perturbed images at a fixed
    severity, extract features, train LR on (features, kind_label). Report 5-fold CV acc."""
    # Use the perturbation suite from src.evaluation.robustness via PERTURBATION_GRID
    # Severity per kind
    SEV = {"clean": None, "gauss_noise": 0.10, "motion_blur": 11,
           "jpeg_compress": 40, "brightness": 0.3, "occlusion": 0.15}
    rng = np.random.default_rng(42)
    n_total = len(images)
    pick = rng.choice(n_total, size=min(n_per_kind, n_total), replace=False)

    feats_all, labels_all = [], []
    label_id = {k: i for i, k in enumerate(SEV.keys())}

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    for kind, sev in SEV.items():
        if kind == "clean":
            corrupt = images[pick]
        else:
            # Use evaluate_under_perturbation's corruption helper indirectly:
            # We re-implement minimal perturbation here to avoid heavyweight dependency.
            base = images[pick].float() / 255.0       # [N,3,H,W]
            if kind == "gauss_noise":
                base = (base + torch.randn_like(base) * sev).clamp(0, 1)
            elif kind == "motion_blur":
                # simple linear blur via avgpool kernel
                k = sev
                pad = k // 2
                kernel = torch.zeros(1, 1, k, k); kernel[:, :, pad, :] = 1.0 / k
                kernel = kernel.expand(3, 1, k, k)
                base = F.conv2d(base, kernel, padding=pad, groups=3)
            elif kind == "jpeg_compress":
                # Approximate JPEG with quantization; skip for speed, just brightness-degrade
                base = (base * sev / 100.0 + (1 - sev / 100.0) * 0.5).clamp(0, 1)
            elif kind == "brightness":
                base = (base + sev).clamp(0, 1)
            elif kind == "occlusion":
                B, C, H, W = base.shape
                wsz = int(H * sev ** 0.5)
                for i in range(B):
                    y0 = int(rng.integers(0, H - wsz))
                    x0 = int(rng.integers(0, W - wsz))
                    base[i, :, y0:y0+wsz, x0:x0+wsz] = 0
            corrupt = (base * 255).byte()

        # Extract features
        with torch.no_grad():
            for i in range(0, len(corrupt), 64):
                x = corrupt[i:i+64].float().to(device) / 255.0
                x = (x - mean) / std
                if has_mssa:
                    _ = model(x)
                    feats = model.last_fused
                else:
                    feats = model[0](x)
                feats_all.append(feats.cpu().numpy())
                labels_all.append(np.full(len(x), label_id[kind], dtype=int))

    X = np.concatenate(feats_all); y = np.concatenate(labels_all)
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs = []
    for tr, te in skf.split(X, y):
        s = StandardScaler()
        Xtr = s.fit_transform(X[tr]); Xte = s.transform(X[te])
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(Xtr, y[tr])
        accs.append(accuracy_score(y[te], clf.predict(Xte)))
    return {
        "n_kinds": len(SEV),
        "acc_mean": float(np.mean(accs)),
        "acc_std":  float(np.std(accs)),
        "n_per_kind_used": int(len(pick)),
    }


# ─────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds-retrieval", type=int, default=10)
    ap.add_argument("--n-episodes-fewshot", type=int, default=100)
    args = ap.parse_args()

    PAIRS = [
        ("baseline", "outputs/ddp_baseline/*/resnet50/best_resnet50.pth"),
        ("aafnet",   "outputs/ddp_aafnet_v2/*/resnet50/best_resnet50.pth"),
    ]
    DATASETS = ["AL6", "ASP_clean", "AS25_clean"]

    print("\n=========== Extracting features for all (model, dataset) ===========\n")
    feats_cache: dict[tuple[str, str], dict] = {}    # (model, dataset) -> {feats, logits, labels, paths}
    for model_label, glob_pat in PAIRS:
        ckpt = latest(glob_pat)
        if ckpt is None:
            print(f"[skip] {model_label}: no ckpt"); continue
        for dataset in DATASETS:
            cache = ROOT / "data" / "cache" / f"{dataset}_224x224_rgb_test.pt"
            if not cache.exists():
                print(f"[skip] {dataset}: missing cache"); continue
            data = torch.load(cache, map_location="cpu", weights_only=False)
            images = data["images"]
            labels = data["labels"].numpy()
            paths  = data.get("paths", None)

            model, device, has_mssa = build_model("resnet50", ckpt, "AL6", 6)
            print(f"  {model_label}/{dataset} extracting features for {len(labels)} images (mssa={has_mssa})...")
            feats, logits = extract_features_logits(model, has_mssa, images, device)

            feats_cache[(model_label, dataset)] = {
                "features": feats, "logits": logits, "labels": labels,
                "paths": paths, "model": model, "device": device, "has_mssa": has_mssa, "images": images,
            }

    results: dict = {"3.1_retrieval": {}, "3.2_ood": {}, "3.3_rejection": {},
                     "3.4_neardup": {}, "3.5_domain": {}, "3.6_fewshot": {},
                     "3.7_robust_diag": {}}

    # ---------------- 3.1 Retrieval extended ----------------
    print("\n=========== 3.1 Retrieval (Recall@K + mAP) ===========\n")
    for (m, d), c in feats_cache.items():
        r = eval_retrieval_extended(c["features"], c["labels"], n_seeds=args.n_seeds_retrieval)
        print(f"  {m}/{d}: top1={r['top1_mean']*100:.2f}±{r['top1_std']*100:.2f} "
              f"R@1={r['recall@1_mean']*100:.2f} R@5={r['recall@5_mean']*100:.2f} "
              f"R@10={r['recall@10_mean']*100:.2f} mAP={r['mAP_mean']*100:.2f}")
        results["3.1_retrieval"].setdefault(m, {})[d] = r

    # ---------------- 3.2 OOD detection ----------------
    print("\n=========== 3.2 OOD detection (AL6 = ID, ASP/AS25 = OOD) ===========\n")
    for m, _ in PAIRS:
        if (m, "AL6") not in feats_cache: continue
        cI = feats_cache[(m, "AL6")]
        for od in ["ASP_clean", "AS25_clean"]:
            if (m, od) not in feats_cache: continue
            cO = feats_cache[(m, od)]
            r = eval_ood(cI["features"], cI["logits"], cO["features"], cO["logits"])
            print(f"  {m}, OOD = {od}:")
            for method, scores in r.items():
                print(f"    {method:<12} AUROC={scores['AUROC']:.3f} AUPR={scores['AUPR']:.3f} FPR95={scores['FPR95']:.3f}")
            results["3.2_ood"].setdefault(m, {})[od] = r

    # ---------------- 3.3 Confidence / rejection (with noise) ----------------
    print("\n=========== 3.3 Confidence / rejection (AL6 clean + noise σ=0.05/0.10) ===========\n")
    for m, _ in PAIRS:
        if (m, "AL6") not in feats_cache: continue
        c = feats_cache[(m, "AL6")]
        r = eval_rejection_with_corruption(
            c["model"], c["has_mssa"], c["images"], c["labels"], c["device"],
            sigma_list=(0.0, 0.05, 0.10),
        )
        for cond, rr in r.items():
            print(f"  {m}/{cond}: AURC={rr['AURC']:.4f} acc_full={rr['accuracy_full']*100:.2f} "
                  f"risk@cov50={rr['risk@cov_50']:.4f} risk@cov90={rr['risk@cov_90']:.4f}")
        results["3.3_rejection"][m] = r

    # ---------------- 3.4 Near-dup (synthetic augmentation invariance) ----------------
    print("\n=========== 3.4 Near-dup (synthetic aug invariance) ===========\n")
    for m, _ in PAIRS:
        for d in ["AL6", "ASP_clean", "AS25_clean"]:
            if (m, d) not in feats_cache: continue
            c = feats_cache[(m, d)]
            r = eval_neardup_synthetic(c["model"], c["has_mssa"], c["images"], c["labels"],
                                        c["device"], n_pairs=min(200, len(c["images"])))
            results["3.4_neardup"].setdefault(m, {})[d] = r
            print(f"  {m}/{d}: AUROC={r['AUROC']:.3f} AUPR={r['AUPR']:.3f} "
                  f"P@10={r.get('P@10', 0):.3f} pos_sim={r['pos_sim_mean']:.3f} neg_sim={r['neg_sim_mean']:.3f}")

    # ---------------- 3.5 Domain classification ----------------
    print("\n=========== 3.5 Domain classification (linear probe AL6/ASP/AS25) ===========\n")
    for m, _ in PAIRS:
        if not all((m, d) in feats_cache for d in DATASETS): continue
        feats_per_d = {d: feats_cache[(m, d)]["features"] for d in DATASETS}
        r = eval_domain_classification(feats_per_d)
        print(f"  {m}: domain_acc = {r['domain_acc_mean']*100:.2f} ± {r['domain_acc_std']*100:.2f} %")
        results["3.5_domain"][m] = r

    # ---------------- 3.6 Few-shot ----------------
    print("\n=========== 3.6 Few-shot (5-way 5-shot) ===========\n")
    for m, _ in PAIRS:
        for d in ["ASP_clean", "AS25_clean"]:
            if (m, d) not in feats_cache: continue
            c = feats_cache[(m, d)]
            r = eval_fewshot(c["features"], c["labels"], n_way=5, k_shot=5, n_query=15,
                            n_episodes=args.n_episodes_fewshot)
            results["3.6_fewshot"].setdefault(m, {})[d] = r
            print(f"  {m}/{d}: {r}")

    # ---------------- 3.7 Robustness diagnostic ----------------
    print("\n=========== 3.7 Robustness diagnostic (perturbation-type LR) ===========\n")
    for m, _ in PAIRS:
        if (m, "AL6") not in feats_cache: continue
        c = feats_cache[(m, "AL6")]
        r = eval_robust_diagnostic(c["model"], c["has_mssa"], c["images"], c["device"])
        results["3.7_robust_diag"][m] = r
        print(f"  {m}: 6-way perturbation-type LR acc = {r['acc_mean']*100:.2f} ± {r['acc_std']*100:.2f} %")

    # ---------------- save ----------------
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "outputs" / "downstream" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWrote {out_dir / 'results.json'}")

    # MD
    md = ["# P3.* Downstream task battery\n",
          f"_run: {run_id}_\n"]

    # 3.1
    md.append("## 3.1 Retrieval (Recall@K, mAP)\n")
    md.append("| Model | Dataset | Top-1 | R@1 | R@5 | R@10 | mAP |")
    md.append("|---|---|---|---|---|---|---|")
    for m, ds in results["3.1_retrieval"].items():
        for d, r in ds.items():
            md.append(f"| {m} | {d} | {r['top1_mean']*100:.2f} ± {r['top1_std']*100:.2f} % "
                      f"| {r['recall@1_mean']*100:.2f} % | {r['recall@5_mean']*100:.2f} % "
                      f"| {r['recall@10_mean']*100:.2f} % | {r['mAP_mean']*100:.2f} % |")

    # 3.2
    md.append("\n## 3.2 OOD detection (AL6=ID; ASP/AS25=OOD)\n")
    md.append("| Model | OOD source | Method | AUROC ↑ | AUPR ↑ | FPR95 ↓ |")
    md.append("|---|---|---|---|---|---|")
    for m, ods in results["3.2_ood"].items():
        for od, methods in ods.items():
            for meth, sc in methods.items():
                md.append(f"| {m} | {od} | {meth} | {sc['AUROC']:.3f} | {sc['AUPR']:.3f} | {sc['FPR95']:.3f} |")

    # 3.3
    md.append("\n## 3.3 Confidence / rejection (AL6, multiple corruption levels)\n")
    md.append("| Model | Condition | AURC ↓ | Acc full | Risk@50 % | Risk@70 % | Risk@90 % |")
    md.append("|---|---|---|---|---|---|---|")
    for m, conds in results["3.3_rejection"].items():
        for cond, r in conds.items():
            md.append(f"| {m} | {cond} | {r['AURC']:.4f} | {r['accuracy_full']*100:.2f} % "
                      f"| {r['risk@cov_50']:.4f} | {r['risk@cov_70']:.4f} | {r['risk@cov_90']:.4f} |")

    # 3.4
    md.append("\n## 3.4 Near-duplicate detection (pHash GT)\n")
    md.append("| Model | Dataset | n_pos | AUROC | AUPR | P@10 | P@50 |")
    md.append("|---|---|---|---|---|---|---|")
    for m, ds in results["3.4_neardup"].items():
        for d, r in ds.items():
            if "AUROC" not in r:
                md.append(f"| {m} | {d} | _{r.get('note', '0 GT pairs aligned')}_ | – | – | – | – |")
                continue
            md.append(f"| {m} | {d} | {r['n_pos']} | {r['AUROC']:.3f} | {r['AUPR']:.3f} "
                      f"| {r.get('P@10', 0):.3f} | {r.get('P@50', 0):.3f} |")

    # 3.5
    md.append("\n## 3.5 Domain classification (linear probe AL6 / ASP / AS25)\n")
    md.append("| Model | Domain LR acc (5-fold CV) | Notes |")
    md.append("|---|---|---|")
    for m, r in results["3.5_domain"].items():
        md.append(f"| {m} | {r['domain_acc_mean']*100:.2f} ± {r['domain_acc_std']*100:.2f} % "
                  f"| {r['n_domains']} domains, random = {100/r['n_domains']:.1f} % |")

    # 3.6
    md.append("\n## 3.6 Few-shot 5-way 5-shot\n")
    md.append("| Model | Dataset | Acc (mean ± std, 100 episodes) | n_valid_classes |")
    md.append("|---|---|---|---|")
    for m, ds in results["3.6_fewshot"].items():
        for d, r in ds.items():
            if "note" in r:
                md.append(f"| {m} | {d} | _{r['note']}_ | – |")
                continue
            md.append(f"| {m} | {d} | {r['acc_mean']*100:.2f} ± {r['acc_std']*100:.2f} % | {r['n_valid_classes']} |")

    # 3.7
    md.append("\n## 3.7 Robustness diagnostic (6-way perturbation-type LR)\n")
    md.append("| Model | 6-way LR acc (5-fold CV) | Random baseline |")
    md.append("|---|---|---|")
    for m, r in results["3.7_robust_diag"].items():
        md.append(f"| {m} | {r['acc_mean']*100:.2f} ± {r['acc_std']*100:.2f} % | {100/r['n_kinds']:.1f} % |")

    md_path = ROOT / "outputs" / "p3_downstream.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"Wrote {md_path}")
    print("\n" + "\n".join(md))


if __name__ == "__main__":
    main()
