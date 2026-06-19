"""
P2 #39 — Calibration metrics (ECE, NLL, Brier) on AL6.

Reports for baseline + AAFNet (full) ckpts under:
  - clean test
  - σ = 0.05 Gaussian noise
  - σ = 0.10 Gaussian noise
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.backbones import get_backbone
from src.utils.config import load_config


def latest(pat: str) -> Path | None:
    matches = sorted(ROOT.glob(pat))
    return matches[-1] if matches else None


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Compute ECE: average |confidence − accuracy| across n_bins equal-width bins."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies  = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i+1]
        in_bin = (confidences > lo) & (confidences <= hi)
        if in_bin.sum() > 0:
            avg_conf = confidences[in_bin].mean()
            avg_acc  = accuracies[in_bin].mean()
            ece += (in_bin.sum() / len(probs)) * abs(avg_conf - avg_acc)
    return float(ece)


def negative_log_likelihood(probs: np.ndarray, labels: np.ndarray) -> float:
    """Compute negative log-likelihood (mean over samples)."""
    eps = 1e-12
    p_true = probs[np.arange(len(labels)), labels]
    return float(-np.log(p_true + eps).mean())


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Brier score: mean squared error between probabilities and one-hot labels."""
    K = probs.shape[1]
    one_hot = np.eye(K)[labels]
    return float(((probs - one_hot) ** 2).sum(axis=1).mean())


def add_gaussian_noise(images: torch.Tensor, sigma: float) -> torch.Tensor:
    """Add gaussian noise to uint8 images. sigma in [0,1] domain (after div 255)."""
    if sigma <= 0:
        return images
    f = images.float() / 255.0
    noise = torch.randn_like(f) * sigma
    f = (f + noise).clamp(0, 1)
    return (f * 255).byte()


def evaluate_model(model: nn.Module, images: torch.Tensor, labels: torch.Tensor,
                   device: torch.device, batch_size: int = 64) -> dict:
    mean = torch.tensor([0.485,0.456,0.406], device=device).view(1,3,1,1)
    std  = torch.tensor([0.229,0.224,0.225], device=device).view(1,3,1,1)

    all_probs = []
    all_labels = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size].float().to(device) / 255.0
            batch = (batch - mean) / std
            out = model(batch)
            if isinstance(out, (tuple, list)):
                out = out[0]
            probs = F.softmax(out, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels[i:i+batch_size].numpy())
    probs = np.concatenate(all_probs)
    labs  = np.concatenate(all_labels)
    preds = probs.argmax(axis=1)
    return {
        "accuracy": float((preds == labs).mean()),
        "ece":      expected_calibration_error(probs, labs, n_bins=15),
        "nll":      negative_log_likelihood(probs, labs),
        "brier":    brier_score(probs, labs),
    }


def build_model(model_name: str, ckpt_path: Path, dataset: str, num_classes: int):
    log_path = ckpt_path.parent / "training_log.json"
    overrides = {
        "model": {"name": model_name},
        "data":  {"dataset": dataset, "img_height": 224, "img_width": 224},
    }
    if log_path.exists():
        snap = json.loads(log_path.read_text()).get("config_snapshot", {})
        if "aafnet" in snap:
            overrides["aafnet"] = snap["aafnet"]
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
    return model.to(instance.device).eval(), instance.device


PAIRS = [
    ("baseline", "outputs/ddp_baseline/*/resnet50/best_resnet50.pth"),
    ("aafnet",   "outputs/ddp_aafnet_v2/*/resnet50/best_resnet50.pth"),
]
CONDITIONS = [
    ("clean",       0.0),
    ("noise_0_05",  0.05),
    ("noise_0_10",  0.10),
]


def main():
    cache = ROOT / "data" / "cache" / "AL6_224x224_rgb_test.pt"
    data = torch.load(cache, map_location="cpu", weights_only=False)
    images = data["images"]
    labels = data["labels"]
    num_classes = int(labels.max().item()) + 1

    results = {}
    torch.manual_seed(42)
    for label, glob_pat in PAIRS:
        ckpt = latest(glob_pat)
        if ckpt is None:
            print(f"[skip] {label}: no ckpt for {glob_pat}")
            continue
        print(f"=== {label} ===  ckpt={ckpt}")
        model, device = build_model("resnet50", ckpt, "AL6", num_classes)
        results[label] = {}
        for cond_name, sigma in CONDITIONS:
            torch.manual_seed(42 + int(sigma * 100))
            corrupted = add_gaussian_noise(images, sigma) if sigma > 0 else images
            metrics = evaluate_model(model, corrupted, labels, device)
            print(f"  [{cond_name}] acc={metrics['accuracy']:.4f}  ECE={metrics['ece']:.4f}  NLL={metrics['nll']:.4f}  Brier={metrics['brier']:.4f}")
            results[label][cond_name] = metrics
        del model
        torch.cuda.empty_cache()

    out = ROOT / "outputs" / "p2_calibration.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out}")

    md = ["# P2 #39 — Calibration metrics on AL6\n",
          "Computed on the AL6 test set under clean and Gaussian-noise conditions.",
          "ECE = expected calibration error (15 bins); NLL = mean negative log-likelihood;",
          "Brier = squared error between softmax probabilities and one-hot labels.\n",
          "| Model | Condition | Acc | ECE ↓ | NLL ↓ | Brier ↓ |",
          "|---|---|---|---|---|---|"]
    for label in results:
        for cond, m in results[label].items():
            md.append(f"| {label} | {cond} | {m['accuracy']*100:.2f} % | {m['ece']:.4f} | {m['nll']:.4f} | {m['brier']:.4f} |")
    md_path = ROOT / "outputs" / "p2_calibration.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"Wrote {md_path}")
    print("\n" + "\n".join(md))


if __name__ == "__main__":
    main()
