"""
P2 #43 — Calibration v2: temperature scaling + reliability diagram + extended metrics.

For each (model, ckpt):
  1. Get logits on val split + test split + test under perturbation.
  2. Fit temperature T on the val split (minimize NLL).
  3. Apply T to test/test+noise softmax outputs.
  4. Report (acc, ECE, NLL, Brier) before AND after T-scaling.
  5. Draw reliability diagram (15-bin) for clean and noise.

Outputs:
  outputs/p2_calibration_v2.json
  outputs/p2_calibration_v2.md
  outputs/figures/F_calibration_reliability.png
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


def add_gaussian_noise(images: torch.Tensor, sigma: float, seed: int = 0) -> torch.Tensor:
    if sigma <= 0:
        return images
    g = torch.Generator().manual_seed(seed)
    f = images.float() / 255.0
    noise = torch.randn(f.shape, generator=g) * sigma
    f = (f + noise).clamp(0, 1)
    return (f * 255).byte()


def get_logits(model: nn.Module, images: torch.Tensor, device, batch_size=64) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    out_all = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size].float().to(device) / 255.0
            batch = (batch - mean) / std
            o = model(batch)
            if isinstance(o, (tuple, list)):
                o = o[0]
            out_all.append(o.cpu().numpy())
    return np.concatenate(out_all, axis=0)


def fit_temperature(logits_val: np.ndarray, labels_val: np.ndarray) -> float:
    """LBFGS on a single scalar T to minimize NLL on val split."""
    logits = torch.from_numpy(logits_val).float()
    labels = torch.from_numpy(labels_val).long()
    T = torch.nn.Parameter(torch.ones(1) * 1.5)
    nll = nn.CrossEntropyLoss()
    opt = torch.optim.LBFGS([T], lr=0.01, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = nll(logits / T, labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(T.item())


def metrics_from_logits(logits: np.ndarray, labels: np.ndarray, T: float = 1.0,
                       n_bins: int = 15) -> dict:
    """Return acc, ECE, NLL, Brier with optional temperature."""
    z = logits / T
    # softmax via numpy
    z_max = z.max(axis=1, keepdims=True)
    e = np.exp(z - z_max)
    probs = e / e.sum(axis=1, keepdims=True)

    preds = probs.argmax(axis=1)
    acc = float((preds == labels).mean())

    confidences = probs.max(axis=1)
    accuracies = (preds == labels).astype(float)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_data = []
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi)
        n_in = int(in_bin.sum())
        if n_in:
            avg_conf = float(confidences[in_bin].mean())
            avg_acc = float(accuracies[in_bin].mean())
            ece += (n_in / len(probs)) * abs(avg_conf - avg_acc)
            bin_data.append({"bin": i, "n": n_in, "avg_conf": avg_conf, "avg_acc": avg_acc})
        else:
            bin_data.append({"bin": i, "n": 0, "avg_conf": (lo + hi) / 2, "avg_acc": 0.0})

    eps = 1e-12
    nll = float(-np.log(probs[np.arange(len(labels)), labels] + eps).mean())
    K = probs.shape[1]
    one_hot = np.eye(K)[labels]
    brier = float(((probs - one_hot) ** 2).sum(axis=1).mean())

    return {"accuracy": acc, "ece": ece, "nll": nll, "brier": brier, "bin_data": bin_data}


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


def reliability_diagram(bin_data_pre: list, bin_data_post: list,
                        title: str, save_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, bins, sub in zip(axes, [bin_data_pre, bin_data_post],
                              ["before T-scaling", "after T-scaling"]):
        confs = [b["avg_conf"] for b in bins]
        accs  = [b["avg_acc"]  for b in bins]
        ns    = [b["n"]        for b in bins]
        # Use bar chart of acc, with confidence diagonal
        x = np.linspace(0.05, 0.95, len(bins))  # bin centers
        ax.bar(x, accs, width=0.06, alpha=0.7, edgecolor='black', label='accuracy')
        ax.plot([0,1], [0,1], 'k--', lw=1, label='perfect calibration')
        ax.scatter(confs, accs, c='red', s=12, label='confidence')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Accuracy')
        ax.set_title(sub)
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


PAIRS = [
    ("baseline", "outputs/ddp_baseline/*/resnet50/best_resnet50.pth"),
    ("aafnet",   "outputs/ddp_aafnet_v2/*/resnet50/best_resnet50.pth"),
]


def main():
    cache_train = ROOT / "data" / "cache" / "AL6_224x224_rgb_train.pt"
    cache_test  = ROOT / "data" / "cache" / "AL6_224x224_rgb_test.pt"
    train_data = torch.load(cache_train, map_location="cpu", weights_only=False)
    test_data  = torch.load(cache_test,  map_location="cpu", weights_only=False)

    # 用 train 的最后 20% 当 val (固定 seed 划分以保证可复现)
    rng = np.random.default_rng(42)
    n_train = len(train_data["labels"])
    perm = rng.permutation(n_train)
    n_val = n_train // 5  # 20%
    val_idx = perm[:n_val]
    val_images = train_data["images"][val_idx]
    val_labels = train_data["labels"][val_idx].numpy()

    test_images = test_data["images"]
    test_labels = test_data["labels"].numpy()
    num_classes = int(test_labels.max() + 1)

    fig_dir = ROOT / "paper" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for label, glob_pat in PAIRS:
        ckpt = latest(glob_pat)
        if ckpt is None:
            print(f"[skip] {label}: no ckpt"); continue
        print(f"=== {label} === ckpt={ckpt}")
        model, device = build_model("resnet50", ckpt, "AL6", num_classes)

        # 收集 val logits 拟合 T
        logits_val = get_logits(model, val_images, device)
        T = fit_temperature(logits_val, val_labels)
        print(f"  fitted T = {T:.4f}")

        # 在三种 condition 下评估
        results[label] = {"T": T, "conditions": {}}
        for cond_name, sigma in [("clean", 0.0), ("noise_0_05", 0.05), ("noise_0_10", 0.10)]:
            corrupt = add_gaussian_noise(test_images, sigma, seed=42 + int(sigma * 100))
            logits = get_logits(model, corrupt, device)
            m_pre  = metrics_from_logits(logits, test_labels, T=1.0)
            m_post = metrics_from_logits(logits, test_labels, T=T)
            results[label]["conditions"][cond_name] = {
                "T_scaled": False,
                "pre":  {k: v for k, v in m_pre.items() if k != "bin_data"},
                "post": {k: v for k, v in m_post.items() if k != "bin_data"},
                "bins_pre":  m_pre["bin_data"],
                "bins_post": m_post["bin_data"],
            }
            print(f"  [{cond_name}] acc={m_pre['accuracy']:.4f}  ECE pre={m_pre['ece']:.4f} post={m_post['ece']:.4f}  NLL pre={m_pre['nll']:.4f} post={m_post['nll']:.4f}")

        # 画 clean reliability diagram
        rel_path = fig_dir / f"F_calibration_{label}.png"
        reliability_diagram(
            results[label]["conditions"]["clean"]["bins_pre"],
            results[label]["conditions"]["clean"]["bins_post"],
            f"{label} on AL6 clean test (T = {T:.3f})",
            rel_path)
        print(f"  reliability diagram -> {rel_path}")

        del model
        torch.cuda.empty_cache()

    out = ROOT / "outputs" / "p2_calibration_v2.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out}")

    md = ["# P2 #43 — Calibration with temperature scaling on AL6\n",
          "Temperature `T` is fitted on a 20%-held-out slice of the training",
          "split (seed=42 random partition) by minimizing NLL with LBFGS.",
          "Pre = raw softmax. Post = softmax(logits / T).\n",
          "| Model | Condition | Acc | T | ECE pre | ECE post | NLL pre | NLL post | Brier pre | Brier post |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for label, r in results.items():
        T = r["T"]
        for cond, c in r["conditions"].items():
            pre = c["pre"]; post = c["post"]
            md.append(f"| {label} | {cond} | {pre['accuracy']*100:.2f} % | {T:.3f} | {pre['ece']:.4f} | **{post['ece']:.4f}** | {pre['nll']:.4f} | **{post['nll']:.4f}** | {pre['brier']:.4f} | **{post['brier']:.4f}** |")
    md_path = ROOT / "outputs" / "p2_calibration_v2.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"Wrote {md_path}")
    print("\n" + "\n".join(md))


if __name__ == "__main__":
    main()
