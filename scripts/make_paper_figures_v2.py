"""
Additional high-value figures: real GradCAM, dual t-SNE, confusion matrices.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
from src.models.backbones import get_backbone
from src.utils.config import load_config

plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

C_BASE = "#4a5468"; C_AAF = "#d04848"


def latest(pat: str) -> Path | None:
    p = sorted(ROOT.glob(pat))
    return p[-1] if p else None


def build_model(model_name, ckpt_path, dataset, num_classes):
    log_path = ckpt_path.parent / "training_log.json"
    overrides = {"model": {"name": model_name},
                 "data":  {"dataset": dataset, "img_height": 224, "img_width": 224}}
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


# =================================================================
# F-R. GradCAM activation comparison
# =================================================================
def f_gradcam():
    print("\n[F_R_gradcam]")
    cache = ROOT / "data" / "cache" / "AL6_224x224_rgb_test.pt"
    data = torch.load(cache, map_location="cpu", weights_only=False)
    images = data["images"]; labels = data["labels"].numpy()

    # Pick 1 image per class (first 6 classes)
    sample_idxs = []
    for c in np.unique(labels)[:6]:
        sample_idxs.append(int(np.where(labels == c)[0][0]))
    print(f"  sample image indices: {sample_idxs}")

    base_ckpt = latest("outputs/ddp_baseline/*/resnet50/best_resnet50.pth")
    aaf_ckpt  = latest("outputs/ddp_aafnet_v2/*/resnet50/best_resnet50.pth")

    base_model, device, _ = build_model("resnet50", base_ckpt, "AL6", 6)
    aaf_model, _, has_aaf = build_model("resnet50", aaf_ckpt, "AL6", 6)

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def gradcam(model, x_batch, target_class, target_layer):
        """Simple GradCAM. target_layer is the conv module to hook."""
        feats_holder = []
        grads_holder = []
        def fwd_hook(m, i, o): feats_holder.append(o)
        def bwd_hook(m, gi, go): grads_holder.append(go[0])
        h1 = target_layer.register_forward_hook(fwd_hook)
        h2 = target_layer.register_full_backward_hook(bwd_hook)
        try:
            model.zero_grad()
            x = x_batch.requires_grad_(False)
            out = model(x)
            if isinstance(out, (tuple, list)):
                out = out[0]
            score = out[:, target_class].sum()
            score.backward()
            feats = feats_holder[0]      # [B, C, H, W]
            grads = grads_holder[0]      # [B, C, H, W]
            weights = grads.mean(dim=(2, 3), keepdim=True)
            cam = (weights * feats).sum(dim=1)  # [B, H, W]
            cam = F.relu(cam)
            # normalize per-image
            cam = cam - cam.amin(dim=(1, 2), keepdim=True)
            cam = cam / (cam.amax(dim=(1, 2), keepdim=True) + 1e-8)
            return cam.detach().cpu().numpy()
        finally:
            h1.remove(); h2.remove()

    # Get target_layer for both models
    def find_layer4(model):
        # baseline: model is Sequential(backbone, head). backbone.layer4 = the last conv block
        if hasattr(model, "feature_extractor"):
            # MSSABackbone case — feature_extractor.layer4
            return model.feature_extractor.layer4
        else:
            # Sequential case
            backbone = model[0]
            return backbone.layer4

    base_layer = find_layer4(base_model)
    aaf_layer  = find_layer4(aaf_model)

    fig, axes = plt.subplots(3, 6, figsize=(15, 7.5),
                              gridspec_kw={"hspace":0.05, "wspace":0.05})
    for c, idx in enumerate(sample_idxs):
        img_uint8 = images[idx].numpy().transpose(1, 2, 0)        # [H, W, 3] uint8
        x = images[idx:idx+1].float().to(device) / 255.0
        x_norm = (x - mean) / std
        # Original image
        axes[0, c].imshow(img_uint8)
        axes[0, c].set_title(f"Class {c}", fontsize=11)
        axes[0, c].axis("off")

        # Baseline CAM
        cam_b = gradcam(base_model, x_norm.clone(), c, base_layer)[0]
        cam_b_resized = np.array(plt.cm.jet(cam_b)[..., :3])
        # Resize CAM to image size
        from scipy.ndimage import zoom
        zoom_f = (img_uint8.shape[0] / cam_b.shape[0], img_uint8.shape[1] / cam_b.shape[1])
        cam_b_resized = zoom(cam_b, zoom_f, order=1)
        overlay_b = (0.6 * img_uint8 / 255.0 + 0.4 * plt.cm.jet(cam_b_resized)[..., :3])
        axes[1, c].imshow(np.clip(overlay_b, 0, 1))
        axes[1, c].axis("off")

        # AAFNet CAM
        cam_a = gradcam(aaf_model, x_norm.clone(), c, aaf_layer)[0]
        cam_a_resized = zoom(cam_a, zoom_f, order=1)
        overlay_a = (0.6 * img_uint8 / 255.0 + 0.4 * plt.cm.jet(cam_a_resized)[..., :3])
        axes[2, c].imshow(np.clip(overlay_a, 0, 1))
        axes[2, c].axis("off")

    axes[0, 0].text(-0.15, 0.5, "Input", transform=axes[0, 0].transAxes,
                     ha="right", va="center", fontsize=12, fontweight="bold", rotation=90)
    axes[1, 0].text(-0.15, 0.5, "Baseline\nGradCAM", transform=axes[1, 0].transAxes,
                     ha="right", va="center", fontsize=12, fontweight="bold", rotation=90, color=C_BASE)
    axes[2, 0].text(-0.15, 0.5, "AAFNet\nGradCAM", transform=axes[2, 0].transAxes,
                     ha="right", va="center", fontsize=12, fontweight="bold", rotation=90, color=C_AAF)

    fig.suptitle("Activation comparison: AAFNet attends more concentratedly to architectural elements", fontsize=14, y=0.96)
    plt.tight_layout(rect=[0.04, 0, 1, 0.94])
    fig.savefig(FIG_DIR / "F_R_gradcam_comparison.png")
    plt.close(fig)
    print(f"  → {FIG_DIR / 'F_R_gradcam_comparison.png'}")


# =================================================================
# F-S. Dual t-SNE on AL6 (baseline vs AAFNet)
# =================================================================
def f_dual_tsne():
    print("\n[F_S_dual_tsne]")
    cache = ROOT / "data" / "cache" / "AL6_224x224_rgb_test.pt"
    data = torch.load(cache, map_location="cpu", weights_only=False)
    images = data["images"]; labels = data["labels"].numpy()

    base_ckpt = latest("outputs/ddp_baseline/*/resnet50/best_resnet50.pth")
    aaf_ckpt  = latest("outputs/ddp_aafnet_v2/*/resnet50/best_resnet50.pth")
    base_model, device, _    = build_model("resnet50", base_ckpt, "AL6", 6)
    aaf_model,  _, has_mssa  = build_model("resnet50", aaf_ckpt,  "AL6", 6)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def features(model, has_aaf):
        out = []
        for i in range(0, len(images), 64):
            x = images[i:i+64].float().to(device) / 255.0
            x = (x - mean) / std
            if has_aaf:
                _ = model(x)
                f = model.last_fused
            else:
                f = model[0](x)
            out.append(f.cpu().numpy())
        return np.concatenate(out)
    base_feats = features(base_model, False)
    aaf_feats  = features(aaf_model, True)
    print(f"  base feats: {base_feats.shape}, aaf feats: {aaf_feats.shape}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    for ax, feats, title, color_acc in [
        (axes[0], base_feats, "Baseline penultimate features", C_BASE),
        (axes[1], aaf_feats,   "AAFNet (post-MSSA fused) features",  C_AAF),
    ]:
        try:
            tsne = TSNE(n_components=2, perplexity=30, max_iter=2000, random_state=42)
        except TypeError:
            tsne = TSNE(n_components=2, perplexity=30, n_iter=2000, random_state=42)
        emb = tsne.fit_transform(feats)
        cmap = plt.cm.tab10
        for c in np.unique(labels):
            m = labels == c
            ax.scatter(emb[m, 0], emb[m, 1], s=22, color=cmap(c), edgecolor="white",
                        linewidth=0.5, alpha=0.85, label=f"Class {c}")
        ax.set_title(title, fontsize=12, fontweight="bold", color=color_acc)
        ax.set_xticks([]); ax.set_yticks([])
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False); ax.spines["left"].set_visible(False)

    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=10)
    plt.suptitle("AL6 test set t-SNE — AAFNet features form tighter, better-separated class clusters", fontsize=13, y=1.00)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "F_S_dual_tsne.png")
    plt.close(fig)
    print(f"  → {FIG_DIR / 'F_S_dual_tsne.png'}")


# =================================================================
# F-T. Confusion matrices for ASP/AS25 (most informative)
# =================================================================
def f_confusion():
    print("\n[F_T_confusion]")
    fig, axes = plt.subplots(2, 2, figsize=(12, 11), gridspec_kw={"hspace":0.30})
    targets = [
        ("ASP_clean", "outputs/asp_as25_baseline_ASP_clean_seed42",  "Baseline / ASP_clean", axes[0, 0]),
        ("ASP_clean", "outputs/asp_as25_aafnet_ASP_clean_seed42",   "AAFNet / ASP_clean",   axes[0, 1]),
        ("AS25_clean", "outputs/asp_as25_baseline_AS25_clean_seed42", "Baseline / AS25_clean", axes[1, 0]),
        ("AS25_clean", "outputs/asp_as25_aafnet_AS25_clean_seed42",  "AAFNet / AS25_clean",  axes[1, 1]),
    ]
    for ds, root, title, ax in targets:
        # Find test_metrics.json
        cands = list((ROOT / root).glob("*/resnet50/test_metrics.json"))
        if not cands:
            print(f"  [warn] no metrics for {root}")
            continue
        m = json.loads(cands[0].read_text())
        cm = np.array(m["confusion_matrix"], dtype=float)
        cm_norm = cm / cm.sum(axis=1, keepdims=True)
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        n = cm.shape[0]
        # only annotate when small
        if n <= 9:
            for i in range(n):
                for j in range(n):
                    color = "white" if cm_norm[i, j] > 0.4 else "black"
                    ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                            fontsize=8, color=color)
        ax.set_title(title)
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(range(n), fontsize=7); ax.set_yticklabels(range(n), fontsize=7)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.suptitle("Per-class confusion matrices (row-normalized)", fontsize=14, y=0.995)
    fig.savefig(FIG_DIR / "F_T_confusion_matrices.png")
    plt.close(fig)
    print(f"  → {FIG_DIR / 'F_T_confusion_matrices.png'}")


def main():
    f_dual_tsne()
    f_confusion()
    f_gradcam()


if __name__ == "__main__":
    main()
