"""
Convert tall (vertical) figures to landscape orientation
to better fit two-column / single-column journal layouts.

Targets (ratio > 1.05):
  - near_duplicate_pairs_*  (3.71, 2.60) — pair grid: stack 2 pairs per row
  - F_X_attention_diff       (1.48)        — 6×4 → 2×12
  - F_AB_rotation_sample_ring (1.11)       — already mostly square, leave
  - F_T_confusion_matrices    (1.10)       — 2×2 → 1×4
  - F_Z_confidence_compare    (0.93, but tall content) — keep

Mermaid diagrams (re-render LR):
  - D1, D6, D7, D10
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "paper" / "figures"
sys.path.insert(0, str(ROOT))
from src.models.backbones import get_backbone
from src.utils.config import load_config

plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 11,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

C_BASE = "#4a5468"; C_AAF = "#d04848"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
STD  = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)


def latest(pat: str) -> Path | None:
    p = sorted(ROOT.glob(pat))
    return p[-1] if p else None


# =================================================================
# 1) Near-duplicate pair grids — stack pairs horizontally
# =================================================================
def fix_near_dup():
    print("\n[fix_near_dup]")
    for dataset in ["AL6", "ASP", "AS25"]:
        ds = dataset
        audit_p = ROOT / "outputs" / "data_audit" / ds / "audit.json"
        if not audit_p.exists():
            continue
        audit = json.loads(audit_p.read_text())
        near = audit.get("near_duplicate_phash", {}).get("examples_top10", [])
        if not near:
            continue

        ds_root = ROOT / "data" / "processed" / ds
        pairs = []
        for d in near:
            test_p = ds_root / "test" / str(d["test_label"]) / d["test_file"]
            train_p = ds_root / "train" / str(d["train_label"]) / d["train_file"]
            if not test_p.exists():
                test_p = next(iter(ds_root.glob(f"test/*/{d['test_file']}")), None)
            if not train_p.exists():
                train_p = next(iter(ds_root.glob(f"train/*/{d['train_file']}")), None)
            if test_p and train_p:
                pairs.append((test_p, train_p, d))

        if not pairs:
            continue

        # Layout: per row stack 3 pairs (6 panels) for ASP/AS25 (10 pairs); 2 pairs (4 panels) for AL6 (7 pairs)
        n = min(len(pairs), 10)
        per_row = 3 if dataset in ("ASP", "AS25") else 2
        nrows = int(np.ceil(n / per_row))
        ncols = per_row * 2
        fig, axes = plt.subplots(nrows, ncols, figsize=(2.4 * ncols, 2.3 * nrows))
        if nrows == 1:
            axes = axes.reshape(1, -1)

        for i, (test_p, train_p, d) in enumerate(pairs[:n]):
            r = i // per_row
            c = (i % per_row) * 2
            try:
                axes[r, c].imshow(Image.open(test_p).convert("RGB"))
                axes[r, c+1].imshow(Image.open(train_p).convert("RGB"))
                axes[r, c].set_title(f"test  cls #{d['test_label']}", fontsize=9)
                axes[r, c+1].set_title(f"train cls #{d['train_label']}  H={d['phash_distance']}", fontsize=9)
            except Exception as e:
                print(f"    failed: {e}")
            axes[r, c].axis("off"); axes[r, c+1].axis("off")

        # Hide unused
        for j in range(n, nrows * per_row):
            r = j // per_row; c = (j % per_row) * 2
            axes[r, c].axis("off"); axes[r, c+1].axis("off")

        fig.suptitle(f"{ds} — top-{n} pHash near-duplicate cross-split pairs (test ↔ train)", fontsize=12, y=1.0)
        plt.tight_layout(rect=[0, 0, 1, 0.985])
        out_dir = FIG_DIR / "data_audit"
        fig.savefig(out_dir / f"near_duplicate_pairs_{ds}.png")
        plt.close(fig)
        print(f"  → {out_dir / f'near_duplicate_pairs_{ds}.png'}")

    # Replace legacy name
    al6 = FIG_DIR / "data_audit" / "near_duplicate_pairs_AL6.png"
    if al6.exists():
        (FIG_DIR / "data_audit" / "near_duplicate_pairs.png").write_bytes(al6.read_bytes())


# =================================================================
# 2) F_X_attention_diff — convert 6×4 to 2×12
# =================================================================
def fix_attention_diff():
    print("\n[fix_attention_diff]")
    cache = ROOT / "data" / "cache" / "AL6_224x224_rgb_test.pt"
    data = torch.load(cache, map_location="cpu", weights_only=False)
    images = data["images"]; labels = data["labels"].numpy()

    def build_model(ckpt_path, num_classes):
        log_path = ckpt_path.parent / "training_log.json"
        overrides = {"model": {"name": "resnet50"},
                     "data":  {"dataset": "AL6", "img_height": 224, "img_width": 224}}
        has_mssa = False
        if log_path.exists():
            snap = json.loads(log_path.read_text()).get("config_snapshot", {})
            if "aafnet" in snap:
                overrides["aafnet"] = snap["aafnet"]
                has_mssa = bool(snap["aafnet"].get("msa", {}).get("enabled", False))
        cfg = load_config(overrides=overrides)
        Cls = get_backbone("resnet50")
        inst = Cls.__new__(Cls)
        inst.config = cfg; inst.num_classes = num_classes
        inst.device = DEVICE; inst._to_rgb = True
        model = inst.build_model()
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        model.load_state_dict(sd, strict=False)
        return model.to(DEVICE).eval(), has_mssa

    def find_layer4(model):
        if hasattr(model, "feature_extractor"):
            return model.feature_extractor.layer4
        return model[0].layer4

    def gradcam(model, x_uint8, target_class, layer):
        captured = {}
        def hk(m, i, o):
            if o.requires_grad:
                o.retain_grad(); captured["feat"] = o
        h = layer.register_forward_hook(hk)
        try:
            x = (x_uint8.unsqueeze(0).float().to(DEVICE) / 255.0)
            x = (x - MEAN) / STD
            x = x.detach().clone().requires_grad_(True)
            model.zero_grad()
            out = model(x)
            if isinstance(out, (tuple, list)): out = out[0]
            out[:, target_class].sum().backward()
            feats = captured.get("feat")
            if feats is None or feats.grad is None: return np.zeros((7, 7))
            grads = feats.grad
            weights = grads.mean(dim=(2, 3), keepdim=True)
            cam = F.relu((weights * feats).sum(dim=1))
            cam = cam - cam.amin(dim=(1, 2), keepdim=True)
            cam = cam / (cam.amax(dim=(1, 2), keepdim=True) + 1e-8)
            return cam.detach().cpu().numpy()[0]
        finally:
            h.remove()

    def overlay(img_u8, cam, alpha=0.42):
        cam_full = zoom(cam, (img_u8.shape[0]/cam.shape[0], img_u8.shape[1]/cam.shape[1]), order=1)
        heat = plt.cm.jet(cam_full)[..., :3]
        return np.clip((1-alpha)*img_u8/255.0 + alpha*heat, 0, 1)

    base_ckpt = latest("outputs/ddp_baseline/*/resnet50/best_resnet50.pth")
    aaf_ckpt  = latest("outputs/ddp_aafnet_v2/*/resnet50/best_resnet50.pth")
    base_model, _ = build_model(base_ckpt, 6)
    aaf_model, _  = build_model(aaf_ckpt, 6)
    bL = find_layer4(base_model); aL = find_layer4(aaf_model)

    rng = np.random.default_rng(101)
    # Layout: 6 classes split 2 rows × 3 classes; each class takes 4 panels (input/baseline/aafnet/diff)
    # Total: 2 rows × 12 cols
    fig, axes = plt.subplots(2, 12, figsize=(22, 8.6),
                              gridspec_kw={"hspace": 0.20, "wspace": 0.04})
    last_im = None
    for cls_idx, cls in enumerate(range(6)):
        row = cls_idx // 3
        col_off = (cls_idx % 3) * 4
        idx = int(rng.choice(np.where(labels == cls)[0]))
        x = images[idx:idx+1].float().to(DEVICE) / 255.0
        xn = (x - MEAN) / STD
        cam_b = gradcam(base_model, images[idx], cls, bL)
        cam_a = gradcam(aaf_model, images[idx], cls, aL)
        diff = cam_a - cam_b

        img_u8 = images[idx].numpy().transpose(1, 2, 0)
        diff_full = zoom(diff, (img_u8.shape[0]/diff.shape[0], img_u8.shape[1]/diff.shape[1]), order=1)

        axes[row, col_off + 0].imshow(img_u8); axes[row, col_off + 0].axis("off")
        axes[row, col_off + 0].set_title(f"Cls {cls} input", fontsize=10, pad=4)
        axes[row, col_off + 1].imshow(overlay(img_u8, cam_b)); axes[row, col_off + 1].axis("off")
        axes[row, col_off + 1].set_title("Baseline", fontsize=10, pad=4, color=C_BASE)
        axes[row, col_off + 2].imshow(overlay(img_u8, cam_a)); axes[row, col_off + 2].axis("off")
        axes[row, col_off + 2].set_title("AAFNet", fontsize=10, pad=4, color=C_AAF)
        axes[row, col_off + 3].imshow(img_u8); axes[row, col_off + 3].axis("off")
        last_im = axes[row, col_off + 3].imshow(diff_full, cmap="RdBu_r", vmin=-1, vmax=1, alpha=0.55)
        axes[row, col_off + 3].set_title("Δ (A−B)", fontsize=10, pad=4)

    cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(), shrink=0.7, pad=0.012, location="right")
    cbar.set_label("Δ attention (red = AAFNet > Baseline)")
    fig.suptitle("Attention difference (landscape) — red regions = AAFNet pays MORE attention vs baseline",
                 fontsize=14, y=0.995)
    fig.savefig(FIG_DIR / "F_X_attention_diff.png")
    plt.close(fig)
    print(f"  → {FIG_DIR / 'F_X_attention_diff.png'}")


# =================================================================
# 3) F_T_confusion_matrices — 2×2 → 1×4
# =================================================================
def fix_confusion():
    print("\n[fix_confusion]")
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.6))
    targets = [
        ("ASP_clean",  "outputs/asp_as25_baseline_ASP_clean_seed42",  "Baseline / ASP_clean", axes[0]),
        ("ASP_clean",  "outputs/asp_as25_aafnet_ASP_clean_seed42",   "AAFNet / ASP_clean",   axes[1]),
        ("AS25_clean", "outputs/asp_as25_baseline_AS25_clean_seed42", "Baseline / AS25_clean", axes[2]),
        ("AS25_clean", "outputs/asp_as25_aafnet_AS25_clean_seed42",  "AAFNet / AS25_clean",  axes[3]),
    ]
    for ds, root, title, ax in targets:
        cands = list((ROOT / root).glob("*/resnet50/test_metrics.json"))
        if not cands:
            ax.set_visible(False); continue
        m = json.loads(cands[0].read_text())
        cm = np.array(m["confusion_matrix"], dtype=float)
        cm_norm = cm / cm.sum(axis=1, keepdims=True)
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        n = cm.shape[0]
        if n <= 9:
            for i in range(n):
                for j in range(n):
                    color = "white" if cm_norm[i, j] > 0.4 else "black"
                    ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                            fontsize=7, color=color)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(range(n), fontsize=6); ax.set_yticklabels(range(n), fontsize=6)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True" if ax is axes[0] else "")
    fig.suptitle("Per-class confusion matrices (row-normalized) — landscape", fontsize=13, y=1.0)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG_DIR / "F_T_confusion_matrices.png")
    plt.close(fig)
    print(f"  → {FIG_DIR / 'F_T_confusion_matrices.png'}")


def main():
    fix_near_dup()
    fix_attention_diff()
    fix_confusion()


if __name__ == "__main__":
    main()
