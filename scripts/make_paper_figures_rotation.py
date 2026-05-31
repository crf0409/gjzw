"""
Closed-loop rotation robustness figures.

  F-AA  Polar accuracy curve (0°→360° closed loop), baseline vs AAFNet
  F-AB  Rotated-sample ring (12 samples around the wheel, predictions annotated)
  F-AC  GradCAM following rotation (12 angles arranged in a circle)
  F-AD  Per-angle Δ bar wrapped on a circle
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import zoom

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
STD  = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)


def latest(pat: str) -> Path | None:
    p = sorted(ROOT.glob(pat))
    return p[-1] if p else None


def build_model(ckpt_path, dataset, num_classes):
    log_path = ckpt_path.parent / "training_log.json"
    overrides = {"model": {"name": "resnet50"},
                 "data":  {"dataset": dataset, "img_height": 224, "img_width": 224}}
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


def rotate_uint8(img_uint8: torch.Tensor, angle_deg: float) -> torch.Tensor:
    """Rotate single image. img_uint8 is [3, H, W] uint8."""
    if abs(angle_deg) < 0.01:
        return img_uint8
    f = (img_uint8.float() / 255.0).unsqueeze(0).to(DEVICE)
    theta = -angle_deg * np.pi / 180.0
    cos, sin = np.cos(theta), np.sin(theta)
    aff = torch.tensor([[cos, -sin, 0.0], [sin, cos, 0.0]], device=f.device, dtype=f.dtype).unsqueeze(0)
    grid = F.affine_grid(aff, f.size(), align_corners=False)
    rot = F.grid_sample(f, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return (rot.squeeze(0) * 255).byte().cpu()


def predict(model, x_uint8):
    """x_uint8: [3, H, W] uint8."""
    f = x_uint8.unsqueeze(0).float().to(DEVICE) / 255.0
    f = (f - MEAN) / STD
    with torch.no_grad():
        out = model(f)
        if isinstance(out, (tuple, list)): out = out[0]
        prob = F.softmax(out, dim=1)[0]
    p, c = prob.max(0)
    return int(c.item()), float(p.item())


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


def overlay_cam(img_u8_HWC, cam, alpha=0.42):
    cam_full = zoom(cam, (img_u8_HWC.shape[0]/cam.shape[0], img_u8_HWC.shape[1]/cam.shape[1]), order=1)
    heat = plt.cm.jet(cam_full)[..., :3]
    return np.clip((1 - alpha) * img_u8_HWC / 255.0 + alpha * heat, 0, 1)


def find_layer4(model):
    if hasattr(model, "feature_extractor"):
        return model.feature_extractor.layer4
    return model[0].layer4


# =================================================================
# F-AA. Closed-loop polar accuracy curve
# =================================================================
def f_aa_polar():
    print("\n[F_AA_polar]")
    d = json.loads((ROOT / "outputs" / "p3_rotation.json").read_text())
    angles = np.array(d["angles"])
    base = np.array(d["models"]["baseline"]) * 100
    aaf  = np.array(d["models"]["aafnet"]) * 100

    # Close the loop by appending angle 0
    angles_loop = np.concatenate([angles, [360]])
    base_loop = np.concatenate([base, [base[0]]])
    aaf_loop  = np.concatenate([aaf,  [aaf[0]]])
    theta = angles_loop * np.pi / 180.0

    fig = plt.figure(figsize=(9.5, 9.5))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_zero_location("N")    # 0° at top
    ax.set_theta_direction(-1)         # clockwise

    # Fill region
    ax.fill(theta, base_loop, color=C_BASE, alpha=0.18)
    ax.fill(theta, aaf_loop, color=C_AAF, alpha=0.18)
    # Lines
    ax.plot(theta, base_loop, "-o", color=C_BASE, linewidth=2.4, markersize=6, label="Baseline")
    ax.plot(theta, aaf_loop, "-o", color=C_AAF,  linewidth=2.4, markersize=6, label="AAFNet")

    ax.set_ylim(70, 102)
    ax.set_yticks([75, 85, 95, 100])
    ax.set_yticklabels(["75 %", "85 %", "95 %", "100 %"], fontsize=9, color="gray")
    ax.set_thetagrids(angles, [f"{a}°" for a in angles], fontsize=9.5)
    ax.set_rlabel_position(90)

    # Mean accuracy lines (dashed circles)
    base_mean = base.mean(); aaf_mean = aaf.mean()
    ax.plot(np.linspace(0, 2*np.pi, 200), [base_mean]*200, "--", color=C_BASE, linewidth=1, alpha=0.6)
    ax.plot(np.linspace(0, 2*np.pi, 200), [aaf_mean]*200,  "--", color=C_AAF,  linewidth=1, alpha=0.6)

    ax.set_title("Rotation-robustness closed-loop\n"
                 f"Baseline mean = {base_mean:.2f} %  vs  AAFNet mean = {aaf_mean:.2f} %  (+{aaf_mean - base_mean:.2f} pp)",
                 pad=22, fontsize=13)
    ax.legend(loc="upper right", bbox_to_anchor=(1.18, 1.06), fontsize=11)
    ax.grid(alpha=0.5)
    fig.savefig(FIG_DIR / "F_AA_rotation_polar.png")
    plt.close(fig)
    print(f"  → {FIG_DIR / 'F_AA_rotation_polar.png'}")


# =================================================================
# F-AB. Rotated-sample ring with predictions
# =================================================================
def f_ab_sample_ring():
    print("\n[F_AB_sample_ring]")
    cache = ROOT / "data" / "cache" / "AL6_224x224_rgb_test.pt"
    data = torch.load(cache, map_location="cpu", weights_only=False)
    images = data["images"]; labels = data["labels"].numpy()
    rng = np.random.default_rng(11)
    idx = int(rng.choice(np.where(labels == 4)[0]))    # pick one fixed image
    cls = int(labels[idx])

    base_model, _ = build_model(latest("outputs/ddp_baseline/*/resnet50/best_resnet50.pth"), "AL6", 6)
    aaf_model, _  = build_model(latest("outputs/ddp_aafnet_v2/*/resnet50/best_resnet50.pth"),  "AL6", 6)

    angles = list(range(0, 360, 30))    # 12 around
    fig = plt.figure(figsize=(13, 13))
    fig.suptitle(f"Rotation closed-loop — image of class {cls} rotated 0°→330° in 30° steps\n"
                 f"each tile shows: input | Baseline pred (✓/✗ + conf) | AAFNet pred",
                 fontsize=12, y=0.96)

    # Center axis showing the original (un-rotated)
    center_ax = fig.add_axes([0.42, 0.42, 0.16, 0.16])
    center_ax.imshow(images[idx].numpy().transpose(1, 2, 0))
    center_ax.set_title("Original (0°)", fontsize=10, pad=4)
    center_ax.axis("off")

    R = 0.34          # radial distance for sub-axes
    cx, cy = 0.5, 0.5
    for i, ang in enumerate(angles):
        theta_rad = np.pi/2 - ang * np.pi / 180.0   # 0 at top, clockwise
        ax_x = cx + R * np.cos(theta_rad) - 0.075
        ax_y = cy + R * np.sin(theta_rad) - 0.075
        ax = fig.add_axes([ax_x, ax_y, 0.15, 0.15])

        rot = rotate_uint8(images[idx], ang)
        ax.imshow(rot.numpy().transpose(1, 2, 0))
        ax.axis("off")
        # Predict
        b_c, b_p = predict(base_model, rot)
        a_c, a_p = predict(aaf_model, rot)
        b_tag = "✓" if b_c == cls else "✗"
        a_tag = "✓" if a_c == cls else "✗"
        ax.text(0.5, -0.12, f"θ={ang}°\nB:{b_c}{b_tag} {b_p:.2f}\nA:{a_c}{a_tag} {a_p:.2f}",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=9.0, fontfamily="monospace",
                color="black",
                bbox=dict(boxstyle="round,pad=0.25",
                          fc="#f0f9ed" if (b_c == cls and a_c == cls) else
                             ("#fdf3f3" if (b_c != cls and a_c != cls) else "#fff8e1"),
                          ec="gray", lw=0.4))

    fig.savefig(FIG_DIR / "F_AB_rotation_sample_ring.png")
    plt.close(fig)
    print(f"  → {FIG_DIR / 'F_AB_rotation_sample_ring.png'}")


# =================================================================
# F-AC. GradCAM ring following rotation
# =================================================================
def f_ac_gradcam_ring():
    print("\n[F_AC_gradcam_ring]")
    cache = ROOT / "data" / "cache" / "AL6_224x224_rgb_test.pt"
    data = torch.load(cache, map_location="cpu", weights_only=False)
    images = data["images"]; labels = data["labels"].numpy()
    rng = np.random.default_rng(99)
    idx = int(rng.choice(np.where(labels == 2)[0]))
    cls = int(labels[idx])

    base_model, _ = build_model(latest("outputs/ddp_baseline/*/resnet50/best_resnet50.pth"), "AL6", 6)
    aaf_model, _  = build_model(latest("outputs/ddp_aafnet_v2/*/resnet50/best_resnet50.pth"),  "AL6", 6)
    bL = find_layer4(base_model); aL = find_layer4(aaf_model)

    angles = list(range(0, 360, 30))    # 12 around

    fig = plt.figure(figsize=(15, 7.5))
    fig.suptitle(f"Activation following rotation — same image, 12 rotations × 2 models\n"
                 f"Class {cls}; left ring = Baseline GradCAM, right ring = AAFNet GradCAM",
                 fontsize=12, y=0.97)

    for j, (model, layer, label, color, cx_off) in enumerate([
        (base_model, bL, "Baseline", C_BASE, 0.255),
        (aaf_model,  aL, "AAFNet",  C_AAF,  0.745),
    ]):
        # Center label
        fig.text(cx_off, 0.5, label, ha="center", va="center",
                 fontsize=15, fontweight="bold", color=color)

        R = 0.21; cy = 0.5
        for i, ang in enumerate(angles):
            theta_rad = np.pi/2 - ang * np.pi / 180.0
            ax_x = cx_off + R * np.cos(theta_rad) - 0.06
            ax_y = cy + R * np.sin(theta_rad) - 0.06
            ax = fig.add_axes([ax_x, ax_y, 0.12, 0.12])
            rot = rotate_uint8(images[idx], ang)
            cam = gradcam(model, rot, cls, layer)
            ax.imshow(overlay_cam(rot.numpy().transpose(1, 2, 0), cam))
            ax.axis("off")
            # Predict
            pc, pp = predict(model, rot)
            tag = "✓" if pc == cls else "✗"
            ax.text(0.5, -0.10, f"{ang}° {tag}", transform=ax.transAxes,
                    ha="center", va="top", fontsize=8.5,
                    color=("#0d3a25" if pc == cls else "#a32a2a"),
                    fontweight="bold")
    fig.savefig(FIG_DIR / "F_AC_rotation_gradcam_ring.png")
    plt.close(fig)
    print(f"  → {FIG_DIR / 'F_AC_rotation_gradcam_ring.png'}")


# =================================================================
# F-AD. Per-angle Δ bar wrapped on a polar wheel
# =================================================================
def f_ad_delta_polar():
    print("\n[F_AD_delta_polar]")
    d = json.loads((ROOT / "outputs" / "p3_rotation.json").read_text())
    angles = np.array(d["angles"])
    base = np.array(d["models"]["baseline"]) * 100
    aaf  = np.array(d["models"]["aafnet"]) * 100
    delta = aaf - base

    fig = plt.figure(figsize=(8.5, 8.5))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    theta = angles * np.pi / 180.0
    width = 2 * np.pi / len(angles) * 0.85

    cmap = plt.cm.RdYlGn
    norm_d = (delta - delta.min()) / max(delta.max() - delta.min(), 1e-6)
    colors = [cmap(0.35 + 0.6 * v) for v in norm_d]
    bars = ax.bar(theta, delta, width=width, bottom=0,
                  color=colors, edgecolor="black", linewidth=0.5)
    # Annotate Δ values
    for ang, d_, col in zip(angles, delta, colors):
        ax.text(ang * np.pi / 180.0, d_ + 0.6, f"{d_:+.1f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold",
                color="black")

    ax.set_ylim(0, max(delta.max() * 1.18, 16))
    ax.set_yticks([4, 8, 12])
    ax.set_yticklabels(["+4 pp", "+8 pp", "+12 pp"], fontsize=9, color="gray")
    ax.set_thetagrids(angles, [f"{a}°" for a in angles], fontsize=9)
    ax.set_title("Closed-loop Δ (AAFNet − Baseline) per rotation angle\n"
                 f"all 24 angles favor AAFNet (mean +{delta.mean():.2f} pp, max +{delta.max():.2f} pp)",
                 pad=22, fontsize=12)
    fig.savefig(FIG_DIR / "F_AD_rotation_delta_polar.png")
    plt.close(fig)
    print(f"  → {FIG_DIR / 'F_AD_rotation_delta_polar.png'}")


def main():
    f_aa_polar()
    f_ab_sample_ring()
    f_ac_gradcam_ring()
    f_ad_delta_polar()


if __name__ == "__main__":
    main()
