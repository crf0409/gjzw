"""
P3.8 — Rotation robustness evaluation.

For each angle θ ∈ {0, 15, 30, …, 345}, rotate the AL6 test images by θ,
evaluate baseline and AAFNet ckpts, report top-1 accuracy.

Output: outputs/p3_rotation.json + outputs/p3_rotation.md
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
    inst.config = cfg
    inst.num_classes = num_classes
    inst.device = DEVICE
    inst._to_rgb = True
    model = inst.build_model()
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd, strict=False)
    return model.to(DEVICE).eval(), has_mssa


def rotate_batch(imgs: torch.Tensor, angle_deg: float) -> torch.Tensor:
    """Rotate a batch of uint8 [N,3,H,W] images by angle (CCW)."""
    if abs(angle_deg) < 0.01:
        return imgs
    f = imgs.float() / 255.0
    theta = -angle_deg * np.pi / 180.0   # negative for grid_sample (CCW visually)
    cos, sin = np.cos(theta), np.sin(theta)
    # Affine matrix [N, 2, 3]
    aff = torch.tensor([[cos, -sin, 0.0], [sin, cos, 0.0]], device=f.device, dtype=f.dtype)
    aff = aff.unsqueeze(0).expand(f.shape[0], -1, -1).contiguous()
    grid = F.affine_grid(aff, f.size(), align_corners=False)
    rot = F.grid_sample(f, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return (rot * 255).byte()


@torch.no_grad()
def eval_at_angle(model, has_mssa, images, labels, angle_deg, batch=64):
    correct = 0; total = 0; preds_all = []; conf_all = []
    for i in range(0, len(images), batch):
        chunk = images[i:i+batch].to(DEVICE)
        rot = rotate_batch(chunk, angle_deg).float() / 255.0
        rot = (rot - MEAN) / STD
        out = model(rot)
        if isinstance(out, (tuple, list)): out = out[0]
        prob = F.softmax(out, dim=1)
        conf, pred = prob.max(dim=1)
        lbl = labels[i:i+batch].to(DEVICE)
        correct += int((pred == lbl).sum().item())
        total += int(lbl.numel())
        preds_all.append(pred.cpu().numpy())
        conf_all.append(conf.cpu().numpy())
    return {
        "accuracy": correct / max(1, total),
        "n": total,
    }


def main():
    cache = ROOT / "data" / "cache" / "AL6_224x224_rgb_test.pt"
    data = torch.load(cache, map_location="cpu", weights_only=False)
    images = data["images"]; labels = data["labels"]

    angles = list(range(0, 360, 15))   # 24 angles
    base_ckpt = latest("outputs/ddp_baseline/*/resnet50/best_resnet50.pth")
    aaf_ckpt  = latest("outputs/ddp_aafnet_v2/*/resnet50/best_resnet50.pth")

    results = {"angles": angles, "models": {}}
    for label, ckpt in [("baseline", base_ckpt), ("aafnet", aaf_ckpt)]:
        print(f"\n=== {label} === ckpt={ckpt}")
        model, has_mssa = build_model(ckpt, "AL6", 6)
        accs = []
        for ang in angles:
            r = eval_at_angle(model, has_mssa, images, labels, ang)
            accs.append(r["accuracy"])
            print(f"  θ={ang:3d}°  acc={r['accuracy']*100:.2f} %")
        results["models"][label] = accs
        del model
        torch.cuda.empty_cache()

    # Compute summary stats
    summary = {}
    for m, accs in results["models"].items():
        accs_np = np.array(accs)
        summary[m] = {
            "min":   float(accs_np.min()),
            "max":   float(accs_np.max()),
            "mean":  float(accs_np.mean()),
            "std":   float(accs_np.std()),
            "acc@0":   float(accs[0]),
            "acc@90":  float(accs[6]),
            "acc@180": float(accs[12]),
            "acc@270": float(accs[18]),
        }
        print(f"\n{m} summary: min={summary[m]['min']*100:.2f}% mean={summary[m]['mean']*100:.2f}% "
              f"max={summary[m]['max']*100:.2f}%")

    out = ROOT / "outputs" / "p3_rotation.json"
    with open(out, "w") as f:
        json.dump({"angles": angles, "models": results["models"], "summary": summary}, f, indent=2)
    print(f"\nWrote {out}")

    md = ["# P3.8 — Rotation robustness on AL6\n",
          "Test images rotated CCW by θ ∈ {0, 15, 30, …, 345}° (24 angles).",
          "Bilinear sampling, zero-fill outside image bounds.\n",
          "## Per-angle accuracy",
          "| θ (deg) | Baseline | AAFNet | Δ |",
          "|---|---|---|---|"]
    for i, ang in enumerate(angles):
        b = results["models"]["baseline"][i] * 100
        a = results["models"]["aafnet"][i] * 100
        md.append(f"| {ang}° | {b:.2f} % | {a:.2f} % | {a-b:+.2f} pp |")

    md.append("\n## Summary statistics")
    md.append("| Model | Min acc | Mean acc | Max acc | Acc @ 0° | Acc @ 90° | Acc @ 180° | Acc @ 270° |")
    md.append("|---|---|---|---|---|---|---|---|")
    for m in ["baseline", "aafnet"]:
        s = summary[m]
        md.append(f"| {m} | {s['min']*100:.2f} % | {s['mean']*100:.2f} % | {s['max']*100:.2f} % | "
                  f"{s['acc@0']*100:.2f} % | {s['acc@90']*100:.2f} % | {s['acc@180']*100:.2f} % | "
                  f"{s['acc@270']*100:.2f} % |")
    md_path = ROOT / "outputs" / "p3_rotation.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
