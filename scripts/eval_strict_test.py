"""
P1.4 — Re-evaluate existing best ckpts on the *_strict test sets.
Uses the same config-restore pattern as run_robustness.py so that AAFNet
(MSSA + SupCon) ckpts load correctly.

Outputs a small JSON + Markdown table comparing main vs strict test accuracy.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.backbones import get_backbone
from src.utils.config import load_config


def latest(pat: str) -> Path | None:
    matches = sorted(ROOT.glob(pat))
    return matches[-1] if matches else None


def evaluate_one(model_name: str, ckpt_path: Path, dataset: str) -> dict:
    cache = ROOT / "data" / "cache" / f"{dataset}_224x224_rgb_test.pt"
    if not cache.exists():
        return {"error": f"missing cache {cache}"}
    data = torch.load(cache, map_location="cpu", weights_only=False)
    images = data["images"]
    labels = data["labels"]
    num_classes = int(labels.max().item()) + 1

    log_path = ckpt_path.parent / "training_log.json"
    overrides = {
        "model": {"name": model_name},
        "data": {"dataset": dataset, "img_height": 224, "img_width": 224},
    }
    if log_path.exists():
        log = json.loads(log_path.read_text())
        snap = log.get("config_snapshot", {})
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

    device = instance.device
    model = model.to(device).eval()

    BS = 64
    preds, gts = [], []
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    with torch.no_grad():
        for i in range(0, len(images), BS):
            batch = images[i:i+BS].float().to(device) / 255.0
            batch = (batch - mean) / std
            out = model(batch)
            if isinstance(out, (tuple, list)):
                out = out[0]
            p = out.argmax(dim=1).cpu().numpy().tolist()
            preds.extend(p)
            gts.extend(labels[i:i+BS].cpu().numpy().tolist())

    return {
        "n_test": len(gts),
        "test_accuracy": float(accuracy_score(gts, preds)),
        "macro_f1":      float(f1_score(gts, preds, average="macro",   zero_division=0)),
        "weighted_f1":   float(f1_score(gts, preds, average="weighted", zero_division=0)),
    }


PAIRS = [
    ("baseline_AL6_clean",  "AL6",        "outputs/ddp_baseline/*/resnet50/best_resnet50.pth"),
    ("baseline_AL6_strict", "AL6_strict", "outputs/ddp_baseline/*/resnet50/best_resnet50.pth"),
    ("aafnet_AL6_clean",    "AL6",        "outputs/ddp_aafnet_v2/*/resnet50/best_resnet50.pth"),
    ("aafnet_AL6_strict",   "AL6_strict", "outputs/ddp_aafnet_v2/*/resnet50/best_resnet50.pth"),
]


def main():
    rows = []
    for label, dataset, glob_pat in PAIRS:
        ckpt = latest(glob_pat)
        if ckpt is None:
            rows.append({"label": label, "dataset": dataset, "error": f"no ckpt match for {glob_pat}"})
            continue
        print(f"=== {label} on {dataset} ===")
        print(f"  ckpt: {ckpt}")
        res = evaluate_one("resnet50", ckpt, dataset)
        print(f"  {res}")
        rows.append({"label": label, "dataset": dataset, "ckpt": str(ckpt), **res})

    out = ROOT / "outputs" / "p1_strict_eval.json"
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Wrote {out}")

    md = ["# P1.4 — Strict-test re-evaluation\n",
          "Same best-ckpt as in main results, evaluated on the original (clean)",
          "and the *_strict test sets (MD5-deduplicated + pHash-near-duplicate stripped).\n",
          "| Label | Dataset | n_test | Test acc | Macro-F1 |",
          "|---|---|---|---|---|"]
    for r in rows:
        if "error" in r:
            md.append(f"| {r['label']} | {r['dataset']} | — | error | error |")
        else:
            md.append(f"| {r['label']} | {r['dataset']} | {r['n_test']} | {r['test_accuracy']*100:.2f} % | {r['macro_f1']*100:.2f} % |")
    md_path = ROOT / "outputs" / "p1_strict_eval.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
