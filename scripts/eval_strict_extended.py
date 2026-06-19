"""
P2 #44 — Extended strict evaluation on ASP/AS25 *_clean vs *_strict.

After P1.2 (AAFNet on ASP_clean / AS25_clean) finishes, this script:
  1. For each (dataset, role, seed=42) — uses the seed=42 ckpt (consistent w/ §5.1)
  2. Evaluates on both *_clean and *_strict test sets
  3. Reports per-pair (clean acc, strict acc, Δ)
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
    }


# Each entry: (label, model_name, base_dataset_for_train, ckpt_glob, eval_dataset)
PAIRS = []
for ds_base in ["AL6", "ASP", "AS25"]:
    if ds_base == "AL6":
        clean_id = "AL6"
        strict_id = "AL6_strict"
        # local AL6 ckpts are in ddp_baseline / ddp_aafnet_v2 (60-ep) or attrib_*_seed42 (30-ep)
        baseline_ckpt = "outputs/ddp_baseline/*/resnet50/best_resnet50.pth"
        aafnet_ckpt   = "outputs/ddp_aafnet_v2/*/resnet50/best_resnet50.pth"
    else:
        clean_id = f"{ds_base}_clean"
        strict_id = f"{ds_base}_strict"
        baseline_ckpt = f"outputs/asp_as25_baseline_{clean_id}_seed42/*/resnet50/best_resnet50.pth"
        aafnet_ckpt   = f"outputs/asp_as25_aafnet_{clean_id}_seed42/*/resnet50/best_resnet50.pth"

    PAIRS.append((f"baseline_{ds_base}", clean_id,  baseline_ckpt))
    PAIRS.append((f"baseline_{ds_base}", strict_id, baseline_ckpt))
    PAIRS.append((f"aafnet_{ds_base}",   clean_id,  aafnet_ckpt))
    PAIRS.append((f"aafnet_{ds_base}",   strict_id, aafnet_ckpt))


def main():
    rows = []
    for label, eval_ds, glob_pat in PAIRS:
        ckpt = latest(glob_pat)
        if ckpt is None:
            rows.append({"label": label, "eval_ds": eval_ds, "error": f"no ckpt for {glob_pat}"})
            continue
        cache = ROOT / "data" / "cache" / f"{eval_ds}_224x224_rgb_test.pt"
        if not cache.exists():
            rows.append({"label": label, "eval_ds": eval_ds, "error": f"missing cache {cache}"})
            continue
        print(f"=== {label} on {eval_ds} ===")
        res = evaluate_one("resnet50", ckpt, eval_ds)
        print(f"  {res}")
        rows.append({"label": label, "eval_ds": eval_ds, "ckpt": str(ckpt), **res})

    out = ROOT / "outputs" / "p2_strict_extended.json"
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)

    md = ["# P2 #44 — Strict-test evaluation on AL6 / ASP / AS25\n",
          "Single seed=42 ckpt evaluated on both *_clean and *_strict test sets.",
          "*_strict additionally removes pHash near-duplicate cross-split pairs.\n",
          "| Label | Eval split | n_test | Test acc | Macro-F1 |",
          "|---|---|---|---|---|"]
    for r in rows:
        if "error" in r:
            md.append(f"| {r['label']} | {r['eval_ds']} | — | _{r['error']}_ | — |")
        else:
            md.append(f"| {r['label']} | {r['eval_ds']} | {r['n_test']} | {r['test_accuracy']*100:.2f} % | {r['macro_f1']*100:.2f} % |")

    md_path = ROOT / "outputs" / "p2_strict_extended.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
