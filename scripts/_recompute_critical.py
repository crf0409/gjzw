#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""快速重测 baseline + AAFNet 两个关键 ckpt, 修 test_metrics.json."""
import sys, json, torch, numpy as np, torch.nn as nn
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.backbones import get_backbone
from src.utils.config import load_config
from sklearn.metrics import (precision_recall_fscore_support, confusion_matrix,
                              classification_report)

ROOT = Path(__file__).resolve().parents[1]

CKPTS = [
    ("outputs/ddp_baseline/20260508_115052/resnet50", "baseline_resnet50"),
    ("outputs/ddp_aafnet/20260508_115922/resnet50", "aafnet_resnet50"),
]

cache = torch.load(ROOT / "data/cache/AL6_224x224_rgb_test.pt",
                    weights_only=False)
imgs, labs = cache["images"], cache["labels"]
nc = int(labs.max()) + 1
device = torch.device("cuda")
mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

results = {}
for sub, name in CKPTS:
    d = ROOT / sub
    ckpt = d / "best_resnet50.pth"
    if not ckpt.exists():
        print(f"[skip] {ckpt}")
        continue
    log_path = d / "training_log.json"
    overrides = {"model": {"name": "resnet50"},
                  "data": {"dataset": "AL6", "img_height": 224,
                            "img_width": 224}}
    if log_path.exists():
        snap = json.load(open(log_path)).get("config_snapshot", {})
        if "aafnet" in snap:
            overrides["aafnet"] = snap["aafnet"]
    cfg = load_config(overrides=overrides)
    Cls = get_backbone("resnet50")
    inst = Cls.__new__(Cls)
    inst.config = cfg
    inst.num_classes = nc
    inst.device = device
    inst._to_rgb = True
    m = inst.build_model().to(device).eval()
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    m.load_state_dict(sd, strict=False)

    pred = []
    with torch.no_grad():
        for i in range(0, len(imgs), 64):
            x = imgs[i:i + 64].float().to(device) / 255.0
            x = (x - mean) / std
            out = m(x)
            if isinstance(out, tuple):
                out = out[0]
            pred.append(out.argmax(1).cpu().numpy())
    y_pred = np.concatenate(pred)
    y_true = labs.numpy()
    acc = float((y_true == y_pred).mean())
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    wt_p, wt_r, wt_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(nc)))

    new_metrics = {
        "test_accuracy": acc,
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(wt_p),
        "weighted_recall": float(wt_r),
        "weighted_f1": float(wt_f1),
        "confusion_matrix": cm.tolist(),
        "classification_report_text": classification_report(
            y_true, y_pred, zero_division=0),
        "n_test": int(len(y_true)),
        "_recomputed": True,
        "_ckpt": str(ckpt),
    }

    old_path = d / "test_metrics.json"
    if old_path.exists():
        bak = old_path.with_suffix(".json.bak")
        if not bak.exists():
            old_path.rename(bak)
        old_data = json.load(open(bak))
        new_metrics["old_test_accuracy"] = old_data.get("test_accuracy")
    with open(old_path, "w") as f:
        json.dump(new_metrics, f, indent=2, ensure_ascii=False)

    print(f"[{name}] old={new_metrics.get('old_test_accuracy')}  "
          f"new={acc:.4f}  macro_f1={macro_f1:.4f}  weighted_f1={wt_f1:.4f}")
    results[name] = {"old_acc": new_metrics.get("old_test_accuracy"),
                      "new_acc": acc, "macro_f1": float(macro_f1)}
    del m
    torch.cuda.empty_cache()

# 写整体修复 summary
out = ROOT / "outputs/data_audit/recompute_summary.json"
with open(out, "w") as f:
    json.dump({"applied": True, "results": results}, f, indent=2,
                ensure_ascii=False)
print(f"\nsaved: {out}")
