#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
聚合所有实验结果 -> 一个论文级 JSON, 便于生成 6 表 13 图.

扫描 outputs/ 下的:
    ddp/<run_id>/<model>/                  # 单 run 训练
    cv/<run_id>/<model>/cv_summary.json     # 5-fold × 3-seed
    data_eff/<run_id>/<model>/results.json  # 数据效率扫描
    robustness/<run_id>/<model>/results.json # 鲁棒性
    ensemble/<run_id>/results.json           # 集成
    inr_clf/<run_id>/<head>/                 # INR 分类
    inr_vs_pruning/<run_id>/results.json     # INR vs 剪枝
    paper/                                    # 原始 baseline (历史)

输出:
    outputs/paper_v2/summary_v2.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"


def safe_read_json(p: Path) -> dict | None:
    if not p.exists():
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def collect_ddp_runs() -> list:
    """ddp/<run_id>/<model>/{training_log,test_metrics}.json"""
    out = []
    base = OUTPUTS / "ddp"
    if not base.exists():
        return out
    for run_dir in sorted([d for d in base.iterdir() if d.is_dir()]):
        if run_dir.name == "latest":
            continue
        for model_dir in sorted([d for d in run_dir.iterdir() if d.is_dir()]):
            log = safe_read_json(model_dir / "training_log.json")
            tm = safe_read_json(model_dir / "test_metrics.json")
            if not log:
                continue
            entry = {
                "section": "ddp_single_run",
                "run_id": run_dir.name,
                "model": model_dir.name,
                "best_val_accuracy": log.get("best_val_accuracy"),
                "epochs_completed": log.get("epochs_completed"),
                "total_training_seconds": log.get("total_training_seconds"),
                "world_size": log.get("world_size"),
                "args": log.get("args", {}),
            }
            if tm:
                entry["test_accuracy"] = tm.get("test_accuracy")
                entry["macro_f1"] = tm.get("macro_f1")
                entry["weighted_f1"] = tm.get("weighted_f1")
            out.append(entry)
    return out


def collect_cv() -> list:
    """cv/<run_id>/<model>/cv_summary.json"""
    out = []
    base = OUTPUTS / "cv"
    if not base.exists():
        return out
    for run_dir in sorted([d for d in base.iterdir() if d.is_dir()]):
        for model_dir in sorted([d for d in run_dir.iterdir() if d.is_dir()]):
            s = safe_read_json(model_dir / "cv_summary.json")
            if not s:
                continue
            out.append({
                "section": "cv",
                "run_id": run_dir.name,
                "model": s.get("model", model_dir.name),
                "test_accuracy_mean": s.get("test_accuracy_mean"),
                "test_accuracy_std": s.get("test_accuracy_std"),
                "macro_f1_mean": s.get("macro_f1_mean"),
                "macro_f1_std": s.get("macro_f1_std"),
                "n_folds_completed": s.get("n_folds_completed"),
                "n_folds_total": s.get("n_folds_total"),
                "fold_accs": [f.get("test_accuracy") for f in s.get("folds", [])
                              if "test_accuracy" in f],
            })
    return out


def collect_data_eff() -> list:
    out = []
    base = OUTPUTS / "data_eff"
    if not base.exists():
        return out
    for run_dir in sorted([d for d in base.iterdir() if d.is_dir()]):
        for model_dir in sorted([d for d in run_dir.iterdir() if d.is_dir()]):
            s = safe_read_json(model_dir / "results.json")
            if not s:
                continue
            out.append({
                "section": "data_efficiency",
                "run_id": run_dir.name,
                "model": s.get("model", model_dir.name),
                "by_fraction": s.get("by_fraction", {}),
            })
    return out


def collect_robustness() -> list:
    out = []
    base = OUTPUTS / "robustness"
    if not base.exists():
        return out
    for run_dir in sorted([d for d in base.iterdir() if d.is_dir()]):
        for model_dir in sorted([d for d in run_dir.iterdir() if d.is_dir()]):
            s = safe_read_json(model_dir / "results.json")
            if not s:
                continue
            out.append({
                "section": "robustness",
                "run_id": run_dir.name,
                "model": s.get("model"),
                "clean_accuracy": s.get("clean_accuracy"),
                "robustness": s.get("robustness", {}),
            })
    return out


def collect_ensemble() -> list:
    out = []
    base = OUTPUTS / "ensemble"
    if not base.exists():
        return out
    for run_dir in sorted([d for d in base.iterdir() if d.is_dir()]):
        s = safe_read_json(run_dir / "results.json")
        if not s:
            continue
        out.append({"section": "ensemble", "run_id": run_dir.name, **s})
    return out


def collect_inr() -> list:
    out = []
    base = OUTPUTS / "inr_clf"
    if not base.exists():
        return out
    for run_dir in sorted([d for d in base.iterdir() if d.is_dir()]):
        for head_dir in sorted([d for d in run_dir.iterdir() if d.is_dir()]):
            s = safe_read_json(head_dir / "training_log.json")
            if not s:
                continue
            out.append({
                "section": "inr_classifier",
                "run_id": run_dir.name,
                "head": head_dir.name,
                "args": s.get("args", {}),
                "best_val_accuracy": s.get("best_val_accuracy"),
                "test_metrics": s.get("test_metrics"),
                "fit_metrics": s.get("fit_metrics"),
            })
    # benchmark 也加入
    base2 = OUTPUTS / "inr_vs_pruning"
    if base2.exists():
        for run_dir in sorted([d for d in base2.iterdir() if d.is_dir()]):
            s = safe_read_json(run_dir / "results.json")
            if s:
                out.append({"section": "inr_vs_pruning",
                             "run_id": run_dir.name, **s})
    return out


def collect_paper_legacy() -> dict | None:
    """原始 paper/summary.json (跑过的 9 baseline)"""
    p = OUTPUTS / "paper" / "summary.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(OUTPUTS / "paper_v2" / "summary_v2.json"))
    args = p.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_by": "scripts/aggregate_results.py",
        "ddp_runs": collect_ddp_runs(),
        "cv": collect_cv(),
        "data_efficiency": collect_data_eff(),
        "robustness": collect_robustness(),
        "ensemble": collect_ensemble(),
        "inr": collect_inr(),
        "paper_legacy": collect_paper_legacy(),
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\n=== Aggregated ===")
    print(f"  ddp_runs:           {len(payload['ddp_runs'])}")
    print(f"  cv runs:            {len(payload['cv'])}")
    print(f"  data_efficiency:    {len(payload['data_efficiency'])}")
    print(f"  robustness:         {len(payload['robustness'])}")
    print(f"  ensemble:           {len(payload['ensemble'])}")
    print(f"  inr:                {len(payload['inr'])}")
    print(f"  paper_legacy:       "
          f"{'present' if payload['paper_legacy'] else 'absent'}")
    print(f"\n  saved -> {out_path}")


if __name__ == "__main__":
    main()
