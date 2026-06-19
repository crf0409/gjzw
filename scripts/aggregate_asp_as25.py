"""
Aggregate P1.2 results: AAFNet on ASP_clean / AS25_clean (3 seeds each).

Reads outputs/asp_as25_<role>_<dataset>_seed<seed>/<run_id>/resnet50/test_metrics.json
Writes outputs/p1_asp_as25_summary.json + .md
"""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

ROLES = ["baseline", "aafnet"]
DATASETS = ["ASP_clean", "AS25_clean"]
SEEDS = [42, 1337, 2024]


def find_test_metrics(dataset: str, role: str, seed: int) -> Path | None:
    out_dir = ROOT / "outputs" / f"asp_as25_{role}_{dataset}_seed{seed}"
    if not out_dir.exists():
        return None
    candidates = list(out_dir.glob("**/resnet50/test_metrics.json"))
    if not candidates:
        return None
    return sorted(candidates)[-1]


def aggregate():
    summary: dict = {}
    for dataset in DATASETS:
        summary[dataset] = {}
        for role in ROLES:
            accs, f1s, weighted_f1s, per_class = [], [], [], []
            for seed in SEEDS:
                p = find_test_metrics(dataset, role, seed)
                if p is None:
                    continue
                with open(p) as f:
                    m = json.load(f)
                accs.append(m.get("test_accuracy", m.get("accuracy")))
                f1s.append(m.get("macro_f1", m.get("test_macro_f1")))
                weighted_f1s.append(m.get("weighted_f1", m.get("test_weighted_f1")))
                if "per_class_f1" in m:
                    per_class.append(m["per_class_f1"])
            if accs:
                summary[dataset][role] = {
                    "n_seeds": len(accs),
                    "test_acc_mean": float(np.mean(accs)),
                    "test_acc_std":  float(np.std(accs, ddof=0)),
                    "macro_f1_mean": float(np.mean(f1s)) if f1s else None,
                    "macro_f1_std":  float(np.std(f1s, ddof=0)) if f1s else None,
                    "weighted_f1_mean": float(np.mean(weighted_f1s)) if weighted_f1s else None,
                    "raw_accs": accs,
                }
    return summary


def render_md(summary: dict) -> str:
    lines = ["# P1.2 — AAFNet on ASP_clean / AS25_clean (3 seeds)\n"]
    lines.append("| Dataset | Model | n_seeds | Test acc (mean ± std) | Macro-F1 |")
    lines.append("|---|---|---|---|---|")
    for ds in DATASETS:
        for role in ROLES:
            r = summary.get(ds, {}).get(role)
            if r is None:
                lines.append(f"| {ds} | {role} | — | (not available) | — |")
                continue
            acc = f"{r['test_acc_mean']*100:.2f} ± {r['test_acc_std']*100:.2f} %"
            f1  = f"{r['macro_f1_mean']*100:.2f} %" if r.get("macro_f1_mean") else "—"
            lines.append(f"| {ds} | {role} | {r['n_seeds']} | {acc} | {f1} |")
    lines.append("")
    # Δ rows
    lines.append("\n**Δ AAFNet − baseline:**\n")
    for ds in DATASETS:
        b = summary.get(ds, {}).get("baseline")
        a = summary.get(ds, {}).get("aafnet")
        if b is None or a is None:
            continue
        delta = (a["test_acc_mean"] - b["test_acc_mean"]) * 100
        lines.append(f"- **{ds}**: baseline {b['test_acc_mean']*100:.2f} % → AAFNet {a['test_acc_mean']*100:.2f} % (**Δ {delta:+.2f} pp**)")
    return "\n".join(lines) + "\n"


def main():
    s = aggregate()
    out_json = ROOT / "outputs" / "p1_asp_as25_summary.json"
    out_md   = ROOT / "outputs" / "p1_asp_as25_summary.md"
    with open(out_json, "w") as f:
        json.dump(s, f, indent=2)
    with open(out_md, "w") as f:
        f.write(render_md(s))
    print(render_md(s))


if __name__ == "__main__":
    main()
