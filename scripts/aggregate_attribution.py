"""
Aggregate P1.1 robustness attribution results.

Reads:
  outputs/attrib_<config>_seed<seed>/.../resnet50/test_metrics.json   (clean acc)
  outputs/attrib_robust_<config>_seed<seed>/.../resnet50/results.json (robustness)
  + existing baseline (cv_baseline) and AAFNet (cv_aafnet_v2) for context.

Writes:
  outputs/p1_attribution_summary.json
  outputs/p1_attribution_summary.md  (markdown table for §5.2)
"""
from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


CONFIGS = [
    ("baseline",            "ResNet-50 baseline"),
    ("nx_only",             "ResNet-50 + GaussianNoise only"),
    ("archaug_no_noise",    "ResNet-50 + ArchAug w/o noise"),
    ("archaug_with_noise",  "ResNet-50 + ArchAug + noise"),
    ("mssa_only",           "ResNet-50 + MSSA"),
    ("mssa_archaug_noise",  "ResNet-50 + MSSA + ArchAug + noise"),
    ("aafnet_full",         "AAFNet (full: MSSA + SupCon + ArchAug + noise)"),
]

SEEDS = [42, 1337, 2024]


def find_latest(pattern: str) -> Path | None:
    paths = sorted(ROOT.glob(pattern))
    return paths[-1] if paths else None


def load_test_metrics(out_dir: Path) -> dict | None:
    """Find test_metrics.json under out_dir for resnet50."""
    candidates = list(out_dir.glob("**/resnet50/test_metrics.json"))
    if not candidates:
        return None
    with open(candidates[0]) as f:
        return json.load(f)


def load_robust_results(out_dir: Path) -> dict | None:
    candidates = list(out_dir.glob("**/resnet50/results.json"))
    if not candidates:
        return None
    with open(candidates[0]) as f:
        return json.load(f)


def collect_config_seed(config: str, seed: int) -> dict:
    """Return dict with 'clean_acc', 'robust' (kind→severity→acc) for one config-seed.
    优先从 attrib_*_seed* 拿 (3-seed 数据齐), 没有再回退到 cv_baseline / cv_aafnet_v2 + 旧 robustness."""

    # 1. 先看新的 attribution-style dir
    train_dir = ROOT / "outputs" / f"attrib_{config}_seed{seed}"
    test = load_test_metrics(train_dir) if train_dir.exists() else None
    robust_dir = ROOT / "outputs" / f"attrib_robust_{config}_seed{seed}"
    robust = load_robust_results(robust_dir) if robust_dir.exists() else None

    if test or robust:
        clean = None
        if test:
            clean = test.get("test_accuracy") or test.get("accuracy")
        if robust and clean is None:
            clean = robust.get("clean_accuracy")
        return {"config": config, "seed": seed, "clean_acc": clean, "robust": robust}

    # 2. 回退到旧路径 (baseline / aafnet_full)
    if config == "baseline":
        cv_dir = {42: "cv_baseline", 1337: "cv_baseline_seed1337", 2024: "cv_baseline_seed2024"}[seed]
        cv_summary = find_latest(f"outputs/{cv_dir}/*/resnet50/cv_summary.json")
        clean = None
        if cv_summary:
            with open(cv_summary) as f:
                cv = json.load(f)
            clean = cv["test_accuracy_mean"]
        robust = None
        if seed == 42:
            res = find_latest("outputs/robustness/*/resnet50/results.json")
            if res:
                with open(res) as f:
                    robust = json.load(f)
        return {"config": config, "seed": seed, "clean_acc": clean, "robust": robust}

    if config == "aafnet_full":
        cv_dir = {42: "cv_aafnet_v2", 1337: "cv_aafnet_v2_seed1337", 2024: "cv_aafnet_v2_seed2024"}[seed]
        cv_summary = find_latest(f"outputs/{cv_dir}/*/resnet50/cv_summary.json")
        clean = None
        if cv_summary:
            with open(cv_summary) as f:
                cv = json.load(f)
            clean = cv["test_accuracy_mean"]
        robust = None
        if seed == 42:
            res = find_latest("outputs/robustness_aafnet_v2/*/resnet50/results.json")
            if res:
                with open(res) as f:
                    robust = json.load(f)
        return {"config": config, "seed": seed, "clean_acc": clean, "robust": robust}

    return {"config": config, "seed": seed, "clean_acc": None, "robust": None}


def aggregate():
    rows = []
    by_config: dict[str, list] = defaultdict(list)
    for cfg, _ in CONFIGS:
        for seed in SEEDS:
            r = collect_config_seed(cfg, seed)
            rows.append(r)
            by_config[cfg].append(r)

    # 输出表: 每 config 一行, 各扰动级 mean ± std (跨 seed)
    summary = {}
    for cfg, label in CONFIGS:
        cfg_rows = by_config[cfg]
        clean_vals = [r["clean_acc"] for r in cfg_rows if r.get("clean_acc") is not None]
        out: dict = {"label": label, "n_seeds": len(clean_vals)}
        if clean_vals:
            out["clean_mean"] = float(np.mean(clean_vals))
            out["clean_std"]  = float(np.std(clean_vals, ddof=0))

        # robust: 收集 (kind, severity) → list across seeds
        rob_collect: dict[tuple, list] = defaultdict(list)
        for r in cfg_rows:
            rb = r.get("robust") or {}
            for kind, lst in rb.get("robustness", {}).items():
                for entry in lst:
                    rob_collect[(kind, entry["severity"])].append(entry["accuracy"])

        out["robust"] = {}
        for (kind, sev), vals in sorted(rob_collect.items()):
            out["robust"][f"{kind}@{sev}"] = {
                "mean": float(np.mean(vals)),
                "std":  float(np.std(vals, ddof=0)) if len(vals) > 1 else 0.0,
                "n":    len(vals),
            }
        summary[cfg] = out

    return summary, rows


def render_md(summary: dict) -> str:
    lines = []
    lines.append("# P1.1 — Robustness attribution summary\n")
    lines.append("Each row averages across the 3 seeds (42, 1337, 2024) where available.\n")
    lines.append("Format: `mean ± std` (omitted if only seed=42 was evaluated for robustness).\n\n")

    # 横向: 选关键扰动列
    cols = [
        ("clean", "Clean"),
        ("gauss_noise@0.05", "Noise σ=0.05"),
        ("gauss_noise@0.1",  "Noise σ=0.10"),
        ("gauss_noise@0.2",  "Noise σ=0.20"),
        ("motion_blur@5",    "Blur k=5"),
        ("motion_blur@11",   "Blur k=11"),
        ("motion_blur@17",   "Blur k=17"),
        ("brightness@0.4",   "Brightness Δ=0.4"),
        ("occlusion@0.3",    "Occlusion 30%"),
    ]
    header = "| Config | " + " | ".join(c[1] for c in cols) + " |"
    sep = "|" + "|".join(["---"] * (len(cols) + 1)) + "|"
    lines.append(header)
    lines.append(sep)
    for cfg, _ in CONFIGS:
        row_data = summary.get(cfg, {})
        row = ["**" + row_data.get("label", cfg) + "**"]
        for col_key, _ in cols:
            if col_key == "clean":
                m = row_data.get("clean_mean")
                s = row_data.get("clean_std", 0)
                if m is None:
                    row.append("—")
                elif row_data.get("n_seeds", 0) > 1:
                    row.append(f"{m*100:.2f} ± {s*100:.2f} %")
                else:
                    row.append(f"{m*100:.2f} %")
            else:
                rb = row_data.get("robust", {}).get(col_key)
                if rb is None:
                    row.append("—")
                elif rb["n"] > 1:
                    row.append(f"{rb['mean']*100:.2f} ± {rb['std']*100:.2f} %")
                else:
                    row.append(f"{rb['mean']*100:.2f} %")
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines) + "\n"


def main():
    summary, raw = aggregate()
    out_json = ROOT / "outputs" / "p1_attribution_summary.json"
    out_md   = ROOT / "outputs" / "p1_attribution_summary.md"
    with open(out_json, "w") as f:
        json.dump({"summary": summary, "raw": raw}, f, indent=2)
    with open(out_md, "w") as f:
        f.write(render_md(summary))
    print(f"[ok] wrote {out_json}")
    print(f"[ok] wrote {out_md}")
    print()
    print(render_md(summary))


if __name__ == "__main__":
    main()
