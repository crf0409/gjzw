"""
Aggregate P1.3 multi-model 5-fold × 3-seed CV results and run:
  - Friedman omnibus
  - Pairwise Wilcoxon signed-rank
  - Holm-Bonferroni adjustment

Models loaded from cv_summary.json files in:
  outputs/cv_baseline*/<run_id>/resnet50/cv_summary.json   (3 seeds, baseline)
  outputs/cv_aafnet_v2*/<run_id>/resnet50/cv_summary.json  (3 seeds, AAFNet)
  outputs/cv_aug_only/<run_id>/resnet50/cv_summary.json    (3 seeds, ResNet50+ArchAug+noise)
  outputs/cv_efficientnetv2/<run_id>/efficientnet_v2_s_tv/cv_summary.json
  outputs/cv_convnext/<run_id>/convnext_tiny_tv/cv_summary.json

Outputs:
  outputs/sig_collect_v2/significance.json
  outputs/sig_collect_v2/pvalue_matrix.png
  outputs/p1_significance_summary.md
"""
from __future__ import annotations
import json
import datetime
from pathlib import Path
from itertools import combinations
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]


# (display_name, glob_patterns_per_seed42_1337_2024, model_subdir)
MODELS = [
    ("baseline", ["cv_baseline", "cv_baseline_seed1337", "cv_baseline_seed2024"], "resnet50"),
    ("aafnet",   ["cv_aafnet_v2", "cv_aafnet_v2_seed1337", "cv_aafnet_v2_seed2024"], "resnet50"),
    ("rn50_aug_only", ["cv_aug_only", "cv_aug_only_seed1337", "cv_aug_only_seed2024"], "resnet50"),
    ("efficientnet_v2_s", ["cv_efficientnetv2", "cv_efficientnetv2_seed1337", "cv_efficientnetv2_seed2024"], "efficientnet_v2_s_tv"),
    ("convnext_tiny",     ["cv_convnext", "cv_convnext_seed1337", "cv_convnext_seed2024"], "convnext_tiny_tv"),
]


def load_per_seed(subdir: str, model_dir: str) -> list[float]:
    """Load 5 fold accuracies from one cv_summary.json."""
    p = sorted((ROOT / "outputs" / subdir).glob(f"*/{model_dir}/cv_summary.json"))
    if not p:
        # Try aggregated form: the cv_baseline+cv_aafnet_v2 single dir for seed=42 has multi-seed in folds[]
        return []
    with open(p[-1]) as f:
        cv = json.load(f)
    return [fold["test_accuracy"] for fold in cv.get("folds", []) if fold.get("status") == "ok"]


def collect_15_folds(name: str, dir_patterns: list[str], model_dir: str) -> list[float]:
    """Return list of 15 accuracies (3 seeds × 5 folds), or as many as exist."""
    # 兼容: 部分旧目录里 single seed 的 cv_summary 已经包含全 15 folds (run_cv 多 seed 时);
    # 新目录是每 seed 一个目录 5 folds
    accs: list[float] = []
    for sub in dir_patterns:
        per = load_per_seed(sub, model_dir)
        accs.extend(per)
    return accs


def holm_bonferroni(pvals: list[float], alpha: float = 0.05) -> list[bool]:
    """Holm step-down. Returns reject (True/False) per index of pvals."""
    n = len(pvals)
    order = sorted(range(n), key=lambda i: pvals[i])
    reject = [False] * n
    for rank, idx in enumerate(order):
        adj_alpha = alpha / (n - rank)
        if pvals[idx] <= adj_alpha:
            reject[idx] = True
        else:
            break
    return reject


def main():
    fold_accs: dict[str, list[float]] = {}
    for name, dirs, model_dir in MODELS:
        accs = collect_15_folds(name, dirs, model_dir)
        if accs:
            fold_accs[name] = accs
            print(f"[{name}] n={len(accs)}  mean={np.mean(accs):.4f}  std={np.std(accs):.4f}")
        else:
            print(f"[{name}] no data found")

    # 至少需要 3 个模型才能 Friedman
    n_models = len(fold_accs)
    n_folds  = min(len(v) for v in fold_accs.values()) if fold_accs else 0

    # 把所有模型截断到相同长度 (按 fold 排序)
    if n_folds >= 2:
        for k in list(fold_accs.keys()):
            fold_accs[k] = fold_accs[k][:n_folds]

    out: dict = {
        "n_models": n_models,
        "n_folds_per_model": n_folds,
        "models": list(fold_accs.keys()),
        "fold_accs": fold_accs,
        "mean": {k: float(np.mean(v)) for k, v in fold_accs.items()},
        "std":  {k: float(np.std(v))  for k, v in fold_accs.items()},
    }

    # Friedman: 要求每模型相同长度
    if n_models >= 3 and n_folds >= 3:
        try:
            stat, pval = stats.friedmanchisquare(*fold_accs.values())
            out["friedman"] = {"statistic": float(stat), "p_value": float(pval), "alpha": 0.05}
        except Exception as e:
            out["friedman"] = {"error": str(e)}
    else:
        out["friedman"] = {"note": f"requires ≥3 models with equal n_folds; got n_models={n_models}, n_folds={n_folds}"}

    # Pairwise Wilcoxon
    pairs = list(combinations(fold_accs.keys(), 2))
    raw_pvals = []
    pair_results = {}
    for a, b in pairs:
        try:
            res = stats.wilcoxon(fold_accs[a], fold_accs[b])
            pval = float(res.pvalue)
        except ValueError as e:
            pval = float("nan")
        pair_results[f"{a}__vs__{b}"] = pval
        raw_pvals.append(pval)
    out["wilcoxon_pvals"] = pair_results

    # Holm-Bonferroni on the pairwise pvals
    valid = [p for p in raw_pvals if not np.isnan(p)]
    if valid:
        reject = holm_bonferroni(raw_pvals)
        out["holm_reject"] = {k: bool(r) for k, r in zip(pair_results.keys(), reject)}

    # Mean ranks (lower = better)
    mat = np.stack([fold_accs[k] for k in fold_accs.keys()])  # [n_models, n_folds]
    ranks = -mat  # acc 高 → rank 低
    rank_vals = np.zeros_like(mat)
    for j in range(mat.shape[1]):
        rank_vals[:, j] = stats.rankdata(ranks[:, j])
    mean_rank = rank_vals.mean(axis=1)
    out["mean_rank"] = {name: float(r) for name, r in zip(fold_accs.keys(), mean_rank)}

    out_dir = ROOT / "outputs" / "sig_collect_v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "significance.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_dir / 'significance.json'}")

    # Markdown
    md = ["# P1.3 — Multi-model significance summary",
          f"_run: {datetime.datetime.now().isoformat(timespec='seconds')}_",
          "",
          f"- Models: {n_models}",
          f"- Folds per model: {n_folds}",
          ""]
    md.append("## Mean test accuracy ± std")
    md.append("| Model | Mean | Std | Mean rank (lower better) |")
    md.append("|---|---|---|---|")
    for k in fold_accs:
        md.append(f"| {k} | {out['mean'][k]*100:.2f} % | {out['std'][k]*100:.2f} % | {out['mean_rank'][k]:.2f} |")

    if "statistic" in out["friedman"]:
        f = out["friedman"]
        md.append(f"\n## Friedman omnibus\n- χ² = {f['statistic']:.3f}, p = {f['p_value']:.4f}")
    elif "note" in out["friedman"]:
        md.append(f"\n## Friedman omnibus\n_{out['friedman']['note']}_")

    md.append("\n## Pairwise Wilcoxon signed-rank (Holm-Bonferroni adjusted)")
    md.append("| Pair | p-value | Reject H₀ at α = 0.05 (Holm) |")
    md.append("|---|---|---|")
    for k in pair_results:
        p = pair_results[k]
        rj = out.get("holm_reject", {}).get(k, False)
        md.append(f"| {k} | {p:.4f} | {'**yes**' if rj else 'no'} |")

    md_path = ROOT / "outputs" / "p1_significance_summary.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"Wrote {md_path}")
    print("\n" + "\n".join(md))


if __name__ == "__main__":
    main()
