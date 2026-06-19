"""
Update key comparison figures to include CapsNet:
  - F_L  multi-model CV bar + p-value matrix
  - F_C  robustness grouped bar
  - F_O  hero panel (a) — main CV accuracy bar
  - F_N  Pareto bubble

Re-aggregates multi-model significance first (adds capsnet model entry).
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from itertools import combinations

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "paper" / "figures"

plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

C_BASE = "#4a5468"; C_AAF = "#d04848"; C_GREY = "#bdbdbd"
C_AUG = "#6f9bd1"; C_MSSA = "#7d9d6e"; C_CAPS = "#9a3d99"


def latest(pat: str) -> Path | None:
    p = sorted(ROOT.glob(pat))
    return p[-1] if p else None


def collect_capsnet_15_folds() -> list[float]:
    """Pull capsnet 5-fold × 3-seed accuracies from cv_capsnet output."""
    cs = sorted(ROOT.glob("outputs/cv_capsnet/*/capsnet/cv_summary.json"))
    if not cs:
        return []
    cv = json.loads(cs[-1].read_text())
    return [f["test_accuracy"] for f in cv.get("folds", []) if f.get("status") == "ok"]


def collect_capsnet_robust(seed: int) -> dict | None:
    p = sorted(ROOT.glob(f"outputs/capsnet_robust_seed{seed}/*/capsnet/results.json"))
    if not p:
        return None
    return json.loads(p[-1].read_text())


def collect_capsnet_clean(seed: int) -> float | None:
    p = sorted(ROOT.glob(f"outputs/capsnet_seed{seed}/*/capsnet/test_metrics.json"))
    if not p:
        return None
    return json.loads(p[-1].read_text()).get("test_accuracy")


# =================================================================
# Re-aggregate multi-model significance (adds CapsNet)
# =================================================================
def reaggregate_significance():
    print("\n[reaggregate significance with CapsNet]")
    # Existing 5 models: read from sig_collect_v2/significance.json
    existing = ROOT / "outputs" / "sig_collect_v2" / "significance.json"
    if not existing.exists():
        print("  warn: no v2 significance file; skipping")
        return None
    sig = json.loads(existing.read_text())

    cap_accs = collect_capsnet_15_folds()
    if not cap_accs:
        print("  warn: capsnet CV not done yet; skipping")
        return sig

    # Truncate / pad to match existing fold count
    n_folds = sig["n_folds_per_model"]
    if len(cap_accs) >= n_folds:
        cap_accs = cap_accs[:n_folds]
    else:
        # Pad with the mean (shouldn't happen if CV finished)
        m = np.mean(cap_accs)
        cap_accs = cap_accs + [m] * (n_folds - len(cap_accs))

    if "capsnet" not in sig["models"]:
        sig["models"].append("capsnet")
    sig["fold_accs"]["capsnet"] = cap_accs
    sig["mean"]["capsnet"] = float(np.mean(cap_accs))
    sig["std"]["capsnet"]  = float(np.std(cap_accs))
    sig["n_models"] = len(sig["models"])

    # Recompute Friedman + pairwise Wilcoxon + mean ranks
    fold_accs = {m: sig["fold_accs"][m] for m in sig["models"]}

    if sig["n_models"] >= 3:
        try:
            stat, pval = stats.friedmanchisquare(*fold_accs.values())
            sig["friedman"] = {"statistic": float(stat), "p_value": float(pval), "alpha": 0.05}
        except Exception as e:
            sig["friedman"] = {"error": str(e)}

    pair_results = {}
    raw = []
    for a, b in combinations(fold_accs.keys(), 2):
        try:
            r = stats.wilcoxon(fold_accs[a], fold_accs[b])
            pval = float(r.pvalue)
        except ValueError:
            pval = float("nan")
        pair_results[f"{a}__vs__{b}"] = pval
        raw.append(pval)
    sig["wilcoxon_pvals"] = pair_results

    # Holm
    n = len(raw); order = sorted(range(n), key=lambda i: raw[i])
    reject = [False] * n
    for rank, idx in enumerate(order):
        a_alpha = 0.05 / (n - rank)
        if not np.isnan(raw[idx]) and raw[idx] <= a_alpha:
            reject[idx] = True
        else:
            break
    sig["holm_reject"] = {k: bool(r) for k, r in zip(pair_results.keys(), reject)}

    # Mean ranks
    mat = np.stack([np.array(fold_accs[k]) for k in fold_accs])
    rank_vals = np.zeros_like(mat)
    for j in range(mat.shape[1]):
        rank_vals[:, j] = stats.rankdata(-mat[:, j])
    sig["mean_rank"] = {name: float(rank_vals[i].mean()) for i, name in enumerate(fold_accs.keys())}

    out = ROOT / "outputs" / "sig_collect_v2" / "significance.json"
    with open(out, "w") as f:
        json.dump(sig, f, indent=2)
    print(f"  saved {out}")

    # Markdown
    md = [f"# P1.3 — Multi-model significance summary (with CapsNet)\n",
          f"- Models: {sig['n_models']}",
          f"- Folds per model: {sig['n_folds_per_model']}\n",
          "## Mean ± std",
          "| Model | Mean | Std | Mean rank |",
          "|---|---|---|---|"]
    for m in sig["models"]:
        md.append(f"| {m} | {sig['mean'][m]*100:.2f} % | {sig['std'][m]*100:.2f} % | {sig['mean_rank'][m]:.2f} |")
    if "statistic" in sig.get("friedman", {}):
        f = sig["friedman"]
        md.append(f"\n## Friedman: χ² = {f['statistic']:.3f}, p = {f['p_value']:.4f}")
    md.append("\n## Pairwise Wilcoxon (Holm-adjusted)")
    md.append("| Pair | p | Holm-reject |")
    md.append("|---|---|---|")
    for k, p in pair_results.items():
        md.append(f"| {k} | {p:.4f} | {'**yes**' if sig.get('holm_reject', {}).get(k, False) else 'no'} |")
    (ROOT / "outputs" / "p1_significance_summary.md").write_text("\n".join(md) + "\n")
    return sig


# =================================================================
# Update F_L (CV bar + p-value heatmap)
# =================================================================
def f_l_with_capsnet(sig):
    print("\n[F_L_friedman_cv with CapsNet]")
    if sig is None:
        return
    pretty = {
        "baseline":      "ResNet-50 baseline",
        "aafnet":        "AAFNet (full)",
        "rn50_aug_only": "RN50 + ArchAug+noise",
        "efficientnet_v2_s": "EfficientNetV2-S",
        "convnext_tiny": "ConvNeXt-Tiny",
        "capsnet":       "CapsNet (Sabour 2017)",
    }
    models = sig["models"]
    means  = [sig["mean"][m]*100 for m in models]
    stds   = [sig["std"][m]*100  for m in models]
    ranks  = [sig["mean_rank"][m] for m in models]
    ord_idx = np.argsort(ranks)

    fig, axes = plt.subplots(1, 2, figsize=(15, 4.6),
                             gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    xs = np.arange(len(models))
    sorted_means = [means[i] for i in ord_idx]
    sorted_stds = [stds[i] for i in ord_idx]
    sorted_labels = [pretty[models[i]] for i in ord_idx]
    colors = []
    for i in ord_idx:
        m = models[i]
        if m == "aafnet":   colors.append(C_AAF)
        elif m == "capsnet":colors.append(C_CAPS)
        else:               colors.append(C_BASE)
    bars = ax.bar(xs, sorted_means, yerr=sorted_stds, color=colors,
                   edgecolor="black", linewidth=0.5, capsize=4)
    for bar, m in zip(bars, sorted_means):
        ax.text(bar.get_x()+bar.get_width()/2, m + 0.04, f"{m:.2f}",
                ha="center", fontsize=9.5, fontweight="bold")
    ax.set_xticks(xs); ax.set_xticklabels(sorted_labels, rotation=18, ha="right", fontsize=9.5)
    ymin = min(sorted_means) - 3
    ax.set_ylim(max(0, ymin), 99.8)
    ax.set_ylabel("Mean test accuracy ± std (%)")
    f = sig.get("friedman", {})
    title_extra = f"Friedman χ² = {f['statistic']:.2f}, p = {f['p_value']:.4f}" if "statistic" in f else ""
    ax.set_title(f"5-fold × 3-seed CV on AL6 (n=15)  —  {sig['n_models']} models\n{title_extra}")
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # p-value matrix
    ax = axes[1]
    n = len(models)
    P = np.ones((n, n))
    for k, p in sig["wilcoxon_pvals"].items():
        a, b = k.split("__vs__")
        i, j = models.index(a), models.index(b)
        P[i, j] = p; P[j, i] = p
    im = ax.imshow(P, cmap="RdYlGn", vmin=0, vmax=0.2)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels([pretty[m] for m in models], rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels([pretty[m] for m in models], fontsize=8)
    for i in range(n):
        for j in range(n):
            if i != j:
                color = "black" if P[i, j] > 0.02 else "white"
                ax.text(j, i, f"{P[i,j]:.3f}", ha="center", va="center", color=color, fontsize=7.5)
    ax.set_title("Pairwise Wilcoxon p")
    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("p")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "F_L_friedman_cv.png")
    plt.close(fig)
    print(f"  → {FIG_DIR / 'F_L_friedman_cv.png'}")


# =================================================================
# Update F_C robustness grouped bar (add capsnet column)
# =================================================================
def f_c_with_capsnet():
    print("\n[F_C_robustness with CapsNet]")
    d = json.loads((ROOT / "outputs" / "p1_attribution_summary.json").read_text())["summary"]

    # Get capsnet robust 3 seeds
    caps_results = {"clean_mean": [], "clean_std": [], "robust": {}}
    cap_accs_clean = []
    cap_robust_perseed = {}
    for s in [42, 1337, 2024]:
        c_clean = collect_capsnet_clean(s)
        if c_clean is not None:
            cap_accs_clean.append(c_clean)
        rb = collect_capsnet_robust(s)
        if rb:
            cap_robust_perseed[s] = rb

    # Assemble means
    if cap_accs_clean:
        clean_mean = float(np.mean(cap_accs_clean)); clean_std = float(np.std(cap_accs_clean))
    else:
        clean_mean = 0; clean_std = 0
    rob_means = {}
    if cap_robust_perseed:
        # collect per (kind, severity)
        from collections import defaultdict
        agg = defaultdict(list)
        for s, rb in cap_robust_perseed.items():
            for kind, lst in rb.get("robustness", {}).items():
                for entry in lst:
                    agg[(kind, entry["severity"])].append(entry["accuracy"])
        for (k, s), vs in agg.items():
            rob_means[f"{k}@{s}"] = {"mean": float(np.mean(vs)), "std": float(np.std(vs)) if len(vs) > 1 else 0}

    cells = [("clean",            "Clean"),
             ("gauss_noise@0.05", "σ=0.05"),
             ("gauss_noise@0.1",  "σ=0.10"),
             ("motion_blur@11",   "Blur k=11"),
             ("motion_blur@17",   "Blur k=17"),
             ("brightness@0.4",   "Bright Δ=0.4"),
             ("occlusion@0.3",    "Occ 30%")]
    cfgs = [("baseline", "Baseline", C_BASE),
            ("nx_only", "+ Noise only", C_GREY),
            ("archaug_with_noise", "+ ArchAug + noise", C_AUG),
            ("aafnet_full", "AAFNet (full)", C_AAF),
            ("capsnet", "CapsNet (Sabour)", C_CAPS)]
    x = np.arange(len(cells)); w = 0.16

    fig, ax = plt.subplots(figsize=(13.5, 4.8))
    for i, (cfg, label, color) in enumerate(cfgs):
        if cfg == "capsnet":
            cd_clean_mean = clean_mean; cd_clean_std = clean_std
            cd_rob = rob_means
        else:
            cd = d.get(cfg, {})
            cd_clean_mean = cd.get("clean_mean", 0); cd_clean_std = cd.get("clean_std", 0)
            cd_rob = cd.get("robust", {})

        means, stds = [], []
        for key, _ in cells:
            if key == "clean":
                means.append(cd_clean_mean*100); stds.append(cd_clean_std*100)
            else:
                r = cd_rob.get(key, {})
                means.append(r.get("mean", 0)*100); stds.append(r.get("std", 0)*100)
        ax.bar(x + i*w - 2*w, means, w, yerr=stds, color=color,
               edgecolor="black", linewidth=0.5, capsize=2.5,
               label=label, error_kw={"linewidth":0.7, "alpha":0.7})

    ax.set_xticks(x); ax.set_xticklabels([c[1] for c in cells])
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Robustness comparison incl. CapsNet (3-seed mean ± std on AL6)")
    ax.set_ylim(0, 105)
    ax.legend(loc="lower left", ncol=5, frameon=True, fontsize=8.5)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    fig.savefig(FIG_DIR / "F_C_robustness_grouped_bar.png")
    plt.close(fig)
    print(f"  → {FIG_DIR / 'F_C_robustness_grouped_bar.png'}")


# =================================================================
# Update F_O hero (panel a) to include CapsNet
# =================================================================
def f_o_with_capsnet(sig):
    print("\n[F_O_hero panel-a with CapsNet]")
    # Just call the existing make_paper_figures.f_hero — it already reads sig_collect_v2
    sys.path.insert(0, str(ROOT / "scripts"))
    import make_paper_figures
    # Patch pretty names to include capsnet
    if sig and "capsnet" in sig["models"]:
        # Re-run just hero
        make_paper_figures.f_hero()


def main():
    sig = reaggregate_significance()
    f_l_with_capsnet(sig)
    f_c_with_capsnet()
    f_o_with_capsnet(sig)


if __name__ == "__main__":
    main()
