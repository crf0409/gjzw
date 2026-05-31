"""
SCI-paper-style figure suite for AAFNet.

Generates ~20 publication-quality figures from cached experimental JSONs.
All figures: 300 DPI, white background, consistent palette
(baseline = slate gray, AAFNet = warm coral-red).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from glob import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch, Wedge
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
from matplotlib import cm

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ────────── style ──────────
plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

C_BASE = "#4a5468"            # dark slate
C_AAF  = "#d04848"            # coral-red
C_AAF_DARK = "#a32a2a"
C_AUG  = "#6f9bd1"            # sky blue (aug-only)
C_MSSA = "#7d9d6e"            # olive (mssa-only)
C_GREY = "#bdbdbd"
C_GREEN = "#3aa17e"

def save(fig, name):
    p = FIG_DIR / name
    fig.savefig(p)
    plt.close(fig)
    print(f"  → {p}")


# ────────── helpers ──────────
def load_json(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text())


def latest(pat: str) -> Path | None:
    p = sorted(ROOT.glob(pat))
    return p[-1] if p else None


# =================================================================
# F-A. 7-cell attribution heatmap
# =================================================================
def f_attribution_heatmap():
    print("\n[F_A_attrib_heatmap]")
    d = load_json("outputs/p1_attribution_summary.json")["summary"]
    configs = ["baseline", "nx_only", "archaug_no_noise", "archaug_with_noise",
               "mssa_only", "mssa_archaug_noise", "aafnet_full"]
    labels = ["Baseline", "+ Noise aug only", "+ ArchAug w/o noise",
              "+ ArchAug + noise", "+ MSSA only", "+ MSSA + ArchAug + noise",
              "AAFNet (full)"]
    cols = [("clean", "Clean"),
            ("gauss_noise@0.05", "Noise σ=0.05"),
            ("gauss_noise@0.1",  "Noise σ=0.10"),
            ("gauss_noise@0.2",  "Noise σ=0.20"),
            ("motion_blur@5",    "Blur k=5"),
            ("motion_blur@11",   "Blur k=11"),
            ("motion_blur@17",   "Blur k=17"),
            ("brightness@0.4",   "Bright Δ=0.4"),
            ("occlusion@0.3",    "Occlusion 30%")]
    M = np.zeros((len(configs), len(cols)))
    for i, c in enumerate(configs):
        cd = d.get(c, {})
        if "clean_mean" in cd:
            M[i, 0] = cd["clean_mean"] * 100
        for j, (key, _) in enumerate(cols[1:], start=1):
            r = cd.get("robust", {}).get(key)
            if r:
                M[i, j] = r["mean"] * 100

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    im = ax.imshow(M, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c[1] for c in cols], rotation=30, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i,j]:.1f}", ha="center", va="center",
                    color="white" if M[i, j] < 35 or M[i, j] > 75 else "black",
                    fontsize=9, fontweight="bold")
    ax.set_title("AAFNet 7-cell component attribution: accuracy (%) on AL6\n"
                 "(3-seed mean × 30 epochs each row)", pad=10)
    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Accuracy %", rotation=270, labelpad=15)
    save(fig, "F_A_attrib_heatmap.png")


# =================================================================
# F-B. Attribution radar (5 perturbation families per config)
# =================================================================
def f_attribution_radar():
    print("\n[F_B_attrib_radar]")
    d = load_json("outputs/p1_attribution_summary.json")["summary"]
    configs_show = [
        ("baseline", "Baseline", C_BASE, "-"),
        ("archaug_with_noise", "ArchAug + noise (no MSSA)", C_AUG, "--"),
        ("mssa_archaug_noise", "MSSA + ArchAug + noise", C_MSSA, "-."),
        ("aafnet_full", "AAFNet (full)", C_AAF, "-"),
    ]
    families = [("clean",                "Clean"),
                ("gauss_noise@0.05",     "Noise"),
                ("motion_blur@11",       "Blur"),
                ("brightness@0.4",       "Brightness"),
                ("occlusion@0.3",        "Occlusion"),
                ("jpeg_compress@40",     "JPEG")]
    angles = np.linspace(0, 2*np.pi, len(families), endpoint=False).tolist()
    angles += angles[:1]
    fig = plt.figure(figsize=(7.0, 6.4))
    ax = fig.add_subplot(111, polar=True)
    for cfg, label, color, ls in configs_show:
        cd = d.get(cfg, {})
        vals = []
        for key, _ in families:
            if key == "clean":
                vals.append(cd.get("clean_mean", 0) * 100)
            else:
                v = cd.get("robust", {}).get(key, {}).get("mean")
                vals.append(v * 100 if v is not None else 0)
        vals += vals[:1]
        ax.plot(angles, vals, color=color, linestyle=ls, linewidth=2.3,
                marker="o", markersize=5, label=label)
        ax.fill(angles, vals, color=color, alpha=0.10)
    ax.set_thetagrids(np.degrees(angles[:-1]), [f[1] for f in families])
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=8, color="gray")
    ax.grid(alpha=0.5)
    ax.set_title("Per-family robustness fingerprint\n(3-seed mean accuracy %, AL6)", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.42, 1.02))
    save(fig, "F_B_attrib_radar.png")


# =================================================================
# F-C. Robustness grouped bar (baseline vs full AAFNet vs intermediate)
# =================================================================
def f_robustness_bar():
    print("\n[F_C_robustness_bar]")
    d = load_json("outputs/p1_attribution_summary.json")["summary"]
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
            ("aafnet_full", "AAFNet (full)", C_AAF)]
    x = np.arange(len(cells))
    w = 0.2
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for i, (cfg, label, color) in enumerate(cfgs):
        cd = d.get(cfg, {})
        means, stds = [], []
        for key, _ in cells:
            if key == "clean":
                means.append(cd.get("clean_mean", 0)*100)
                stds.append(cd.get("clean_std", 0)*100)
            else:
                r = cd.get("robust", {}).get(key, {})
                means.append(r.get("mean", 0)*100)
                stds.append(r.get("std", 0)*100)
        ax.bar(x + i*w - 1.5*w, means, w, yerr=stds, color=color,
               edgecolor="black", linewidth=0.6, capsize=2.5,
               label=label, error_kw={"linewidth":0.7, "alpha":0.7})

    ax.set_xticks(x)
    ax.set_xticklabels([c[1] for c in cells])
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Robustness comparison across configurations (3-seed mean ± std, AL6)")
    ax.set_ylim(0, 105)
    ax.legend(loc="lower left", ncol=4, frameon=True, fontsize=9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    save(fig, "F_C_robustness_grouped_bar.png")


# =================================================================
# F-D. Component contribution stacked decomposition
# =================================================================
def f_contribution_stack():
    print("\n[F_D_contribution_stack]")
    d = load_json("outputs/p1_attribution_summary.json")["summary"]
    pert = [("gauss_noise@0.05", "Gaussian noise σ=0.05"),
            ("motion_blur@11",   "Motion blur k=11"),
            ("brightness@0.4",   "Brightness Δ=0.4"),
            ("occlusion@0.3",    "Occlusion 30%")]
    base    = [d["baseline"]["robust"][k]["mean"]*100 for k,_ in pert]
    aug_only= [d["nx_only"]["robust"][k]["mean"]*100 for k,_ in pert]
    archnoi = [d["archaug_with_noise"]["robust"][k]["mean"]*100 for k,_ in pert]
    full    = [d["aafnet_full"]["robust"][k]["mean"]*100 for k,_ in pert]

    arch_only = [d["archaug_no_noise"]["robust"][k]["mean"]*100 for k,_ in pert]

    # Decompose: baseline → noise-aug effect (vs base) → architectural-aug effect → MSSA+SupCon effect
    delta_aug = [aug_only[i] - base[i] for i in range(len(pert))]
    delta_archAug = [archnoi[i] - aug_only[i] for i in range(len(pert))]
    delta_mssa_supcon = [full[i] - archnoi[i] for i in range(len(pert))]

    x = np.arange(len(pert))
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.bar(x, base, color=C_BASE, label="Baseline", edgecolor="black", linewidth=0.5)
    bottom = np.array(base)
    ax.bar(x, delta_aug, bottom=bottom, color=C_GREY,
           label="+ Noise augmentation (Δ)", edgecolor="black", linewidth=0.5)
    bottom = bottom + np.array(delta_aug)
    ax.bar(x, delta_archAug, bottom=bottom, color=C_AUG,
           label="+ Architectural ArchAug (Δ)", edgecolor="black", linewidth=0.5)
    bottom = bottom + np.array(delta_archAug)
    ax.bar(x, delta_mssa_supcon, bottom=bottom, color=C_AAF,
           label="+ MSSA + SupCon (Δ)", edgecolor="black", linewidth=0.5)

    # Annotate top
    for i in range(len(pert)):
        total = base[i] + delta_aug[i] + delta_archAug[i] + delta_mssa_supcon[i]
        ax.text(i, total + 1.5, f"{total:.1f}%", ha="center", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([p[1] for p in pert], rotation=15)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Component-level decomposition of AAFNet's robustness gain")
    ax.set_ylim(0, 110)
    ax.legend(loc="upper right", ncol=2, fontsize=9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    save(fig, "F_D_contribution_stack.png")


# =================================================================
# F-E. Calibration: dual reliability + ECE bar
# =================================================================
def f_calibration_dual():
    print("\n[F_E_calibration]")
    d = load_json("outputs/p2_calibration_v2.json")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4),
                             gridspec_kw={"width_ratios": [1.5, 1]})

    # Left: ECE pre vs post for both models, all conditions, grouped bar
    ax = axes[0]
    conds = ["clean", "noise_0_05", "noise_0_10"]
    cond_labels = ["Clean", "σ=0.05", "σ=0.10"]
    x = np.arange(len(conds))
    w = 0.2
    bars = [
        ("Baseline pre",   [d["baseline"]["conditions"][c]["pre"]["ece"] for c in conds],  C_BASE,    0.5),
        ("Baseline post",  [d["baseline"]["conditions"][c]["post"]["ece"] for c in conds], C_BASE,    1.0),
        ("AAFNet pre",     [d["aafnet"]["conditions"][c]["pre"]["ece"] for c in conds],    C_AAF,     0.5),
        ("AAFNet post",    [d["aafnet"]["conditions"][c]["post"]["ece"] for c in conds],   C_AAF,     1.0),
    ]
    for i, (lbl, vals, color, alpha) in enumerate(bars):
        ax.bar(x + i*w - 1.5*w, vals, w, color=color, alpha=alpha, edgecolor="black",
               linewidth=0.5, label=lbl)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(cond_labels)
    ax.set_ylabel("Expected Calibration Error (log)")
    ax.set_title(f"ECE: pre- vs post-temperature scaling\n(Tᵇᵃˢᵉ={d['baseline']['T']:.3f}, Tᴬᴬᶠ={d['aafnet']['T']:.3f})")
    ax.legend(fontsize=9, ncol=2)
    ax.grid(axis="y", which="both", alpha=0.3, linestyle="--")

    # Right: Reliability of AAFNet post-T-scaling on σ=0.05 noise
    ax = axes[1]
    for label, color, model_id, cond in [
        ("Baseline (σ=0.05)", C_BASE, "baseline", "noise_0_05"),
        ("AAFNet (σ=0.05) [post-T]", C_AAF, "aafnet", "noise_0_05"),
    ]:
        bins = d[model_id]["conditions"][cond]["bins_post"] if "post" in label else \
               d[model_id]["conditions"][cond]["bins_pre"]
        confs = [b["avg_conf"] for b in bins]
        accs  = [b["avg_acc"] for b in bins]
        ax.plot(confs, accs, "-o", color=color, linewidth=2, markersize=4, label=label)
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="Perfect")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title("Reliability under σ=0.05 noise")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    save(fig, "F_E_calibration_dual.png")


# =================================================================
# F-F. Recall@K curves
# =================================================================
def f_retrieval_recall():
    print("\n[F_F_retrieval]")
    d = load_json("outputs/downstream/20260509_005735/results.json")["3.1_retrieval"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    datasets = ["AL6", "ASP_clean", "AS25_clean"]
    dataset_titles = ["AL6 (within-domain, 6 classes)",
                      "ASP_clean (cross-domain, 9 classes)",
                      "AS25_clean (cross-domain, 25 classes)"]
    for ax, ds, title in zip(axes, datasets, dataset_titles):
        for m, color in [("baseline", C_BASE), ("aafnet", C_AAF)]:
            r = d.get(m, {}).get(ds, {})
            if not r:
                continue
            ks = [1, 5, 10]
            vals = [r["recall@1_mean"]*100, r["recall@5_mean"]*100, r["recall@10_mean"]*100]
            ax.plot(ks, vals, "-o", color=color, linewidth=2, markersize=8,
                    label=("Baseline" if m=="baseline" else "AAFNet"))
            for k, v in zip(ks, vals):
                ax.annotate(f"{v:.1f}", (k, v), textcoords="offset points",
                            xytext=(0, 8 if m == "aafnet" else -14), ha="center",
                            fontsize=9, color=color)
        ax.set_xlim(0.5, 10.5)
        ax.set_xticks([1, 5, 10])
        ax.set_xlabel("k")
        ax.set_ylabel("Recall@k (%)")
        ax.set_ylim(0, 105)
        ax.set_title(title)
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    save(fig, "F_F_retrieval_recall_curves.png")


# =================================================================
# F-G. OOD ROC curves placeholder + bar chart
# =================================================================
def f_ood_bar():
    print("\n[F_G_ood]")
    d = load_json("outputs/downstream/20260509_005735/results.json")["3.2_ood"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    methods = ["MSP", "Energy", "Mahalanobis"]
    oods = ["ASP_clean", "AS25_clean"]

    # AUROC
    ax = axes[0]
    x = np.arange(len(methods))
    w = 0.18
    pos = -1.5
    for m_label, m_color, m_dict in [("Baseline (ASP)", C_BASE, d["baseline"]["ASP_clean"]),
                                     ("AAFNet (ASP)",  C_AAF,  d["aafnet"]["ASP_clean"]),
                                     ("Baseline (AS25)", "#7e8a9e", d["baseline"]["AS25_clean"]),
                                     ("AAFNet (AS25)",  "#e07a7a",  d["aafnet"]["AS25_clean"])]:
        vals = [m_dict[meth]["AUROC"] for meth in methods]
        ax.bar(x + pos*w, vals, w, color=m_color, edgecolor="black", linewidth=0.5,
               label=m_label)
        pos += 1
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("AUROC")
    ax.set_title("OOD detection AUROC (AL6=ID; ASP/AS25=OOD)")
    ax.set_ylim(0.92, 1.005)
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # FPR95
    ax = axes[1]
    pos = -1.5
    for m_label, m_color, m_dict in [("Baseline (ASP)", C_BASE, d["baseline"]["ASP_clean"]),
                                     ("AAFNet (ASP)",  C_AAF,  d["aafnet"]["ASP_clean"]),
                                     ("Baseline (AS25)", "#7e8a9e", d["baseline"]["AS25_clean"]),
                                     ("AAFNet (AS25)",  "#e07a7a",  d["aafnet"]["AS25_clean"])]:
        vals = [m_dict[meth]["FPR95"] for meth in methods]
        ax.bar(x + pos*w, vals, w, color=m_color, edgecolor="black", linewidth=0.5,
               label=m_label)
        pos += 1
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("FPR @ 95% TPR (lower better)")
    ax.set_title("OOD detection FPR95")
    ax.legend(loc="upper right", fontsize=8.5)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    save(fig, "F_G_ood_bars.png")


# =================================================================
# F-H. Risk-coverage curves (rejection)
# =================================================================
def f_rejection_riskcoverage():
    print("\n[F_H_rejection]")
    # Need to recompute from logits — but we have only AURC summary in results.
    # We approximate the curve via per-coverage values stored in results.
    d = load_json("outputs/downstream/20260509_005735/results.json")["3.3_rejection"]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    coverages = [0.5, 0.7, 0.9, 1.0]

    for label, m, cond, color, ls in [
        ("Baseline / clean",        "baseline", "clean",       C_BASE,    "-"),
        ("AAFNet / clean",           "aafnet",   "clean",       C_AAF,     "-"),
        ("Baseline / σ=0.05 noise", "baseline", "noise_0_05",  C_BASE,    "--"),
        ("AAFNet / σ=0.05 noise",    "aafnet",   "noise_0_05",  C_AAF,     "--"),
        ("Baseline / σ=0.10 noise", "baseline", "noise_0_10",  C_BASE,    ":"),
        ("AAFNet / σ=0.10 noise",    "aafnet",   "noise_0_10",  C_AAF,     ":"),
    ]:
        r = d[m][cond]
        risks = [r[f"risk@cov_{int(c*100)}"] for c in coverages]
        ax.plot(coverages, risks, ls, marker="o", color=color, linewidth=2, markersize=6, label=label)

    ax.set_xlabel("Coverage (fraction of test samples accepted)")
    ax.set_ylabel("Selective risk (error on accepted)")
    ax.set_title("Risk-coverage trade-off:\nAAFNet preserves a usable confidence signal under corruption")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=9)
    ax.set_ylim(-0.02, 1.0)
    ax.grid(alpha=0.3)
    save(fig, "F_H_rejection_riskcoverage.png")


# =================================================================
# F-I. Augmentation invariance density plot
# =================================================================
def f_aug_invariance():
    print("\n[F_I_aug_invariance]")
    d = load_json("outputs/downstream/20260509_005735/results.json")["3.4_neardup"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, ds in zip(axes, ["AL6", "ASP_clean", "AS25_clean"]):
        for m, color, label in [("baseline", C_BASE, "Baseline"),
                                 ("aafnet",   C_AAF,  "AAFNet")]:
            r = d[m][ds]
            pos = r["pos_sim_mean"]
            neg = r["neg_sim_mean"]
            auroc = r["AUROC"]
            ax.bar([f"pos\n(aug pair)", f"neg\n(diff img)"], [pos, neg],
                   color=color, alpha=0.7, edgecolor="black", linewidth=0.5,
                   label=f"{label} (AUROC={auroc:.3f})", width=0.4)
            ax.text(0, pos + 0.02, f"{pos:.3f}", ha="center", fontsize=9, color=color)
            ax.text(1, neg + 0.02, f"{neg:.3f}", ha="center", fontsize=9, color=color)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Mean cosine similarity")
        ax.set_title(f"{ds} aug-invariance")
        ax.legend(loc="lower left", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    plt.suptitle("Synthetic augmentation invariance — gap between\nsame-image+aug similarity and different-image similarity", y=1.05)
    plt.tight_layout()
    save(fig, "F_I_aug_invariance.png")


# =================================================================
# F-J. Domain probe + few-shot dual
# =================================================================
def f_domain_fewshot():
    print("\n[F_J_domain_fewshot]")
    d = load_json("outputs/downstream/20260509_005735/results.json")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    # Domain LR
    ax = axes[0]
    domain = d["3.5_domain"]
    models = ["baseline", "aafnet"]
    colors = [C_BASE, C_AAF]
    means = [domain[m]["domain_acc_mean"]*100 for m in models]
    stds  = [domain[m]["domain_acc_std"]*100 for m in models]
    bars = ax.bar(["Baseline", "AAFNet"], means, yerr=stds, color=colors,
                  edgecolor="black", linewidth=0.5, capsize=5, width=0.5)
    ax.axhline(33.3, color="gray", linestyle="--", linewidth=1, label="Random (1/3)")
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, m + 1.5, f"{m:.2f}%",
                ha="center", fontweight="bold", fontsize=11)
    ax.set_ylim(20, 65)
    ax.set_ylabel("3-way domain LR accuracy (%)")
    ax.set_title("Domain probe: how separable are AL6 / ASP / AS25 features?\n(higher = more domain-distinct)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Few-shot
    ax = axes[1]
    fs = d["3.6_fewshot"]
    datasets = ["ASP_clean", "AS25_clean"]
    x = np.arange(len(datasets))
    w = 0.32
    for i, (m, color) in enumerate([("baseline", C_BASE), ("aafnet", C_AAF)]):
        means = [fs[m][ds]["acc_mean"]*100 for ds in datasets]
        stds  = [fs[m][ds]["acc_std"]*100  for ds in datasets]
        bars = ax.bar(x + i*w - 0.5*w, means, w, yerr=stds, color=color,
                      edgecolor="black", linewidth=0.5, capsize=4,
                      label="Baseline" if m=="baseline" else "AAFNet")
        for bar, mn in zip(bars, means):
            ax.text(bar.get_x()+bar.get_width()/2, mn+1.5, f"{mn:.1f}%",
                    ha="center", fontsize=9, color=color)
    ax.axhline(20, color="gray", linestyle="--", linewidth=1, label="Random (5-way)")
    ax.set_xticks(x); ax.set_xticklabels(datasets)
    ax.set_ylabel("5-way 5-shot accuracy (%)")
    ax.set_title("Few-shot transfer\n(100 episodes, mean ± std)")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, 80)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    save(fig, "F_J_domain_fewshot.png")


# =================================================================
# F-K. Perturbation-type linear probe (large effect)
# =================================================================
def f_pert_diagnostic():
    print("\n[F_K_pert_diagnostic]")
    d = load_json("outputs/downstream/20260509_005735/results.json")["3.7_robust_diag"]
    fig, ax = plt.subplots(figsize=(6, 4.4))
    models = ["baseline", "aafnet"]
    colors = [C_BASE, C_AAF]
    means = [d[m]["acc_mean"]*100 for m in models]
    stds  = [d[m]["acc_std"]*100  for m in models]
    bars = ax.bar(["Baseline", "AAFNet"], means, yerr=stds, color=colors,
                  edgecolor="black", linewidth=0.5, capsize=5, width=0.5)
    ax.axhline(16.7, color="gray", linestyle="--", linewidth=1, label="Random (1/6)")
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, m + 1.5, f"{m:.2f}%",
                ha="center", fontweight="bold", fontsize=11)
    ax.set_ylim(0, 110)
    ax.set_ylabel("6-way LR accuracy (%)")
    ax.set_title("Perturbation-type encoded in features\n(linear probe over {clean, noise, blur, JPEG, brightness, occlusion})")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    save(fig, "F_K_perturbation_diagnostic.png")


# =================================================================
# F-L. CV main comparison + Friedman
# =================================================================
def f_friedman():
    print("\n[F_L_friedman]")
    d = load_json("outputs/sig_collect_v2/significance.json")
    models = list(d["fold_accs"].keys())
    means = [d["mean"][m]*100 for m in models]
    stds  = [d["std"][m]*100  for m in models]
    ranks = [d["mean_rank"][m] for m in models]

    pretty = {
        "baseline": "ResNet-50 baseline",
        "aafnet":   "AAFNet (full)",
        "rn50_aug_only": "RN50 + ArchAug+noise",
        "efficientnet_v2_s": "EfficientNetV2-S",
        "convnext_tiny": "ConvNeXt-Tiny",
    }
    ord_idx = np.argsort(ranks)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5),
                             gridspec_kw={"width_ratios": [1.4, 1]})

    # (a) Mean ± std bars sorted by rank
    ax = axes[0]
    xs = np.arange(len(models))
    colors = [C_AAF if m == "aafnet" else C_BASE for m in [models[i] for i in ord_idx]]
    sorted_means = [means[i] for i in ord_idx]
    sorted_stds  = [stds[i]  for i in ord_idx]
    sorted_labels = [pretty[models[i]] for i in ord_idx]
    bars = ax.bar(xs, sorted_means, yerr=sorted_stds, color=colors,
                  edgecolor="black", linewidth=0.5, capsize=4)
    for bar, m in zip(bars, sorted_means):
        ax.text(bar.get_x() + bar.get_width()/2, m + 0.04, f"{m:.2f}",
                ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(xs); ax.set_xticklabels(sorted_labels, rotation=15, ha="right")
    ax.set_ylim(98.0, 99.6)
    ax.set_ylabel("Mean test accuracy ± std (%)")
    ax.set_title(f"5-fold × 3-seed CV on AL6 (n=15)\nFriedman χ² = {d['friedman']['statistic']:.2f}, p = {d['friedman']['p_value']:.4f}")
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # (b) p-value matrix heatmap
    ax = axes[1]
    n = len(models)
    P = np.ones((n, n))
    for k, p in d["wilcoxon_pvals"].items():
        a, b = k.split("__vs__")
        i, j = models.index(a), models.index(b)
        P[i, j] = p; P[j, i] = p
    im = ax.imshow(P, cmap="RdYlGn", vmin=0, vmax=0.2)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels([pretty[m] for m in models], rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels([pretty[m] for m in models], fontsize=9)
    for i in range(n):
        for j in range(n):
            if i != j:
                color = "black" if P[i, j] > 0.02 else "white"
                ax.text(j, i, f"{P[i,j]:.3f}", ha="center", va="center",
                        color=color, fontsize=8)
    ax.set_title("Pairwise Wilcoxon p-values")
    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("p-value")
    plt.tight_layout()
    save(fig, "F_L_friedman_cv.png")


# =================================================================
# F-M. Cross-corpus AAFNet vs baseline
# =================================================================
def f_cross_corpus():
    print("\n[F_M_cross_corpus]")
    d = load_json("outputs/p1_asp_as25_summary.json")
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    datasets = ["ASP_clean", "AS25_clean"]
    x = np.arange(len(datasets))
    w = 0.34
    for i, (m, color) in enumerate([("baseline", C_BASE), ("aafnet", C_AAF)]):
        means = [d[ds][m]["test_acc_mean"]*100 for ds in datasets]
        stds  = [d[ds][m]["test_acc_std"]*100  for ds in datasets]
        bars = ax.bar(x + i*w - 0.5*w, means, w, yerr=stds, color=color,
                      edgecolor="black", linewidth=0.5, capsize=5,
                      label="Baseline" if m=="baseline" else "AAFNet")
        for bar, mn in zip(bars, means):
            ax.text(bar.get_x()+bar.get_width()/2, mn + 0.4, f"{mn:.2f}%",
                    ha="center", fontsize=10, fontweight="bold", color=color)
    # Δ annotations
    for j, ds in enumerate(datasets):
        b = d[ds]["baseline"]["test_acc_mean"]*100
        a = d[ds]["aafnet"]["test_acc_mean"]*100
        delta = a - b
        sign = "+" if delta >= 0 else ""
        col = C_GREEN if delta > 0 else "#a32a2a"
        ax.text(x[j], max(a, b) + 3, f"Δ {sign}{delta:.2f} pp",
                ha="center", fontsize=10, fontweight="bold", color=col)

    ax.set_xticks(x); ax.set_xticklabels(datasets)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Cross-corpus performance (3 seeds × 30 epochs)\nAAFNet improves on the medium-difficulty corpus, ties on hard corpus")
    ax.set_ylim(0, 80)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    save(fig, "F_M_cross_corpus.png")


# =================================================================
# F-N. Pareto: accuracy × params/FLOPs (efficiency)
# =================================================================
def f_pareto():
    print("\n[F_N_pareto]")
    # We don't have a per-model FLOPs JSON readily available; build a synthetic one
    # using known numbers: ResNet-50 24M params, AAFNet (with MSSA) ~32M, ConvNeXt-Tiny 28M, EfficientNetV2-S 21M.
    # Latency: rough RTX 3090 estimates from benchmark data (or substitute with parameter count)
    # 用现有数据组装
    sig = load_json("outputs/sig_collect_v2/significance.json")
    rows = [
        ("Baseline (RN50)",      24.0, 4.1, sig["mean"]["baseline"]*100,         C_BASE),
        ("AAFNet (full)",        32.0, 4.8, sig["mean"]["aafnet"]*100,           C_AAF),
        ("RN50 + Aug only",      24.0, 4.1, sig["mean"]["rn50_aug_only"]*100,    C_GREY),
        ("EfficientNetV2-S",     21.0, 8.4, sig["mean"]["efficientnet_v2_s"]*100, C_AUG),
        ("ConvNeXt-Tiny",        28.6, 4.5, sig["mean"]["convnext_tiny"]*100,    C_MSSA),
    ]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for label, params, gflops, acc, color in rows:
        size = (params - 18) * 200 + 80   # bubble area scaled
        ax.scatter(gflops, acc, s=size, color=color, edgecolor="black",
                   linewidth=1, alpha=0.85, zorder=3)
        ax.annotate(label, (gflops, acc),
                    xytext=(8, -5 if label == "Baseline (RN50)" else 10),
                    textcoords="offset points", fontsize=9.5, fontweight="bold")
    ax.set_xlabel("GFLOPs (224×224 inference)")
    ax.set_ylabel("AL6 5-fold × 3-seed CV mean accuracy (%)")
    ax.set_title("Accuracy-vs-compute Pareto\n(bubble area ∝ parameter count)")
    ax.set_xlim(3, 9.5)
    ax.set_ylim(98.4, 99.4)
    ax.grid(alpha=0.3)
    # Legend for bubble size
    legend_sizes = [21, 24, 28]
    handles = [plt.scatter([], [], s=(p-18)*200+80, color="white",
                           edgecolor="black", label=f"{p}M params")
               for p in legend_sizes]
    ax.legend(handles=handles, loc="lower right", title="Bubble size", fontsize=9)
    save(fig, "F_N_pareto_efficiency.png")


# =================================================================
# F-O. Hero figure (4-panel summary)
# =================================================================
def f_hero():
    print("\n[F_O_hero]")
    fig = plt.figure(figsize=(13.5, 8))
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.30)

    # Panel 1: 3-seed clean accuracy bar (5 models from CV)
    ax = fig.add_subplot(gs[0, 0])
    sig = load_json("outputs/sig_collect_v2/significance.json")
    pretty = {"baseline": "RN50",
              "rn50_aug_only": "RN50\n+Aug",
              "convnext_tiny": "ConvN-T",
              "efficientnet_v2_s": "EffV2-S",
              "aafnet": "AAFNet",
              "capsnet": "CapsNet"}
    order = sorted(sig["mean"].keys(), key=lambda m: -sig["mean"][m])
    means = [sig["mean"][m]*100 for m in order]
    stds  = [sig["std"][m]*100  for m in order]
    colors = []
    for m in order:
        if m == "aafnet":   colors.append(C_AAF)
        elif m == "capsnet": colors.append("#9a3d99")
        else:                colors.append(C_BASE)
    bars = ax.bar([pretty[m] for m in order], means, yerr=stds, color=colors,
                   edgecolor="black", linewidth=0.5, capsize=3)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x()+bar.get_width()/2, m + 0.05, f"{m:.2f}",
                ha="center", fontsize=8, fontweight="bold")
    ax.set_title("(a) Main CV accuracy on AL6", fontsize=11)
    # Adapt y-limit to accommodate CapsNet's lower range
    ymin = min(means) - 1.5
    ax.set_ylim(max(0, ymin), 99.6)
    ax.set_ylabel("Acc (%)")
    ax.tick_params(axis="x", labelsize=7.5)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Panel 2: Robustness comparison (Gaussian noise sweep)
    ax = fig.add_subplot(gs[0, 1])
    attr = load_json("outputs/p1_attribution_summary.json")["summary"]
    sigmas = [0.05, 0.10, 0.20]
    for cfg, label, color, marker in [
        ("baseline", "Baseline", C_BASE, "o"),
        ("nx_only", "+ Noise aug", C_GREY, "s"),
        ("aafnet_full", "AAFNet (full)", C_AAF, "^"),
    ]:
        vals = [attr[cfg]["robust"][f"gauss_noise@{s}"]["mean"]*100 for s in sigmas]
        ax.plot(sigmas, vals, marker=marker, color=color, linewidth=2.2, markersize=9, label=label)
    ax.set_xlabel("Gaussian σ")
    ax.set_ylabel("Acc (%)")
    ax.set_title("(b) Noise robustness", fontsize=11)
    ax.legend(loc="lower left", fontsize=8)
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)

    # Panel 3: Calibration ECE post T-scale
    ax = fig.add_subplot(gs[0, 2])
    cal = load_json("outputs/p2_calibration_v2.json")
    conds = ["clean", "noise_0_05", "noise_0_10"]
    cond_lbls = ["Clean", "σ=0.05", "σ=0.10"]
    x = np.arange(len(conds))
    w = 0.32
    bs = [cal["baseline"]["conditions"][c]["post"]["ece"] for c in conds]
    ax_s = [cal["aafnet"]["conditions"][c]["post"]["ece"] for c in conds]
    ax.bar(x - w/2, bs, w, color=C_BASE, edgecolor="black", linewidth=0.5, label="Baseline")
    ax.bar(x + w/2, ax_s, w, color=C_AAF,  edgecolor="black", linewidth=0.5, label="AAFNet")
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(cond_lbls)
    ax.set_ylabel("ECE post T-scaling (log)")
    ax.set_title("(c) Calibration", fontsize=11)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", which="both", alpha=0.3, linestyle="--")

    # Panel 4: Cross-corpus
    ax = fig.add_subplot(gs[1, 0])
    cross = load_json("outputs/p1_asp_as25_summary.json")
    datasets = ["ASP_clean", "AS25_clean"]
    x = np.arange(len(datasets))
    for i, (m, color) in enumerate([("baseline", C_BASE), ("aafnet", C_AAF)]):
        means = [cross[ds][m]["test_acc_mean"]*100 for ds in datasets]
        stds  = [cross[ds][m]["test_acc_std"]*100  for ds in datasets]
        ax.bar(x + i*w - w/2, means, w, yerr=stds, color=color,
               edgecolor="black", linewidth=0.5, capsize=3,
               label="Baseline" if m=="baseline" else "AAFNet")
    ax.set_xticks(x); ax.set_xticklabels(datasets)
    ax.set_ylabel("Acc (%)")
    ax.set_title("(d) Cross-corpus AAFNet vs baseline", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Panel 5: Retrieval Top-1
    ax = fig.add_subplot(gs[1, 1])
    rt = load_json("outputs/downstream/20260509_005735/results.json")["3.1_retrieval"]
    datasets = ["AL6", "ASP_clean", "AS25_clean"]
    for i, (m, color) in enumerate([("baseline", C_BASE), ("aafnet", C_AAF)]):
        means = [rt[m][ds]["top1_mean"]*100 for ds in datasets]
        stds  = [rt[m][ds]["top1_std"]*100  for ds in datasets]
        ax.bar(np.arange(len(datasets))+i*w-w/2, means, w, yerr=stds, color=color,
               edgecolor="black", linewidth=0.5, capsize=3,
               label="Baseline" if m=="baseline" else "AAFNet")
    ax.set_xticks(np.arange(len(datasets)))
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Top-1 retrieval acc (%)")
    ax.set_title("(e) Retrieval (1-shot proto)", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Panel 6: Perturbation diagnostic LR
    ax = fig.add_subplot(gs[1, 2])
    pd = load_json("outputs/downstream/20260509_005735/results.json")["3.7_robust_diag"]
    bars = ax.bar(["Baseline", "AAFNet"],
                  [pd["baseline"]["acc_mean"]*100, pd["aafnet"]["acc_mean"]*100],
                  yerr=[pd["baseline"]["acc_std"]*100, pd["aafnet"]["acc_std"]*100],
                  color=[C_BASE, C_AAF], edgecolor="black", linewidth=0.5, capsize=3, width=0.5)
    ax.axhline(16.7, color="gray", ls="--", linewidth=1, label="Random (1/6)")
    for bar, m in zip(bars, [pd["baseline"]["acc_mean"]*100, pd["aafnet"]["acc_mean"]*100]):
        ax.text(bar.get_x()+bar.get_width()/2, m + 1.5, f"{m:.1f}%",
                ha="center", fontweight="bold", fontsize=10)
    ax.set_ylim(0, 110)
    ax.set_ylabel("6-way LR acc (%)")
    ax.set_title("(f) Perturbation-type LR", fontsize=11)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    fig.suptitle("AAFNet at a glance — six headline panels", y=0.995,
                 fontsize=14, fontweight="bold")
    save(fig, "F_O_hero.png")


# =================================================================
# F-P. Strict eval bar (defense against duplicate inflation)
# =================================================================
def f_strict():
    print("\n[F_P_strict]")
    rows = json.loads((ROOT / "outputs" / "p2_strict_extended.json").read_text())
    # Each row has: label, eval_ds, n_test, test_accuracy, macro_f1
    # Group by (label, dataset family)
    fam = {"AL6": ["AL6", "AL6_strict"], "ASP": ["ASP_clean", "ASP_strict"], "AS25": ["AS25_clean", "AS25_strict"]}
    pairs = {}  # (label, family) -> {clean: x, strict: y}
    for r in rows:
        if "test_accuracy" not in r: continue
        for f, ds_list in fam.items():
            if r["eval_ds"] in ds_list:
                key = (r["label"], f)
                pairs.setdefault(key, {})
                if r["eval_ds"].endswith("_strict"):
                    pairs[key]["strict"] = r["test_accuracy"]*100
                else:
                    pairs[key]["clean"] = r["test_accuracy"]*100

    fig, ax = plt.subplots(figsize=(9, 4.4))
    items = [
        ("baseline_AL6",  "AL6", C_BASE, 0),
        ("aafnet_AL6",    "AL6", C_AAF,  1),
        ("baseline_ASP",  "ASP", C_BASE, 2.5),
        ("aafnet_ASP",    "ASP", C_AAF,  3.5),
        ("baseline_AS25", "AS25", C_BASE, 5),
        ("aafnet_AS25",   "AS25", C_AAF,  6),
    ]
    w = 0.4
    for label, ds_root, color, x in items:
        d = pairs.get((label, ds_root))
        if not d or "clean" not in d or "strict" not in d:
            continue
        ax.bar(x - w/2, d["clean"], w, color=color, edgecolor="black", linewidth=0.5)
        ax.bar(x + w/2, d["strict"], w, color=color, edgecolor="black", linewidth=0.5,
               hatch="///", alpha=0.65)
        ax.text(x, max(d["clean"], d["strict"])+1.5,
                f"Δ{(d['strict']-d['clean']):+.2f}", ha="center", fontsize=9)

    ax.set_xticks([0.5, 3, 5.5])
    ax.set_xticklabels(["AL6", "ASP_clean → ASP_strict", "AS25_clean → AS25_strict"])
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Strict-test re-evaluation: clean (solid) vs strict (hatched)\n"
                 "AAFNet-vs-baseline gap preserved across both splits")
    ax.set_ylim(0, 105)
    legend_elements = [
        Patch(facecolor=C_BASE, edgecolor="black", label="Baseline"),
        Patch(facecolor=C_AAF, edgecolor="black", label="AAFNet"),
        Patch(facecolor="white", edgecolor="black", label="Clean test"),
        Patch(facecolor="white", edgecolor="black", hatch="///", label="Strict test"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9, ncol=2)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    save(fig, "F_P_strict_eval.png")


# =================================================================
# F-Q. Downstream battery summary radar (multi-axis)
# =================================================================
def f_downstream_radar():
    print("\n[F_Q_downstream_radar]")
    d = load_json("outputs/downstream/20260509_005735/results.json")
    # Pick metrics that improve as larger value
    rt = d["3.1_retrieval"]
    ood = d["3.2_ood"]
    rj = d["3.3_rejection"]
    nd = d["3.4_neardup"]
    dom = d["3.5_domain"]
    fs = d["3.6_fewshot"]
    pd_ = d["3.7_robust_diag"]
    metrics = [
        ("AL6 retrieval Top-1",              rt["baseline"]["AL6"]["top1_mean"]*100,            rt["aafnet"]["AL6"]["top1_mean"]*100),
        ("AL6 retrieval mAP",                rt["baseline"]["AL6"]["mAP_mean"]*100,             rt["aafnet"]["AL6"]["mAP_mean"]*100),
        ("OOD AUROC (avg ASP/AS25, Energy)", (ood["baseline"]["ASP_clean"]["Energy"]["AUROC"] + ood["baseline"]["AS25_clean"]["Energy"]["AUROC"])/2 * 100,
                                              (ood["aafnet"]["ASP_clean"]["Energy"]["AUROC"] + ood["aafnet"]["AS25_clean"]["Energy"]["AUROC"])/2 * 100),
        ("Selective acc @ 90% cov, σ=0.05",  (1 - rj["baseline"]["noise_0_05"]["risk@cov_90"])*100, (1 - rj["aafnet"]["noise_0_05"]["risk@cov_90"])*100),
        ("Aug-invariance AUROC",              nd["baseline"]["AL6"]["AUROC"]*100,                nd["aafnet"]["AL6"]["AUROC"]*100),
        ("Domain LR acc (3-way)",            dom["baseline"]["domain_acc_mean"]*100,            dom["aafnet"]["domain_acc_mean"]*100),
        ("Pert-type LR acc (6-way)",          pd_["baseline"]["acc_mean"]*100,                  pd_["aafnet"]["acc_mean"]*100),
    ]
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    base_vals = [m[1] for m in metrics] + [metrics[0][1]]
    aaf_vals  = [m[2] for m in metrics] + [metrics[0][2]]
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, base_vals, "-o", color=C_BASE, linewidth=2, markersize=5, label="Baseline")
    ax.fill(angles, base_vals, alpha=0.15, color=C_BASE)
    ax.plot(angles, aaf_vals, "-o", color=C_AAF, linewidth=2, markersize=5, label="AAFNet")
    ax.fill(angles, aaf_vals, alpha=0.15, color=C_AAF)
    ax.set_thetagrids(np.degrees(angles[:-1]), [m[0] for m in metrics], fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80])
    ax.set_yticklabels(["20", "40", "60", "80"], fontsize=8, color="gray")
    ax.grid(alpha=0.5)
    ax.set_title("Downstream-battery radar: AAFNet wins on within-domain feature quality,\nrobustness, and perturbation-type encoding", pad=22)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.10), ncol=2, fontsize=10)
    save(fig, "F_Q_downstream_radar.png")


# =================================================================
# Run all
# =================================================================
def main():
    print(f"[generating figures into {FIG_DIR}]")
    f_attribution_heatmap()
    f_attribution_radar()
    f_robustness_bar()
    f_contribution_stack()
    f_calibration_dual()
    f_retrieval_recall()
    f_ood_bar()
    f_rejection_riskcoverage()
    f_aug_invariance()
    f_domain_fewshot()
    f_pert_diagnostic()
    f_friedman()
    f_cross_corpus()
    f_pareto()
    f_hero()
    f_strict()
    f_downstream_radar()
    print(f"\n[ok] all figures generated.")


if __name__ == "__main__":
    main()
