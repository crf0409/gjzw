#!/usr/bin/env python
"""Redraw the curated manuscript figures in a Nature-style Python workflow.

The script reads only cached experiment outputs and existing interpretability
assets. It overwrites the PNG paths referenced by the manuscript builder and
also writes SVG/PDF companions plus source-data CSV files for auditability.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "paper" / "figures"
DIAGRAM_DIR = FIG_DIR / "diagrams"
EXPORT_DIR = FIG_DIR / "nature_exports"
SOURCE_DIR = FIG_DIR / "nature_source_data"
BACKUP_DIR = FIG_DIR / "_pre_nature_backup"
QA_PATH = FIG_DIR / "nature_redraw_qa.md"

for directory in (DIAGRAM_DIR, EXPORT_DIR, SOURCE_DIR, BACKUP_DIR):
    directory.mkdir(parents=True, exist_ok=True)


# Mandatory editable-text settings from the Nature figure workflow.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"

mpl.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 7,
        "axes.titlesize": 7,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.65,
        "xtick.major.width": 0.55,
        "ytick.major.width": 0.55,
        "xtick.major.size": 2.4,
        "ytick.major.size": 2.4,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
    }
)

MM = 1 / 25.4
DOUBLE_W = 183 * MM
SINGLE_W = 89 * MM
MAX_H = 170 * MM

COL = {
    "baseline": "#3A3A3A",
    "aafnet": "#E69F00",
    "aug": "#56B4E9",
    "mssa": "#CC79A7",
    "teal": "#009E73",
    "blue": "#0072B2",
    "purple": "#7851A9",
    "neutral": "#8A8A8A",
    "light": "#D9D9D9",
    "pale": "#F3F3F3",
    "ink": "#1F1F1F",
}

MODEL_LABELS = {
    "baseline": "RN50",
    "aafnet": "AAFNet",
    "rn50_aug_only": "RN50+Aug",
    "efficientnet_v2_s": "EffNetV2-S",
    "convnext_tiny": "ConvNeXt-T",
    "capsnet": "CapsNet",
}

MODEL_COLORS = {
    "baseline": COL["baseline"],
    "aafnet": COL["aafnet"],
    "rn50_aug_only": COL["aug"],
    "efficientnet_v2_s": COL["teal"],
    "convnext_tiny": COL["blue"],
    "capsnet": COL["purple"],
}

TARGETS = [
    "F_O_hero.png",
    "F_P_strict_eval.png",
    "F_N_pareto_efficiency.png",
    "F_A_attrib_heatmap.png",
    "F_C_robustness_grouped_bar.png",
    "F_AA_rotation_polar.png",
    "F_M_cross_corpus.png",
    "F_L_friedman_cv.png",
    "F_E_calibration_dual.png",
    "F_Q_downstream_radar.png",
    "F_R_gradcam_comparison.png",
    "diagrams/D1_aafnet_arch.png",
]


def load_json(rel_path: str):
    return json.loads((ROOT / rel_path).read_text())


def pct(value: float) -> float:
    return float(value) * 100.0


def backup_targets() -> None:
    for rel in TARGETS:
        src = FIG_DIR / rel
        if not src.exists():
            continue
        dst = BACKUP_DIR / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(src, dst)


def write_csv(name: str, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path = SOURCE_DIR / name
    keys: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def panel_label(ax, label: str, x: float = -0.13, y: float = 1.04, color: str = "black") -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        fontweight="bold",
        color=color,
    )


def polish_axes(ax, grid: bool = False) -> None:
    ax.tick_params(direction="out", pad=2)
    if grid:
        ax.yaxis.grid(True, color="#E6E6E6", linewidth=0.45)
        ax.set_axisbelow(True)


def save_figure(fig, rel_path: str, close: bool = True) -> None:
    out_png = FIG_DIR / rel_path
    out_png.parent.mkdir(parents=True, exist_ok=True)
    stem = rel_path.replace("/", "__").replace(".png", "")
    out_svg = EXPORT_DIR / f"{stem}.svg"
    out_pdf = EXPORT_DIR / f"{stem}.pdf"
    fig.savefig(out_png, dpi=600, bbox_inches="tight", pad_inches=0.025)
    fig.savefig(out_svg, bbox_inches="tight", pad_inches=0.025)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.025)
    if close:
        plt.close(fig)
    print(f"  -> {out_png}")


def add_direct_label(ax, x, y, text, color, dx=3, dy=0, **kwargs) -> None:
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=6,
        color=COL["ink"],
        arrowprops=dict(arrowstyle="-", lw=0.45, color=color, shrinkA=0, shrinkB=0),
        **kwargs,
    )


def f_hero(sig, attr, cal, cross, downstream, rotation) -> None:
    fig = plt.figure(figsize=(DOUBLE_W, 156 * MM), constrained_layout=False)
    gs = fig.add_gridspec(2, 3, left=0.055, right=0.985, top=0.965, bottom=0.085, wspace=0.42, hspace=0.62)

    # a. CV accuracy
    ax = fig.add_subplot(gs[0, 0])
    order = sorted(sig["mean"], key=lambda m: sig["mean"][m], reverse=True)
    x = np.arange(len(order))
    means = np.array([pct(sig["mean"][m]) for m in order])
    stds = np.array([pct(sig["std"][m]) for m in order])
    ax.bar(x, means, yerr=stds, color=[MODEL_COLORS[m] for m in order], edgecolor="black", linewidth=0.45, capsize=1.6)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in order], rotation=35, ha="right")
    ax.set_ylim(88, 100)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("CV accuracy")
    panel_label(ax, "a")
    polish_axes(ax, grid=True)

    # b. Noise robustness
    ax = fig.add_subplot(gs[0, 1])
    sigmas = np.array([0.05, 0.10, 0.20])
    series = [
        ("baseline", "RN50", COL["baseline"], "o"),
        ("nx_only", "Noise aug.", COL["aug"], "s"),
        ("aafnet_full", "AAFNet", COL["aafnet"], "^"),
    ]
    for cfg, label, color, marker in series:
        vals = [pct(attr[cfg]["robust"][f"gauss_noise@{s}"]["mean"]) for s in sigmas]
        ax.plot(sigmas, vals, color=color, marker=marker, lw=1.4, ms=3.2, label=label)
    ax.set_xlabel("Gaussian noise sigma")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 102)
    ax.set_title("Noise robustness")
    ax.legend(loc="center right", handlelength=1.5)
    panel_label(ax, "b")
    polish_axes(ax, grid=True)

    # c. Rotation summary
    ax = fig.add_subplot(gs[0, 2])
    rot_rows = ["baseline", "aafnet"]
    rot_means = [pct(rotation["summary"][m]["mean"]) for m in rot_rows]
    rot_mins = [pct(rotation["summary"][m]["min"]) for m in rot_rows]
    x = np.arange(2)
    ax.bar(x, rot_means, color=[COL["baseline"], COL["aafnet"]], edgecolor="black", linewidth=0.45, width=0.55)
    ax.scatter(x, rot_mins, marker="_", s=110, color="black", linewidths=1.0)
    for i, (mean, minimum) in enumerate(zip(rot_means, rot_mins)):
        ax.plot([i, i], [minimum, mean], color="black", lw=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(["RN50", "AAFNet"])
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(75, 101)
    ax.set_title("Rotation sweep\nmean bars, min ticks")
    panel_label(ax, "c")
    polish_axes(ax, grid=True)

    # d. Cross-corpus accuracy
    ax = fig.add_subplot(gs[1, 0])
    datasets = ["ASP_clean", "AS25_clean"]
    x = np.arange(len(datasets))
    width = 0.32
    for i, (model, label, color) in enumerate([("baseline", "RN50", COL["baseline"]), ("aafnet", "AAFNet", COL["aafnet"])]):
        vals = [pct(cross[ds][model]["test_acc_mean"]) for ds in datasets]
        stds = [pct(cross[ds][model]["test_acc_std"]) for ds in datasets]
        ax.bar(x + (i - 0.5) * width, vals, width, yerr=stds, color=color, edgecolor="black", linewidth=0.45, capsize=1.5, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(["ASP", "AS25"])
    ax.set_ylim(60, 76)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Cross-corpus")
    ax.legend(loc="upper right")
    panel_label(ax, "d")
    polish_axes(ax, grid=True)

    # e. Calibration ECE after temperature scaling
    ax = fig.add_subplot(gs[1, 1])
    conds = ["clean", "noise_0_05", "noise_0_10"]
    labels = ["Clean", "0.05", "0.10"]
    x = np.arange(len(conds))
    for i, (model, label, color) in enumerate([("baseline", "RN50", COL["baseline"]), ("aafnet", "AAFNet", COL["aafnet"])]):
        vals = [cal[model]["conditions"][c]["post"]["ece"] for c in conds]
        ax.bar(x + (i - 0.5) * width, vals, width, color=color, edgecolor="black", linewidth=0.45, label=label)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Noise sigma")
    ax.set_ylabel("ECE")
    ax.set_title("Post-T calibration")
    ax.legend(loc="upper left")
    panel_label(ax, "e")
    polish_axes(ax, grid=False)

    # f. Downstream probes
    ax = fig.add_subplot(gs[1, 2])
    rt = downstream["3.1_retrieval"]
    ood = downstream["3.2_ood"]
    rj = downstream["3.3_rejection"]
    pd = downstream["3.7_robust_diag"]
    metrics = [
        ("Retr. Top-1", pct(rt["baseline"]["AL6"]["top1_mean"]), pct(rt["aafnet"]["AL6"]["top1_mean"])),
        ("OOD AUROC", pct((ood["baseline"]["ASP_clean"]["Energy"]["AUROC"] + ood["baseline"]["AS25_clean"]["Energy"]["AUROC"]) / 2), pct((ood["aafnet"]["ASP_clean"]["Energy"]["AUROC"] + ood["aafnet"]["AS25_clean"]["Energy"]["AUROC"]) / 2)),
        ("Sel. acc.", pct(1 - rj["baseline"]["noise_0_05"]["risk@cov_90"]), pct(1 - rj["aafnet"]["noise_0_05"]["risk@cov_90"])),
        ("Pert. probe", pct(pd["baseline"]["acc_mean"]), pct(pd["aafnet"]["acc_mean"])),
    ]
    y = np.arange(len(metrics))
    base = [m[1] for m in metrics]
    ours = [m[2] for m in metrics]
    ax.hlines(y, base, ours, color="#BDBDBD", lw=1.0)
    ax.scatter(base, y, color=COL["baseline"], marker="o", s=18, label="RN50", zorder=3)
    ax.scatter(ours, y, color=COL["aafnet"], marker="^", s=22, label="AAFNet", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([m[0] for m in metrics])
    ax.set_xlim(0, 103)
    ax.set_xlabel("Score (%)")
    ax.set_title("Feature-quality probes")
    ax.legend(loc="lower right")
    panel_label(ax, "f")
    polish_axes(ax, grid=True)

    rows = []
    for model in sig["mean"]:
        rows.append({"panel": "a", "model": model, "mean_accuracy_pct": pct(sig["mean"][model]), "std_pct": pct(sig["std"][model]), "n": sig["n_folds_per_model"]})
    for cfg, label, _, _ in series:
        for sigma in sigmas:
            rows.append({"panel": "b", "model": cfg, "condition": f"gauss_noise@{sigma}", "accuracy_pct": pct(attr[cfg]["robust"][f"gauss_noise@{sigma}"]["mean"])})
    write_csv("F_O_hero_source.csv", rows)
    save_figure(fig, "F_O_hero.png")


def draw_box(ax, xy, width, height, text, fill, edge, fontsize=6, lw=0.7) -> None:
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.014",
        facecolor=fill,
        edgecolor=edge,
        linewidth=lw,
    )
    ax.add_patch(box)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=fontsize, color=COL["ink"], linespacing=1.25)


def draw_arrow(ax, start, end, color="#333333", lw=0.75, style="->", rad=0.0) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=8,
        color=color,
        linewidth=lw,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arrow)


def f_architecture() -> None:
    fig, ax = plt.subplots(figsize=(DOUBLE_W, 92 * MM))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.02, 0.955, "AAFNet architecture", fontsize=7, fontweight="bold", ha="left", va="top", color=COL["ink"])
    ax.text(
        0.02,
        0.925,
        "Forward path: multi-scale features -> MSSA -> gated fusion -> dual heads. Training adds focal/label-smoothing, supervised contrastive and distillation losses.",
        fontsize=5.7,
        ha="left",
        va="top",
        color="#555555",
    )

    # Clean section bands.
    ax.add_patch(Rectangle((0.02, 0.48), 0.96, 0.38, facecolor="#FAFAFA", edgecolor="#D5D5D5", linewidth=0.45))
    ax.add_patch(Rectangle((0.18, 0.11), 0.64, 0.24, facecolor="#FAFAFA", edgecolor="#D5D5D5", linewidth=0.45))
    ax.text(0.035, 0.84, "Inference path", fontsize=6, fontweight="bold", ha="left", va="top")
    ax.text(0.195, 0.325, "Training objectives", fontsize=6, fontweight="bold", ha="left", va="top")

    # Inference path.
    draw_box(ax, (0.045, 0.66), 0.075, 0.075, "Image\n224 x 224", "#FFFFFF", COL["neutral"], fontsize=5.7)
    draw_box(ax, (0.145, 0.66), 0.100, 0.075, "ResNet-50\nbackbone", "#FFFFFF", COL["baseline"], fontsize=5.7)
    draw_arrow(ax, (0.12, 0.697), (0.145, 0.697))

    draw_box(ax, (0.075, 0.535), 0.135, 0.055, "ArchAug (train only)", "#FFF8E6", COL["aafnet"], fontsize=5.5)
    draw_arrow(ax, (0.142, 0.590), (0.185, 0.66), color=COL["aafnet"], lw=0.65, rad=-0.15)

    ys = [0.735, 0.655, 0.575]
    feature_labels = ["F2\n512 x 28 x 28", "F3\n1024 x 14 x 14", "F4\n2048 x 7 x 7"]
    for y, label in zip(ys, feature_labels):
        draw_box(ax, (0.285, y - 0.030), 0.105, 0.060, label, "#FFFFFF", COL["blue"], fontsize=5.3)
        draw_box(ax, (0.425, y - 0.030), 0.075, 0.060, "MSSA", "#EAF5FF", COL["blue"], fontsize=5.8)
        draw_arrow(ax, (0.245, 0.697), (0.285, y), rad=0.18 if y > 0.69 else (-0.18 if y < 0.62 else 0.0))
        draw_arrow(ax, (0.390, y), (0.425, y))

    draw_box(ax, (0.555, 0.685), 0.095, 0.065, "Project\n512-D", "#FFFFFF", COL["aafnet"], fontsize=5.5)
    draw_box(ax, (0.555, 0.585), 0.095, 0.065, "Gate\nsoftmax", "#FFFFFF", COL["aafnet"], fontsize=5.5)
    draw_box(ax, (0.690, 0.635), 0.105, 0.075, "Fused\n512-D vector", "#FFF8E6", COL["aafnet"], fontsize=5.6)
    for y in ys:
        draw_arrow(ax, (0.500, y), (0.555, 0.718 if y >= 0.655 else 0.618), rad=0.08)
    draw_arrow(ax, (0.650, 0.718), (0.690, 0.675))
    draw_arrow(ax, (0.650, 0.618), (0.690, 0.675))

    draw_box(ax, (0.845, 0.710), 0.105, 0.060, "ClsHead\n6 logits", "#FFFFFF", COL["teal"], fontsize=5.5)
    draw_box(ax, (0.845, 0.595), 0.105, 0.060, "ProjHead\n128-D z", "#FFFFFF", COL["purple"], fontsize=5.5)
    draw_arrow(ax, (0.795, 0.675), (0.845, 0.740), rad=0.10)
    draw_arrow(ax, (0.795, 0.675), (0.845, 0.625), rad=-0.10)

    # Training objectives, laid out below to avoid line crossings.
    draw_box(ax, (0.245, 0.210), 0.110, 0.060, "FLS loss\nclass logits", "#FFFFFF", COL["teal"], fontsize=5.5)
    draw_box(ax, (0.445, 0.210), 0.110, 0.060, "SupCon loss\nprojection z", "#FFFFFF", COL["purple"], fontsize=5.5)
    draw_box(ax, (0.645, 0.210), 0.110, 0.060, "KD loss\nRN50 teacher", "#FFFFFF", COL["blue"], fontsize=5.5)
    draw_box(ax, (0.405, 0.125), 0.190, 0.055, "Total loss\nweighted sum", "#FFF8E6", COL["aafnet"], fontsize=5.5)
    ax.text(
        0.405,
        0.292,
        "Inputs to losses: class logits, projection z and frozen-teacher logits",
        fontsize=5.2,
        color="#555555",
        ha="left",
        va="center",
    )
    draw_arrow(ax, (0.300, 0.210), (0.430, 0.180), rad=-0.05)
    draw_arrow(ax, (0.500, 0.210), (0.500, 0.180))
    draw_arrow(ax, (0.700, 0.210), (0.570, 0.180), rad=0.05)

    save_figure(fig, "diagrams/D1_aafnet_arch.png")


def f_strict(strict_rows) -> None:
    fam = {"AL6": ["AL6", "AL6_strict"], "ASP": ["ASP_clean", "ASP_strict"], "AS25": ["AS25_clean", "AS25_strict"]}
    pairs: dict[tuple[str, str], dict[str, float]] = {}
    nvals: dict[tuple[str, str], dict[str, int]] = {}
    for row in strict_rows:
        for family, datasets in fam.items():
            if row["eval_ds"] not in datasets:
                continue
            key = (row["label"], family)
            slot = "strict" if row["eval_ds"].endswith("_strict") else "clean"
            pairs.setdefault(key, {})[slot] = pct(row["test_accuracy"])
            nvals.setdefault(key, {})[slot] = int(row["n_test"])

    fig, ax = plt.subplots(figsize=(DOUBLE_W, 72 * MM))
    base_x = np.arange(3)
    offsets = {"baseline": -0.16, "aafnet": 0.16}
    colors = {"baseline": COL["baseline"], "aafnet": COL["aafnet"]}
    labels = {"baseline": "RN50", "aafnet": "AAFNet"}
    families = ["AL6", "ASP", "AS25"]
    rows = []
    for model in ["baseline", "aafnet"]:
        for i, family in enumerate(families):
            key = (f"{model}_{family}", family)
            if key not in pairs:
                continue
            clean = pairs[key]["clean"]
            strict = pairs[key]["strict"]
            x0 = base_x[i] + offsets[model] - 0.045
            x1 = base_x[i] + offsets[model] + 0.045
            ax.plot([x0, x1], [clean, strict], color=colors[model], lw=1.1)
            ax.scatter([x0], [clean], facecolor="white", edgecolor=colors[model], marker="o", s=28, linewidth=0.9, zorder=3)
            ax.scatter([x1], [strict], facecolor=colors[model], edgecolor="black", marker="o", s=28, linewidth=0.45, zorder=3)
            ax.text((x0 + x1) / 2, max(clean, strict) + 1.0, f"{strict - clean:+.2f}", ha="center", va="bottom", fontsize=5.5)
            rows.append({"dataset": family, "model": model, "clean_acc_pct": clean, "strict_acc_pct": strict, "delta_pct": strict - clean, "n_clean": nvals[key]["clean"], "n_strict": nvals[key]["strict"]})
    ax.set_xticks(base_x)
    ax.set_xticklabels(["AL6", "ASP", "AS25"])
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(58, 102)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("")
    ax.scatter([], [], facecolor="white", edgecolor="black", marker="o", s=25, label="Clean")
    ax.scatter([], [], facecolor="black", edgecolor="black", marker="o", s=25, label="Strict")
    for model in ["baseline", "aafnet"]:
        ax.plot([], [], color=colors[model], lw=1.1, label=labels[model])
    ax.legend(ncol=4, loc="lower left", bbox_to_anchor=(0, 1.02), borderaxespad=0)
    polish_axes(ax, grid=True)
    write_csv("F_P_strict_eval_source.csv", rows)
    save_figure(fig, "F_P_strict_eval.png")


def f_pareto(sig) -> None:
    rows = [
        {"model": "baseline", "label": "RN50", "params_m": 24.0, "gflops": 4.1},
        {"model": "aafnet", "label": "AAFNet", "params_m": 32.0, "gflops": 4.8},
        {"model": "rn50_aug_only", "label": "RN50+Aug", "params_m": 24.0, "gflops": 4.1},
        {"model": "efficientnet_v2_s", "label": "EffNetV2-S", "params_m": 21.0, "gflops": 8.4},
        {"model": "convnext_tiny", "label": "ConvNeXt-T", "params_m": 28.6, "gflops": 4.5},
    ]
    fig, ax = plt.subplots(figsize=(SINGLE_W * 1.45, 74 * MM))
    for row in rows:
        model = row["model"]
        acc = pct(sig["mean"][model])
        err = pct(sig["std"][model])
        size = 18 + (row["params_m"] - 18) * 8
        ax.errorbar(row["gflops"], acc, yerr=err, fmt="none", ecolor="#555555", elinewidth=0.7, capsize=1.5, zorder=2)
        ax.scatter(row["gflops"], acc, s=size, color=MODEL_COLORS[model], edgecolor="black", linewidth=0.45, zorder=3)
        offset = (4, -8) if model == "baseline" else (4, 5)
        ax.annotate(row["label"], (row["gflops"], acc), xytext=offset, textcoords="offset points", fontsize=5.8, ha="left", va="center")
        row["acc_mean_pct"] = acc
        row["acc_std_pct"] = err
    ax.set_xlabel("Inference cost (GFLOPs)")
    ax.set_ylabel("CV accuracy (%)")
    ax.set_xlim(3.6, 8.9)
    ax.set_ylim(98.35, 99.45)
    ax.set_title("Accuracy-compute trade-off")
    polish_axes(ax, grid=True)
    write_csv("F_N_pareto_efficiency_source.csv", rows)
    save_figure(fig, "F_N_pareto_efficiency.png")


def f_attrib_heatmap(attr) -> None:
    configs = [
        ("baseline", "RN50"),
        ("nx_only", "+Noise"),
        ("archaug_no_noise", "+ArchAug"),
        ("archaug_with_noise", "+ArchAug+N"),
        ("mssa_only", "+MSSA"),
        ("mssa_archaug_noise", "+MSSA+Aug"),
        ("aafnet_full", "AAFNet"),
    ]
    cols = [
        ("clean", "Clean"),
        ("gauss_noise@0.05", "N0.05"),
        ("gauss_noise@0.1", "N0.10"),
        ("gauss_noise@0.2", "N0.20"),
        ("motion_blur@5", "B5"),
        ("motion_blur@11", "B11"),
        ("motion_blur@17", "B17"),
        ("brightness@0.4", "Bright"),
        ("occlusion@0.3", "Occ"),
    ]
    mat = np.zeros((len(configs), len(cols)))
    rows = []
    for i, (cfg, cfg_label) in enumerate(configs):
        for j, (key, col_label) in enumerate(cols):
            if key == "clean":
                val = pct(attr[cfg]["clean_mean"])
            else:
                val = pct(attr[cfg]["robust"][key]["mean"])
            mat[i, j] = val
            rows.append({"config": cfg, "config_label": cfg_label, "condition": key, "condition_label": col_label, "accuracy_pct": val})

    cmap = LinearSegmentedColormap.from_list("nature_blue", ["#F7FBFF", "#C6DBEF", "#6BAED6", "#2171B5", "#08306B"])
    fig, ax = plt.subplots(figsize=(DOUBLE_W, 84 * MM))
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=100)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels([c[1] for c in cols], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(configs)))
    ax.set_yticklabels([c[1] for c in configs])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            color = "white" if mat[i, j] > 78 else "black"
            ax.text(j, i, f"{mat[i, j]:.1f}", ha="center", va="center", fontsize=5.5, color=color)
    ax.set_title("Component attribution across corruptions")
    cbar = fig.colorbar(im, ax=ax, fraction=0.023, pad=0.012)
    cbar.set_label("Accuracy (%)")
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(configs), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.45)
    ax.tick_params(which="minor", bottom=False, left=False)
    write_csv("F_A_attrib_heatmap_source.csv", rows)
    save_figure(fig, "F_A_attrib_heatmap.png")


def f_robustness(attr) -> None:
    cells = [
        ("clean", "Clean"),
        ("gauss_noise@0.05", "N0.05"),
        ("gauss_noise@0.1", "N0.10"),
        ("motion_blur@11", "B11"),
        ("motion_blur@17", "B17"),
        ("brightness@0.4", "Bright"),
        ("occlusion@0.3", "Occ"),
    ]
    cfgs = [
        ("baseline", "RN50", COL["baseline"], "o"),
        ("nx_only", "Noise aug.", COL["aug"], "s"),
        ("archaug_with_noise", "ArchAug+N", COL["teal"], "D"),
        ("aafnet_full", "AAFNet", COL["aafnet"], "^"),
    ]
    fig, ax = plt.subplots(figsize=(DOUBLE_W, 74 * MM))
    x = np.arange(len(cells))
    rows = []
    for cfg, label, color, marker in cfgs:
        vals = []
        errs = []
        for key, short in cells:
            if key == "clean":
                mean = pct(attr[cfg]["clean_mean"])
                err = pct(attr[cfg].get("clean_std", 0))
            else:
                mean = pct(attr[cfg]["robust"][key]["mean"])
                err = pct(attr[cfg]["robust"][key].get("std", 0))
            vals.append(mean)
            errs.append(err)
            rows.append({"config": cfg, "condition": key, "accuracy_pct": mean, "std_pct": err})
        ax.errorbar(x, vals, yerr=errs, color=color, marker=marker, lw=1.15, ms=3.0, capsize=1.2, elinewidth=0.55, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([c[1] for c in cells])
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 103)
    ax.set_title("Robustness profiles", loc="left", pad=10)
    ax.legend(ncol=4, loc="lower right", bbox_to_anchor=(1, 1.04), borderaxespad=0, handlelength=1.5)
    polish_axes(ax, grid=True)
    write_csv("F_C_robustness_grouped_bar_source.csv", rows)
    save_figure(fig, "F_C_robustness_grouped_bar.png")


def f_rotation(rotation) -> None:
    angles = np.array(rotation["angles"])
    theta = np.deg2rad(np.r_[angles, 360])
    base = np.r_[np.array(rotation["models"]["baseline"]) * 100, pct(rotation["models"]["baseline"][0])]
    aaf = np.r_[np.array(rotation["models"]["aafnet"]) * 100, pct(rotation["models"]["aafnet"][0])]

    fig = plt.figure(figsize=(SINGLE_W * 1.25, SINGLE_W * 1.25))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.plot(theta, base, color=COL["baseline"], lw=1.25, marker="o", ms=2.5, label="RN50")
    ax.plot(theta, aaf, color=COL["aafnet"], lw=1.25, marker="^", ms=2.5, label="AAFNet")
    ax.fill(theta, base, color=COL["baseline"], alpha=0.08)
    ax.fill(theta, aaf, color=COL["aafnet"], alpha=0.08)
    ax.set_ylim(75, 101)
    ax.set_yticks([80, 90, 100])
    ax.set_yticklabels(["80", "90", "100"], fontsize=5.5)
    ax.set_thetagrids(np.arange(0, 360, 45), [str(a) for a in np.arange(0, 360, 45)], fontsize=5.5)
    ax.grid(color="#D0D0D0", linewidth=0.45)
    ax.legend(loc="upper right", bbox_to_anchor=(1.16, 1.12), handlelength=1.4)
    ax.set_title("Rotation accuracy (%)", pad=9)
    ax.text(0.5, -0.08, f"Mean delta = {pct(rotation['summary']['aafnet']['mean'] - rotation['summary']['baseline']['mean']):+.2f} pp", transform=ax.transAxes, ha="center", va="top", fontsize=6)

    rows = []
    for model in ["baseline", "aafnet"]:
        for angle, acc in zip(rotation["angles"], rotation["models"][model]):
            rows.append({"model": model, "angle_deg": angle, "accuracy_pct": pct(acc)})
    write_csv("F_AA_rotation_polar_source.csv", rows)
    save_figure(fig, "F_AA_rotation_polar.png")


def f_cross_corpus(cross) -> None:
    datasets = ["ASP_clean", "AS25_clean"]
    fig, ax = plt.subplots(figsize=(SINGLE_W * 1.18, 72 * MM))
    x = np.arange(len(datasets))
    width = 0.28
    rng = np.random.default_rng(7)
    rows = []
    for i, (model, label, color, marker) in enumerate([("baseline", "RN50", COL["baseline"], "o"), ("aafnet", "AAFNet", COL["aafnet"], "^")]):
        means = [pct(cross[ds][model]["test_acc_mean"]) for ds in datasets]
        stds = [pct(cross[ds][model]["test_acc_std"]) for ds in datasets]
        xpos = x + (i - 0.5) * width
        ax.bar(xpos, means, width, color=color, edgecolor="black", linewidth=0.45, yerr=stds, capsize=1.6, label=label)
        for j, ds in enumerate(datasets):
            raw = [pct(v) for v in cross[ds][model]["raw_accs"]]
            jitter = rng.normal(0, 0.015, len(raw))
            ax.scatter(np.full(len(raw), xpos[j]) + jitter, raw, marker=marker, s=11, facecolor="white", edgecolor="black", linewidth=0.35, zorder=3)
            rows.append({"dataset": ds, "model": model, "mean_acc_pct": means[j], "std_pct": stds[j], "raw_acc_pct": ";".join(f"{v:.4f}" for v in raw)})
    for j, ds in enumerate(datasets):
        delta = pct(cross[ds]["aafnet"]["test_acc_mean"] - cross[ds]["baseline"]["test_acc_mean"])
        ax.text(x[j], max(pct(cross[ds]["aafnet"]["test_acc_mean"]), pct(cross[ds]["baseline"]["test_acc_mean"])) + 1.2, f"{delta:+.2f} pp", ha="center", va="bottom", fontsize=5.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["ASP", "AS25"])
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(60, 76)
    ax.set_title("Cross-corpus evaluation")
    ax.legend(loc="upper right")
    polish_axes(ax, grid=True)
    write_csv("F_M_cross_corpus_source.csv", rows)
    save_figure(fig, "F_M_cross_corpus.png")


def f_friedman(sig) -> None:
    models = sig["models"]
    ordered = sorted(models, key=lambda m: sig["mean_rank"][m])
    fig = plt.figure(figsize=(DOUBLE_W, 82 * MM))
    gs = fig.add_gridspec(1, 2, left=0.07, right=0.985, top=0.82, bottom=0.20, wspace=0.42, width_ratios=[1.08, 1.0])

    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(ordered))
    rng = np.random.default_rng(3)
    rows = []
    for i, model in enumerate(ordered):
        vals = np.array(sig["fold_accs"][model]) * 100
        jitter = rng.normal(0, 0.035, len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=8, color=MODEL_COLORS[model], alpha=0.55, linewidth=0)
        mean = pct(sig["mean"][model])
        std = pct(sig["std"][model])
        ax.errorbar(i, mean, yerr=std, color="black", marker="_", markersize=9, elinewidth=0.75, capsize=1.8)
        rows.append({"model": model, "mean_acc_pct": mean, "std_pct": std, "mean_rank": sig["mean_rank"][model], "fold_acc_pct": ";".join(f"{v:.4f}" for v in vals)})
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in ordered], rotation=35, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(88, 100)
    ax.set_title("Fold-seed accuracy")
    panel_label(ax, "a")
    polish_axes(ax, grid=True)

    ax = fig.add_subplot(gs[0, 1])
    n = len(models)
    pmat = np.ones((n, n))
    for key, value in sig["wilcoxon_pvals"].items():
        a, b = key.split("__vs__")
        i, j = models.index(a), models.index(b)
        pmat[i, j] = value
        pmat[j, i] = value
    logp = -np.log10(np.clip(pmat, 1e-6, 1.0))
    cmap = LinearSegmentedColormap.from_list("pvals", ["#F7F7F7", "#BFD3E6", "#8C96C6", "#4D4D8F"])
    im = ax.imshow(logp, cmap=cmap, vmin=0, vmax=4.0)
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels([MODEL_LABELS[m] for m in models], rotation=45, ha="right")
    ax.set_yticklabels([MODEL_LABELS[m] for m in models])
    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, "-", ha="center", va="center", fontsize=5.2)
            else:
                p = pmat[i, j]
                shown = "<0.001" if p < 0.001 else f"{p:.3f}"
                ax.text(j, i, shown, ha="center", va="center", fontsize=4.8, color="white" if logp[i, j] > 2.2 else "black")
    ax.set_title("Wilcoxon p values")
    panel_label(ax, "b")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("-log10(p)")
    ax.tick_params(length=0)
    fig.text(0.07, 0.965, f"Friedman chi2 = {sig['friedman']['statistic']:.2f}, p = {sig['friedman']['p_value']:.2e}; n = {sig['n_folds_per_model']} fold-seed scores per model", ha="left", va="top", fontsize=6.2)
    write_csv("F_L_friedman_cv_source.csv", rows)
    save_figure(fig, "F_L_friedman_cv.png")


def f_calibration(cal) -> None:
    fig = plt.figure(figsize=(DOUBLE_W, 76 * MM))
    gs = fig.add_gridspec(1, 2, left=0.07, right=0.985, top=0.88, bottom=0.17, wspace=0.36, width_ratios=[1.05, 1.0])

    ax = fig.add_subplot(gs[0, 0])
    conds = ["clean", "noise_0_05", "noise_0_10"]
    labels = ["Clean", "0.05", "0.10"]
    x = np.arange(len(conds))
    markers = {("baseline", "pre"): ("o", "white", COL["baseline"]), ("baseline", "post"): ("o", COL["baseline"], COL["baseline"]), ("aafnet", "pre"): ("^", "white", COL["aafnet"]), ("aafnet", "post"): ("^", COL["aafnet"], COL["aafnet"])}
    rows = []
    for model, model_label in [("baseline", "RN50"), ("aafnet", "AAFNet")]:
        for phase, ls in [("pre", "--"), ("post", "-")]:
            vals = [cal[model]["conditions"][c][phase]["ece"] for c in conds]
            marker, face, edge = markers[(model, phase)]
            ax.plot(x, vals, color=edge, linestyle=ls, lw=1.0, marker=marker, ms=4, markerfacecolor=face, markeredgecolor=edge, label=f"{model_label} {phase}")
            for c, val in zip(conds, vals):
                rows.append({"model": model, "phase": phase, "condition": c, "ece": val})
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Noise sigma")
    ax.set_ylabel("Expected calibration error")
    ax.set_title("")
    ax.legend(ncol=2, loc="lower left", bbox_to_anchor=(0, 1.02), borderaxespad=0)
    panel_label(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    for model, label, color, marker in [("baseline", "RN50", COL["baseline"], "o"), ("aafnet", "AAFNet", COL["aafnet"], "^")]:
        bins = [b for b in cal[model]["conditions"]["noise_0_05"]["bins_post"] if b["n"] > 0]
        conf = [b["avg_conf"] for b in bins]
        acc = [b["avg_acc"] for b in bins]
        ax.plot(conf, acc, color=color, marker=marker, ms=3.2, lw=1.1, label=label)
    ax.plot([0, 1], [0, 1], color="#777777", lw=0.7, linestyle="--", label="Perfect")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title("Reliability, sigma = 0.05")
    ax.legend(loc="lower right")
    panel_label(ax, "b")
    polish_axes(ax, grid=True)
    write_csv("F_E_calibration_dual_source.csv", rows)
    save_figure(fig, "F_E_calibration_dual.png")


def f_downstream_radar(downstream) -> None:
    rt = downstream["3.1_retrieval"]
    ood = downstream["3.2_ood"]
    rj = downstream["3.3_rejection"]
    nd = downstream["3.4_neardup"]
    dom = downstream["3.5_domain"]
    pd = downstream["3.7_robust_diag"]
    metrics = [
        ("Retr.\nTop-1", pct(rt["baseline"]["AL6"]["top1_mean"]), pct(rt["aafnet"]["AL6"]["top1_mean"])),
        ("Retr.\nmAP", pct(rt["baseline"]["AL6"]["mAP_mean"]), pct(rt["aafnet"]["AL6"]["mAP_mean"])),
        ("OOD\nAUROC", pct((ood["baseline"]["ASP_clean"]["Energy"]["AUROC"] + ood["baseline"]["AS25_clean"]["Energy"]["AUROC"]) / 2), pct((ood["aafnet"]["ASP_clean"]["Energy"]["AUROC"] + ood["aafnet"]["AS25_clean"]["Energy"]["AUROC"]) / 2)),
        ("Selective\nacc.", pct(1 - rj["baseline"]["noise_0_05"]["risk@cov_90"]), pct(1 - rj["aafnet"]["noise_0_05"]["risk@cov_90"])),
        ("Aug.\nAUROC", pct(nd["baseline"]["AL6"]["AUROC"]), pct(nd["aafnet"]["AL6"]["AUROC"])),
        ("Domain\nprobe", pct(dom["baseline"]["domain_acc_mean"]), pct(dom["aafnet"]["domain_acc_mean"])),
        ("Perturb.\nprobe", pct(pd["baseline"]["acc_mean"]), pct(pd["aafnet"]["acc_mean"])),
    ]
    theta = np.linspace(0, 2 * math.pi, len(metrics), endpoint=False)
    theta = np.r_[theta, theta[0]]
    base = np.r_[[m[1] for m in metrics], metrics[0][1]]
    aaf = np.r_[[m[2] for m in metrics], metrics[0][2]]

    fig = plt.figure(figsize=(SINGLE_W * 1.22, SINGLE_W * 1.16))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.plot(theta, base, color=COL["baseline"], marker="o", ms=3, lw=1.15, label="RN50")
    ax.fill(theta, base, color=COL["baseline"], alpha=0.08)
    ax.plot(theta, aaf, color=COL["aafnet"], marker="^", ms=3, lw=1.15, label="AAFNet")
    ax.fill(theta, aaf, color=COL["aafnet"], alpha=0.08)
    ax.set_thetagrids(np.degrees(theta[:-1]), [m[0] for m in metrics], fontsize=5.8)
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(["25", "50", "75", "100"], fontsize=5.2)
    ax.grid(color="#D0D0D0", linewidth=0.45)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    ax.set_title("Downstream battery (%)", pad=10)
    write_csv("F_Q_downstream_radar_source.csv", [{"metric": m[0].replace("\n", " "), "baseline_pct": m[1], "aafnet_pct": m[2]} for m in metrics])
    save_figure(fig, "F_Q_downstream_radar.png")


def _card_box(ax, x, y, w, h, text, edge, fill="#FFFFFF", fontsize=5.3, lw=0.55) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        transform=ax.transAxes,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        facecolor=fill,
        edgecolor=edge,
        linewidth=lw,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=COL["ink"],
        linespacing=1.12,
    )


def _card_arrow(ax, x0, y0, x1, y1, color="#555555") -> None:
    arrow = FancyArrowPatch(
        (x0, y0),
        (x1, y1),
        transform=ax.transAxes,
        arrowstyle="->",
        mutation_scale=6,
        linewidth=0.55,
        color=color,
    )
    ax.add_patch(arrow)


def _fmt_value(value: float, scale: str) -> str:
    if scale == "pct":
        return f"{value:.1f}%"
    if abs(value) < 0.01:
        return f"{value:.4f}"
    return f"{value:.3f}"


def _draw_inference_card(
    ax,
    panel: str,
    title: str,
    flow: tuple[str, str, str],
    metric: str,
    baseline: float,
    aafnet: float,
    *,
    xlim: tuple[float, float],
    scale: str = "raw",
    higher_is_better: bool = True,
    note: str = "",
) -> dict[str, object]:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.add_patch(Rectangle((0.0, 0.0), 1.0, 1.0, transform=ax.transAxes, facecolor="#FFFFFF", edgecolor="#D9D9D9", linewidth=0.55))
    panel_label(ax, panel, x=0.02, y=0.91)
    ax.text(0.10, 0.935, title, transform=ax.transAxes, ha="left", va="top", fontsize=6.7, fontweight="bold", color=COL["ink"])

    xs = [0.08, 0.38, 0.68]
    widths = [0.22, 0.22, 0.22]
    fills = ["#FAFAFA", "#EEF7FB", "#FFF8E6"]
    edges = [COL["neutral"], COL["blue"], COL["aafnet"]]
    for i, step in enumerate(flow):
        _card_box(ax, xs[i], 0.62, widths[i], 0.16, step, edges[i], fills[i])
    _card_arrow(ax, xs[0] + widths[0], 0.70, xs[1], 0.70)
    _card_arrow(ax, xs[1] + widths[1], 0.70, xs[2], 0.70)

    x0, x1 = xlim
    rng = max(x1 - x0, 1e-9)
    b_norm = np.clip((baseline - x0) / rng, 0, 1)
    a_norm = np.clip((aafnet - x0) / rng, 0, 1)
    bar_x = 0.10
    bar_w = 0.78
    y_base = 0.36
    y_aaf = 0.24
    ax.text(0.10, 0.49, metric, transform=ax.transAxes, ha="left", va="center", fontsize=5.8, color="#555555")
    ax.add_patch(Rectangle((bar_x, y_base), bar_w, 0.055, transform=ax.transAxes, facecolor="#F0F0F0", edgecolor="none"))
    ax.add_patch(Rectangle((bar_x, y_aaf), bar_w, 0.055, transform=ax.transAxes, facecolor="#F0F0F0", edgecolor="none"))
    ax.add_patch(Rectangle((bar_x, y_base), bar_w * b_norm, 0.055, transform=ax.transAxes, facecolor=COL["baseline"], edgecolor="black", linewidth=0.35))
    ax.add_patch(Rectangle((bar_x, y_aaf), bar_w * a_norm, 0.055, transform=ax.transAxes, facecolor=COL["aafnet"], edgecolor="black", linewidth=0.35))
    ax.text(0.035, y_base + 0.027, "RN50", transform=ax.transAxes, ha="left", va="center", fontsize=5.3)
    ax.text(0.035, y_aaf + 0.027, "AAF", transform=ax.transAxes, ha="left", va="center", fontsize=5.3)
    ax.text(bar_x + bar_w + 0.015, y_base + 0.027, _fmt_value(baseline, scale), transform=ax.transAxes, ha="left", va="center", fontsize=5.3)
    ax.text(bar_x + bar_w + 0.015, y_aaf + 0.027, _fmt_value(aafnet, scale), transform=ax.transAxes, ha="left", va="center", fontsize=5.3)

    delta = aafnet - baseline
    improved = delta >= 0 if higher_is_better else delta <= 0
    delta_color = COL["teal"] if improved else "#C44E52"
    if scale == "pct":
        delta_text = f"{delta:+.2f} pp"
    else:
        delta_text = f"{delta:+.3f}"
    ax.text(0.10, 0.115, delta_text, transform=ax.transAxes, ha="left", va="center", fontsize=6.3, fontweight="bold", color=delta_color)
    if note:
        ax.text(0.90, 0.115, note, transform=ax.transAxes, ha="right", va="center", fontsize=5.1, color="#555555")
    return {
        "task": title,
        "metric": metric,
        "baseline": baseline,
        "aafnet": aafnet,
        "delta": delta,
        "higher_is_better": higher_is_better,
        "note": note,
    }


def f_downstream_inference_cards(downstream) -> None:
    rt = downstream["3.1_retrieval"]
    ood = downstream["3.2_ood"]
    rj = downstream["3.3_rejection"]
    nd = downstream["3.4_neardup"]
    dom = downstream["3.5_domain"]
    fs = downstream["3.6_fewshot"]

    aug_base = float(np.mean([nd["baseline"][ds]["AUROC"] for ds in ["AL6", "ASP_clean", "AS25_clean"]]))
    aug_aaf = float(np.mean([nd["aafnet"][ds]["AUROC"] for ds in ["AL6", "ASP_clean", "AS25_clean"]]))

    cards = [
        (
            "a",
            "AL6 retrieval",
            ("Query\nimage", "Feature\nz", "Prototype\nrank"),
            "Top-1 retrieval",
            pct(rt["baseline"]["AL6"]["top1_mean"]),
            pct(rt["aafnet"]["AL6"]["top1_mean"]),
            (90, 100),
            "pct",
            True,
            "mAP +0.74 pp",
        ),
        (
            "b",
            "OOD: AL6 -> ASP",
            ("AL6 ID", "Energy\nscore", "OOD\nflag"),
            "Energy AUROC",
            ood["baseline"]["ASP_clean"]["Energy"]["AUROC"],
            ood["aafnet"]["ASP_clean"]["Energy"]["AUROC"],
            (0.90, 1.00),
            "raw",
            True,
            "FPR95 0.218 -> 0.152",
        ),
        (
            "c",
            "OOD: AL6 -> AS25",
            ("AL6 ID", "Energy\nscore", "OOD\nflag"),
            "Energy AUROC",
            ood["baseline"]["AS25_clean"]["Energy"]["AUROC"],
            ood["aafnet"]["AS25_clean"]["Energy"]["AUROC"],
            (0.90, 1.00),
            "raw",
            True,
            "FPR95 0.236 -> 0.161",
        ),
        (
            "d",
            "Selective prediction",
            ("Noisy\nimage", "Confidence\nrank", "Accept /\nreject"),
            "AURC (lower)",
            rj["baseline"]["noise_0_05"]["AURC"],
            rj["aafnet"]["noise_0_05"]["AURC"],
            (0.00, 0.90),
            "raw",
            False,
            "sigma = 0.05",
        ),
        (
            "e",
            "Aug-invariance",
            ("Image", "Augmented\nview", "Pair\nscore"),
            "AUROC",
            aug_base,
            aug_aaf,
            (0.90, 1.00),
            "raw",
            True,
            "3-dataset mean",
        ),
        (
            "f",
            "Domain probe",
            ("Feature\nz", "Linear\nprobe", "AL6/ASP/\nAS25"),
            "3-domain accuracy",
            pct(dom["baseline"]["domain_acc_mean"]),
            pct(dom["aafnet"]["domain_acc_mean"]),
            (33, 60),
            "pct",
            True,
            "random 33.3%",
        ),
        (
            "g",
            "Few-shot ASP",
            ("5-shot\nsupport", "Prototype\nhead", "Query\nlabel"),
            "5-way accuracy",
            pct(fs["baseline"]["ASP_clean"]["acc_mean"]),
            pct(fs["aafnet"]["ASP_clean"]["acc_mean"]),
            (20, 60),
            "pct",
            True,
            "100 episodes",
        ),
        (
            "h",
            "Few-shot AS25",
            ("5-shot\nsupport", "Prototype\nhead", "Query\nlabel"),
            "5-way accuracy",
            pct(fs["baseline"]["AS25_clean"]["acc_mean"]),
            pct(fs["aafnet"]["AS25_clean"]["acc_mean"]),
            (30, 70),
            "pct",
            True,
            "100 episodes",
        ),
    ]

    fig = plt.figure(figsize=(DOUBLE_W, 138 * MM))
    gs = fig.add_gridspec(2, 4, left=0.035, right=0.99, top=0.93, bottom=0.055, wspace=0.12, hspace=0.22)
    rows: list[dict[str, object]] = []
    for idx, card in enumerate(cards):
        ax = fig.add_subplot(gs[idx // 4, idx % 4])
        rows.append(
            _draw_inference_card(
                ax,
                card[0],
                card[1],
                card[2],
                card[3],
                card[4],
                card[5],
                xlim=card[6],
                scale=card[7],
                higher_is_better=card[8],
                note=card[9],
            )
        )
    fig.text(0.035, 0.985, "Eight downstream inference tasks", ha="left", va="top", fontsize=7.2, fontweight="bold", color=COL["ink"])
    fig.text(0.035, 0.958, "Each panel shows the inference signal used by the probe and the baseline-to-AAFNet metric change.", ha="left", va="top", fontsize=5.8, color="#555555")
    write_csv("F_Q2_downstream_inference_cards_source.csv", rows)
    save_figure(fig, "F_Q2_downstream_inference_cards.png")


def f_gradcam_plate() -> None:
    src = BACKUP_DIR / "F_R_gradcam_comparison.png"
    if not src.exists():
        src = FIG_DIR / "F_R_gradcam_comparison.png"
    img = Image.open(src).convert("RGB")

    # Crop the original 3 x 6 image tiles and redraw the plate so that rotated
    # samples are corrected consistently across input and GradCAM rows.
    tile_x = [(219, 776), (805, 1362), (1391, 1948), (1977, 2535), (2563, 3120), (3149, 3706)]
    tile_y = [(210, 767), (797, 1354), (1384, 1941)]
    rotations = [0, 0, 90, 90, 90, 270]
    row_labels = ["Input", "Baseline\nGradCAM", "AAFNet\nGradCAM"]
    row_colors = [COL["ink"], COL["baseline"], "#D04A4A"]

    fig = plt.figure(figsize=(DOUBLE_W, 86 * MM))
    gs = fig.add_gridspec(
        3,
        6,
        left=0.055,
        right=0.995,
        top=0.925,
        bottom=0.045,
        wspace=0.035,
        hspace=0.055,
    )
    axes = np.empty((3, 6), dtype=object)
    for r, (y0, y1) in enumerate(tile_y):
        for c, (x0, x1) in enumerate(tile_x):
            ax = fig.add_subplot(gs[r, c])
            tile = img.crop((x0, y0, x1, y1))
            if rotations[c]:
                tile = tile.rotate(rotations[c], expand=False, resample=Image.Resampling.BICUBIC)
            ax.imshow(tile)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if r == 0:
                ax.set_title(f"Class {c}", fontsize=6.2, pad=2.5)
            axes[r, c] = ax

    fig.text(0.012, 0.94, "a", ha="left", va="top", fontsize=8, fontweight="bold")
    for r, (label, color) in enumerate(zip(row_labels, row_colors)):
        pos = axes[r, 0].get_position()
        fig.text(
            0.030,
            (pos.y0 + pos.y1) / 2,
            label,
            ha="center",
            va="center",
            rotation=90,
            fontsize=6.5,
            fontweight="bold",
            color=color,
        )

    write_csv(
        "F_R_gradcam_comparison_source.csv",
        [
            {
                "source": "existing_gradcam_plate_tiles",
                "note": "Reformatted from the cached GradCAM comparison image; Class 2, 3 and 4 tiles were rotated 90 degrees counter-clockwise and Class 5 tiles 90 degrees clockwise for visual orientation consistency. No local contrast manipulation was applied.",
                "column_rotations_deg": ";".join(str(v) for v in rotations),
            }
        ],
    )
    save_figure(fig, "F_R_gradcam_comparison.png")


def write_qa() -> None:
    lines = [
        "# Nature-style Figure Redraw QA",
        "",
        "- Backend: Python / matplotlib only.",
        "- Figure contract: quantitative grid for statistical panels; schematic-led composite for D1; image plate for GradCAM.",
        "- Export contract: each cited PNG has SVG and PDF companions in `paper/figures/nature_exports/`.",
        "- Source-data contract: quantitative panels have CSV files in `paper/figures/nature_source_data/`.",
        "- Style contract: sans-serif font stack, editable SVG text, PDF fonttype 42, RGB artwork, restrained colour-blind-aware palette.",
        "- Integrity note: GradCAM panel was reformatted from the cached generated GradCAM plate; rotated columns were corrected consistently across all three rows, and source model inference was not rerun in this redraw script.",
        "",
        "## Redrawn manuscript figures",
        "",
    ]
    for rel in TARGETS:
        png = FIG_DIR / rel
        stem = rel.replace("/", "__").replace(".png", "")
        svg = EXPORT_DIR / f"{stem}.svg"
        pdf = EXPORT_DIR / f"{stem}.pdf"
        ok = png.exists() and svg.exists() and pdf.exists()
        lines.append(f"- `{rel}`: {'OK' if ok else 'MISSING'}")
    QA_PATH.write_text("\n".join(lines) + "\n")


def main() -> None:
    backup_targets()
    sig = load_json("outputs/sig_collect_v2/significance.json")
    attr = load_json("outputs/p1_attribution_summary.json")["summary"]
    cross = load_json("outputs/p1_asp_as25_summary.json")
    cal = load_json("outputs/p2_calibration_v2.json")
    downstream = load_json("outputs/downstream/20260509_005735/results.json")
    rotation = load_json("outputs/p3_rotation.json")
    strict_rows = load_json("outputs/p2_strict_extended.json")

    print("[redrawing Nature-style manuscript figures]")
    f_hero(sig, attr, cal, cross, downstream, rotation)
    f_architecture()
    f_strict(strict_rows)
    f_pareto(sig)
    f_attrib_heatmap(attr)
    f_robustness(attr)
    f_rotation(rotation)
    f_cross_corpus(cross)
    f_friedman(sig)
    f_calibration(cal)
    f_downstream_radar(downstream)
    f_gradcam_plate()
    write_qa()
    print(f"[ok] QA written to {QA_PATH}")


if __name__ == "__main__":
    main()
