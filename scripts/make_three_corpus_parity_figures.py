#!/usr/bin/env python
"""Draw three-corpus seed-42 parity figures from existing evaluation outputs.

The script reads completed AL6, ASP_clean, and AS25_clean robustness,
calibration, and rotation JSON files. It does not train or evaluate models.
It writes manuscript PNG, editable SVG/PDF companions, and source-data CSV.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "paper" / "figures"
EXPORT_DIR = FIG_DIR / "nature_exports"
SOURCE_DIR = FIG_DIR / "nature_source_data"
QA_PATH = FIG_DIR / "three_corpus_parity_qa.md"

for directory in (FIG_DIR, EXPORT_DIR, SOURCE_DIR):
    directory.mkdir(parents=True, exist_ok=True)

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
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }
)

MM = 1 / 25.4
DOUBLE_W = 183 * MM

COL = {
    "baseline": "#3A3A3A",
    "aafnet": "#E69F00",
    "blue": "#0072B2",
    "teal": "#009E73",
    "red": "#C44E52",
    "neutral": "#8A8A8A",
    "ink": "#1F1F1F",
}

DATASETS = ["AL6", "ASP_clean", "AS25_clean"]
DISPLAY = {"AL6": "AL6", "ASP_clean": "ASP", "AS25_clean": "AS25"}
MODEL_LABEL = {"baseline": "RN50", "aafnet": "AAFNet"}
FAMILY_LABEL = {
    "gauss_noise": "Gaussian",
    "motion_blur": "Blur",
    "jpeg_compress": "JPEG",
    "brightness": "Brightness",
    "occlusion": "Occlusion",
}
FAMILIES = list(FAMILY_LABEL)


def load_json(rel_path: str) -> dict:
    return json.loads((ROOT / rel_path).read_text(encoding="utf-8"))


def write_csv(name: str, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with (SOURCE_DIR / name).open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig: plt.Figure, name: str) -> None:
    png = FIG_DIR / name
    stem = name.replace(".png", "")
    fig.savefig(png, dpi=600, bbox_inches="tight", pad_inches=0.025)
    fig.savefig(EXPORT_DIR / f"{stem}.svg", bbox_inches="tight", pad_inches=0.025)
    fig.savefig(EXPORT_DIR / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)
    print(f"  -> {png}")


def panel_label(ax: plt.Axes, label: str, x: float = -0.12, y: float = 1.04) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        fontweight="bold",
        color=COL["ink"],
    )


def polish(ax: plt.Axes, grid: bool = False) -> None:
    ax.tick_params(direction="out", pad=2)
    if grid:
        ax.yaxis.grid(True, color="#E6E6E6", linewidth=0.45)
        ax.set_axisbelow(True)


def mean_family(values: list[dict]) -> float:
    return float(np.mean([float(v["accuracy"]) for v in values]))


def collect_payload() -> dict[str, dict]:
    public = load_json("outputs/three_corpus_parity/latest/results.json")
    al6_rob = {
        "baseline": load_json("outputs/attrib_robust_baseline_seed42/20260508_222138/resnet50/results.json"),
        "aafnet": load_json("outputs/attrib_robust_aafnet_full_seed42/20260508_221648/resnet50/results.json"),
    }
    al6_cal = load_json("outputs/p2_calibration_v2.json")
    al6_rot = load_json("outputs/p3_rotation.json")

    payload: dict[str, dict] = {"AL6": {"models": {}}}
    for model in ["baseline", "aafnet"]:
        family_means = {fam: mean_family(al6_rob[model]["robustness"][fam]) for fam in FAMILIES}
        payload["AL6"]["models"][model] = {
            "clean_accuracy": al6_rob[model]["clean_accuracy"],
            "robustness": {
                "all_mean": float(np.mean(list(family_means.values()))),
                "family_means": family_means,
            },
            "calibration": {
                "temperature": al6_cal[model]["T"],
                "conditions": al6_cal[model]["conditions"],
            },
            "rotation": {
                "summary": al6_rot["summary"][model],
                "angles": al6_rot["angles"],
                "accuracies": al6_rot["models"][model],
            },
        }

    for dataset in ["ASP_clean", "AS25_clean"]:
        payload[dataset] = {"models": {}}
        for model, model_payload in public["datasets"][dataset]["models"].items():
            payload[dataset]["models"][model] = model_payload
    return payload


def collect_rows(payload: dict[str, dict]) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    robust_rows: list[dict[str, object]] = []
    delta_rows: list[dict[str, object]] = []
    cal_rows: list[dict[str, object]] = []
    rot_rows: list[dict[str, object]] = []

    for dataset in DATASETS:
        dres = payload[dataset]
        for model in ["baseline", "aafnet"]:
            mres = dres["models"][model]
            robust_rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "metric": "all_corruption_mean",
                    "value": mres["robustness"]["all_mean"],
                }
            )
            for fam in FAMILIES:
                robust_rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "metric": fam,
                        "value": mres["robustness"]["family_means"][fam],
                    }
                )
            for condition in ["clean", "noise_0_05", "noise_0_10"]:
                c = mres["calibration"]["conditions"][condition]
                cal_rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "condition": condition,
                        "temperature": mres["calibration"]["temperature"],
                        "pre_ece": c["pre"]["ece"],
                        "post_ece": c["post"]["ece"],
                        "post_nll": c["post"]["nll"],
                        "post_brier": c["post"]["brier"],
                        "accuracy": c["post"]["accuracy"],
                    }
                )
            summary = mres["rotation"]["summary"]
            for metric in ["mean", "min", "acc@0", "acc@90", "acc@180", "acc@270"]:
                rot_rows.append({"dataset": dataset, "model": model, "metric": metric, "value": summary[metric]})

        for fam in FAMILIES:
            base = dres["models"]["baseline"]["robustness"]["family_means"][fam]
            aaf = dres["models"]["aafnet"]["robustness"]["family_means"][fam]
            delta_rows.append({"dataset": dataset, "family": fam, "delta_aafnet_minus_baseline": aaf - base})

    return robust_rows, delta_rows, cal_rows, rot_rows


def draw_parity_figure(payload: dict[str, dict]) -> None:
    robust_rows, delta_rows, cal_rows, rot_rows = collect_rows(payload)
    write_csv("F_TC6_three_corpus_parity_robustness_source.csv", robust_rows)
    write_csv("F_TC6_three_corpus_parity_delta_source.csv", delta_rows)
    write_csv("F_TC6_three_corpus_parity_calibration_source.csv", cal_rows)
    write_csv("F_TC6_three_corpus_parity_rotation_source.csv", rot_rows)

    fig = plt.figure(figsize=(DOUBLE_W, 116 * MM))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], wspace=0.34, hspace=0.46)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # Panel a: all-corruption mean accuracy.
    x = np.arange(len(DATASETS))
    width = 0.32
    for offset, model in [(-width / 2, "baseline"), (width / 2, "aafnet")]:
        vals = [payload[d]["models"][model]["robustness"]["all_mean"] * 100 for d in DATASETS]
        bars = ax_a.bar(x + offset, vals, width=width, color=COL[model], label=MODEL_LABEL[model])
        for bar, val in zip(bars, vals):
            ax_a.text(bar.get_x() + bar.get_width() / 2, val + 1.0, f"{val:.1f}", ha="center", va="bottom", fontsize=5.5)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([DISPLAY[d] for d in DATASETS])
    ax_a.set_ylim(0, 105)
    ax_a.set_ylabel("Accuracy (%)")
    ax_a.set_title("All-corruption mean")
    ax_a.legend(loc="upper right", ncols=2, handlelength=1.1, columnspacing=0.8)
    polish(ax_a, grid=True)
    panel_label(ax_a, "a")

    # Panel b: family-level AAFNet-minus-baseline deltas.
    delta_matrix = np.array(
        [
            [
                payload[d]["models"]["aafnet"]["robustness"]["family_means"][fam]
                - payload[d]["models"]["baseline"]["robustness"]["family_means"][fam]
                for fam in FAMILIES
            ]
            for d in DATASETS
        ]
    ) * 100
    norm = TwoSlopeNorm(vmin=-5, vcenter=0, vmax=max(5, float(delta_matrix.max())))
    im = ax_b.imshow(delta_matrix, cmap="RdBu_r", norm=norm, aspect="auto")
    ax_b.set_xticks(np.arange(len(FAMILIES)))
    ax_b.set_xticklabels([FAMILY_LABEL[f] for f in FAMILIES], rotation=25, ha="right")
    ax_b.set_yticks(np.arange(len(DATASETS)))
    ax_b.set_yticklabels([DISPLAY[d] for d in DATASETS])
    ax_b.set_title("AAFNet minus RN50 under corruption")
    for i in range(len(DATASETS)):
        for j in range(len(FAMILIES)):
            val = delta_matrix[i, j]
            ax_b.text(j, i, f"{val:+.1f}", ha="center", va="center", fontsize=5.8, color="white" if abs(val) > 22 else COL["ink"])
    cbar = fig.colorbar(im, ax=ax_b, shrink=0.82, pad=0.02)
    cbar.ax.tick_params(labelsize=5.5, length=2)
    cbar.set_label("pp", fontsize=6)
    panel_label(ax_b, "b")

    # Panel c: post-temperature ECE on clean and sigma=0.05 noise.
    conditions = ["clean", "noise_0_05"]
    cond_label = {"clean": "Clean", "noise_0_05": "Noise 0.05"}
    group_x = np.arange(len(DATASETS) * len(conditions))
    labels = []
    for d in DATASETS:
        for cond in conditions:
            labels.append(f"{DISPLAY[d]}\n{cond_label[cond]}")
    for offset, model in [(-width / 2, "baseline"), (width / 2, "aafnet")]:
        vals = []
        for d in DATASETS:
            for cond in conditions:
                vals.append(payload[d]["models"][model]["calibration"]["conditions"][cond]["post"]["ece"])
        ax_c.bar(group_x + offset, vals, width=width, color=COL[model], label=MODEL_LABEL[model])
    ax_c.set_xticks(group_x)
    ax_c.set_xticklabels(labels)
    ax_c.set_ylabel("ECE after T-scaling")
    ax_c.set_ylim(0, 0.92)
    ax_c.set_title("Calibration parity")
    ax_c.legend(loc="upper left", ncols=2, handlelength=1.1, columnspacing=0.8)
    polish(ax_c, grid=True)
    panel_label(ax_c, "c")

    # Panel d: rotation mean and worst-case accuracy.
    metric_labels = ["Mean", "Worst"]
    group_x = np.arange(len(DATASETS) * len(metric_labels))
    labels = []
    for d in DATASETS:
        for metric in metric_labels:
            labels.append(f"{DISPLAY[d]}\n{metric}")
    for offset, model in [(-width / 2, "baseline"), (width / 2, "aafnet")]:
        vals = []
        for d in DATASETS:
            summary = payload[d]["models"][model]["rotation"]["summary"]
            vals.extend([summary["mean"] * 100, summary["min"] * 100])
        ax_d.bar(group_x + offset, vals, width=width, color=COL[model], label=MODEL_LABEL[model])
    ax_d.set_xticks(group_x)
    ax_d.set_xticklabels(labels)
    ax_d.set_ylim(0, 105)
    ax_d.set_ylabel("Accuracy (%)")
    ax_d.set_title("24-angle rotation")
    ax_d.legend(loc="lower left", ncols=2, handlelength=1.1, columnspacing=0.8)
    polish(ax_d, grid=True)
    panel_label(ax_d, "d")

    save_figure(fig, "F_TC6_three_corpus_parity.png")


def write_qa(payload: dict[str, dict]) -> None:
    robust_rows, _, cal_rows, rot_rows = collect_rows(payload)
    deltas = []
    for dataset in DATASETS:
        base = payload[dataset]["models"]["baseline"]
        aaf = payload[dataset]["models"]["aafnet"]
        deltas.append(
            {
                "dataset": dataset,
                "robustness_all_pp": (aaf["robustness"]["all_mean"] - base["robustness"]["all_mean"]) * 100,
                "clean_ece_delta": aaf["calibration"]["conditions"]["clean"]["post"]["ece"]
                - base["calibration"]["conditions"]["clean"]["post"]["ece"],
                "rotation_mean_pp": (aaf["rotation"]["summary"]["mean"] - base["rotation"]["summary"]["mean"]) * 100,
            }
        )
    lines = [
        "# Three-Corpus Parity Figure QA",
        "",
        "- Source files: AL6 seed-42 attribution/calibration/rotation outputs; ASP_clean and AS25_clean `outputs/three_corpus_parity/latest/results.json`.",
        "- Scope: seed-42 checkpoint parity evaluation only; no new training was performed.",
        f"- Robustness source rows: {len(robust_rows)}.",
        f"- Calibration source rows: {len(cal_rows)}.",
        f"- Rotation source rows: {len(rot_rows)}.",
        "",
        "| Dataset | Robustness all-cell Δ (pp) | Clean ECE Δ | Rotation mean Δ (pp) |",
        "|---|---:|---:|---:|",
    ]
    for row in deltas:
        lines.append(
            f"| {DISPLAY[row['dataset']]} | {row['robustness_all_pp']:+.2f} | "
            f"{row['clean_ece_delta']:+.4f} | {row['rotation_mean_pp']:+.2f} |"
        )
    QA_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  -> {QA_PATH}")


def main() -> None:
    payload = collect_payload()
    draw_parity_figure(payload)
    write_qa(payload)


if __name__ == "__main__":
    main()
