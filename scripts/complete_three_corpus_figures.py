#!/usr/bin/env python
"""Complete three-corpus manuscript figures from cached experiment outputs.

This script does not train models. It reads audited data splits and existing
AL6 / ASP_clean / AS25_clean results, then writes Nature-style PNG figures,
editable SVG/PDF companions, source-data CSV files, and a short QA report.
"""

from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "paper" / "figures"
EXPORT_DIR = FIG_DIR / "nature_exports"
SOURCE_DIR = FIG_DIR / "nature_source_data"
OUT_DIR = ROOT / "outputs" / "three_corpus_completion"
QA_PATH = FIG_DIR / "three_corpus_completion_qa.md"

for directory in (EXPORT_DIR, SOURCE_DIR, OUT_DIR):
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
SINGLE_W = 89 * MM

COL = {
    "baseline": "#3A3A3A",
    "aafnet": "#E69F00",
    "blue": "#0072B2",
    "teal": "#009E73",
    "red": "#C44E52",
    "purple": "#7851A9",
    "neutral": "#8A8A8A",
    "pale": "#F2F2F2",
    "ink": "#1F1F1F",
}

RAW_DATASETS = ["AL6", "ASP", "AS25"]
ANALYSIS_DATASETS = ["AL6", "ASP_clean", "AS25_clean"]
DISPLAY = {"AL6": "AL6", "ASP_clean": "ASP", "AS25_clean": "AS25"}
MODEL_LABEL = {"baseline": "RN50", "aafnet": "AAFNet"}

TARGETS = [
    "F_TC0_three_corpus_samples.png",
    "F_TC1_three_corpus_audit.png",
    "F_TC2_three_corpus_performance.png",
    "F_TC3_three_corpus_confusions.png",
    "F_TC4_three_corpus_training.png",
    "F_TC5_three_corpus_probe_matrix.png",
    "F_TC6_three_corpus_parity.png",
    "F_TC7_public_interpretability.png",
    "F_TC8_public_experiment_coverage.png",
]

AL6_SAMPLE_FILES = [
    "1/train_0002_label1.png",
    "2/train_0003_label2.png",
    "3/train_0005_label3.png",
    "4/train_0052_label4.png",
    "5/train_0048_label5.png",
    "6/train_0001_label6.png",
    "4/train_0101_label4.png",
    "6/train_0034_label6.png",
]


def load_json(rel_path: str):
    return json.loads((ROOT / rel_path).read_text(encoding="utf-8"))


def pct(value: float | None) -> float | None:
    if value is None:
        return None
    return float(value) * 100.0


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


def read_source_csv(name: str) -> list[dict[str, str]]:
    path = SOURCE_DIR / name
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def save_figure(fig, rel_path: str) -> None:
    out_png = FIG_DIR / rel_path
    out_png.parent.mkdir(parents=True, exist_ok=True)
    stem = rel_path.replace("/", "__").replace(".png", "")
    fig.savefig(out_png, dpi=600, bbox_inches="tight", pad_inches=0.025)
    fig.savefig(EXPORT_DIR / f"{stem}.svg", bbox_inches="tight", pad_inches=0.025)
    fig.savefig(EXPORT_DIR / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)
    print(f"  -> {out_png}")


def polish(ax, grid: bool = False) -> None:
    ax.tick_params(direction="out", pad=2)
    if grid:
        ax.yaxis.grid(True, color="#E6E6E6", linewidth=0.45)
        ax.set_axisbelow(True)


def panel_label(ax, label: str, x: float = -0.10, y: float = 1.04) -> None:
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


def short_label(text: str, n: int = 15) -> str:
    text = text.replace("_", " ")
    if len(text) <= n:
        return text
    return text[: n - 1] + "."


def class_names(dataset: str) -> list[str]:
    p = ROOT / "data" / "processed" / dataset / "class_map.json"
    if not p.exists():
        return []
    data = json.loads(p.read_text(encoding="utf-8"))
    return [data[k] for k in sorted(data, key=lambda x: int(x))]


def split_mapping(dataset: str, split: str) -> list[dict[str, str]]:
    p = ROOT / "data" / "processed" / dataset / f"{split}_mapping.csv"
    with p.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def split_counts(dataset: str, split: str) -> dict[int, int]:
    counts: dict[int, int] = {}
    for row in split_mapping(dataset, split):
        lab = int(row["标签"])
        counts[lab] = counts.get(lab, 0) + 1
    return dict(sorted(counts.items()))


def load_metrics(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def public_cv_summary(dataset: str, model: str) -> dict | None:
    prefix = dataset.split("_")[0].lower()
    pattern = f"outputs/cv_{prefix}_{model}/*/resnet50/cv_summary.json"
    for path in sorted(ROOT.glob(pattern), reverse=True):
        data = load_json(str(path.relative_to(ROOT)))
        completed = int(data.get("n_folds_completed", 0))
        total = int(data.get("n_folds_total", 15))
        if completed == total == 15:
            data["_source_path"] = str(path.relative_to(ROOT))
            return data
    status = load_json("outputs/public_cv_parity/status.json")
    role = model
    for row in status.get("rows", []):
        if row.get("dataset") != dataset or row.get("role") != role:
            continue
        completed = int(row.get("n_folds_completed") or 0)
        total = int(row.get("n_folds_total") or 15)
        if completed <= 0:
            continue
        raw_accs: list[float] = []
        for item in str(row.get("completed_metric_paths") or "").split(";"):
            if not item:
                continue
            metric_path = ROOT / item
            if not metric_path.exists():
                continue
            metric = json.loads(metric_path.read_text(encoding="utf-8"))
            if metric.get("test_accuracy") is not None:
                raw_accs.append(float(metric["test_accuracy"]))
        return {
            "test_accuracy_mean": row.get("test_accuracy_mean"),
            "test_accuracy_std": row.get("test_accuracy_std"),
            "macro_f1_mean": row.get("macro_f1_mean"),
            "macro_f1_std": row.get("macro_f1_std"),
            "n_folds_completed": completed,
            "n_folds_total": total,
            "folds": [{"test_accuracy": value} for value in raw_accs],
            "_source_path": "outputs/public_cv_parity/status.json",
            "_source_label": f"partial 5-fold x 3-seed CV ({completed}/{total} completed)",
        }
    return None


def first_metric(pattern: str) -> Path:
    matches = [p for p in ROOT.glob(pattern) if "latest" not in p.parts]
    if not matches:
        raise FileNotFoundError(pattern)
    return sorted(matches)[-1]


def metric_path(dataset: str, model: str, seed: int = 42) -> Path:
    if dataset == "AL6":
        if model == "baseline":
            return first_metric("outputs/ddp_baseline/*/resnet50/test_metrics.json")
        return first_metric("outputs/ddp_aafnet_v2/*/resnet50/test_metrics.json")
    return first_metric(f"outputs/asp_as25_{model}_{dataset}_seed{seed}/*/resnet50/test_metrics.json")


def curve_path(dataset: str, model: str, seed: int = 42) -> Path:
    if dataset == "AL6":
        if model == "baseline":
            return first_metric("outputs/ddp_baseline/*/resnet50/training_curve.csv")
        return first_metric("outputs/ddp_aafnet_v2/*/resnet50/training_curve.csv")
    return first_metric(f"outputs/asp_as25_{model}_{dataset}_seed{seed}/*/resnet50/training_curve.csv")


def read_curve(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out: dict[str, float] = {}
            for key, value in row.items():
                try:
                    out[key] = float(value)
                except (TypeError, ValueError):
                    out[key] = math.nan
            rows.append(out)
    return rows


def collect_audit_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for raw in RAW_DATASETS:
        audit = load_json(f"outputs/data_audit/{raw}/audit.json")
        clean = raw if raw == "AL6" else f"{raw}_clean"
        strict = f"{raw}_strict"
        clean_action = None
        if raw != "AL6":
            clean_action = load_json(f"outputs/data_audit/{raw}/dedup_action_clean.json")
        strict_action = load_json(f"outputs/data_audit/{raw}/dedup_action_strict.json")
        variants = [raw]
        if clean != raw:
            variants.append(clean)
        variants.append(strict)
        for variant in variants:
            if not (ROOT / "data" / "processed" / variant).exists():
                continue
            train_counts = split_counts(variant, "train")
            test_counts = split_counts(variant, "test")
            total = sum(train_counts.values()) + sum(test_counts.values())
            min_class = min((train_counts.get(k, 0) + test_counts.get(k, 0)) for k in set(train_counts) | set(test_counts))
            max_class = max((train_counts.get(k, 0) + test_counts.get(k, 0)) for k in set(train_counts) | set(test_counts))
            rows.append(
                {
                    "raw_dataset": raw,
                    "analysis_variant": variant,
                    "n_train": sum(train_counts.values()),
                    "n_test": sum(test_counts.values()),
                    "n_total": total,
                    "n_classes": len(set(train_counts) | set(test_counts)),
                    "class_imbalance_max_min": max_class / max(min_class, 1),
                    "exact_cross_split_duplicates_raw": audit["exact_duplicate"]["train_test_cross_count"],
                    "near_duplicate_pairs_raw": audit["near_duplicate_phash"]["n_pairs_found"],
                    "clean_train_dropped": 0 if clean_action is None else clean_action["n_train_dropped"],
                    "clean_test_dropped": 0 if clean_action is None else clean_action["n_test_dropped"],
                    "strict_train_dropped": strict_action["n_train_dropped"],
                    "strict_test_dropped": strict_action["n_test_dropped"],
                }
            )
    return rows


def collect_performance_rows() -> list[dict[str, object]]:
    sig = load_json("outputs/sig_collect_v2/significance.json")
    asp = load_json("outputs/p1_asp_as25_summary.json")
    strict_rows = load_json("outputs/p2_strict_extended.json")
    out: list[dict[str, object]] = []

    for model in ["baseline", "aafnet"]:
        m = load_metrics(metric_path("AL6", model))
        fold_accs = sig["fold_accs"][model]
        out.append(
            {
                "dataset": "AL6",
                "model": model,
                "test_acc_mean": sig["mean"][model],
                "test_acc_std": sig["std"][model],
                "macro_f1_mean": m.get("macro_f1"),
                "macro_f1_std": None,
                "n_repeats": len(fold_accs),
                "raw_accs": ";".join(str(v) for v in fold_accs),
                "source": "5-fold x 3-seed CV for accuracy; single headline ckpt for macro-F1",
            }
        )

    for dataset in ["ASP_clean", "AS25_clean"]:
        for model in ["baseline", "aafnet"]:
            cv = public_cv_summary(dataset, model)
            if cv is not None:
                raw_accs = [str(f["test_accuracy"]) for f in cv.get("folds", []) if "test_accuracy" in f]
                out.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "test_acc_mean": cv["test_accuracy_mean"],
                        "test_acc_std": cv["test_accuracy_std"],
                        "macro_f1_mean": cv.get("macro_f1_mean"),
                        "macro_f1_std": cv.get("macro_f1_std"),
                        "n_repeats": cv["n_folds_completed"],
                        "raw_accs": ";".join(raw_accs),
                        "source": cv.get("_source_label", "5-fold x 3-seed CV"),
                        "source_path": cv["_source_path"],
                    }
                )
                continue
            r = asp[dataset][model]
            out.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "test_acc_mean": r["test_acc_mean"],
                    "test_acc_std": r["test_acc_std"],
                    "macro_f1_mean": r.get("macro_f1_mean"),
                    "macro_f1_std": r.get("macro_f1_std"),
                    "n_repeats": r["n_seeds"],
                    "raw_accs": ";".join(str(v) for v in r["raw_accs"]),
                    "source": "3-seed single split fallback",
                }
            )

    strict_lookup = {(r["label"], r["eval_ds"]): r for r in strict_rows if "error" not in r}
    for dataset in ANALYSIS_DATASETS:
        stem = dataset.split("_")[0]
        strict_ds = f"{stem}_strict" if dataset != "AL6" else "AL6_strict"
        for model in ["baseline", "aafnet"]:
            key_clean = (f"{model}_{stem}", dataset)
            key_strict = (f"{model}_{stem}", strict_ds)
            if key_clean in strict_lookup and key_strict in strict_lookup:
                for r, split_type in [(strict_lookup[key_clean], "clean"), (strict_lookup[key_strict], "strict")]:
                    out.append(
                        {
                            "dataset": dataset,
                            "model": model,
                            "split_type": split_type,
                            "strict_eval_dataset": r["eval_ds"],
                            "strict_eval_accuracy": r["test_accuracy"],
                            "strict_eval_macro_f1": r["macro_f1"],
                            "n_test": r["n_test"],
                            "source": "single seed=42 strict evaluation",
                        }
                    )
    return out


def f_tc0_samples() -> None:
    rng = random.Random(42)
    rows: list[dict[str, object]] = []
    fig = plt.figure(figsize=(DOUBLE_W, 90 * MM))
    gs = fig.add_gridspec(3, 8, left=0.055, right=0.99, top=0.92, bottom=0.055, wspace=0.025, hspace=0.16)
    for r, dataset in enumerate(ANALYSIS_DATASETS):
        mapping = split_mapping(dataset, "train")
        labels = class_names(dataset)
        grouped: dict[str, list[dict[str, str]]] = {}
        for row in mapping:
            grouped.setdefault(row["标签"], []).append(row)
        labs = sorted(grouped, key=lambda x: int(x))
        if dataset == "AL6":
            by_file = {row["文件名"]: row for row in mapping}
            chosen = [by_file[name] for name in AL6_SAMPLE_FILES if name in by_file]
        else:
            chosen = []
            for lab in labs:
                chosen.append(rng.choice(grouped[lab]))
            while len(chosen) < 8:
                chosen.append(rng.choice(mapping))
            chosen = chosen[:8]
        for c, row in enumerate(chosen):
            ax = fig.add_subplot(gs[r, c])
            img_path = ROOT / "data" / "processed" / dataset / "train" / row["文件名"]
            try:
                img = Image.open(img_path).convert("RGB")
                img = ImageOps.fit(img, (180, 180), method=Image.Resampling.LANCZOS)
                ax.imshow(img)
            except Exception:
                ax.text(0.5, 0.5, "missing", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            lab_i = int(row["标签"])
            cname = labels[lab_i - 1] if 0 < lab_i <= len(labels) else f"C{lab_i}"
            ax.set_title(short_label(cname, 13), fontsize=4.8, pad=1.5)
            rows.append(
                {
                    "dataset": dataset,
                    "split": "train",
                    "panel_row": r,
                    "panel_col": c,
                    "label": lab_i,
                    "class_name": cname,
                    "file_name": row["文件名"],
                }
            )
        pos = fig.axes[r * 8].get_position()
        fig.text(0.015, (pos.y0 + pos.y1) / 2, DISPLAY[dataset], rotation=90, ha="center", va="center", fontsize=7, fontweight="bold")
    fig.text(0.055, 0.975, "Three-corpus image samples", ha="left", va="top", fontsize=7.3, fontweight="bold")
    fig.text(0.055, 0.948, "Representative train images from the analysis split used in experiments.", ha="left", va="top", fontsize=5.8, color="#555555")
    write_csv("F_TC0_three_corpus_samples_source.csv", rows)
    save_figure(fig, "F_TC0_three_corpus_samples.png")


def f_tc1_audit() -> None:
    rows = collect_audit_rows()
    write_csv("F_TC1_three_corpus_audit_source.csv", rows)
    analysis = [r for r in rows if r["analysis_variant"] in ANALYSIS_DATASETS]
    raw_rows = {r["raw_dataset"]: r for r in rows if r["analysis_variant"] == r["raw_dataset"]}
    strict_rows = {r["raw_dataset"]: r for r in rows if r["analysis_variant"].endswith("_strict")}

    fig = plt.figure(figsize=(DOUBLE_W, 112 * MM))
    gs = fig.add_gridspec(2, 2, left=0.075, right=0.985, top=0.92, bottom=0.10, wspace=0.36, hspace=0.48)

    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(analysis))
    train = np.array([r["n_train"] for r in analysis], dtype=float)
    test = np.array([r["n_test"] for r in analysis], dtype=float)
    ax.bar(x, train, color="#B8D8E8", edgecolor="black", linewidth=0.45, label="Train")
    ax.bar(x, test, bottom=train, color="#EBCB8B", edgecolor="black", linewidth=0.45, label="Test")
    for i, total in enumerate(train + test):
        ax.text(i, total + max(train + test) * 0.025, f"{int(total):,}", ha="center", va="bottom", fontsize=5.5)
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY[r["analysis_variant"]] for r in analysis])
    ax.set_ylabel("Images")
    ax.set_title("Analysis split size")
    ax.legend(loc="upper left")
    panel_label(ax, "a")
    polish(ax, grid=True)

    ax = fig.add_subplot(gs[0, 1])
    n_cls = [r["n_classes"] for r in analysis]
    imb = [r["class_imbalance_max_min"] for r in analysis]
    ax.bar(x - 0.17, n_cls, width=0.34, color=COL["blue"], edgecolor="black", linewidth=0.45, label="Classes")
    ax2 = ax.twinx()
    ax2.bar(x + 0.17, imb, width=0.34, color=COL["purple"], edgecolor="black", linewidth=0.45, label="Max/min")
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY[r["analysis_variant"]] for r in analysis])
    ax.set_ylabel("Classes")
    ax2.set_ylabel("Class imbalance")
    ax.set_title("Label-space difficulty")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    panel_label(ax, "b")
    polish(ax, grid=True)

    ax = fig.add_subplot(gs[1, 0])
    width = 0.28
    exact = [raw_rows[d]["exact_cross_split_duplicates_raw"] for d in RAW_DATASETS]
    near = [raw_rows[d]["near_duplicate_pairs_raw"] for d in RAW_DATASETS]
    strict_drop = [strict_rows[d]["strict_test_dropped"] for d in RAW_DATASETS]
    ax.bar(x - width, exact, width=width, color="#D9D9D9", edgecolor="black", linewidth=0.45, label="Exact cross")
    ax.bar(x, near, width=width, color="#A6CEE3", edgecolor="black", linewidth=0.45, label="pHash pairs")
    ax.bar(x + width, strict_drop, width=width, color="#FDBF6F", edgecolor="black", linewidth=0.45, label="Strict test drops")
    ax.set_xticks(x)
    ax.set_xticklabels(["AL6", "ASP", "AS25"])
    ax.set_ylabel("Count")
    ax.set_title("Leakage and near-duplicate audit")
    ax.legend(loc="upper left", ncol=1)
    panel_label(ax, "c")
    polish(ax, grid=True)

    ax = fig.add_subplot(gs[1, 1])
    for i, raw in enumerate(RAW_DATASETS):
        rr = raw_rows[raw]
        cr = next(r for r in rows if r["raw_dataset"] == raw and r["analysis_variant"] == (raw if raw == "AL6" else f"{raw}_clean"))
        sr = strict_rows[raw]
        vals = [rr["n_total"], cr["n_total"], sr["n_total"]]
        base = vals[0]
        ax.plot([0, 1, 2], [v / base * 100 for v in vals], marker="o", lw=1.2, ms=3.5, label=raw)
        for j, v in enumerate(vals):
            ax.text(j, v / base * 100 - 2.3, f"{int(v):,}", ha="center", va="top", fontsize=5.1)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Raw", "Clean", "Strict"])
    ax.set_ylim(90, 101)
    ax.set_ylabel("Retained images (%)")
    ax.set_title("Audit-cleaning retention")
    ax.legend(loc="lower left")
    panel_label(ax, "d")
    polish(ax, grid=True)

    save_figure(fig, "F_TC1_three_corpus_audit.png")


def f_tc2_performance() -> None:
    rows = collect_performance_rows()
    perf = [r for r in rows if "test_acc_mean" in r]
    write_csv("F_TC2_three_corpus_performance_source.csv", perf)
    by = {(r["dataset"], r["model"]): r for r in perf}
    datasets = ANALYSIS_DATASETS
    x = np.arange(len(datasets))
    width = 0.32
    fig = plt.figure(figsize=(DOUBLE_W, 114 * MM))
    gs = fig.add_gridspec(2, 2, left=0.075, right=0.985, top=0.92, bottom=0.12, wspace=0.34, hspace=0.50)

    ax = fig.add_subplot(gs[0, 0])
    for i, model in enumerate(["baseline", "aafnet"]):
        vals = [pct(by[(d, model)]["test_acc_mean"]) for d in datasets]
        errs = [pct(by[(d, model)]["test_acc_std"]) for d in datasets]
        ax.bar(x + (i - 0.5) * width, vals, width=width, yerr=errs, color=COL[model], edgecolor="black", linewidth=0.45, capsize=1.6, label=MODEL_LABEL[model])
        for j, d in enumerate(datasets):
            raw = [float(v) * 100 for v in str(by[(d, model)]["raw_accs"]).split(";") if v]
            jitter = np.linspace(-0.055, 0.055, len(raw)) if raw else []
            ax.scatter(np.full(len(raw), x[j] + (i - 0.5) * width) + jitter, raw, s=8, color="white", edgecolor="black", linewidth=0.3, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY[d] for d in datasets])
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(58, 101)
    ax.set_title("Accuracy across all corpora")
    ax.legend(loc="lower left")
    panel_label(ax, "a")
    polish(ax, grid=True)

    ax = fig.add_subplot(gs[0, 1])
    for i, model in enumerate(["baseline", "aafnet"]):
        vals = [pct(by[(d, model)]["macro_f1_mean"]) for d in datasets]
        ax.bar(x + (i - 0.5) * width, vals, width=width, color=COL[model], edgecolor="black", linewidth=0.45, label=MODEL_LABEL[model])
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY[d] for d in datasets])
    ax.set_ylabel("Macro-F1 (%)")
    ax.set_ylim(58, 101)
    ax.set_title("Class-balanced performance")
    panel_label(ax, "b")
    polish(ax, grid=True)

    ax = fig.add_subplot(gs[1, 0])
    deltas = [pct(by[(d, "aafnet")]["test_acc_mean"] - by[(d, "baseline")]["test_acc_mean"]) for d in datasets]
    colors = [COL["teal"] if v >= 0 else COL["red"] for v in deltas]
    ax.axhline(0, color="#666666", lw=0.7)
    ax.bar(x, deltas, color=colors, edgecolor="black", linewidth=0.45, width=0.55)
    for i, v in enumerate(deltas):
        ax.text(i, v + (0.16 if v >= 0 else -0.16), f"{v:+.2f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=5.8)
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY[d] for d in datasets])
    ax.set_ylabel("AAFNet - RN50 (pp)")
    ax.set_title("Effect direction")
    panel_label(ax, "c")
    polish(ax, grid=True)

    ax = fig.add_subplot(gs[1, 1])
    strict = [r for r in rows if "strict_eval_accuracy" in r]
    sby = {(r["dataset"], r["model"], r["split_type"]): r for r in strict}
    for i, model in enumerate(["baseline", "aafnet"]):
        clean = [pct(sby[(d, model, "clean")]["strict_eval_accuracy"]) for d in datasets]
        strict_acc = [pct(sby[(d, model, "strict")]["strict_eval_accuracy"]) for d in datasets]
        xpos = x + (i - 0.5) * width
        ax.scatter(xpos - 0.045, clean, marker="o", facecolor="white", edgecolor=COL[model], s=26, linewidth=0.9)
        ax.scatter(xpos + 0.045, strict_acc, marker="o", facecolor=COL[model], edgecolor="black", s=26, linewidth=0.45)
        for j in range(len(datasets)):
            ax.plot([xpos[j] - 0.045, xpos[j] + 0.045], [clean[j], strict_acc[j]], color=COL[model], lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY[d] for d in datasets])
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(58, 101)
    ax.set_title("Clean-to-strict stability")
    ax.scatter([], [], marker="o", facecolor="white", edgecolor="black", s=24, label="Clean")
    ax.scatter([], [], marker="o", facecolor="black", edgecolor="black", s=24, label="Strict")
    ax.legend(loc="lower left", ncol=2)
    panel_label(ax, "d")
    polish(ax, grid=True)

    save_figure(fig, "F_TC2_three_corpus_performance.png")


def f_tc3_confusions() -> None:
    rows: list[dict[str, object]] = []
    fig = plt.figure(figsize=(DOUBLE_W, 150 * MM))
    gs = fig.add_gridspec(3, 2, left=0.070, right=0.945, top=0.860, bottom=0.065, wspace=0.20, hspace=0.46)
    last_im = None
    for r, dataset in enumerate(ANALYSIS_DATASETS):
        names = class_names(dataset)
        for c, model in enumerate(["baseline", "aafnet"]):
            ax = fig.add_subplot(gs[r, c])
            metrics = load_metrics(metric_path(dataset, model))
            cm = np.array(metrics["confusion_matrix"], dtype=float)
            denom = cm.sum(axis=1, keepdims=True)
            norm = np.divide(cm, denom, out=np.zeros_like(cm), where=denom != 0)
            last_im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
            n = cm.shape[0]
            tick_step = 1 if n <= 10 else 3
            ticks = np.arange(0, n, tick_step)
            tick_labels = [str(i + 1) for i in ticks]
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.set_xticklabels(tick_labels)
            ax.set_yticklabels(tick_labels)
            if r == 2:
                ax.set_xlabel("Predicted class")
            if c == 0:
                ax.set_ylabel(f"{DISPLAY[dataset]}\nTrue class")
            ax.set_title(f"{MODEL_LABEL[model]}  acc={metrics['test_accuracy']*100:.1f}%")
            if n <= 9:
                for i in range(n):
                    for j in range(n):
                        val = norm[i, j]
                        if val >= 0.08 or i == j:
                            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=4.5, color="white" if val > 0.55 else "black")
            for i in range(n):
                for j in range(n):
                    rows.append(
                        {
                            "dataset": dataset,
                            "model": model,
                            "true_class_index": i + 1,
                            "pred_class_index": j + 1,
                            "true_class_name": names[i] if i < len(names) else f"C{i+1}",
                            "pred_class_name": names[j] if j < len(names) else f"C{j+1}",
                            "count": int(cm[i, j]),
                            "row_normalized": float(norm[i, j]),
                        }
                    )
            if r == 0 and c == 0:
                panel_label(ax, "a")
            elif r == 0 and c == 1:
                panel_label(ax, "b")
    if last_im is not None:
        cax = fig.add_axes([0.958, 0.16, 0.012, 0.68])
        cb = fig.colorbar(last_im, cax=cax)
        cb.set_label("Row-normalized")
    fig.text(0.070, 0.975, "Confusion matrices for all analysis corpora", ha="left", va="top", fontsize=7.2, fontweight="bold")
    fig.text(0.070, 0.945, "Labels use class indices; full class names are recorded in the source-data CSV.", ha="left", va="top", fontsize=5.8, color="#555555")
    write_csv("F_TC3_three_corpus_confusions_source.csv", rows)
    save_figure(fig, "F_TC3_three_corpus_confusions.png")


def f_tc4_training() -> None:
    rows: list[dict[str, object]] = []
    fig = plt.figure(figsize=(DOUBLE_W, 116 * MM))
    gs = fig.add_gridspec(1, 3, left=0.070, right=0.985, top=0.86, bottom=0.17, wspace=0.32)
    for i, dataset in enumerate(ANALYSIS_DATASETS):
        ax = fig.add_subplot(gs[0, i])
        for model, color in [("baseline", COL["baseline"]), ("aafnet", COL["aafnet"])]:
            p = curve_path(dataset, model, seed=42)
            curve = read_curve(p)
            epochs = [int(r["epoch"]) for r in curve]
            train = [r.get("train_acc", math.nan) * 100 for r in curve]
            val = [r.get("val_acc", math.nan) * 100 for r in curve]
            ax.plot(epochs, val, color=color, lw=1.35, label=f"{MODEL_LABEL[model]} val")
            ax.plot(epochs, train, color=color, lw=0.75, linestyle="--", alpha=0.45, label=f"{MODEL_LABEL[model]} train")
            for erow in curve:
                rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "source_curve": str(p.relative_to(ROOT)),
                        "epoch": int(erow["epoch"]),
                        "train_acc": erow.get("train_acc", math.nan),
                        "val_acc": erow.get("val_acc", math.nan),
                        "train_loss": erow.get("train_loss", math.nan),
                        "val_loss": erow.get("val_loss", math.nan),
                    }
                )
        ax.set_title(DISPLAY[dataset])
        ax.set_xlabel("Epoch")
        if i == 0:
            ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 102)
        polish(ax, grid=True)
        panel_label(ax, chr(ord("a") + i))
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles[:4], labels[:4], loc="upper center", ncol=4, bbox_to_anchor=(0.54, 0.945))
    fig.text(0.070, 0.985, "Seed-42/reference training dynamics", ha="left", va="top", fontsize=7.2, fontweight="bold")
    fig.text(0.070, 0.918, "Solid lines are validation accuracy; dashed lines are training accuracy.", ha="left", va="top", fontsize=5.8, color="#555555")
    write_csv("F_TC4_three_corpus_training_source.csv", rows)
    save_figure(fig, "F_TC4_three_corpus_training.png")


def f_tc5_probe_matrix() -> None:
    downstream = load_json("outputs/downstream/20260509_005735/results.json")
    strict_rows = collect_performance_rows()
    sby = {(r["dataset"], r["model"], r.get("split_type")): r for r in strict_rows if "strict_eval_accuracy" in r}
    perf = [r for r in strict_rows if "test_acc_mean" in r]
    pby = {(r["dataset"], r["model"]): r for r in perf}

    metrics = [
        ("Clean acc", "clean_acc", "higher"),
        ("Strict acc", "strict_acc", "higher"),
        ("Retr. Top1", "retr_top1", "higher"),
        ("Retr. mAP", "retr_map", "higher"),
        ("Near-dup AUROC", "near_dup", "higher"),
        ("OOD AUROC", "ood", "higher"),
        ("Few-shot acc", "fewshot", "higher"),
    ]
    values: dict[tuple[str, str, str], float | None] = {}
    for d in ANALYSIS_DATASETS:
        for model in ["baseline", "aafnet"]:
            values[(d, model, "clean_acc")] = pby[(d, model)]["test_acc_mean"]
            values[(d, model, "strict_acc")] = sby[(d, model, "strict")]["strict_eval_accuracy"]
            values[(d, model, "retr_top1")] = downstream["3.1_retrieval"][model][d]["top1_mean"]
            values[(d, model, "retr_map")] = downstream["3.1_retrieval"][model][d]["mAP_mean"]
            values[(d, model, "near_dup")] = downstream["3.4_neardup"][model][d]["AUROC"]
            values[(d, model, "ood")] = None
            values[(d, model, "fewshot")] = None
    for d in ["ASP_clean", "AS25_clean"]:
        for model in ["baseline", "aafnet"]:
            values[(d, model, "ood")] = downstream["3.2_ood"][model][d]["Energy"]["AUROC"]
            values[(d, model, "fewshot")] = downstream["3.6_fewshot"][model][d]["acc_mean"]

    delta = np.full((len(ANALYSIS_DATASETS), len(metrics)), np.nan, dtype=float)
    rows: list[dict[str, object]] = []
    for i, d in enumerate(ANALYSIS_DATASETS):
        for j, (label, key, direction) in enumerate(metrics):
            b = values[(d, "baseline", key)]
            a = values[(d, "aafnet", key)]
            if b is not None and a is not None:
                delta[i, j] = (a - b) * 100
            rows.append(
                {
                    "dataset": d,
                    "metric": label,
                    "baseline": "" if b is None else b,
                    "aafnet": "" if a is None else a,
                    "delta_pp": "" if b is None or a is None else (a - b) * 100,
                    "higher_is_better": direction == "higher",
                }
            )
    write_csv("F_TC5_three_corpus_probe_matrix_source.csv", rows)

    fig, ax = plt.subplots(figsize=(DOUBLE_W, 82 * MM))
    masked = np.ma.masked_invalid(delta)
    norm = TwoSlopeNorm(vcenter=0, vmin=-4.0, vmax=6.0)
    cmap = mpl.colors.LinearSegmentedColormap.from_list("delta", ["#C44E52", "#F7F7F7", "#009E73"])
    im = ax.imshow(masked, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(len(ANALYSIS_DATASETS)))
    ax.set_xticklabels([m[0] for m in metrics], rotation=35, ha="right")
    ax.set_yticklabels([DISPLAY[d] for d in ANALYSIS_DATASETS])
    ax.set_title("AAFNet - RN50 deltas across completed probes")
    for i, d in enumerate(ANALYSIS_DATASETS):
        for j, (label, key, _) in enumerate(metrics):
            b = values[(d, "baseline", key)]
            a = values[(d, "aafnet", key)]
            if b is None or a is None:
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=5.4, color="#666666")
                continue
            dv = (a - b) * 100
            text = f"{dv:+.1f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=5.5, color="white" if abs(dv) > 2.4 else "black")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.02)
    cbar.set_label("Delta (percentage points)")
    fig.text(0.085, 0.085, "OOD and few-shot cells are defined only for the two external public corpora.", ha="left", va="center", fontsize=5.8, color="#555555")
    save_figure(fig, "F_TC5_three_corpus_probe_matrix.png")


def f_tc8_public_experiment_coverage() -> None:
    task_order = [
        ("ASP_clean", "baseline", "ASP RN50"),
        ("ASP_clean", "aafnet", "ASP AAFNet"),
        ("AS25_clean", "baseline", "AS25 RN50"),
        ("AS25_clean", "aafnet", "AS25 AAFNet"),
    ]
    columns = [
        ("cv_done", "CV\ncomplete"),
        ("robust_ready", "Robust\nready"),
        ("robust_done", "Robust\ndone"),
        ("cal_ready", "Calib.\nready"),
        ("cal_done", "Calib.\ndone"),
        ("rot_ready", "Rotation\nready"),
        ("rot_done", "Rotation\ndone"),
    ]

    cv_rows = read_source_csv("public_cv_remaining_queue_source.csv")
    robust_rows = read_source_csv("public_robustness_attribution_queue_source.csv")
    calrot_rows = read_source_csv("public_calibration_rotation_queue_source.csv")
    follow_rows = read_source_csv("public_followup_probe_status_source.csv")
    follow = {(r["dataset"], r["role"]): r for r in follow_rows}

    def count(rows: list[dict[str, str]], dataset: str, role: str, predicate) -> int:
        return sum(1 for row in rows if row.get("dataset") == dataset and row.get("role") == role and predicate(row))

    source_rows: list[dict[str, object]] = []
    matrix = np.zeros((len(task_order), len(columns)), dtype=float)
    for i, (dataset, role, label) in enumerate(task_order):
        frow = follow.get((dataset, role), {})
        values = {
            "cv_done": count(cv_rows, dataset, role, lambda r: r.get("status") == "complete"),
            "robust_ready": count(robust_rows, dataset, role, lambda r: r.get("status") in {"checkpoint_ready", "robustness_complete"}),
            "robust_done": int(frow.get("robustness_complete") or 0),
            "cal_ready": count(calrot_rows, dataset, role, lambda r: r.get("status") == "checkpoint_ready"),
            "cal_done": int(frow.get("calibration_complete") or 0),
            "rot_ready": count(calrot_rows, dataset, role, lambda r: r.get("status") == "checkpoint_ready"),
            "rot_done": int(frow.get("rotation_complete") or 0),
        }
        for j, (key, display) in enumerate(columns):
            value = values[key]
            matrix[i, j] = value / 15.0
            source_rows.append(
                {
                    "dataset": dataset,
                    "role": role,
                    "row_label": label,
                    "metric": key,
                    "metric_label": display.replace("\n", " "),
                    "completed_cells": value,
                    "target_cells": 15,
                    "completion_fraction": value / 15.0,
                }
            )

    write_csv("F_TC8_public_experiment_coverage_source.csv", source_rows)

    fig, ax = plt.subplots(figsize=(DOUBLE_W, 82 * MM))
    cmap = mpl.colors.LinearSegmentedColormap.from_list("coverage", ["#F4F4F4", "#B8D8E8", "#0072B2"])
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap=cmap, aspect="auto")
    ax.set_xticks(np.arange(len(columns)))
    ax.set_xticklabels([label for _, label in columns])
    ax.set_yticks(np.arange(len(task_order)))
    ax.set_yticklabels([label for _, _, label in task_order])
    ax.set_title("Public-corpus experiment coverage")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = int(round(matrix[i, j] * 15))
            ax.text(j, i, f"{value}/15", ha="center", va="center", fontsize=5.8, color="white" if matrix[i, j] > 0.55 else "#1F1F1F")
    cbar = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.02)
    cbar.set_label("Completion fraction")
    fig.text(
        0.078,
        0.060,
        "Ready means a completed public-CV checkpoint exists; done means the corresponding fold-level probe JSON has been produced.",
        ha="left",
        va="center",
        fontsize=5.8,
        color="#555555",
    )
    save_figure(fig, "F_TC8_public_experiment_coverage.png")


def rebuild_summary_json() -> None:
    support_rows = [
        {
            "requirement": "Three-corpus data audit and representative samples",
            "status": "supported",
            "evidence": "outputs/data_audit/{AL6,ASP,AS25}/audit.json; data/processed/*_mapping.csv",
            "boundary": "Source AL6 images include some orientation variation; paper sample panel uses fixed upright display examples.",
        },
        {
            "requirement": "Three-corpus clean performance comparison",
            "status": "partially supported",
            "evidence": "outputs/sig_collect_v2/significance.json; outputs/p1_asp_as25_summary.json; outputs/public_cv_parity/status.json",
            "boundary": "Public corpora use completed or partial public-CV summaries where available, with partial rows explicitly labeled by completed fold count; otherwise they remain 3-seed single-split fallbacks.",
        },
        {
            "requirement": "Three-corpus confusion matrices",
            "status": "supported",
            "evidence": "outputs/ddp_{baseline,aafnet_v2}/*/test_metrics.json; outputs/asp_as25_*/*/test_metrics.json",
            "boundary": "AL6 confusion uses the headline single checkpoint; ASP/AS25 use seed-42 reference checkpoints.",
        },
        {
            "requirement": "Three-corpus training dynamics",
            "status": "supported",
            "evidence": "training_curve.csv files for AL6 headline checkpoints and ASP/AS25 seed-42 runs",
            "boundary": "Reference curves only, not fold-averaged curves.",
        },
        {
            "requirement": "Three-corpus strict-test stability",
            "status": "supported",
            "evidence": "outputs/p2_strict_extended.json",
            "boundary": "Single seed=42 checkpoint evaluation.",
        },
        {
            "requirement": "Three-corpus downstream feature probes",
            "status": "partially supported",
            "evidence": "outputs/downstream/20260509_005735/results.json",
            "boundary": "Retrieval and near-duplicate probes cover all three corpora; OOD/few-shot are defined only for ASP_clean/AS25_clean as external corpora.",
        },
        {
            "requirement": "ASP/AS25 seed-42 robustness, calibration and rotation parity",
            "status": "partially supported",
            "evidence": "outputs/three_corpus_parity/latest/results.json; paper/figures/F_TC6_three_corpus_parity.png; outputs/public_calibration_rotation/queue.md; outputs/public_followup_probes/summary.md",
            "boundary": "Existing checkpoints support paired seed-42 probes; fold-level calibration/rotation is now queued and has a result aggregator for completed public-CV checkpoints, but fold-level probe JSON outputs are not yet complete.",
        },
        {
            "requirement": "ASP/AS25 seed-42 inference and GradCAM parity",
            "status": "partially supported",
            "evidence": "paper/figures/F_TC7_public_interpretability.png; paper/figures/nature_source_data/F_TC7_public_interpretability_predictions_source.csv; paper/figures/figure_rotation_audit.md",
            "boundary": "Existing checkpoints support qualitative public-corpus inference traces. Display-only rotations are audited separately; this is not a multi-seed interpretability study.",
        },
        {
            "requirement": "Full AL6-level public-corpus validation",
            "status": "in progress",
            "evidence": "outputs/public_cv_parity/status.md; outputs/public_cv_parity/remaining_queue.md; outputs/public_cv_parity/smoke_status.md; outputs/cv_asp_baseline/public_cv_parity_v1/resnet50/seed42_fold*_train/*/resnet50/test_metrics.json",
            "boundary": "Public CV launcher/status scanner, remaining-cell queue, resumable fold skipping, 2-fold smoke checks, and the completed ASP_clean baseline full-CV summary now exist; the remaining public-corpus baseline/AAFNet summaries are still required before claiming identical statistical depth across all three corpora.",
        },
        {
            "requirement": "ASP/AS25 full robustness attribution and multi-seed interpretability parity",
            "status": "in progress",
            "evidence": "scripts/run_public_robustness_attribution.py; outputs/public_robustness_attribution/queue.md; paper/figures/nature_source_data/public_robustness_attribution_queue_source.csv",
            "boundary": "Fold-level public-corpus robustness attribution is now queued and resumable for completed CV checkpoints; matched multi-seed GradCAM/activation panels remain to be generated.",
        },
    ]
    summary = {
        "audit_rows": collect_audit_rows(),
        "performance_rows": collect_performance_rows(),
        "training_support_audit": support_rows,
        "figures": TARGETS,
        "source_csv": [
            "F_TC0_three_corpus_samples_source.csv",
            "F_TC1_three_corpus_audit_source.csv",
            "F_TC2_three_corpus_performance_source.csv",
            "F_TC3_three_corpus_confusions_source.csv",
            "F_TC4_three_corpus_training_source.csv",
            "F_TC5_three_corpus_probe_matrix_source.csv",
            "F_TC6_three_corpus_parity_source.csv",
            "F_TC7_public_interpretability_selected_source.csv",
            "F_TC7_public_interpretability_predictions_source.csv",
            "F_TC8_public_experiment_coverage_source.csv",
            "figure_rotation_audit.csv",
            "public_cv_remaining_queue_source.csv",
            "public_robustness_attribution_queue_source.csv",
            "public_calibration_rotation_queue_source.csv",
            "public_followup_probe_status_source.csv",
            "public_robustness_fold_metrics_source.csv",
            "public_calibration_fold_metrics_source.csv",
            "public_rotation_fold_metrics_source.csv",
            "public_robustness_summary_source.csv",
            "public_calibration_summary_source.csv",
            "public_rotation_summary_source.csv",
        ],
    }
    (OUT_DIR / "three_corpus_completion_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv("F_TC_training_support_audit.csv", support_rows)

    lines = [
        "# Three-Corpus Completion Summary",
        "",
        "The manuscript originally had the richest experimental presentation for AL6. This package completes a parallel, auditable figure layer for AL6, ASP_clean and AS25_clean using existing cached results.",
        "",
        "## Generated figures",
        "",
    ]
    for rel in TARGETS:
        lines.append(f"- `{rel}`")
    lines += [
        "",
        "## Evidence boundaries",
        "",
        "- No new model training is performed by this script.",
        "- Public-corpus performance uses completed or partial public-CV summaries where available; partial rows are labeled by completed fold count, and missing public-corpus CV cells fall back to existing 3-seed single-split runs.",
        "- AL6 accuracy uses the existing 5-fold x 3-seed CV summary; AL6 confusion and training curves use the headline single checkpoint.",
        "- Robustness, calibration and rotation parity for ASP_clean/AS25_clean uses existing seed-42 checkpoints only.",
        "- Full public-corpus robustness attribution now has a resumable fold-level queue and result aggregator, but robustness JSON outputs are still pending.",
        "- Full public-corpus calibration and rotation now have a resumable fold-level queue and result aggregator, but fold-level JSON outputs are still pending.",
        "- Public-corpus interpretability parity uses selected ASP_clean/AS25_clean seed-42 inference traces, with selected-case and full prediction CSVs.",
        "- Display-only rotations for inference thumbnails and GradCAM overlays are audited in `paper/figures/figure_rotation_audit.md`.",
        "- OOD and few-shot probes are not defined for AL6 in the current downstream battery; those cells are explicitly marked n/a.",
        "",
        "## Training-artifact support audit",
        "",
        "| Requirement | Status | Evidence | Boundary |",
        "|---|---|---|---|",
    ]
    for row in support_rows:
        lines.append(f"| {row['requirement']} | {row['status']} | {row['evidence']} | {row['boundary']} |")
    lines.append("")
    (OUT_DIR / "three_corpus_completion_summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_qa() -> None:
    lines = [
        "# Three-Corpus Figure Completion QA",
        "",
        "- Backend: Python / matplotlib only.",
        "- Core conclusion: the paper now has a parallel audit, performance, confusion, training and feature-probe evidence chain for AL6, ASP_clean and AS25_clean.",
        "- Export contract: every F-TC PNG has SVG and PDF companions in `paper/figures/nature_exports/`.",
        "- Source-data contract: every F-TC figure has a CSV in `paper/figures/nature_source_data/`.",
        "- Rotation QA: display-only rotations are recorded in source-data fields ending in `rotation_deg`/`rotation_degs` and summarized in `paper/figures/figure_rotation_audit.md`.",
        "- Integrity note: no model was retrained; figures read existing cached split metadata, test metrics, training curves and downstream result JSON files.",
        "- Training support boundary: current artifacts support a completed visual/data-analysis layer, but not a claim that ASP_clean and AS25_clean have AL6-level 5-fold statistical depth.",
        "",
        "## Generated figure checks",
        "",
    ]
    for rel in TARGETS:
        stem = rel.replace("/", "__").replace(".png", "")
        png = FIG_DIR / rel
        svg = EXPORT_DIR / f"{stem}.svg"
        pdf = EXPORT_DIR / f"{stem}.pdf"
        lines.append(f"- `{rel}`: {'OK' if png.exists() and svg.exists() and pdf.exists() else 'MISSING'}")
    QA_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    print("[completing three-corpus figures]")
    f_tc0_samples()
    f_tc1_audit()
    f_tc2_performance()
    f_tc3_confusions()
    f_tc4_training()
    f_tc5_probe_matrix()
    f_tc8_public_experiment_coverage()
    rebuild_summary_json()
    write_qa()
    print(f"[ok] QA written to {QA_PATH}")


if __name__ == "__main__":
    main()
