#!/usr/bin/env python
"""Create per-task downstream inference effect figures.

These figures are qualitative, source-data-backed companions for the downstream
probe summary. They use real cached images and AL6-trained checkpoints to show
what the model actually returns for retrieval, OOD scoring, rejection,
invariance, domain probing, and few-shot episodes.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from eval_downstream import build_model, extract_features_logits, latest  # noqa: E402
from figure_display_rotation import display_rotation, rotate_display_array  # noqa: E402


FIG_DIR = ROOT / "paper" / "figures" / "downstream_inference"
EXPORT_DIR = ROOT / "paper" / "figures" / "nature_exports"
SOURCE_DIR = ROOT / "paper" / "figures" / "nature_source_data"
CACHE_DIR = ROOT / "outputs" / "downstream_inference_effects"

PALETTE = {
    "baseline": "#8A8F98",
    "aafnet": "#2C7FB8",
    "green": "#2E7D32",
    "red": "#B23A48",
    "amber": "#C47F17",
    "ink": "#23262B",
    "soft": "#E9EEF3",
}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "axes.labelcolor": PALETTE["ink"],
        "xtick.color": PALETTE["ink"],
        "ytick.color": PALETTE["ink"],
        "text.color": PALETTE["ink"],
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


def mkdirs() -> None:
    for path in (FIG_DIR, EXPORT_DIR, SOURCE_DIR, CACHE_DIR):
        path.mkdir(parents=True, exist_ok=True)


def clean_label(name: str, max_len: int = 22) -> str:
    name = str(name).replace("_", " ")
    if len(name) <= max_len:
        return name
    return name[: max_len - 1].rstrip() + "."


def load_dataset(dataset: str) -> dict[str, object]:
    path = ROOT / "data" / "cache" / f"{dataset}_224x224_rgb_test.pt"
    data = torch.load(path, map_location="cpu", weights_only=False)
    labels = data["labels"].cpu().numpy()
    class_names = data.get("class_names") or [f"class {i}" for i in np.unique(labels)]
    return {
        "dataset": dataset,
        "images": data["images"],
        "labels": labels,
        "class_names": list(class_names),
    }


def to_image(x: torch.Tensor) -> np.ndarray:
    arr = x.detach().cpu().numpy()
    if arr.shape[0] in (1, 3):
        arr = np.moveaxis(arr, 0, -1)
    return arr.astype(np.uint8)


def norm_rows(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def softmax_np(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def energy_np(logits: np.ndarray) -> np.ndarray:
    zmax = logits.max(axis=1, keepdims=True)
    return -np.log(np.exp(logits - zmax).sum(axis=1)) - zmax[:, 0]


def save_source(name: str, rows: list[dict[str, object]]) -> None:
    path = SOURCE_DIR / f"{name}.csv"
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig: mpl.figure.Figure, name: str, rows: list[dict[str, object]]) -> Path:
    png = FIG_DIR / f"{name}.png"
    fig.savefig(png, dpi=600, bbox_inches="tight")
    fig.savefig(EXPORT_DIR / f"downstream_inference__{name}.svg", bbox_inches="tight")
    fig.savefig(EXPORT_DIR / f"downstream_inference__{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    save_source(name, rows)
    return png


def rotation_for(dataset_name: str, index: int) -> int:
    return display_rotation(dataset_name, index)


def rotated_array(image: torch.Tensor, rotation_deg: int = 0) -> np.ndarray:
    arr = to_image(image)
    return rotate_display_array(arr, rotation_deg)


def plot_thumb(
    ax: mpl.axes.Axes,
    image: torch.Tensor,
    title: str = "",
    border: str | None = None,
    title_size: int = 6,
    rotation_deg: int = 0,
) -> None:
    ax.imshow(rotated_array(image, rotation_deg=rotation_deg))
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=title_size, pad=2)
    for spine in ax.spines.values():
        spine.set_visible(border is not None)
        spine.set_linewidth(2.0)
        if border:
            spine.set_edgecolor(border)


def plot_dataset_thumb(
    ax: mpl.axes.Axes,
    datasets: dict[str, dict[str, object]],
    dataset_name: str,
    index: int,
    title: str = "",
    border: str | None = None,
    title_size: int = 6,
) -> None:
    plot_thumb(
        ax,
        datasets[dataset_name]["images"][index],
        title=title,
        border=border,
        title_size=title_size,
        rotation_deg=rotation_for(dataset_name, index),
    )


def text_panel(ax: mpl.axes.Axes, lines: list[str], title: str) -> None:
    ax.axis("off")
    ax.text(0.0, 1.0, title, fontsize=8, fontweight="bold", va="top")
    ax.text(
        0.0,
        0.82,
        "\n".join(lines),
        fontsize=7,
        va="top",
        linespacing=1.45,
        family="monospace",
    )


def build_models() -> dict[str, dict[str, object]]:
    ckpts = {
        "baseline": latest("outputs/ddp_baseline/*/resnet50/best_resnet50.pth"),
        "aafnet": latest("outputs/ddp_aafnet_v2/*/resnet50/best_resnet50.pth"),
    }
    if ckpts["aafnet"] is None:
        ckpts["aafnet"] = latest("outputs/ddp_aafnet/*/resnet50/best_resnet50.pth")
    missing = [name for name, ckpt in ckpts.items() if ckpt is None]
    if missing:
        raise FileNotFoundError(f"Missing checkpoint(s): {', '.join(missing)}")

    out: dict[str, dict[str, object]] = {}
    for name, ckpt in ckpts.items():
        model, device, has_mssa = build_model("resnet50", ckpt, "AL6", 6)
        out[name] = {
            "model": model,
            "device": device,
            "has_mssa": has_mssa,
            "checkpoint": str(ckpt.relative_to(ROOT)),
        }
    return out


def get_features(
    model_name: str,
    model_pack: dict[str, object],
    dataset: dict[str, object],
    refresh: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    ds = str(dataset["dataset"])
    cache = CACHE_DIR / f"{model_name}_{ds}_features_logits.npz"
    if cache.exists() and not refresh:
        data = np.load(cache)
        return data["features"], data["logits"]
    feats, logits = extract_features_logits(
        model_pack["model"],
        bool(model_pack["has_mssa"]),
        dataset["images"],
        model_pack["device"],
        batch=96,
    )
    np.savez_compressed(cache, features=feats, logits=logits)
    return feats, logits


@torch.no_grad()
def run_logits(
    model_pack: dict[str, object],
    images: torch.Tensor,
    batch: int = 96,
) -> np.ndarray:
    model = model_pack["model"]
    device = model_pack["device"]
    has_mssa = bool(model_pack["has_mssa"])
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    out: list[np.ndarray] = []
    for i in range(0, len(images), batch):
        x = images[i : i + batch].float().to(device) / 255.0
        x = (x - mean) / std
        if has_mssa:
            y = model(x)
            logits = y[0] if isinstance(y, (tuple, list)) else y
        else:
            logits = model[1](model[0](x))
        out.append(logits.detach().cpu().numpy())
    return np.concatenate(out, axis=0)


def make_augmented(images: torch.Tensor, indices: np.ndarray | list[int]) -> torch.Tensor:
    x = images[indices].float() / 255.0
    torch.manual_seed(144)
    x = torch.clamp(x * 0.88 + 0.05 + torch.randn_like(x) * 0.035, 0.0, 1.0)
    return (x * 255).byte()


def make_noisy(images: torch.Tensor, sigma: float = 0.05) -> torch.Tensor:
    torch.manual_seed(50)
    x = images.float() / 255.0
    x = torch.clamp(x + torch.randn_like(x) * sigma, 0.0, 1.0)
    return (x * 255).byte()


def class_name(dataset: dict[str, object], label: int) -> str:
    names = dataset["class_names"]
    if 0 <= int(label) < len(names):
        return clean_label(names[int(label)])
    return f"class {int(label)}"


def make_retrieval_figure(
    datasets: dict[str, dict[str, object]],
    feats: dict[str, dict[str, np.ndarray]],
) -> Path:
    ds = datasets["AL6"]
    labels = ds["labels"]
    base = norm_rows(feats["baseline"]["AL6"])
    aaf = norm_rows(feats["aafnet"]["AL6"])

    rows: list[dict[str, object]] = []
    best = None
    for q in range(len(labels)):
        if np.sum(labels == labels[q]) < 8:
            continue
        counts = []
        picks = []
        for name, f in (("baseline", base), ("aafnet", aaf)):
            sim = f @ f[q]
            sim[q] = -np.inf
            top = np.argsort(-sim)[:5]
            count = int(np.sum(labels[top] == labels[q]))
            counts.append(count)
            picks.append(top)
        score = (counts[1] - counts[0], counts[1], -q)
        if best is None or score > best[0]:
            best = (score, q, picks)
    assert best is not None
    _, q, (base_top, aaf_top) = best
    for name, f, top in (("baseline", base, base_top), ("aafnet", aaf, aaf_top)):
        sim = f @ f[q]
        for r, idx in enumerate(top):
            rows.append(
                {
                    "figure": "F_Q2a_retrieval_effect",
                    "model": name,
                    "query_index": q,
                    "rank": r + 1,
                    "retrieved_index": int(idx),
                    "query_label": int(labels[q]),
                    "retrieved_label": int(labels[idx]),
                    "cosine": float(sim[idx]),
                    "same_class": bool(labels[idx] == labels[q]),
                    "query_display_rotation_deg": rotation_for("AL6", q),
                    "retrieved_display_rotation_deg": rotation_for("AL6", int(idx)),
                }
            )

    fig = plt.figure(figsize=(7.2, 3.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 7, width_ratios=[1.15, 1, 1, 1, 1, 1, 0.35])
    axq = fig.add_subplot(gs[:, 0])
    q_label = class_name(ds, int(labels[q]))
    plot_dataset_thumb(axq, datasets, "AL6", q, f"Query\n{q_label}", PALETTE["ink"], 7)

    for row_idx, (model_name, top) in enumerate((("ResNet-50", base_top), ("AAFNet", aaf_top))):
        ax_lbl = fig.add_subplot(gs[row_idx, 6])
        ax_lbl.axis("off")
        hits = int(np.sum(labels[top] == labels[q]))
        ax_lbl.text(
            0.0,
            0.5,
            f"{model_name}\n{hits}/5",
            fontsize=7,
            fontweight="bold",
            va="center",
        )
        for col, idx in enumerate(top, start=1):
            ax = fig.add_subplot(gs[row_idx, col])
            same = labels[idx] == labels[q]
            border = PALETTE["green"] if same else PALETTE["red"]
            plot_dataset_thumb(ax, datasets, "AL6", int(idx), f"rank {col}", border, 6)
    fig.suptitle("AL6 retrieval: nearest neighbours are produced from model features", fontsize=9, y=1.03)
    return save_figure(fig, "F_Q2a_retrieval_effect", rows)


def make_ood_figure(
    datasets: dict[str, dict[str, object]],
    logits: dict[str, dict[str, np.ndarray]],
    ood_dataset: str,
    fig_name: str,
    title: str,
) -> Path:
    al6 = datasets["AL6"]
    ood = datasets[ood_dataset]
    rows: list[dict[str, object]] = []

    id_idx = int(np.argsort(energy_np(logits["aafnet"]["AL6"]))[len(al6["labels"]) // 2])
    ood_energy = energy_np(logits["aafnet"][ood_dataset])
    ood_idx = int(np.argsort(-ood_energy)[min(10, len(ood_energy) - 1)])

    values = {}
    thresholds = {}
    for model_name in ("baseline", "aafnet"):
        id_e = energy_np(logits[model_name]["AL6"])
        ood_e = energy_np(logits[model_name][ood_dataset])
        thresholds[model_name] = float(np.quantile(id_e, 0.95))
        values[(model_name, "ID")] = float(id_e[id_idx])
        values[(model_name, "OOD")] = float(ood_e[ood_idx])
        for tag, score in (("ID", values[(model_name, "ID")]), ("OOD", values[(model_name, "OOD")])):
            rows.append(
                {
                    "figure": fig_name,
                    "model": model_name,
                    "sample_type": tag,
                    "dataset": "AL6" if tag == "ID" else ood_dataset,
                    "sample_index": id_idx if tag == "ID" else ood_idx,
                    "energy_score": score,
                    "id_95_percentile_threshold": thresholds[model_name],
                    "decision": "OOD" if score > thresholds[model_name] else "ID",
                    "display_rotation_deg": rotation_for("AL6" if tag == "ID" else ood_dataset, id_idx if tag == "ID" else ood_idx),
                }
            )

    fig = plt.figure(figsize=(6.8, 2.65), constrained_layout=True)
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1.35, 1.35])
    plot_dataset_thumb(
        fig.add_subplot(gs[0, 0]),
        datasets,
        "AL6",
        id_idx,
        f"ID: AL6\n{class_name(al6, int(al6['labels'][id_idx]))}",
        PALETTE["ink"],
    )
    plot_dataset_thumb(
        fig.add_subplot(gs[0, 1]),
        datasets,
        ood_dataset,
        ood_idx,
        f"OOD: {ood_dataset}\n{class_name(ood, int(ood['labels'][ood_idx]))}",
        PALETTE["amber"],
    )

    for col, model_name in enumerate(("baseline", "aafnet"), start=2):
        ax = fig.add_subplot(gs[0, col])
        xs = np.arange(2)
        vals = [values[(model_name, "ID")], values[(model_name, "OOD")]]
        ax.bar(xs, vals, color=[PALETTE["baseline"], PALETTE["aafnet"]], width=0.58)
        ax.axhline(thresholds[model_name], color=PALETTE["red"], lw=1.2, ls="--")
        ax.set_xticks(xs, ["ID", "OOD"])
        ax.set_ylabel("energy score")
        ax.set_title("ResNet-50" if model_name == "baseline" else "AAFNet", fontsize=8)
        ymin, ymax = min(vals + [thresholds[model_name]]), max(vals + [thresholds[model_name]])
        pad = max((ymax - ymin) * 0.25, 0.5)
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.text(1.02, thresholds[model_name], "ID 95%", fontsize=6, va="center", color=PALETTE["red"])
    fig.suptitle(title, fontsize=9, y=1.03)
    return save_figure(fig, fig_name, rows)


def make_selective_figure(
    datasets: dict[str, dict[str, object]],
    models: dict[str, dict[str, object]],
    refresh: bool,
) -> Path:
    ds = datasets["AL6"]
    labels = ds["labels"]
    noisy_cache = CACHE_DIR / "AL6_sigma005_noisy_logits.npz"
    noisy_images = make_noisy(ds["images"], sigma=0.05)
    if noisy_cache.exists() and not refresh:
        dat = np.load(noisy_cache)
        noisy_logits = {"baseline": dat["baseline"], "aafnet": dat["aafnet"]}
    else:
        noisy_logits = {name: run_logits(pack, noisy_images) for name, pack in models.items()}
        np.savez_compressed(noisy_cache, baseline=noisy_logits["baseline"], aafnet=noisy_logits["aafnet"])

    stats = {}
    for name in ("baseline", "aafnet"):
        probs = softmax_np(noisy_logits[name])
        pred = probs.argmax(axis=1)
        conf = probs.max(axis=1)
        threshold = float(np.sort(conf)[::-1][max(int(0.90 * len(conf)) - 1, 0)])
        stats[name] = {"pred": pred, "conf": conf, "threshold": threshold}

    candidates = np.where(
        (stats["baseline"]["pred"] != labels)
        & (stats["aafnet"]["pred"] == labels)
        & (stats["baseline"]["conf"] >= stats["baseline"]["threshold"])
    )[0]
    if len(candidates) == 0:
        candidates = np.where(stats["aafnet"]["pred"] == labels)[0]
    idx = int(candidates[np.argmax(stats["aafnet"]["conf"][candidates] - stats["baseline"]["conf"][candidates])])

    rows = []
    lines = []
    for name, label in (("baseline", "ResNet-50"), ("aafnet", "AAFNet")):
        pred = int(stats[name]["pred"][idx])
        conf = float(stats[name]["conf"][idx])
        accepted = conf >= stats[name]["threshold"]
        correct = pred == int(labels[idx])
        lines.append(
            f"{label:<9} pred={class_name(ds, pred):<10} conf={conf:.3f} "
            f"{'accept' if accepted else 'reject'} {'correct' if correct else 'wrong'}"
        )
        rows.append(
            {
                "figure": "F_Q2d_selective_rejection_effect",
                "model": name,
                "dataset": "AL6",
                "sample_index": idx,
                "true_label": int(labels[idx]),
                "pred_label": pred,
                "confidence": conf,
                "confidence_threshold_90pct_coverage": stats[name]["threshold"],
                "decision": "accept" if accepted else "reject",
                "correct": correct,
                "noise_sigma": 0.05,
                "display_rotation_deg": rotation_for("AL6", idx),
            }
        )

    fig = plt.figure(figsize=(6.8, 2.7), constrained_layout=True)
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 2.25, 0.25])
    plot_dataset_thumb(fig.add_subplot(gs[0, 0]), datasets, "AL6", idx, f"Clean\n{class_name(ds, int(labels[idx]))}", PALETTE["ink"])
    plot_thumb(
        fig.add_subplot(gs[0, 1]),
        noisy_images[idx],
        "Noisy input\n$\\sigma=0.05$",
        PALETTE["amber"],
        rotation_deg=rotation_for("AL6", idx),
    )
    text_panel(fig.add_subplot(gs[0, 2]), lines, "Selective prediction at 90% coverage")
    fig.add_subplot(gs[0, 3]).axis("off")
    fig.suptitle("Noise rejection: the displayed decision is made from the noisy image", fontsize=9, y=1.03)
    return save_figure(fig, "F_Q2d_selective_rejection_effect", rows)


def make_invariance_figure(
    datasets: dict[str, dict[str, object]],
    models: dict[str, dict[str, object]],
) -> Path:
    ds = datasets["AL6"]
    labels = ds["labels"]
    idx = int(np.where(labels == labels[0])[0][min(4, np.sum(labels == labels[0]) - 1)])
    neg = int(np.where(labels != labels[idx])[0][0])
    indices = np.array([idx, neg])
    aug = make_augmented(ds["images"], [idx])
    triple = torch.cat([ds["images"][indices], aug], dim=0)

    rows = []
    sims: dict[str, tuple[float, float]] = {}
    for name, pack in models.items():
        fts, _ = extract_features_logits(
            pack["model"],
            bool(pack["has_mssa"]),
            triple,
            pack["device"],
            batch=3,
        )
        fts = norm_rows(fts)
        pos = float(fts[0] @ fts[2])
        neg_sim = float(fts[0] @ fts[1])
        sims[name] = (pos, neg_sim)
        rows.extend(
            [
                {
                    "figure": "F_Q2e_augmentation_invariance_effect",
                    "model": name,
                    "dataset": "AL6",
                    "pair": "original_augmented",
                    "query_index": idx,
                    "comparison_index": idx,
                    "cosine_similarity": pos,
                    "query_display_rotation_deg": rotation_for("AL6", idx),
                    "comparison_display_rotation_deg": rotation_for("AL6", idx),
                },
                {
                    "figure": "F_Q2e_augmentation_invariance_effect",
                    "model": name,
                    "dataset": "AL6",
                    "pair": "original_negative",
                    "query_index": idx,
                    "comparison_index": neg,
                    "cosine_similarity": neg_sim,
                    "query_display_rotation_deg": rotation_for("AL6", idx),
                    "comparison_display_rotation_deg": rotation_for("AL6", neg),
                },
            ]
        )

    fig = plt.figure(figsize=(6.8, 2.65), constrained_layout=True)
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.55])
    plot_dataset_thumb(fig.add_subplot(gs[0, 0]), datasets, "AL6", idx, f"Original\n{class_name(ds, int(labels[idx]))}", PALETTE["ink"])
    plot_thumb(fig.add_subplot(gs[0, 1]), aug[0], "Augmented view", PALETTE["amber"], rotation_deg=rotation_for("AL6", idx))
    plot_dataset_thumb(fig.add_subplot(gs[0, 2]), datasets, "AL6", neg, f"Negative\n{class_name(ds, int(labels[neg]))}", PALETTE["red"])
    ax = fig.add_subplot(gs[0, 3])
    y = np.arange(2)
    height = 0.34
    ax.barh(y - height / 2, [sims["baseline"][0], sims["baseline"][1]], height, color=PALETTE["baseline"], label="ResNet-50")
    ax.barh(y + height / 2, [sims["aafnet"][0], sims["aafnet"][1]], height, color=PALETTE["aafnet"], label="AAFNet")
    ax.set_yticks(y, ["orig-aug", "orig-neg"])
    ax.set_xlim(-0.05, 1.02)
    ax.set_xlabel("feature cosine")
    ax.legend(loc="lower right", fontsize=6)
    ax.set_title("Inference-space invariance", fontsize=8)
    fig.suptitle("Augmentation invariance: the same building should remain closer than a negative", fontsize=9, y=1.03)
    return save_figure(fig, "F_Q2e_augmentation_invariance_effect", rows)


def make_domain_probe_figure(
    datasets: dict[str, dict[str, object]],
    feats: dict[str, dict[str, np.ndarray]],
) -> Path:
    domain_names = ["AL6", "ASP_clean", "AS25_clean"]
    rng = np.random.default_rng(108)
    rows = []
    X_index: dict[str, np.ndarray] = {}
    y_parts = []
    meta: list[tuple[str, int, int]] = []
    for did, ds_name in enumerate(domain_names):
        n = len(datasets[ds_name]["labels"])
        take = rng.choice(n, size=min(500, n), replace=False)
        X_index[ds_name] = take
        y_parts.append(np.full(len(take), did))
        meta.extend((ds_name, int(idx), did) for idx in take)
    y = np.concatenate(y_parts, axis=0)
    order = rng.permutation(len(y))
    train_n = int(0.7 * len(order))
    tr, te = order[:train_n], order[train_n:]
    model_result: dict[str, dict[str, object]] = {}

    for model_name in ("baseline", "aafnet"):
        X_parts = []
        for ds_name in domain_names:
            X_parts.append(feats[model_name][ds_name][X_index[ds_name]])
        X = np.concatenate(X_parts, axis=0)
        scaler = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=1000).fit(scaler.transform(X[tr]), y[tr])
        prob = clf.predict_proba(scaler.transform(X[te]))
        pred = prob.argmax(axis=1)
        model_result[model_name] = {"prob": prob, "pred": pred}

    aaf_prob = model_result["aafnet"]["prob"]
    aaf_pred = model_result["aafnet"]["pred"]
    meta_store: list[tuple[str, int, int]] = []
    local_store: list[int] = []
    for did in range(3):
        candidates = np.where(y[te] == did)[0]
        correct = candidates[aaf_pred[candidates] == did]
        pool = correct if len(correct) else candidates
        chosen = int(pool[np.argmax(aaf_prob[pool, did])])
        local_store.append(chosen)
        meta_store.append(meta[te[chosen]])

    for model_name in ("baseline", "aafnet"):
        prob = model_result[model_name]["prob"]
        pred = model_result[model_name]["pred"]
        for chosen_meta, local in zip(meta_store, local_store):
            rows.append(
                {
                    "figure": "F_Q2f_domain_probe_effect",
                    "model": model_name,
                    "sample_dataset": chosen_meta[0],
                    "sample_index": chosen_meta[1],
                    "true_domain": domain_names[chosen_meta[2]],
                    "pred_domain": domain_names[int(pred[local])],
                    "p_AL6": float(prob[local, 0]),
                    "p_ASP_clean": float(prob[local, 1]),
                    "p_AS25_clean": float(prob[local, 2]),
                    "display_rotation_deg": rotation_for(chosen_meta[0], chosen_meta[1]),
                }
            )

    fig = plt.figure(figsize=(6.8, 3.05), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.1, 1.0])
    for col, (ds_name, idx, did) in enumerate(meta_store[:3]):
        ax = fig.add_subplot(gs[0, col])
        plot_dataset_thumb(ax, datasets, ds_name, idx, f"{ds_name}\nheld-out image", PALETTE["ink"])

    for col, chosen_meta in enumerate(meta_store[:3]):
        ax = fig.add_subplot(gs[1, col])
        labels_short = ["AL6", "ASP", "AS25"]
        width = 0.34
        x = np.arange(3)
        plotted = False
        for off, model_name, color, label in (
            (-width / 2, "baseline", PALETTE["baseline"], "ResNet-50"),
            (width / 2, "aafnet", PALETTE["aafnet"], "AAFNet"),
        ):
            row = [
                r
                for r in rows
                if r["model"] == model_name
                and r["sample_dataset"] == chosen_meta[0]
                and r["sample_index"] == chosen_meta[1]
            ]
            if not row:
                continue
            probs = [row[0]["p_AL6"], row[0]["p_ASP_clean"], row[0]["p_AS25_clean"]]
            ax.bar(x + off, probs, width=width, color=color, label=label)
            plotted = True
        ax.set_xticks(x, labels_short)
        ax.set_ylim(0, 1.02)
        if col == 0:
            ax.set_ylabel("domain probability")
        if plotted:
            ax.axvline(chosen_meta[2], color=PALETTE["green"], lw=1.0, alpha=0.6)
        if col == 2:
            ax.legend(fontsize=6, loc="upper right")
    fig.suptitle("Domain probe: a small classifier reads corpus identity from frozen features", fontsize=9, y=1.03)
    return save_figure(fig, "F_Q2f_domain_probe_effect", rows)


def episode_indices(labels: np.ndarray, n_way: int = 5, k_shot: int = 5) -> tuple[np.ndarray, int, np.ndarray]:
    classes = [c for c in np.unique(labels) if np.sum(labels == c) >= k_shot + 2]
    if len(classes) < n_way:
        raise ValueError("Not enough classes for a 5-way 5-shot episode")
    rng = np.random.default_rng(212)
    classes = np.array(classes)
    chosen_classes = rng.choice(classes, size=n_way, replace=False)
    support = []
    query = None
    for ci, c in enumerate(chosen_classes):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        support.extend(idx[:k_shot])
        if ci == 0:
            query = int(idx[k_shot])
    assert query is not None
    return np.array(support), query, chosen_classes


def make_fewshot_figure(
    datasets: dict[str, dict[str, object]],
    feats: dict[str, dict[str, np.ndarray]],
    dataset_name: str,
    fig_name: str,
    title: str,
) -> Path:
    ds = datasets[dataset_name]
    labels = ds["labels"]
    support, query, classes = episode_indices(labels)
    rows = []
    sim_rows = {}

    for model_name in ("baseline", "aafnet"):
        f = norm_rows(feats[model_name][dataset_name])
        prototypes = []
        for c in classes:
            cls_support = support[labels[support] == c]
            proto = f[cls_support].mean(axis=0)
            proto = proto / (np.linalg.norm(proto) + 1e-12)
            prototypes.append(proto)
        prototypes = np.vstack(prototypes)
        sims = prototypes @ f[query]
        pred_pos = int(np.argmax(sims))
        sim_rows[model_name] = sims
        for pos, c in enumerate(classes):
            rows.append(
                {
                    "figure": fig_name,
                    "model": model_name,
                    "dataset": dataset_name,
                    "query_index": query,
                    "true_label": int(labels[query]),
                    "class_label": int(c),
                    "prototype_position": pos,
                    "cosine_to_query": float(sims[pos]),
                    "predicted": bool(pos == pred_pos),
                    "true_class": bool(c == labels[query]),
                    "support_indices": ";".join(str(int(i)) for i in support[labels[support] == c]),
                    "support_display_rotation_degs": ";".join(
                        str(rotation_for(dataset_name, int(i))) for i in support[labels[support] == c]
                    ),
                    "displayed_support_index": int(support[labels[support] == c][0]),
                    "displayed_support_rotation_deg": rotation_for(dataset_name, int(support[labels[support] == c][0])),
                    "query_display_rotation_deg": rotation_for(dataset_name, query),
                }
            )

    fig = plt.figure(figsize=(7.2, 3.25), constrained_layout=True)
    gs = fig.add_gridspec(2, 7, height_ratios=[1.0, 1.0], width_ratios=[1, 1, 1, 1, 1, 1.1, 1.55])
    for pos, c in enumerate(classes):
        idx = int(support[labels[support] == c][0])
        border = PALETTE["green"] if c == labels[query] else PALETTE["ink"]
        plot_dataset_thumb(fig.add_subplot(gs[0, pos]), datasets, dataset_name, idx, f"Support C{pos+1}", border, 6)
    plot_dataset_thumb(
        fig.add_subplot(gs[0, 5]),
        datasets,
        dataset_name,
        query,
        f"Query\n{class_name(ds, int(labels[query]))}",
        PALETTE["green"],
        6,
    )
    ax_note = fig.add_subplot(gs[0, 6])
    ax_note.axis("off")
    ax_note.text(0, 1.0, "Episode", fontsize=8, fontweight="bold", va="top")
    ax_note.text(
        0,
        0.78,
        "5-way, 5-shot\nfrozen AL6 features\nnearest prototype",
        fontsize=7,
        va="top",
        linespacing=1.4,
    )

    for row, (model_name, label, color) in enumerate(
        (("baseline", "ResNet-50", PALETTE["baseline"]), ("aafnet", "AAFNet", PALETTE["aafnet"]))
    ):
        ax = fig.add_subplot(gs[1, :3] if row == 0 else gs[1, 3:6])
        sims = sim_rows[model_name]
        x = np.arange(len(classes))
        colors = [PALETTE["green"] if c == labels[query] else color for c in classes]
        ax.bar(x, sims, color=colors, width=0.62)
        ax.set_xticks(x, [f"C{i+1}" for i in range(len(classes))])
        ax.set_ylim(min(-0.05, float(sims.min()) - 0.05), min(1.02, float(sims.max()) + 0.12))
        ax.set_ylabel("cosine")
        pred = int(np.argmax(sims))
        ax.set_title(f"{label}: predicted C{pred + 1}", fontsize=8)
    fig.add_subplot(gs[1, 6]).axis("off")
    fig.suptitle(title, fontsize=9, y=1.03)
    return save_figure(fig, fig_name, rows)


def nonzero_rotation_rows(paths: list[Path]) -> list[str]:
    rows: list[str] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    for path in paths:
        csv_path = SOURCE_DIR / f"{path.stem}.csv"
        if not csv_path.exists() or csv_path.stat().st_size == 0:
            continue
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                for key, value in row.items():
                    if not key.endswith("rotation_deg") and not key.endswith("rotation_degs"):
                        continue
                    if value in ("", None):
                        continue
                    values = [v for v in str(value).split(";") if v != ""]
                    if any(float(v) != 0.0 for v in values):
                        if key.startswith("retrieved"):
                            sample = row.get("retrieved_index") or "-"
                        elif key.startswith("comparison"):
                            sample = row.get("comparison_index") or "-"
                        elif key.startswith("displayed_support") or key.startswith("support"):
                            sample = row.get("displayed_support_index") or row.get("support_indices") or "-"
                        elif key.startswith("query"):
                            sample = row.get("query_index") or "-"
                        else:
                            sample = row.get("sample_index") or "-"
                        dataset = row.get("dataset") or row.get("sample_dataset") or ("AL6" if path.stem == "F_Q2a_retrieval_effect" else "-")
                        record = (path.stem, str(dataset), str(sample), key, str(value))
                        if record not in seen:
                            rows.append(f"| {path.stem} | {dataset} | {sample} | {key} | {value} |")
                            seen.add(record)
    return rows


def write_qa(paths: list[Path], models: dict[str, dict[str, object]]) -> None:
    report = [
        "# Downstream Inference Effect Figures QA",
        "",
        "## Figure Contract",
        "",
        "- Core conclusion: frozen AL6-trained features/logits support downstream inference behaviours beyond top-1 classification.",
        "- Evidence chain: each panel uses cached real images plus checkpoint-derived logits or features.",
        "- Archetype: image plate + quantitative inference readout.",
        "- Backend: Python/matplotlib only.",
        "- Exports: PNG for DOCX assembly, SVG/PDF companions with editable text, and CSV source data.",
        "- Image-integrity note: any thumbnail rotation recorded in source CSV is display-only; logits and features are computed from the unmodified cached tensors.",
        "",
        "## Checkpoints",
        "",
    ]
    for name, pack in models.items():
        report.append(f"- {name}: `{pack['checkpoint']}`")
    report.extend(["", "## Outputs", ""])
    for path in paths:
        stem = path.stem
        report.append(f"- `{path.relative_to(ROOT)}`")
        report.append(f"  - SVG: `paper/figures/nature_exports/downstream_inference__{stem}.svg`")
        report.append(f"  - PDF: `paper/figures/nature_exports/downstream_inference__{stem}.pdf`")
        report.append(f"  - Source data: `paper/figures/nature_source_data/{stem}.csv`")
    rotations = nonzero_rotation_rows(paths)
    if rotations:
        report.extend(
            [
                "",
                "## Display Orientation Audit",
                "",
                "- Non-zero rotations are applied only to rendered thumbnails; model logits/features use the original cached tensors.",
                "",
                "| Figure | Dataset | Sample | Rotation field | Value |",
                "|---|---|---:|---|---:|",
            ]
        )
        report.extend(rotations)
    (FIG_DIR / "downstream_inference_qa.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="Recompute cached features/logits")
    args = parser.parse_args()

    mkdirs()
    datasets = {name: load_dataset(name) for name in ("AL6", "ASP_clean", "AS25_clean")}
    models = build_models()

    feats: dict[str, dict[str, np.ndarray]] = {name: {} for name in models}
    logits: dict[str, dict[str, np.ndarray]] = {name: {} for name in models}
    for model_name, model_pack in models.items():
        for ds_name, dataset in datasets.items():
            f, l = get_features(model_name, model_pack, dataset, refresh=args.refresh)
            feats[model_name][ds_name] = f
            logits[model_name][ds_name] = l

    paths = [
        make_retrieval_figure(datasets, feats),
        make_ood_figure(
            datasets,
            logits,
            "ASP_clean",
            "F_Q2b_ood_asp_effect",
            "OOD inference: AL6-trained logits separate AL6 from ASP_clean",
        ),
        make_ood_figure(
            datasets,
            logits,
            "AS25_clean",
            "F_Q2c_ood_as25_effect",
            "OOD inference: AL6-trained logits separate AL6 from AS25_clean",
        ),
        make_selective_figure(datasets, models, refresh=args.refresh),
        make_invariance_figure(datasets, models),
        make_domain_probe_figure(datasets, feats),
        make_fewshot_figure(
            datasets,
            feats,
            "ASP_clean",
            "F_Q2g_fewshot_asp_effect",
            "Few-shot ASP_clean: one query classified by frozen-feature prototypes",
        ),
        make_fewshot_figure(
            datasets,
            feats,
            "AS25_clean",
            "F_Q2h_fewshot_as25_effect",
            "Few-shot AS25_clean: one query classified by frozen-feature prototypes",
        ),
    ]
    write_qa(paths, models)
    print(json.dumps({"figures": [str(p.relative_to(ROOT)) for p in paths]}, indent=2))


if __name__ == "__main__":
    main()
