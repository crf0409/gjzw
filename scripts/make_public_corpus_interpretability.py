#!/usr/bin/env python
"""Draw public-corpus inference and GradCAM parity panels.

This script uses existing seed-42 ASP_clean and AS25_clean checkpoints. It does
not train models. For each public corpus it runs full-test inference, selects
four auditable case types, and renders input / RN50 GradCAM / AAFNet GradCAM
traces with prediction and confidence labels.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.models.backbones import get_backbone
from src.utils.config import load_config
from figure_display_rotation import display_rotation as shared_display_rotation
from figure_display_rotation import rotate_display_array


FIG_DIR = ROOT / "paper" / "figures"
EXPORT_DIR = FIG_DIR / "nature_exports"
SOURCE_DIR = FIG_DIR / "nature_source_data"
QA_PATH = FIG_DIR / "public_corpus_interpretability_qa.md"

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
    "red": "#C44E52",
    "teal": "#009E73",
    "light": "#F2F2F2",
    "ink": "#1F1F1F",
}

DATASETS = ["ASP_clean", "AS25_clean"]
DISPLAY = {"ASP_clean": "ASP", "AS25_clean": "AS25"}
MODEL_LABEL = {"baseline": "RN50", "aafnet": "AAFNet"}
MEAN_VALUES = [0.485, 0.456, 0.406]
STD_VALUES = [0.229, 0.224, 0.225]
CKPT_PATTERNS = {
    "ASP_clean": {
        "baseline": "outputs/asp_as25_baseline_ASP_clean_seed42/*/resnet50/best_resnet50.pth",
        "aafnet": "outputs/asp_as25_aafnet_ASP_clean_seed42/*/resnet50/best_resnet50.pth",
    },
    "AS25_clean": {
        "baseline": "outputs/asp_as25_baseline_AS25_clean_seed42/*/resnet50/best_resnet50.pth",
        "aafnet": "outputs/asp_as25_aafnet_AS25_clean_seed42/*/resnet50/best_resnet50.pth",
    },
}


def latest(pattern: str) -> Path:
    matches = [p for p in ROOT.glob(pattern) if "latest" not in p.parts]
    if not matches:
        raise FileNotFoundError(pattern)
    return sorted(matches)[-1]


def short_label(text: str, n: int = 18) -> str:
    text = str(text).replace("_", " ")
    return text if len(text) <= n else text[: n - 1] + "."


def load_cache(dataset: str) -> dict:
    return torch.load(ROOT / "data" / "cache" / f"{dataset}_224x224_rgb_test.pt", map_location="cpu", weights_only=False)


def normalizer(device: torch.device):
    mean = torch.tensor(MEAN_VALUES, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD_VALUES, device=device).view(1, 3, 1, 1)

    def fn(x: torch.Tensor) -> torch.Tensor:
        return (x.float().to(device) / 255.0 - mean) / std

    return fn


def build_model(ckpt_path: Path, dataset: str, num_classes: int, device: torch.device):
    log_path = ckpt_path.parent / "training_log.json"
    overrides = {
        "model": {"name": "resnet50"},
        "data": {"dataset": dataset, "img_height": 224, "img_width": 224},
    }
    if log_path.exists():
        snap = json.loads(log_path.read_text(encoding="utf-8")).get("config_snapshot", {})
        if "aafnet" in snap:
            overrides["aafnet"] = snap["aafnet"]

    config = load_config(overrides=overrides)
    cls = get_backbone("resnet50")
    instance = cls.__new__(cls)
    instance.config = config
    instance.num_classes = num_classes
    instance.device = device
    instance._to_rgb = True
    model = instance.build_model()

    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    return model.to(device).eval()


def find_layer4(model):
    if hasattr(model, "feature_extractor"):
        return model.feature_extractor.layer4
    return model[0].layer4


@torch.no_grad()
def predict_all(model, images: torch.Tensor, device: torch.device, batch_size: int = 128) -> tuple[np.ndarray, np.ndarray]:
    norm = normalizer(device)
    preds: list[np.ndarray] = []
    confs: list[np.ndarray] = []
    for i in range(0, len(images), batch_size):
        out = model(norm(images[i : i + batch_size]))
        if isinstance(out, (tuple, list)):
            out = out[0]
        prob = out.softmax(dim=1)
        conf, pred = prob.max(dim=1)
        preds.append(pred.cpu().numpy())
        confs.append(conf.cpu().numpy())
    return np.concatenate(preds), np.concatenate(confs)


def gradcam(model, image: torch.Tensor, target_class: int, layer, device: torch.device) -> np.ndarray:
    captured: dict[str, torch.Tensor] = {}

    def keep_feature(_module, _inputs, output):
        if output.requires_grad:
            output.retain_grad()
        captured["feature"] = output

    hook = layer.register_forward_hook(keep_feature)
    try:
        x = normalizer(device)(image.unsqueeze(0)).detach().clone().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        out = model(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        out[:, int(target_class)].sum().backward()
        feat = captured.get("feature")
        if feat is None or feat.grad is None:
            return np.zeros((7, 7), dtype=np.float32)
        weights = feat.grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * feat).sum(dim=1, keepdim=True))
        cam = cam - cam.amin(dim=(2, 3), keepdim=True)
        cam = cam / (cam.amax(dim=(2, 3), keepdim=True) + 1e-8)
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        return cam.detach().cpu().numpy()[0, 0].astype(np.float32)
    finally:
        hook.remove()


def overlay(image: torch.Tensor, cam: np.ndarray, alpha: float = 0.42) -> np.ndarray:
    img = image.numpy().transpose(1, 2, 0).astype(np.float32) / 255.0
    heat = plt.cm.magma(cam)[..., :3]
    return np.clip((1.0 - alpha) * img + alpha * heat, 0, 1)


def select_cases(labels: np.ndarray, pred_b: np.ndarray, conf_b: np.ndarray, pred_a: np.ndarray, conf_a: np.ndarray) -> list[dict[str, object]]:
    correct_b = pred_b == labels
    correct_a = pred_a == labels
    candidates = {
        "both_correct": np.where(correct_b & correct_a)[0],
        "aafnet_fix": np.where((~correct_b) & correct_a)[0],
        "rn50_fix": np.where(correct_b & (~correct_a))[0],
        "both_wrong": np.where((~correct_b) & (~correct_a))[0],
    }
    scores = {
        "both_correct": conf_b + conf_a,
        "aafnet_fix": conf_a - conf_b,
        "rn50_fix": conf_b - conf_a,
        "both_wrong": np.maximum(conf_b, conf_a),
    }

    selected: list[dict[str, object]] = []
    used: set[int] = set()
    for case in ["both_correct", "aafnet_fix", "rn50_fix", "both_wrong"]:
        idxs = [int(i) for i in candidates[case] if int(i) not in used]
        if not idxs:
            idxs = [int(i) for i in np.argsort(-scores[case]) if int(i) not in used]
        chosen = idxs[int(np.argmax(scores[case][idxs]))]
        used.add(chosen)
        selected.append({"case": case, "index": chosen})
    return selected


def display_rotation(dataset: str, index: int) -> int:
    return shared_display_rotation(dataset, index)


def rotate_display(arr: np.ndarray, rotation_deg: int) -> np.ndarray:
    return rotate_display_array(arr, rotation_deg)


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


def render() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    per_dataset = {}
    prediction_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []

    for dataset in DATASETS:
        print(f"\n=== {dataset} ===")
        cache = load_cache(dataset)
        images = cache["images"]
        labels = cache["labels"].numpy()
        class_names = cache.get("class_names") or [f"C{i}" for i in range(int(labels.max()) + 1)]
        num_classes = int(labels.max()) + 1
        models = {}
        layers = {}
        preds = {}
        confs = {}
        for model_name in ["baseline", "aafnet"]:
            ckpt = latest(CKPT_PATTERNS[dataset][model_name])
            print(f"  {model_name}: {ckpt.relative_to(ROOT)}")
            model = build_model(ckpt, dataset, num_classes, device)
            models[model_name] = model
            layers[model_name] = find_layer4(model)
            preds[model_name], confs[model_name] = predict_all(model, images, device)
            acc = float((preds[model_name] == labels).mean())
            print(f"    acc={acc:.4f}")

        for idx in range(len(labels)):
            for model_name in ["baseline", "aafnet"]:
                pred = int(preds[model_name][idx])
                true = int(labels[idx])
                prediction_rows.append(
                    {
                        "dataset": dataset,
                        "model": model_name,
                        "index": idx,
                        "true": true,
                        "pred": pred,
                        "true_name": class_names[true],
                        "pred_name": class_names[pred],
                        "confidence": float(confs[model_name][idx]),
                        "correct": bool(pred == true),
                    }
                )

        cases = select_cases(labels, preds["baseline"], confs["baseline"], preds["aafnet"], confs["aafnet"])
        for item in cases:
            idx = int(item["index"])
            true = int(labels[idx])
            item.update(
                {
                    "dataset": dataset,
                    "selection_note": "automatic full-test selection",
                    "display_rotation_deg": display_rotation(dataset, idx),
                    "true": true,
                    "true_name": class_names[true],
                    "baseline_pred": int(preds["baseline"][idx]),
                    "baseline_pred_name": class_names[int(preds["baseline"][idx])],
                    "baseline_conf": float(confs["baseline"][idx]),
                    "baseline_correct": bool(preds["baseline"][idx] == true),
                    "aafnet_pred": int(preds["aafnet"][idx]),
                    "aafnet_pred_name": class_names[int(preds["aafnet"][idx])],
                    "aafnet_conf": float(confs["aafnet"][idx]),
                    "aafnet_correct": bool(preds["aafnet"][idx] == true),
                }
            )
            selected_rows.append(dict(item))
        per_dataset[dataset] = {
            "images": images,
            "labels": labels,
            "class_names": class_names,
            "models": models,
            "layers": layers,
            "preds": preds,
            "confs": confs,
            "cases": cases,
        }

    write_csv("F_TC7_public_interpretability_predictions_source.csv", prediction_rows)
    write_csv("F_TC7_public_interpretability_selected_source.csv", selected_rows)

    fig = plt.figure(figsize=(DOUBLE_W, 166 * MM))
    gs = fig.add_gridspec(
        6,
        4,
        left=0.075,
        right=0.995,
        top=0.890,
        bottom=0.025,
        wspace=0.055,
        hspace=0.040,
    )
    case_title = {
        "both_correct": "Both correct",
        "aafnet_fix": "AAFNet fixes",
        "rn50_fix": "RN50 fixes",
        "both_wrong": "Both difficult",
    }
    row_defs = [
        ("ASP_clean", "input"),
        ("ASP_clean", "baseline"),
        ("ASP_clean", "aafnet"),
        ("AS25_clean", "input"),
        ("AS25_clean", "baseline"),
        ("AS25_clean", "aafnet"),
    ]

    for r, (dataset, row_kind) in enumerate(row_defs):
        d = per_dataset[dataset]
        for c, item in enumerate(d["cases"]):
            ax = fig.add_subplot(gs[r, c])
            idx = int(item["index"])
            image = d["images"][idx]
            true = int(item["true"])
            if row_kind == "input":
                arr = image.numpy().transpose(1, 2, 0)
                ax.imshow(rotate_display(arr, int(item["display_rotation_deg"])))
                label = f"True: {short_label(item['true_name'])}"
                color = COL["ink"]
            else:
                model_name = row_kind
                target = int(d["preds"][model_name][idx])
                cam = gradcam(d["models"][model_name], image, target, d["layers"][model_name], device)
                ax.imshow(rotate_display(overlay(image, cam), int(item["display_rotation_deg"])))
                pred_name = d["class_names"][target]
                ok = "OK" if target == true else "ERR"
                label = f"{MODEL_LABEL[model_name]} {ok}: {short_label(pred_name)} ({d['confs'][model_name][idx]:.2f})"
                color = COL[model_name]
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.text(
                0.020,
                0.020,
                label,
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=4.6,
                color=color,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=1.4),
            )
            if r == 0:
                ax.set_title(case_title[str(item["case"])], fontsize=7, pad=2.5)
            if c == 0:
                row_label = f"{DISPLAY[dataset]}\n" + ("Input" if row_kind == "input" else MODEL_LABEL[row_kind])
                ax.text(
                    -0.18,
                    0.5,
                    row_label,
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    fontsize=6.2,
                    fontweight="bold",
                    color=COL[row_kind] if row_kind in ["baseline", "aafnet"] else COL["ink"],
                    rotation=90,
                )

    fig.text(0.010, 0.975, "a", ha="left", va="top", fontsize=8, fontweight="bold", color=COL["ink"])
    fig.text(
        0.075,
        0.985,
        "Public-corpus inference traces: input, RN50 GradCAM and AAFNet GradCAM",
        ha="left",
        va="top",
        fontsize=8.2,
        fontweight="bold",
        color=COL["ink"],
    )
    fig.text(
        0.075,
        0.950,
        "Cases are selected from full-test predictions; GradCAM targets each model's predicted class.",
        ha="left",
        va="top",
        fontsize=6.2,
        color=COL["ink"],
    )
    save_figure(fig, "F_TC7_public_interpretability.png")

    qa_lines = [
        "# Public-Corpus Interpretability QA",
        "",
        "- Backend: Python / matplotlib.",
        "- Scope: ASP_clean and AS25_clean seed-42 checkpoints only; no new training.",
        "- Figure contract: image plate + quant; each case is selected from full-test predictions.",
        "- Image-integrity note: `display_rotation_deg` is display-only; predictions, confidence scores and GradCAM targets are computed from the unmodified cached tensors, and input/heatmap panels are rotated together for readability.",
        f"- Prediction source rows: {len(prediction_rows)}.",
        f"- Selected-case rows: {len(selected_rows)}.",
        "",
        "| Dataset | Case | Index | Display rotation | True | RN50 pred/conf | AAFNet pred/conf |",
        "|---|---|---:|---:|---|---|---|",
    ]
    for row in selected_rows:
        qa_lines.append(
            f"| {DISPLAY[str(row['dataset'])]} | {row['case']} | {row['index']} | {row['display_rotation_deg']} | "
            f"{short_label(str(row['true_name']), 24)} | "
            f"{short_label(str(row['baseline_pred_name']), 20)} / {float(row['baseline_conf']):.2f} | "
            f"{short_label(str(row['aafnet_pred_name']), 20)} / {float(row['aafnet_conf']):.2f} |"
        )
    QA_PATH.write_text("\n".join(qa_lines) + "\n", encoding="utf-8")
    print(f"  -> {QA_PATH}")


if __name__ == "__main__":
    render()
