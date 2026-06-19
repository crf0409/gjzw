#!/usr/bin/env python
"""Seed-42 parity evaluations for AL6, ASP_clean and AS25_clean.

This fills the non-training experimental gaps that were previously AL6-only:
robustness, temperature calibration, and 24-angle rotation sweeps. It reuses
existing checkpoints and PT caches; it does not train new models.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.robustness import PERTURBATION_GRID, evaluate_full_robustness
from src.models.backbones import get_backbone
from src.utils.config import load_config


MEAN_VALUES = [0.485, 0.456, 0.406]
STD_VALUES = [0.229, 0.224, 0.225]

CKPT_PATTERNS = {
    "AL6": {
        "baseline": "outputs/ddp_baseline/*/resnet50/best_resnet50.pth",
        "aafnet": "outputs/ddp_aafnet_v2/*/resnet50/best_resnet50.pth",
    },
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


def load_cache(dataset: str, split: str, h: int, w: int) -> dict:
    path = ROOT / "data" / "cache" / f"{dataset}_{h}x{w}_rgb_{split}.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu", weights_only=False)


def normalizer(device: torch.device):
    mean = torch.tensor(MEAN_VALUES, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD_VALUES, device=device).view(1, 3, 1, 1)

    def fn(x: torch.Tensor) -> torch.Tensor:
        x = x.float().to(device) / 255.0
        return (x - mean) / std

    return fn


def build_model(model_name: str, ckpt_path: Path, dataset: str, num_classes: int, device: torch.device):
    log_path = ckpt_path.parent / "training_log.json"
    overrides = {
        "model": {"name": model_name},
        "data": {"dataset": dataset, "img_height": 224, "img_width": 224},
    }
    if log_path.exists():
        snap = json.loads(log_path.read_text(encoding="utf-8")).get("config_snapshot", {})
        if "aafnet" in snap:
            overrides["aafnet"] = snap["aafnet"]

    config = load_config(overrides=overrides)
    cls = get_backbone(model_name)
    instance = cls.__new__(cls)
    instance.config = config
    instance.num_classes = num_classes
    instance.device = device
    instance._to_rgb = True
    model = instance.build_model()

    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    return model.to(device).eval(), {"missing": len(missing), "unexpected": len(unexpected)}


@torch.no_grad()
def get_logits(model: nn.Module, images: torch.Tensor, device: torch.device, batch_size: int) -> np.ndarray:
    norm = normalizer(device)
    chunks: list[np.ndarray] = []
    for i in range(0, len(images), batch_size):
        out = model(norm(images[i : i + batch_size]))
        if isinstance(out, (tuple, list)):
            out = out[0]
        chunks.append(out.cpu().numpy())
    return np.concatenate(chunks, axis=0)


def add_gaussian_noise(images: torch.Tensor, sigma: float, seed: int) -> torch.Tensor:
    if sigma <= 0:
        return images
    generator = torch.Generator().manual_seed(seed)
    x = images.float() / 255.0
    noise = torch.randn(x.shape, generator=generator) * sigma
    return ((x + noise).clamp(0, 1) * 255).to(torch.uint8)


def fit_temperature(logits_val: np.ndarray, labels_val: np.ndarray) -> float:
    logits = torch.from_numpy(logits_val).float()
    labels = torch.from_numpy(labels_val).long()
    log_temp = torch.nn.Parameter(torch.log(torch.ones(1) * 1.5))
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.LBFGS([log_temp], lr=0.01, max_iter=200)

    def closure():
        opt.zero_grad()
        temp = torch.exp(log_temp).clamp(0.05, 10.0)
        loss = loss_fn(logits / temp, labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.exp(log_temp.detach()).clamp(0.05, 10.0).item())


def metrics_from_logits(logits: np.ndarray, labels: np.ndarray, temp: float = 1.0, n_bins: int = 15) -> dict:
    z = logits / max(temp, 1e-4)
    z = z - z.max(axis=1, keepdims=True)
    probs = np.exp(z)
    probs = probs / probs.sum(axis=1, keepdims=True)
    preds = probs.argmax(axis=1)
    acc = float((preds == labels).mean())
    conf = probs.max(axis=1)
    ok = (preds == labels).astype(float)

    bins = []
    ece = 0.0
    edges = np.linspace(0, 1, n_bins + 1)
    for idx in range(n_bins):
        lo, hi = edges[idx], edges[idx + 1]
        mask = (conf > lo) & (conf <= hi)
        n = int(mask.sum())
        if n:
            avg_conf = float(conf[mask].mean())
            avg_acc = float(ok[mask].mean())
            ece += (n / len(labels)) * abs(avg_conf - avg_acc)
        else:
            avg_conf = float((lo + hi) / 2)
            avg_acc = 0.0
        bins.append({"bin": idx, "n": n, "avg_conf": avg_conf, "avg_acc": avg_acc})

    eps = 1e-12
    nll = float(-np.log(probs[np.arange(len(labels)), labels] + eps).mean())
    one_hot = np.eye(probs.shape[1])[labels]
    brier = float(((probs - one_hot) ** 2).sum(axis=1).mean())
    return {"accuracy": acc, "ece": ece, "nll": nll, "brier": brier, "bins": bins}


def calibration_eval(model: nn.Module, train_cache: dict, test_cache: dict, device: torch.device, batch_size: int, seed: int) -> dict:
    labels_train = train_cache["labels"].numpy()
    rng = np.random.default_rng(seed)
    val_size = max(1, len(labels_train) // 5)
    val_idx = rng.permutation(len(labels_train))[:val_size]
    val_images = train_cache["images"][val_idx]
    val_labels = labels_train[val_idx]

    logits_val = get_logits(model, val_images, device, batch_size)
    temp = fit_temperature(logits_val, val_labels)

    labels_test = test_cache["labels"].numpy()
    out = {"temperature": temp, "val_size": int(val_size), "conditions": {}}
    for name, sigma in [("clean", 0.0), ("noise_0_05", 0.05), ("noise_0_10", 0.10)]:
        images = add_gaussian_noise(test_cache["images"], sigma, seed + int(sigma * 1000))
        logits = get_logits(model, images, device, batch_size)
        pre = metrics_from_logits(logits, labels_test, temp=1.0)
        post = metrics_from_logits(logits, labels_test, temp=temp)
        out["conditions"][name] = {
            "pre": {k: v for k, v in pre.items() if k != "bins"},
            "post": {k: v for k, v in post.items() if k != "bins"},
            "bins_pre": pre["bins"],
            "bins_post": post["bins"],
        }
    return out


def rotate_batch(images: torch.Tensor, angle_deg: float, device: torch.device) -> torch.Tensor:
    if abs(angle_deg) < 1e-8:
        return images
    x = images.float().to(device) / 255.0
    theta = -angle_deg * math.pi / 180.0
    cos_v, sin_v = math.cos(theta), math.sin(theta)
    affine = torch.tensor([[cos_v, -sin_v, 0.0], [sin_v, cos_v, 0.0]], device=device, dtype=x.dtype)
    affine = affine.unsqueeze(0).expand(x.shape[0], -1, -1).contiguous()
    grid = F.affine_grid(affine, x.size(), align_corners=False)
    rot = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return (rot * 255).to(torch.uint8).cpu()


@torch.no_grad()
def accuracy_on_images(model: nn.Module, images: torch.Tensor, labels: torch.Tensor, device: torch.device, batch_size: int) -> float:
    norm = normalizer(device)
    correct = 0
    total = 0
    for i in range(0, len(images), batch_size):
        lbl = labels[i : i + batch_size].to(device)
        out = model(norm(images[i : i + batch_size]))
        if isinstance(out, (tuple, list)):
            out = out[0]
        correct += int((out.argmax(1) == lbl).sum().item())
        total += int(lbl.numel())
    return correct / max(total, 1)


def rotation_eval(model: nn.Module, test_cache: dict, device: torch.device, batch_size: int, angles: list[int]) -> dict:
    images = test_cache["images"]
    labels = test_cache["labels"]
    accs = []
    for angle in angles:
        rotated_batches = []
        for i in range(0, len(images), batch_size):
            rotated_batches.append(rotate_batch(images[i : i + batch_size], angle, device))
        rotated = torch.cat(rotated_batches, dim=0)
        accs.append(accuracy_on_images(model, rotated, labels, device, batch_size))
    arr = np.array(accs, dtype=float)
    return {
        "angles": angles,
        "accuracies": [float(v) for v in accs],
        "summary": {
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "acc@0": float(accs[0]),
            "acc@90": float(accs[angles.index(90)]) if 90 in angles else None,
            "acc@180": float(accs[angles.index(180)]) if 180 in angles else None,
            "acc@270": float(accs[angles.index(270)]) if 270 in angles else None,
        },
    }


def robustness_eval(model: nn.Module, test_cache: dict, device: torch.device, batch_size: int, seed: int) -> dict:
    torch.manual_seed(seed)
    rob = evaluate_full_robustness(
        model,
        test_cache["images"],
        test_cache["labels"],
        device,
        batch_size=batch_size,
        normalize_fn=normalizer(device),
    )
    family_means = {}
    all_acc = []
    for kind, values in rob.items():
        accs = [float(v["accuracy"]) for v in values]
        family_means[kind] = float(np.mean(accs))
        all_acc.extend(accs)
    return {"grid": PERTURBATION_GRID, "results": rob, "family_means": family_means, "all_mean": float(np.mean(all_acc))}


def evaluate_dataset(dataset: str, tasks: set[str], batch_size: int, device: torch.device, angles: list[int], seed: int) -> dict:
    train_cache = load_cache(dataset, "train", 224, 224)
    test_cache = load_cache(dataset, "test", 224, 224)
    num_classes = int(test_cache["labels"].max().item()) + 1
    payload = {"n_train": len(train_cache["labels"]), "n_test": len(test_cache["labels"]), "num_classes": num_classes, "models": {}}

    for model_label in ["baseline", "aafnet"]:
        ckpt = latest(CKPT_PATTERNS[dataset][model_label])
        print(f"\n=== {dataset} / {model_label} ===")
        print(f"ckpt: {ckpt.relative_to(ROOT)}")
        model, load_info = build_model("resnet50", ckpt, dataset, num_classes, device)
        model_payload = {
            "ckpt": str(ckpt.relative_to(ROOT)),
            "load_info": load_info,
            "clean_accuracy": accuracy_on_images(model, test_cache["images"], test_cache["labels"], device, batch_size),
        }
        print(f"clean accuracy = {model_payload['clean_accuracy']:.4f}")
        if "calibration" in tasks:
            print("[calibration]")
            model_payload["calibration"] = calibration_eval(model, train_cache, test_cache, device, batch_size, seed)
        if "rotation" in tasks:
            print("[rotation]")
            model_payload["rotation"] = rotation_eval(model, test_cache, device, batch_size, angles)
        if "robustness" in tasks:
            print("[robustness]")
            model_payload["robustness"] = robustness_eval(model, test_cache, device, batch_size, seed)
        payload["models"][model_label] = model_payload
        del model
        torch.cuda.empty_cache()
    return payload


def render_md(results: dict, tasks: set[str]) -> str:
    def pct_cell(value) -> str:
        return "n/a" if value is None else f"{value * 100:.2f} %"

    lines = [
        "# Three-Corpus Seed-42 Parity Evaluation",
        "",
        "This evaluation reuses existing seed-42 checkpoints and does not train new models.",
        "",
        "## Clean Accuracy",
        "",
        "| Dataset | Model | n_test | Clean acc |",
        "|---|---|---:|---:|",
    ]
    for dataset, dres in results["datasets"].items():
        for model_label, mres in dres["models"].items():
            lines.append(f"| {dataset} | {model_label} | {dres['n_test']} | {mres['clean_accuracy']*100:.2f} % |")

    if "robustness" in tasks:
        lines += ["", "## Robustness Aggregate", "", "| Dataset | Model | All 15 cells | Gaussian | Blur | JPEG | Brightness | Occlusion |", "|---|---|---:|---:|---:|---:|---:|---:|"]
        for dataset, dres in results["datasets"].items():
            for model_label, mres in dres["models"].items():
                rob = mres["robustness"]
                fam = rob["family_means"]
                lines.append(
                    f"| {dataset} | {model_label} | {rob['all_mean']*100:.2f} % | "
                    f"{fam['gauss_noise']*100:.2f} % | {fam['motion_blur']*100:.2f} % | "
                    f"{fam['jpeg_compress']*100:.2f} % | {fam['brightness']*100:.2f} % | {fam['occlusion']*100:.2f} % |"
                )

    if "calibration" in tasks:
        lines += ["", "## Calibration After Temperature Scaling", "", "| Dataset | Model | T | Clean ECE post | Noise 0.05 ECE post | Noise 0.10 ECE post |", "|---|---|---:|---:|---:|---:|"]
        for dataset, dres in results["datasets"].items():
            for model_label, mres in dres["models"].items():
                cal = mres["calibration"]
                lines.append(
                    f"| {dataset} | {model_label} | {cal['temperature']:.3f} | "
                    f"{cal['conditions']['clean']['post']['ece']:.4f} | "
                    f"{cal['conditions']['noise_0_05']['post']['ece']:.4f} | "
                    f"{cal['conditions']['noise_0_10']['post']['ece']:.4f} |"
                )

    if "rotation" in tasks:
        lines += ["", "## Rotation Summary", "", "| Dataset | Model | Mean acc | Min acc | Acc@90 | Acc@180 | Acc@270 |", "|---|---|---:|---:|---:|---:|---:|"]
        for dataset, dres in results["datasets"].items():
            for model_label, mres in dres["models"].items():
                summary = mres["rotation"]["summary"]
                lines.append(
                    f"| {dataset} | {model_label} | {pct_cell(summary['mean'])} | {pct_cell(summary['min'])} | "
                    f"{pct_cell(summary['acc@90'])} | {pct_cell(summary['acc@180'])} | {pct_cell(summary['acc@270'])} |"
                )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["ASP_clean", "AS25_clean"], choices=list(CKPT_PATTERNS))
    parser.add_argument("--tasks", nargs="+", default=["robustness", "calibration", "rotation"], choices=["robustness", "calibration", "rotation"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--angles", type=int, nargs="+", default=list(range(0, 360, 15)))
    parser.add_argument("--output-subdir", default="three_corpus_parity")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tasks = set(args.tasks)
    run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "run_id": run_id,
        "datasets_requested": args.datasets,
        "tasks": args.tasks,
        "batch_size": args.batch_size,
        "device": str(device),
        "seed": args.seed,
        "datasets": {},
    }

    for dataset in args.datasets:
        results["datasets"][dataset] = evaluate_dataset(dataset, tasks, args.batch_size, device, args.angles, args.seed)

    out_dir = ROOT / "outputs" / args.output_subdir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (out_dir / "summary.md").write_text(render_md(results, tasks), encoding="utf-8")

    latest_dir = ROOT / "outputs" / args.output_subdir / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    (latest_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (latest_dir / "summary.md").write_text(render_md(results, tasks), encoding="utf-8")
    print(f"\n[ok] wrote {out_dir}")
    print(f"[ok] refreshed {latest_dir}")


if __name__ == "__main__":
    main()
