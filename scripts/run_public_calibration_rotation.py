#!/usr/bin/env python
"""Run or audit public-corpus calibration and rotation probes per CV fold.

The manuscript currently has seed-42 public-corpus calibration/rotation parity.
This wrapper makes the same probes resumable at the 5-fold x 3-seed public-CV
depth once fold checkpoints become available.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "outputs" / "public_calibration_rotation"
SOURCE_DIR = ROOT / "paper" / "figures" / "nature_source_data"
SEEDS = [42, 1337, 2024]
FOLDS = list(range(5))
DEFAULT_ANGLES = list(range(0, 360, 15))


@dataclass(frozen=True)
class Task:
    key: str
    dataset: str
    role: str
    cv_output_subdir: str


TASKS = [
    Task("asp_baseline", "ASP_clean", "baseline", "cv_asp_baseline"),
    Task("asp_aafnet", "ASP_clean", "aafnet", "cv_asp_aafnet"),
    Task("as25_baseline", "AS25_clean", "baseline", "cv_as25_baseline"),
    Task("as25_aafnet", "AS25_clean", "aafnet", "cv_as25_aafnet"),
]


def load_parity_module():
    path = ROOT / "scripts" / "eval_three_corpus_parity.py"
    spec = importlib.util.spec_from_file_location("eval_three_corpus_parity", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def latest_completed_ckpt(task: Task, run_id: str, seed: int, fold: int) -> Path | None:
    fold_root = (
        ROOT
        / "outputs"
        / task.cv_output_subdir
        / run_id
        / "resnet50"
        / f"seed{seed}_fold{fold}_train"
    )
    if not fold_root.exists():
        return None
    candidates: list[Path] = []
    for ckpt in sorted(fold_root.glob("*/resnet50/best_resnet50.pth")):
        if "latest" in ckpt.parts:
            continue
        if (ckpt.parent / "test_metrics.json").exists():
            candidates.append(ckpt)
    return candidates[-1] if candidates else None


def result_path(task: Task, run_id: str, seed: int, fold: int) -> Path:
    return OUT_ROOT / task.key / run_id / f"seed{seed}_fold{fold}" / "results.json"


def result_has(path: Path, probe: str) -> bool:
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return probe in data


def build_rows(run_id: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for task in TASKS:
        for seed in SEEDS:
            for fold in FOLDS:
                ckpt = latest_completed_ckpt(task, run_id, seed, fold)
                out = result_path(task, run_id, seed, fold)
                if ckpt is None:
                    base_status = "waiting_for_cv"
                else:
                    base_status = "checkpoint_ready"
                rows.append(
                    {
                        "dataset": task.dataset,
                        "role": task.role,
                        "task": task.key,
                        "seed": str(seed),
                        "fold": str(fold),
                        "status": base_status,
                        "calibration": "complete" if result_has(out, "calibration") else ("pending" if ckpt else "waiting_for_cv"),
                        "rotation": "complete" if result_has(out, "rotation") else ("pending" if ckpt else "waiting_for_cv"),
                        "checkpoint": "" if ckpt is None else str(ckpt.relative_to(ROOT)),
                        "result": "" if not out.exists() else str(out.relative_to(ROOT)),
                    }
                )
    return rows


def write_status(rows: list[dict[str, str]], run_id: str) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = SOURCE_DIR / "public_calibration_rotation_queue_source.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT_ROOT / "queue.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    lines = [
        "# Public-Corpus Calibration/Rotation Queue",
        "",
        "This file tracks fold-level calibration and 24-angle rotation readiness for ASP_clean and AS25_clean.",
        "",
        f"- CV run id: `{run_id}`",
        f"- Source data: `paper/figures/nature_source_data/{csv_path.name}`",
        "",
        "| Dataset | Role | CV checkpoints ready | Calibration complete | Rotation complete | Waiting for CV |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for task in TASKS:
        task_rows = [r for r in rows if r["task"] == task.key]
        ready = sum(r["status"] == "checkpoint_ready" for r in task_rows)
        cal = sum(r["calibration"] == "complete" for r in task_rows)
        rot = sum(r["rotation"] == "complete" for r in task_rows)
        waiting = sum(r["status"] == "waiting_for_cv" for r in task_rows)
        lines.append(f"| {task.dataset} | {task.role} | {ready}/15 | {cal}/15 | {rot}/15 | {waiting}/15 |")
    (OUT_ROOT / "queue.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def selected_tasks(value: str) -> list[Task]:
    if value == "all":
        return TASKS
    keys = {part.strip() for part in value.split(",") if part.strip()}
    unknown = keys - {task.key for task in TASKS}
    if unknown:
        raise SystemExit(f"unknown task(s): {', '.join(sorted(unknown))}")
    return [task for task in TASKS if task.key in keys]


def run_cell(task: Task, run_id: str, seed: int, fold: int, args: argparse.Namespace) -> bool:
    ckpt = latest_completed_ckpt(task, run_id, seed, fold)
    if ckpt is None:
        return False
    out = result_path(task, run_id, seed, fold)
    requested = set(args.probes)
    if out.exists() and not args.force and all(result_has(out, probe) for probe in requested):
        print(f"[skip] {task.key} seed={seed} fold={fold}: requested probes exist")
        return False

    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--run-id",
        run_id,
        "--task",
        task.key,
        "--seed",
        str(seed),
        "--fold",
        str(fold),
        "--batch-size",
        str(args.batch_size),
        "--device",
        args.device,
        "--probes",
        *args.probes,
    ]
    if args.force:
        cmd.append("--force")
    print("[cmd] " + " ".join(cmd))
    if args.dry_run:
        return True
    subprocess.run(cmd, cwd=ROOT, check=True)
    return True


def evaluate_one(args: argparse.Namespace) -> None:
    task = selected_tasks(args.task)[0]
    if args.seed is None or args.fold is None:
        raise SystemExit("--seed and --fold are required when running a single cell")
    ckpt = latest_completed_ckpt(task, args.run_id, args.seed, args.fold)
    if ckpt is None:
        raise SystemExit(f"missing completed checkpoint for {task.key} seed={args.seed} fold={args.fold}")

    out = result_path(task, args.run_id, args.seed, args.fold)
    out.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if out.exists() and not args.force:
        existing = json.loads(out.read_text(encoding="utf-8"))

    parity = load_parity_module()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train_cache = parity.load_cache(task.dataset, "train", 224, 224)
    test_cache = parity.load_cache(task.dataset, "test", 224, 224)
    num_classes = int(test_cache["labels"].max().item()) + 1
    model, load_info = parity.build_model("resnet50", ckpt, task.dataset, num_classes, device)

    payload = {
        **existing,
        "dataset": task.dataset,
        "role": task.role,
        "task": task.key,
        "cv_run_id": args.run_id,
        "seed": args.seed,
        "fold": args.fold,
        "ckpt": str(ckpt.relative_to(ROOT)),
        "load_info": load_info,
        "batch_size": args.batch_size,
        "device": str(device),
        "clean_accuracy": parity.accuracy_on_images(model, test_cache["images"], test_cache["labels"], device, args.batch_size),
    }
    probes = set(args.probes)
    if "calibration" in probes and (args.force or "calibration" not in payload):
        payload["calibration"] = parity.calibration_eval(model, train_cache, test_cache, device, args.batch_size, args.seed)
    if "rotation" in probes and (args.force or "rotation" not in payload):
        payload["rotation"] = parity.rotation_eval(model, test_cache, device, args.batch_size, args.angles)

    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[ok] wrote {out.relative_to(ROOT)}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", default="public_cv_parity_v1")
    p.add_argument("--task", default="all", help="all or comma-separated task keys")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--fold", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--probes", nargs="+", default=["calibration", "rotation"], choices=["calibration", "rotation"])
    p.add_argument("--angles", type=int, nargs="+", default=DEFAULT_ANGLES)
    p.add_argument("--limit", type=int, default=None, help="maximum new probe jobs to launch")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    if args.seed is not None or args.fold is not None:
        evaluate_one(args)
        rows = build_rows(args.run_id)
        write_status(rows, args.run_id)
        return

    rows = build_rows(args.run_id)
    write_status(rows, args.run_id)
    print((OUT_ROOT / "queue.md").read_text(encoding="utf-8"))

    tasks = selected_tasks(args.task)
    launched = 0
    for task in tasks:
        for seed in SEEDS:
            for fold in FOLDS:
                if args.limit is not None and launched >= args.limit:
                    return
                if run_cell(task, args.run_id, seed, fold, args):
                    launched += 1


if __name__ == "__main__":
    main()
