#!/usr/bin/env python
"""Run or audit public-corpus robustness attribution for completed CV folds.

This is the public-corpus counterpart of the AL6 attribution scripts. It scans
completed fold checkpoints from ``run_public_cv_parity.py`` and optionally runs
``scripts/run_robustness.py`` on each completed cell. The default workflow is
resumable: completed robustness JSON files are skipped unless ``--force`` is
used.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "outputs" / "public_robustness_attribution"
SOURCE_DIR = ROOT / "paper" / "figures" / "nature_source_data"
SEEDS = [42, 1337, 2024]
FOLDS = list(range(5))


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
    return (
        OUT_ROOT
        / task.key
        / run_id
        / f"seed{seed}_fold{fold}"
        / "eval"
        / "resnet50"
        / "results.json"
    )


def build_rows(run_id: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for task in TASKS:
        for seed in SEEDS:
            for fold in FOLDS:
                ckpt = latest_completed_ckpt(task, run_id, seed, fold)
                out = result_path(task, run_id, seed, fold)
                if out.exists():
                    status = "robustness_complete"
                elif ckpt is not None:
                    status = "checkpoint_ready"
                else:
                    status = "waiting_for_cv"
                rows.append(
                    {
                        "dataset": task.dataset,
                        "role": task.role,
                        "task": task.key,
                        "seed": str(seed),
                        "fold": str(fold),
                        "status": status,
                        "checkpoint": "" if ckpt is None else str(ckpt.relative_to(ROOT)),
                        "result": "" if not out.exists() else str(out.relative_to(ROOT)),
                    }
                )
    return rows


def write_status(rows: list[dict[str, str]], run_id: str) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = SOURCE_DIR / "public_robustness_attribution_queue_source.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT_ROOT / "queue.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    lines = [
        "# Public-Corpus Robustness Attribution Queue",
        "",
        "This file tracks fold-level robustness attribution readiness for ASP_clean and AS25_clean.",
        "",
        f"- CV run id: `{run_id}`",
        f"- Source data: `paper/figures/nature_source_data/{csv_path.name}`",
        "",
        "| Dataset | Role | CV checkpoints ready | Robustness complete | Waiting for CV |",
        "|---|---|---:|---:|---:|",
    ]
    for task in TASKS:
        task_rows = [r for r in rows if r["task"] == task.key]
        ready = sum(r["status"] in {"checkpoint_ready", "robustness_complete"} for r in task_rows)
        complete = sum(r["status"] == "robustness_complete" for r in task_rows)
        waiting = sum(r["status"] == "waiting_for_cv" for r in task_rows)
        lines.append(f"| {task.dataset} | {task.role} | {ready}/15 | {complete}/15 | {waiting}/15 |")
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
    if out.exists() and not args.force:
        print(f"[skip] {task.key} seed={seed} fold={fold}: robustness exists")
        return False

    output_subdir = f"public_robustness_attribution/{task.key}/{run_id}/seed{seed}_fold{fold}"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_robustness.py"),
        "--model",
        "resnet50",
        "--dataset",
        task.dataset,
        "--img-size",
        "224",
        "224",
        "--ckpt",
        str(ckpt),
        "--batch-size",
        str(args.batch_size),
        "--device",
        args.device,
        "--output-subdir",
        output_subdir,
        "--run-id",
        "eval",
    ]
    print("[cmd] " + " ".join(cmd))
    if args.dry_run:
        return True
    subprocess.run(cmd, cwd=ROOT, check=True)
    return True


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", default="public_cv_parity_v1")
    p.add_argument("--task", default="all", help="all or comma-separated task keys")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--fold", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=None, help="maximum new robustness jobs to launch")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    tasks = selected_tasks(args.task)
    rows = build_rows(args.run_id)
    write_status(rows, args.run_id)
    print((OUT_ROOT / "queue.md").read_text(encoding="utf-8"))

    if args.seed is not None or args.fold is not None:
        if args.seed is None or args.fold is None:
            raise SystemExit("--seed and --fold must be provided together")
        if len(tasks) != 1:
            raise SystemExit("--task must select exactly one task when --seed/--fold are provided")
        run_cell(tasks[0], args.run_id, args.seed, args.fold, args)
        rows = build_rows(args.run_id)
        write_status(rows, args.run_id)
        return

    launched = 0
    for task in tasks:
        for seed in SEEDS:
            for fold in FOLDS:
                if args.limit is not None and launched >= args.limit:
                    return
                before = result_path(task, args.run_id, seed, fold).exists()
                did_run = run_cell(task, args.run_id, seed, fold, args)
                after = result_path(task, args.run_id, seed, fold).exists()
                if did_run or (not before and after):
                    launched += 1


if __name__ == "__main__":
    main()
