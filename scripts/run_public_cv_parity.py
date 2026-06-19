#!/usr/bin/env python
"""Run or print the public-corpus CV parity jobs.

The full mode launches the four expensive jobs needed to match the AL6
5-fold x 3-seed protocol on ASP_clean and AS25_clean. Smoke mode runs the same
path with a tiny fold/epoch setting to verify that dataset caches, fold indices,
AAFNet flags and output aggregation work before a long run.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

AAFNET_ARGS = [
    "--mssa",
    "--loss-type",
    "focalls_supcon",
    "--supcon-weight",
    "0.3",
    "--archaug",
    "--perspective",
    "0.3",
    "--arch-occlusion",
    "0.3",
    "--weather",
    "0.3",
    "--gauss-noise",
    "0.5",
]


@dataclass(frozen=True)
class Task:
    dataset: str
    role: str
    output_subdir: str
    extra_args: tuple[str, ...] = ()

    @property
    def key(self) -> str:
        return f"{self.dataset}:{self.role}"


TASKS = [
    Task("ASP_clean", "baseline", "cv_asp_baseline"),
    Task("ASP_clean", "aafnet", "cv_asp_aafnet", tuple(AAFNET_ARGS)),
    Task("AS25_clean", "baseline", "cv_as25_baseline"),
    Task("AS25_clean", "aafnet", "cv_as25_aafnet", tuple(AAFNET_ARGS)),
]


def build_command(task: Task, args: argparse.Namespace) -> list[str]:
    output_subdir = task.output_subdir
    folds = args.folds
    seeds = args.seeds
    epochs = args.epochs
    if args.smoke:
        tag = task.output_subdir.replace("cv_", "cv_smoke_")
        output_subdir = args.output_subdir or tag
        folds = args.smoke_folds
        seeds = args.smoke_seeds
        epochs = args.smoke_epochs
    elif args.output_subdir:
        output_subdir = args.output_subdir

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_cv.py"),
        "--model",
        "resnet50",
        "--dataset",
        task.dataset,
        "--img-size",
        "224",
        "224",
        "--folds",
        str(folds),
        "--seeds",
        *[str(seed) for seed in seeds],
        "--epochs",
        str(epochs),
        "--batch-size",
        str(args.batch_size),
        "--nproc",
        str(args.nproc),
        "--output-subdir",
        output_subdir,
    ]
    if args.run_id:
        cmd.extend(["--run-id", args.run_id])
    if args.resume:
        cmd.append("--resume")
    if task.extra_args:
        cmd.extend(["--extra-args", " ".join(task.extra_args)])
    return cmd


def selected_tasks(selection: str | None) -> list[Task]:
    if not selection or selection == "all":
        return TASKS
    wanted = {part.strip() for part in selection.split(",") if part.strip()}
    tasks = [task for task in TASKS if task.key in wanted or task.output_subdir in wanted]
    missing = wanted - {task.key for task in tasks} - {task.output_subdir for task in tasks}
    if missing:
        valid = ", ".join(task.key for task in TASKS)
        raise SystemExit(f"Unknown task(s): {', '.join(sorted(missing))}. Valid: {valid}")
    return tasks


def write_manifest(tasks: list[Task], commands: list[list[str]], args: argparse.Namespace) -> None:
    out_dir = ROOT / "outputs" / "public_cv_parity"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "mode": "smoke" if args.smoke else "full",
        "dry_run": args.dry_run,
        "run_id": args.run_id,
        "resume": args.resume,
        "nproc": args.nproc,
        "batch_size": args.batch_size,
        "tasks": [
            {
                "key": task.key,
                "output_subdir": task.output_subdir,
                "command": shlex.join(cmd),
            }
            for task, cmd in zip(tasks, commands)
        ],
    }
    (out_dir / "last_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="all", help="all, output_subdir, or comma-separated DATASET:role keys")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them")
    parser.add_argument("--smoke", action="store_true", help="Use a tiny 2-fold/1-seed/1-epoch run")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1337, 2024])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--nproc", type=int, default=4)
    parser.add_argument("--run-id", default=None, help="Fixed run_id forwarded to run_cv.py")
    parser.add_argument("--resume", action="store_true", help="Forward --resume to run_cv.py")
    parser.add_argument("--output-subdir", default=None, help="Override output subdir; intended for single-task runs")
    parser.add_argument("--smoke-folds", type=int, default=2)
    parser.add_argument("--smoke-seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--smoke-epochs", type=int, default=1)
    args = parser.parse_args()

    tasks = selected_tasks(args.task)
    if args.output_subdir and len(tasks) != 1:
        raise SystemExit("--output-subdir override is allowed only for a single selected task")

    commands = [build_command(task, args) for task in tasks]
    write_manifest(tasks, commands, args)

    for task, cmd in zip(tasks, commands):
        print(f"\n=== {task.key} ===")
        print(shlex.join(cmd))
        if args.dry_run:
            continue
        result = subprocess.run(cmd, cwd=str(ROOT))
        if result.returncode != 0:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
