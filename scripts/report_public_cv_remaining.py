#!/usr/bin/env python
"""Report remaining public-corpus CV parity work.

This is a lightweight companion to summarize_public_cv_parity.py. It lists the
exact seed/fold cells still needed for each public-corpus baseline/AAFNet task
so interrupted long runs can be audited and resumed without ambiguity.
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "public_cv_parity"
SOURCE_DIR = ROOT / "paper" / "figures" / "nature_source_data"
SEEDS = [42, 1337, 2024]
FOLDS = [0, 1, 2, 3, 4]
TASKS = [
    ("ASP_clean", "baseline", "cv_asp_baseline"),
    ("ASP_clean", "aafnet", "cv_asp_aafnet"),
    ("AS25_clean", "baseline", "cv_as25_baseline"),
    ("AS25_clean", "aafnet", "cv_as25_aafnet"),
]


def completed_cells(output_subdir: str) -> dict[tuple[int, int], Path]:
    root = ROOT / "outputs" / output_subdir
    cells: dict[tuple[int, int], Path] = {}
    if not root.exists():
        return cells
    for path in root.glob("*/resnet50/seed*_fold*_train/*/resnet50/test_metrics.json"):
        if "latest" in path.parts:
            continue
        key = None
        for part in path.parts:
            match = re.fullmatch(r"seed(\d+)_fold(\d+)_train", part)
            if match:
                key = (int(match.group(1)), int(match.group(2)))
                break
        if key is None:
            continue
        old = cells.get(key)
        if old is None or path.stat().st_mtime > old.stat().st_mtime:
            cells[key] = path
    return cells


def active_runs() -> list[dict[str, object]]:
    runs: list[dict[str, object]] = []
    for pid_path in sorted(OUT_DIR.glob("*.pid")):
        run_id = pid_path.stem
        try:
            pid = int(pid_path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            continue
        ps = subprocess.run(
            ["ps", "-p", str(pid), "-o", "pid=,stat=,etime=,cmd="],
            check=False,
            capture_output=True,
            text=True,
        )
        row: dict[str, object] = {"run_id": run_id, "pid": pid, "alive": ps.returncode == 0 and bool(ps.stdout.strip())}
        if row["alive"]:
            parts = ps.stdout.strip().split(None, 3)
            if len(parts) >= 3:
                row["stat"] = parts[1]
                row["elapsed"] = parts[2]
            if len(parts) == 4:
                row["cmd"] = parts[3]
        log_path_file = OUT_DIR / f"{run_id}.logpath"
        if log_path_file.exists():
            row["log_path"] = log_path_file.read_text(encoding="utf-8").strip()
        runs.append(row)
    return runs


def collect_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset, role, output_subdir in TASKS:
        done = completed_cells(output_subdir)
        for seed in SEEDS:
            for fold in FOLDS:
                key = (seed, fold)
                path = done.get(key)
                rows.append(
                    {
                        "dataset": dataset,
                        "role": role,
                        "output_subdir": output_subdir,
                        "seed": seed,
                        "fold": fold,
                        "status": "complete" if path else "missing",
                        "metric_path": str(path.relative_to(ROOT)) if path else "",
                    }
                )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = ["dataset", "role", "output_subdir", "seed", "fold", "status", "metric_path"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def render_md(rows: list[dict[str, object]], runs: list[dict[str, object]]) -> str:
    lines = [
        "# Public-Corpus CV Remaining Queue",
        "",
        "This file enumerates the exact 5-fold x 3-seed cells still required before ASP_clean and AS25_clean can be treated as AL6-level statistical peers.",
        "",
        "| Dataset | Role | Complete | Missing | Next missing cells |",
        "|---|---|---:|---:|---|",
    ]
    for dataset, role, output_subdir in TASKS:
        task_rows = [row for row in rows if row["output_subdir"] == output_subdir]
        complete = sum(1 for row in task_rows if row["status"] == "complete")
        missing_rows = [row for row in task_rows if row["status"] != "complete"]
        next_cells = ", ".join(f"s{row['seed']}-f{row['fold']}" for row in missing_rows[:8])
        if len(missing_rows) > 8:
            next_cells += ", ..."
        lines.append(f"| {dataset} | {role} | {complete}/15 | {len(missing_rows)} | {next_cells or 'none'} |")
    if runs:
        lines += ["", "## Active Runs", "", "| Run | PID | Alive | Elapsed | Log |", "|---|---:|---|---:|---|"]
        for run in runs:
            lines.append(
                f"| {run['run_id']} | {run['pid']} | {run['alive']} | "
                f"{run.get('elapsed', 'n/a')} | `{run.get('log_path', 'n/a')}` |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    rows = collect_rows()
    runs = active_runs()
    payload = {"rows": rows, "active_runs": runs}
    (OUT_DIR / "remaining_queue.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (OUT_DIR / "remaining_queue.md").write_text(render_md(rows, runs), encoding="utf-8")
    write_csv(SOURCE_DIR / "public_cv_remaining_queue_source.csv", rows)
    print(render_md(rows, runs))


if __name__ == "__main__":
    main()
