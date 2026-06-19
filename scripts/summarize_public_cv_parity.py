#!/usr/bin/env python
"""Summarize public-corpus CV parity completion status."""

from __future__ import annotations

import csv
import json
import re
import statistics
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "public_cv_parity"
SOURCE_DIR = ROOT / "paper" / "figures" / "nature_source_data"

TASKS = [
    ("ASP_clean", "baseline", "cv_asp_baseline"),
    ("ASP_clean", "aafnet", "cv_asp_aafnet"),
    ("AS25_clean", "baseline", "cv_as25_baseline"),
    ("AS25_clean", "aafnet", "cv_as25_aafnet"),
]

SMOKE_TASKS = [
    ("ASP_clean", "baseline", "cv_smoke_asp_baseline"),
    ("ASP_clean", "aafnet", "cv_smoke_asp_aafnet"),
    ("AS25_clean", "baseline", "cv_smoke_as25_baseline"),
    ("AS25_clean", "aafnet", "cv_smoke_as25_aafnet"),
]


def latest_summary(output_subdir: str) -> Path | None:
    root = ROOT / "outputs" / output_subdir
    if not root.exists():
        return None
    candidates = [p for p in root.glob("*/resnet50/cv_summary.json") if "latest" not in p.parts]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]


def summarize_completed_fold_metrics(output_subdir: str, expected_total: int) -> dict[str, object] | None:
    root = ROOT / "outputs" / output_subdir
    if not root.exists():
        return None
    candidates = [
        p
        for p in root.glob("*/resnet50/seed*_fold*_train/*/resnet50/test_metrics.json")
        if "latest" not in p.parts
    ]
    if not candidates:
        return None

    by_fold: dict[tuple[int, int], Path] = {}
    for path in candidates:
        fold_key = None
        for part in path.parts:
            match = re.fullmatch(r"seed(\d+)_fold(\d+)_train", part)
            if match:
                fold_key = (int(match.group(1)), int(match.group(2)))
                break
        if fold_key is None:
            continue
        current = by_fold.get(fold_key)
        if current is None or path.stat().st_mtime > current.stat().st_mtime:
            by_fold[fold_key] = path
    if not by_fold:
        return None

    accs: list[float] = []
    f1s: list[float] = []
    paths: list[str] = []
    for path in sorted(by_fold.values()):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("test_accuracy") is not None:
            accs.append(float(data["test_accuracy"]))
        if data.get("macro_f1") is not None:
            f1s.append(float(data["macro_f1"]))
        paths.append(str(path.relative_to(ROOT)))

    def mean_std(values: list[float]) -> tuple[float | None, float | None]:
        if not values:
            return None, None
        if len(values) == 1:
            return values[0], 0.0
        return statistics.mean(values), statistics.stdev(values)

    acc_mean, acc_std = mean_std(accs)
    f1_mean, f1_std = mean_std(f1s)
    latest_path = max(by_fold.values(), key=lambda p: p.stat().st_mtime)
    return {
        "status": "partial",
        "summary_path": f"partial scan; latest {latest_path.relative_to(ROOT)}",
        "run_id": latest_path.parts[latest_path.parts.index(output_subdir) + 1] if output_subdir in latest_path.parts else "",
        "n_folds_completed": len(by_fold),
        "n_folds_total": expected_total,
        "test_accuracy_mean": acc_mean,
        "test_accuracy_std": acc_std,
        "macro_f1_mean": f1_mean,
        "macro_f1_std": f1_std,
        "completed_metric_paths": ";".join(paths),
    }


def pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.2f} %"


def collect_rows(tasks=TASKS, expected_total: int = 15) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset, role, output_subdir in tasks:
        path = latest_summary(output_subdir)
        if path is None:
            partial = summarize_completed_fold_metrics(output_subdir, expected_total)
            if partial is not None:
                rows.append(
                    {
                        "dataset": dataset,
                        "role": role,
                        "output_subdir": output_subdir,
                        **partial,
                    }
                )
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "role": role,
                    "output_subdir": output_subdir,
                    "status": "missing",
                    "summary_path": "",
                    "n_folds_completed": 0,
                    "n_folds_total": expected_total,
                    "test_accuracy_mean": None,
                    "test_accuracy_std": None,
                    "macro_f1_mean": None,
                    "macro_f1_std": None,
                }
            )
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        completed = int(data.get("n_folds_completed", 0))
        total = int(data.get("n_folds_total", expected_total))
        rows.append(
            {
                "dataset": dataset,
                "role": role,
                "output_subdir": output_subdir,
                "status": "complete" if completed == total and total == expected_total else "partial",
                "summary_path": str(path.relative_to(ROOT)),
                "run_id": data.get("run_id", path.parents[1].name),
                "n_folds_completed": completed,
                "n_folds_total": total,
                "test_accuracy_mean": data.get("test_accuracy_mean"),
                "test_accuracy_std": data.get("test_accuracy_std"),
                "macro_f1_mean": data.get("macro_f1_mean"),
                "macro_f1_std": data.get("macro_f1_std"),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def collect_active_runs() -> list[dict[str, object]]:
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
        row: dict[str, object] = {
            "run_id": run_id,
            "pid": pid,
            "alive": ps.returncode == 0 and bool(ps.stdout.strip()),
        }
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


def render_md(rows: list[dict[str, object]], active_runs: list[dict[str, object]] | None = None) -> str:
    lines = [
        "# Public-Corpus CV Parity Status",
        "",
        "This status file tracks whether ASP_clean and AS25_clean have the same 5-fold x 3-seed confirmation depth as AL6.",
        "",
        "| Dataset | Role | Status | Folds | Test accuracy | Macro-F1 | Summary |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        folds = f"{row['n_folds_completed']}/{row['n_folds_total']}"
        acc = "n/a"
        if row["test_accuracy_mean"] is not None:
            acc = f"{pct(float(row['test_accuracy_mean']))} +/- {pct(float(row['test_accuracy_std'] or 0.0))}"
        f1 = "n/a"
        if row["macro_f1_mean"] is not None:
            f1 = f"{pct(float(row['macro_f1_mean']))} +/- {pct(float(row['macro_f1_std'] or 0.0))}"
        summary = row["summary_path"] or "not found"
        lines.append(
            f"| {row['dataset']} | {row['role']} | {row['status']} | {folds} | {acc} | {f1} | `{summary}` |"
        )
    complete = all(row["status"] == "complete" for row in rows)
    lines += [
        "",
        f"Overall status: **{'complete' if complete else 'incomplete'}**.",
        "",
    ]
    if not complete:
        lines.append("Remaining requirement: each public-corpus baseline/AAFNet task must reach 15/15 folds before the public corpora can be described as statistical peers of AL6.")
    if active_runs:
        lines += ["", "## Active run monitor", "", "| Run | PID | Alive | Elapsed | Log |", "|---|---:|---|---:|---|"]
        for run in active_runs:
            log_path = run.get("log_path", "")
            lines.append(
                f"| {run['run_id']} | {run['pid']} | {run['alive']} | "
                f"{run.get('elapsed', 'n/a')} | `{log_path or 'n/a'}` |"
            )
    return "\n".join(lines) + "\n"


def render_smoke_md(rows: list[dict[str, object]]) -> str:
    lines = [
        "# Public-Corpus CV Smoke Status",
        "",
        "Smoke runs use 2 folds x 1 seed x 1 epoch and are only used to verify the training path.",
        "",
        "| Dataset | Role | Status | Folds | Test accuracy | Summary |",
        "|---|---|---|---:|---:|---|",
    ]
    for row in rows:
        folds = f"{row['n_folds_completed']}/{row['n_folds_total']}"
        acc = "n/a"
        if row["test_accuracy_mean"] is not None:
            acc = f"{pct(float(row['test_accuracy_mean']))} +/- {pct(float(row['test_accuracy_std'] or 0.0))}"
        summary = row["summary_path"] or "not found"
        lines.append(f"| {row['dataset']} | {row['role']} | {row['status']} | {folds} | {acc} | `{summary}` |")
    lines += ["", f"Smoke overall status: **{'complete' if all(row['status'] == 'complete' for row in rows) else 'incomplete'}**.", ""]
    return "\n".join(lines) + "\n"


def main() -> None:
    rows = collect_rows(TASKS, expected_total=15)
    smoke_rows = collect_rows(SMOKE_TASKS, expected_total=2)
    active_runs = collect_active_runs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "all_complete": all(row["status"] == "complete" for row in rows),
        "smoke_complete": all(row["status"] == "complete" for row in smoke_rows),
        "rows": rows,
        "smoke_rows": smoke_rows,
        "active_runs": active_runs,
    }
    (OUT_DIR / "status.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (OUT_DIR / "status.md").write_text(render_md(rows, active_runs), encoding="utf-8")
    (OUT_DIR / "smoke_status.md").write_text(render_smoke_md(smoke_rows), encoding="utf-8")
    write_csv(SOURCE_DIR / "public_cv_parity_status_source.csv", rows)
    write_csv(SOURCE_DIR / "public_cv_smoke_status_source.csv", smoke_rows)
    print(render_md(rows, active_runs))
    print(render_smoke_md(smoke_rows))


if __name__ == "__main__":
    main()
