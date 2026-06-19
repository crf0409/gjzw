#!/usr/bin/env python
"""Watch public-CV fold counts and refresh manuscript status on changes.

The watcher is non-training. It scans completed ``test_metrics.json`` files for
the four public-CV tasks and runs ``refresh_three_corpus_public_status.py`` only
when the counts change or when ``--force-refresh`` is supplied. This keeps
tables and F-TC coverage figures aligned with long-running CV jobs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "public_cv_parity"
STATE_PATH = OUT_DIR / "progress_watch_state.json"
REPORT_PATH = OUT_DIR / "progress_watch.md"

TASKS = [
    ("ASP_clean", "baseline", "cv_asp_baseline"),
    ("ASP_clean", "aafnet", "cv_asp_aafnet"),
    ("AS25_clean", "baseline", "cv_as25_baseline"),
    ("AS25_clean", "aafnet", "cv_as25_aafnet"),
]


def count_metrics(output_subdir: str, run_id: str) -> tuple[int, list[str]]:
    root = ROOT / "outputs" / output_subdir / run_id / "resnet50"
    if not root.exists():
        return 0, []
    paths = sorted(
        p
        for p in root.glob("seed*_fold*_train/*/resnet50/test_metrics.json")
        if "latest" not in p.parts
    )
    return len(paths), [str(path.relative_to(ROOT)) for path in paths]


def collect(run_id: str) -> dict[str, object]:
    rows = []
    for dataset, role, output_subdir in TASKS:
        count, paths = count_metrics(output_subdir, run_id)
        rows.append(
            {
                "dataset": dataset,
                "role": role,
                "output_subdir": output_subdir,
                "count": count,
                "target": 15,
                "paths": paths,
            }
        )
    return {
        "run_id": run_id,
        "rows": rows,
        "counts": {f"{row['dataset']}:{row['role']}": row["count"] for row in rows},
        "all_complete": all(int(row["count"]) == 15 for row in rows),
    }


def load_previous() -> dict[str, object] | None:
    if not STATE_PATH.exists():
        return None
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def run_refresh() -> dict[str, object]:
    started = time.time()
    proc = subprocess.run(
        [sys.executable, "scripts/refresh_three_corpus_public_status.py"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return {
        "returncode": proc.returncode,
        "elapsed_sec": round(time.time() - started, 3),
        "output_tail": "\n".join(proc.stdout.strip().splitlines()[-50:]),
    }


def write_state(payload: dict[str, object]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_report(current: dict[str, object], previous: dict[str, object] | None, refreshed: bool, refresh_result: dict[str, object] | None) -> None:
    lines = [
        "# Public-CV Progress Watch",
        "",
        f"- Run id: `{current['run_id']}`",
        f"- Counts changed: {previous is None or current.get('counts') != previous.get('counts')}",
        f"- Refreshed derived artifacts: {refreshed}",
        f"- All public-CV tasks complete: {current['all_complete']}",
        "",
        "| Dataset | Role | Complete | Target |",
        "|---|---|---:|---:|",
    ]
    for row in current["rows"]:
        lines.append(f"| {row['dataset']} | {row['role']} | {row['count']} | {row['target']} |")
    if refresh_result is not None:
        lines.extend(
            [
                "",
                "## Refresh Result",
                "",
                f"- Return code: {refresh_result['returncode']}",
                f"- Seconds: {refresh_result['elapsed_sec']}",
                "",
                "```text",
                str(refresh_result["output_tail"]),
                "```",
            ]
        )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def tick(args: argparse.Namespace) -> bool:
    previous = load_previous()
    current = collect(args.run_id)
    changed = previous is None or current.get("counts") != previous.get("counts")
    refreshed = bool(args.force_refresh or changed)
    refresh_result = run_refresh() if refreshed else None
    current["last_checked_epoch"] = time.time()
    current["last_refresh"] = refresh_result
    write_state(current)
    write_report(current, previous, refreshed, refresh_result)
    print(f"counts: {current['counts']}")
    print(f"changed={changed} refreshed={refreshed} all_complete={current['all_complete']}")
    if refresh_result is not None and int(refresh_result["returncode"]) != 0:
        raise SystemExit(int(refresh_result["returncode"]))
    return bool(current["all_complete"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="public_cv_parity_v1")
    parser.add_argument("--interval-sec", type=int, default=300)
    parser.add_argument("--once", action="store_true", help="Run one watch tick and exit.")
    parser.add_argument("--force-refresh", action="store_true", help="Refresh even if counts did not change.")
    parser.add_argument("--exit-when-complete", action="store_true")
    args = parser.parse_args()

    while True:
        complete = tick(args)
        if args.once or (args.exit_when_complete and complete):
            return
        time.sleep(args.interval_sec)


if __name__ == "__main__":
    main()
