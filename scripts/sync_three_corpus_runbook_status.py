#!/usr/bin/env python
"""Sync the three-corpus missing-experiment runbook with current artifacts.

The runbook is the human-facing checklist for the remaining three-corpus
evidence gap. This script keeps its live status block synchronized with the
machine-readable status files produced by the public-CV and follow-up probe
scanners. It does not train models or run probes.
"""

from __future__ import annotations

import csv
import json
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUNBOOK = ROOT / "outputs" / "three_corpus_completion" / "missing_experiment_runbook.md"
SOURCE_DIR = ROOT / "paper" / "figures" / "nature_source_data"
SOURCE_CSV = SOURCE_DIR / "three_corpus_live_status_source.csv"

CV_STATUS_JSON = ROOT / "outputs" / "public_cv_parity" / "status.json"
ROBUSTNESS_QUEUE_JSON = ROOT / "outputs" / "public_robustness_attribution" / "queue.json"
CALROT_QUEUE_JSON = ROOT / "outputs" / "public_calibration_rotation" / "queue.json"
FOLLOWUP_SUMMARY_JSON = ROOT / "outputs" / "public_followup_probes" / "summary.json"
FOLLOWUP_OUT_DIR = ROOT / "outputs" / "public_followup_probes"

START = "<!-- THREE_CORPUS_LIVE_STATUS_START -->"
END = "<!-- THREE_CORPUS_LIVE_STATUS_END -->"

TASK_ORDER = [
    ("ASP_clean", "baseline"),
    ("ASP_clean", "aafnet"),
    ("AS25_clean", "baseline"),
    ("AS25_clean", "aafnet"),
]


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def percent(mean_value: Any, std_value: Any) -> str:
    if mean_value is None:
        return "n/a"
    mean_pct = float(mean_value) * 100.0
    std_pct = float(std_value or 0.0) * 100.0
    return f"{mean_pct:.2f}% +/- {std_pct:.2f}%"


def count_by_task(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, int]]:
    out: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        key = (str(row.get("dataset", "")), str(row.get("role", "")))
        for field in row:
            value = row[field]
            if isinstance(value, int):
                out[key][field] += value
    return out


def robustness_counts(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, int]]:
    out: dict[tuple[str, str], dict[str, int]] = {}
    for dataset, role in TASK_ORDER:
        task_rows = [r for r in rows if r.get("dataset") == dataset and r.get("role") == role]
        out[(dataset, role)] = {
            "ready": sum(r.get("status") in {"checkpoint_ready", "robustness_complete"} for r in task_rows),
            "complete": sum(r.get("status") == "robustness_complete" for r in task_rows),
            "pending_ready": sum(r.get("status") == "checkpoint_ready" for r in task_rows),
            "waiting": sum(r.get("status") == "waiting_for_cv" for r in task_rows),
        }
    return out


def calrot_counts(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, int]]:
    out: dict[tuple[str, str], dict[str, int]] = {}
    for dataset, role in TASK_ORDER:
        task_rows = [r for r in rows if r.get("dataset") == dataset and r.get("role") == role]
        out[(dataset, role)] = {
            "ready": sum(r.get("status") != "waiting_for_cv" for r in task_rows),
            "calibration_complete": sum(r.get("calibration") == "complete" for r in task_rows),
            "calibration_pending_ready": sum(
                r.get("status") != "waiting_for_cv" and r.get("calibration") != "complete" for r in task_rows
            ),
            "rotation_complete": sum(r.get("rotation") == "complete" for r in task_rows),
            "rotation_pending_ready": sum(
                r.get("status") != "waiting_for_cv" and r.get("rotation") != "complete" for r in task_rows
            ),
            "waiting": sum(r.get("status") == "waiting_for_cv" for r in task_rows),
        }
    return out


def followup_counts(payload: dict[str, Any]) -> dict[tuple[str, str], dict[str, int]]:
    rows = payload.get("status", []) if isinstance(payload, dict) else []
    return count_by_task(rows)


def active_run_line(active_runs: list[dict[str, Any]]) -> str:
    if not active_runs:
        return "- Active public-CV process: not detected in status.json."
    bits = []
    for run in active_runs:
        alive = "alive" if run.get("alive") else "not alive"
        log_path = run.get("log_path") or "n/a"
        bits.append(
            f"`{run.get('run_id', 'unknown')}` PID {run.get('pid', 'n/a')} ({alive}, elapsed {run.get('elapsed', 'n/a')}, log `{log_path}`)"
        )
    return "- Active public-CV process: " + "; ".join(bits) + "."


def pid_alive(pid: int) -> bool:
    return subprocess.run(["ps", "-p", str(pid)], check=False, stdout=subprocess.DEVNULL).returncode == 0


def post_cv_launcher_line() -> str:
    launchers: list[str] = []
    for path in sorted(FOLLOWUP_OUT_DIR.glob("*_launcher.json")):
        payload = load_json(path, {})
        if not isinstance(payload, dict):
            continue
        label = payload.get("label", path.name.replace("_launcher.json", ""))
        pid = payload.get("pid")
        alive = False
        if isinstance(pid, int):
            alive = pid_alive(pid)
        log_path = payload.get("log_path") or "n/a"
        launchers.append(f"`{label}` PID {pid or 'n/a'} ({'alive' if alive else 'not alive'}, log `{log_path}`)")
    if not launchers:
        return "- Post-CV follow-up launcher: not started."
    return "- Post-CV follow-up launcher: " + "; ".join(launchers) + "."


def build_rows() -> tuple[list[dict[str, Any]], dict[str, int], list[dict[str, Any]]]:
    cv_payload = load_json(CV_STATUS_JSON, {})
    cv_rows = cv_payload.get("rows", []) if isinstance(cv_payload, dict) else []
    robust_rows = load_json(ROBUSTNESS_QUEUE_JSON, [])
    calrot_rows = load_json(CALROT_QUEUE_JSON, [])
    followup_payload = load_json(FOLLOWUP_SUMMARY_JSON, {})

    robust = robustness_counts(robust_rows if isinstance(robust_rows, list) else [])
    calrot = calrot_counts(calrot_rows if isinstance(calrot_rows, list) else [])
    followup = followup_counts(followup_payload if isinstance(followup_payload, dict) else {})

    rows: list[dict[str, Any]] = []
    by_task = {(row.get("dataset"), row.get("role")): row for row in cv_rows}
    for dataset, role in TASK_ORDER:
        key = (dataset, role)
        cv = by_task.get(key, {})
        completed = int(cv.get("n_folds_completed") or 0)
        total = int(cv.get("n_folds_total") or 15)
        r = robust.get(key, {})
        cr = calrot.get(key, {})
        f = followup.get(key, {})
        rows.append(
            {
                "dataset": dataset,
                "role": role,
                "cv_status": cv.get("status", "missing"),
                "cv_completed": completed,
                "cv_total": total,
                "cv_missing": max(total - completed, 0),
                "test_accuracy": percent(cv.get("test_accuracy_mean"), cv.get("test_accuracy_std")),
                "macro_f1": percent(cv.get("macro_f1_mean"), cv.get("macro_f1_std")),
                "robustness_ready": int(r.get("ready", 0)),
                "robustness_complete": int(f.get("robustness_complete", r.get("complete", 0))),
                "robustness_pending_ready": int(r.get("pending_ready", 0)),
                "calibration_ready": int(cr.get("ready", 0)),
                "calibration_complete": int(f.get("calibration_complete", cr.get("calibration_complete", 0))),
                "calibration_pending_ready": int(cr.get("calibration_pending_ready", 0)),
                "rotation_ready": int(cr.get("ready", 0)),
                "rotation_complete": int(f.get("rotation_complete", cr.get("rotation_complete", 0))),
                "rotation_pending_ready": int(cr.get("rotation_pending_ready", 0)),
                "waiting_for_cv": int(max(total - completed, 0)),
            }
        )

    totals = {
        "cv_completed": sum(int(row["cv_completed"]) for row in rows),
        "cv_total": sum(int(row["cv_total"]) for row in rows),
        "cv_missing": sum(int(row["cv_missing"]) for row in rows),
        "probe_complete": sum(
            int(row["robustness_complete"]) + int(row["calibration_complete"]) + int(row["rotation_complete"])
            for row in rows
        ),
        "probe_total": sum(int(row["cv_total"]) for row in rows) * 3,
        "probe_ready_to_run": sum(
            int(row["robustness_pending_ready"])
            + int(row["calibration_pending_ready"])
            + int(row["rotation_pending_ready"])
            for row in rows
        ),
        "probe_waiting_for_cv": sum(int(row["cv_missing"]) for row in rows) * 3,
    }
    active_runs = cv_payload.get("active_runs", []) if isinstance(cv_payload, dict) else []
    return rows, totals, active_runs


def write_source_csv(rows: list[dict[str, Any]]) -> None:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with SOURCE_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def render_block(rows: list[dict[str, Any]], totals: dict[str, int], active_runs: list[dict[str, Any]]) -> str:
    generated = time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime())
    complete = totals["cv_completed"] == totals["cv_total"] and totals["probe_complete"] == totals["probe_total"]
    lines = [
        START,
        "## Live Evidence Snapshot",
        "",
        "<!-- Auto-generated by scripts/sync_three_corpus_runbook_status.py; do not edit this block by hand. -->",
        "",
        f"- Generated at: {generated}.",
        f"- Public-corpus CV: {totals['cv_completed']}/{totals['cv_total']} fold-seed cells complete; {totals['cv_missing']} cells remain.",
        f"- Fold-level follow-up probes: {totals['probe_complete']}/{totals['probe_total']} outputs complete; {totals['probe_ready_to_run']} outputs are ready to run from existing checkpoints; {totals['probe_waiting_for_cv']} outputs are still waiting for CV checkpoints.",
        f"- Evidence state: {'complete' if complete else 'incomplete'} for full three-corpus parity.",
        active_run_line(active_runs),
        post_cv_launcher_line(),
        f"- Source data: `paper/figures/nature_source_data/{SOURCE_CSV.name}`.",
        "",
        "| Dataset | Role | CV | Test accuracy | Macro-F1 | Robustness | Calibration | Rotation | CV cells still missing |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {dataset} | {role} | {cv_completed}/{cv_total} ({cv_status}) | {test_accuracy} | {macro_f1} | "
            "{robustness_complete}/15 complete; {robustness_pending_ready} ready | "
            "{calibration_complete}/15 complete; {calibration_pending_ready} ready | "
            "{rotation_complete}/15 complete; {rotation_pending_ready} ready | {cv_missing}/15 |".format(**row)
        )
    lines.extend(["", END, ""])
    return "\n".join(lines)


def replace_block(text: str, block: str) -> str:
    if START in text and END in text:
        before = text.split(START, 1)[0].rstrip()
        after = text.split(END, 1)[1].lstrip()
        return f"{before}\n\n{block}\n{after}"
    marker = "\n## 1. Public-Corpus 5-Fold x 3-Seed CV"
    if marker in text:
        before, after = text.split(marker, 1)
        return f"{before.rstrip()}\n\n{block}\n{marker}{after}"
    return f"{block}\n{text}"


def main() -> None:
    rows, totals, active_runs = build_rows()
    write_source_csv(rows)
    block = render_block(rows, totals, active_runs)
    if not RUNBOOK.exists():
        raise SystemExit(f"missing runbook: {RUNBOOK}")
    old_text = RUNBOOK.read_text(encoding="utf-8")
    new_text = replace_block(old_text, block)
    RUNBOOK.write_text(new_text, encoding="utf-8")
    print(f"wrote: {RUNBOOK.relative_to(ROOT)}")
    print(f"wrote: {SOURCE_CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
