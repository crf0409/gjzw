#!/usr/bin/env python
"""Refresh public-corpus parity status, queues, figures and QA files.

This script is intentionally non-training: it only scans completed artifacts,
updates queue/source-data files, redraws the lightweight three-corpus status
figures, and reruns rotation QA. Use it after each public-CV fold finishes so
the manuscript evidence layer stays synchronized with the current worktree.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "three_corpus_completion"
REPORT_JSON = OUT_DIR / "refresh_three_corpus_public_status.json"
REPORT_MD = OUT_DIR / "refresh_three_corpus_public_status.md"


DEFAULT_STEPS: list[tuple[str, list[str]]] = [
    ("public_cv_status", ["scripts/summarize_public_cv_parity.py"]),
    ("public_cv_remaining", ["scripts/report_public_cv_remaining.py"]),
    ("public_robustness_queue", ["scripts/run_public_robustness_attribution.py", "--dry-run", "--limit", "0"]),
    ("public_calibration_rotation_queue", ["scripts/run_public_calibration_rotation.py", "--dry-run", "--limit", "0"]),
    ("public_followup_summary", ["scripts/summarize_public_followup_probes.py"]),
    ("three_corpus_runbook_status", ["scripts/sync_three_corpus_runbook_status.py"]),
    ("three_corpus_figures", ["scripts/complete_three_corpus_figures.py"]),
    ("figure_rotation_audit", ["scripts/audit_figure_rotations.py"]),
    ("three_corpus_completion_audit", ["scripts/audit_three_corpus_parity_completion.py"]),
]


def run_step(name: str, command: list[str]) -> dict[str, object]:
    started = time.time()
    proc = subprocess.run(
        [sys.executable, *command],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    elapsed = time.time() - started
    output = proc.stdout.strip()
    return {
        "name": name,
        "command": " ".join([sys.executable, *command]),
        "returncode": proc.returncode,
        "elapsed_sec": round(elapsed, 3),
        "output_tail": "\n".join(output.splitlines()[-40:]),
    }


def write_reports(rows: list[dict[str, object]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ok = all(int(row["returncode"]) == 0 for row in rows)
    payload = {
        "ok": ok,
        "generated_at_epoch": time.time(),
        "steps": rows,
    }
    REPORT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Three-Corpus Public Status Refresh",
        "",
        f"- Overall status: {'ok' if ok else 'failed'}",
        f"- Steps: {len(rows)}",
        "",
        "| Step | Return code | Seconds |",
        "|---|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['name']} | {row['returncode']} | {row['elapsed_sec']} |")
    lines.append("")
    lines.extend(["## Output Tails", ""])
    for row in rows:
        lines.extend(
            [
                f"### {row['name']}",
                "",
                "```text",
                str(row["output_tail"]),
                "```",
                "",
            ]
        )
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-figures", action="store_true", help="Skip complete_three_corpus_figures.py.")
    args = parser.parse_args()

    steps = [step for step in DEFAULT_STEPS if not (args.skip_figures and step[0] == "three_corpus_figures")]
    rows = [run_step(name, command) for name, command in steps]
    write_reports(rows)
    for row in rows:
        print(f"{row['name']}: rc={row['returncode']} elapsed={row['elapsed_sec']}s")
    print(f"wrote: {REPORT_MD.relative_to(ROOT)}")
    if any(int(row["returncode"]) != 0 for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
