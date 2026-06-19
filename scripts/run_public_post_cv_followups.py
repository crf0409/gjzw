#!/usr/bin/env python
"""Run public-corpus follow-up probes after full public CV is complete.

Default mode is conservative: refresh status and verify completion only. Add
``--run-probes`` to launch fold-level robustness, calibration and rotation
probes for all completed public-CV cells. Add ``--wait`` to block until the
recorded public-CV process exits before checking completion.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CV_DIR = ROOT / "outputs" / "public_cv_parity"
OUT_DIR = ROOT / "outputs" / "public_followup_probes"


def pid_alive(pid: int) -> bool:
    return subprocess.run(["ps", "-p", str(pid)], check=False, stdout=subprocess.DEVNULL).returncode == 0


def read_pid(run_id: str) -> int | None:
    path = CV_DIR / f"{run_id}.pid"
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def run_command(name: str, command: list[str], report_rows: list[dict[str, object]]) -> None:
    started = time.time()
    proc = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    row = {
        "name": name,
        "command": " ".join(command),
        "returncode": proc.returncode,
        "elapsed_sec": round(time.time() - started, 3),
        "output_tail": "\n".join(proc.stdout.strip().splitlines()[-80:]),
    }
    report_rows.append(row)
    print(f"{name}: rc={proc.returncode} elapsed={row['elapsed_sec']}s")
    if proc.returncode != 0:
        raise RuntimeError(f"{name} failed with return code {proc.returncode}")


def refresh(report_rows: list[dict[str, object]]) -> dict[str, object]:
    run_command(
        "refresh_three_corpus_public_status",
        [sys.executable, "scripts/refresh_three_corpus_public_status.py"],
        report_rows,
    )
    status_path = CV_DIR / "status.json"
    if not status_path.exists():
        raise FileNotFoundError(status_path)
    return json.loads(status_path.read_text(encoding="utf-8"))


def write_report(rows: list[dict[str, object]], status_payload: dict[str, object], label: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "label": label,
        "generated_at_epoch": time.time(),
        "status_all_complete": bool(status_payload.get("all_complete")),
        "steps": rows,
    }
    json_path = OUT_DIR / f"{label}.json"
    md_path = OUT_DIR / f"{label}.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Public Post-CV Follow-Up Runner",
        "",
        f"- Label: `{label}`",
        f"- CV all complete: {bool(status_payload.get('all_complete'))}",
        f"- Steps: {len(rows)}",
        "",
        "| Step | Return code | Seconds |",
        "|---|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['name']} | {row['returncode']} | {row['elapsed_sec']} |")
    lines.extend(["", "## Output Tails", ""])
    for row in rows:
        lines.extend([f"### {row['name']}", "", "```text", str(row["output_tail"]), "```", ""])
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote: {md_path.relative_to(ROOT)}")


def wait_for_cv(run_id: str, poll_sec: int) -> None:
    while True:
        pid = read_pid(run_id)
        if pid is None or not pid_alive(pid):
            return
        print(f"[wait] public CV run {run_id} still alive: pid={pid}; sleeping {poll_sec}s")
        time.sleep(poll_sec)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="public_cv_parity_v1")
    parser.add_argument("--wait", action="store_true", help="Wait until the recorded public-CV PID exits.")
    parser.add_argument("--poll-sec", type=int, default=300)
    parser.add_argument("--run-probes", action="store_true", help="Launch robustness, calibration and rotation probes.")
    parser.add_argument("--allow-partial", action="store_true", help="Run probes for currently ready cells even if CV is incomplete.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--robust-batch-size", type=int, default=64)
    parser.add_argument("--calrot-batch-size", type=int, default=128)
    parser.add_argument("--label", default=None)
    args = parser.parse_args()

    label = args.label or f"post_cv_followups_{time.strftime('%Y%m%d_%H%M%S')}"
    rows: list[dict[str, object]] = []
    status_payload: dict[str, object] = {}
    try:
        if args.wait:
            wait_for_cv(args.run_id, args.poll_sec)
        status_payload = refresh(rows)
        all_complete = bool(status_payload.get("all_complete"))
        if not all_complete and not args.allow_partial:
            print("[stop] public CV is not complete; rerun with --allow-partial to probe ready cells.")
            return
        if not args.run_probes:
            print("[stop] --run-probes not set; refreshed status only.")
            return

        run_command(
            "public_robustness_attribution",
            [
                sys.executable,
                "scripts/run_public_robustness_attribution.py",
                "--run-id",
                args.run_id,
                "--task",
                "all",
                "--batch-size",
                str(args.robust_batch_size),
                "--device",
                args.device,
            ],
            rows,
        )
        run_command(
            "public_calibration_rotation",
            [
                sys.executable,
                "scripts/run_public_calibration_rotation.py",
                "--run-id",
                args.run_id,
                "--task",
                "all",
                "--batch-size",
                str(args.calrot_batch_size),
                "--device",
                args.device,
                "--probes",
                "calibration",
                "rotation",
            ],
            rows,
        )
        run_command("public_followup_summary", [sys.executable, "scripts/summarize_public_followup_probes.py"], rows)
        status_payload = refresh(rows)
    finally:
        write_report(rows, status_payload, label)


if __name__ == "__main__":
    main()
