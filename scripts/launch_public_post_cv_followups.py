#!/usr/bin/env python
"""Launch the public post-CV follow-up runner in the background.

The launched process waits for the recorded public-CV PID to exit, refreshes
the public-corpus status, and only starts fold-level probes when the public CV
is complete. It is a process supervisor only; the heavy probe logic remains in
``scripts/run_public_post_cv_followups.py``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "public_followup_probes"
LOG_DIR = OUT_DIR / "logs"


def pid_alive(pid: int) -> bool:
    return subprocess.run(["ps", "-p", str(pid)], check=False, stdout=subprocess.DEVNULL).returncode == 0


def write_status(label: str, payload: dict[str, object]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / f"{label}_launcher.json"
    md_path = OUT_DIR / f"{label}_launcher.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Public Post-CV Follow-Up Launcher",
        "",
        f"- Label: `{label}`",
        f"- PID: {payload.get('pid', 'n/a')}",
        f"- Alive: {payload.get('alive', 'n/a')}",
        f"- Started at epoch: {payload.get('started_at_epoch', 'n/a')}",
        f"- Log: `{payload.get('log_path', 'n/a')}`",
        "",
        "## Command",
        "",
        "```text",
        str(payload.get("command", "")),
        "```",
    ]
    if payload.get("message"):
        lines.extend(["", "## Message", "", str(payload["message"])])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_existing_status(label: str) -> dict[str, object]:
    json_path = OUT_DIR / f"{label}_launcher.json"
    if not json_path.exists():
        return {}
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="public_cv_parity_v1")
    parser.add_argument("--label", default=None)
    parser.add_argument("--poll-sec", type=int, default=300)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--devices", nargs="+", default=None, help="Device list for the pool runner.")
    parser.add_argument("--robust-batch-size", type=int, default=64)
    parser.add_argument("--calrot-batch-size", type=int, default=128)
    parser.add_argument("--pool", action="store_true", help="Launch the multi-device probe pool runner.")
    parser.add_argument("--force", action="store_true", help="Start a new launcher even if the label PID is alive.")
    args = parser.parse_args()

    label = args.label or f"post_cv_followups_{args.run_id}_auto"
    pid_path = OUT_DIR / f"{label}.pid"
    log_path = LOG_DIR / f"{label}.log"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if pid_path.exists() and not args.force:
        try:
            existing_pid = int(pid_path.read_text(encoding="utf-8").strip())
        except ValueError:
            existing_pid = -1
        if existing_pid > 0 and pid_alive(existing_pid):
            existing = load_existing_status(label)
            payload = {
                **existing,
                "label": label,
                "pid": existing_pid,
                "alive": True,
                "log_path": str(log_path.relative_to(ROOT)),
                "message": "existing launcher is still alive; no new process started",
            }
            write_status(label, payload)
            print(f"[skip] existing launcher alive: pid={existing_pid}")
            print(f"status: {(OUT_DIR / f'{label}_launcher.md').relative_to(ROOT)}")
            return

    command = [
        sys.executable,
        "-u",
        "scripts/run_public_followup_probe_pool.py" if args.pool else "scripts/run_public_post_cv_followups.py",
        "--run-id",
        args.run_id,
        "--wait",
        "--poll-sec",
        str(args.poll_sec),
        "--run-probes",
        "--device",
        args.device,
        "--robust-batch-size",
        str(args.robust_batch_size),
        "--calrot-batch-size",
        str(args.calrot_batch_size),
        "--label",
        label,
    ]
    if args.pool and args.devices:
        command.extend(["--devices", *args.devices])
    log_handle = log_path.open("a", encoding="utf-8")
    proc = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    pid_path.write_text(f"{proc.pid}\n", encoding="utf-8")
    payload = {
        "label": label,
        "pid": proc.pid,
        "alive": True,
        "started_at_epoch": time.time(),
        "command": " ".join(command),
        "log_path": str(log_path.relative_to(ROOT)),
        "pid_path": str(pid_path.relative_to(ROOT)),
    }
    write_status(label, payload)
    print(f"started: pid={proc.pid}")
    print(f"log: {log_path.relative_to(ROOT)}")
    print(f"status: {(OUT_DIR / f'{label}_launcher.md').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
