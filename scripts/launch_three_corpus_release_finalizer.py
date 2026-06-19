#!/usr/bin/env python
"""Launch the final three-corpus package/upload finalizer in the background."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "three_corpus_completion"
LOG_DIR = OUT_DIR / "logs"
DEFAULT_CONFIG_DOC = Path("/home/siton02/disk_sdg/md0_backup_2026-04-28/crf/tos.md")


def pid_alive(pid: int) -> bool:
    return subprocess.run(["ps", "-p", str(pid)], check=False, stdout=subprocess.DEVNULL).returncode == 0


def load_existing_status(label: str) -> dict[str, object]:
    path = OUT_DIR / f"{label}_launcher.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def write_status(label: str, payload: dict[str, object]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / f"{label}_launcher.json"
    md_path = OUT_DIR / f"{label}_launcher.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Three-Corpus Release Finalizer Launcher",
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="three_corpus_release_final_auto")
    parser.add_argument("--run-id", default="public_cv_parity_v1")
    parser.add_argument("--followup-label", default="post_cv_followups_public_cv_parity_v1_pool")
    parser.add_argument("--poll-sec", type=int, default=300)
    parser.add_argument("--config-doc", default=str(DEFAULT_CONFIG_DOC))
    parser.add_argument("--expires-days", type=float, default=3.0)
    parser.add_argument("--include-checkpoints", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    pid_path = OUT_DIR / f"{args.label}.pid"
    log_path = LOG_DIR / f"{args.label}.log"

    if pid_path.exists() and not args.force:
        try:
            existing_pid = int(pid_path.read_text(encoding="utf-8").strip())
        except ValueError:
            existing_pid = -1
        if existing_pid > 0 and pid_alive(existing_pid):
            existing = load_existing_status(args.label)
            payload = {
                **existing,
                "label": args.label,
                "pid": existing_pid,
                "alive": True,
                "log_path": str(log_path.relative_to(ROOT)),
                "message": "existing finalizer is still alive; no new process started",
            }
            write_status(args.label, payload)
            print(f"[skip] existing finalizer alive: pid={existing_pid}")
            print(f"status: {(OUT_DIR / f'{args.label}_launcher.md').relative_to(ROOT)}")
            return

    command = [
        sys.executable,
        "-u",
        "scripts/finalize_three_corpus_release.py",
        "--label",
        args.label,
        "--wait",
        "--poll-sec",
        str(args.poll_sec),
        "--run-id",
        args.run_id,
        "--followup-label",
        args.followup_label,
        "--config-doc",
        args.config_doc,
        "--expires-days",
        str(args.expires_days),
    ]
    if args.include_checkpoints:
        command.append("--include-checkpoints")

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
        "label": args.label,
        "pid": proc.pid,
        "alive": True,
        "started_at_epoch": time.time(),
        "command": " ".join(command),
        "log_path": str(log_path.relative_to(ROOT)),
        "pid_path": str(pid_path.relative_to(ROOT)),
    }
    write_status(args.label, payload)
    print(f"started: pid={proc.pid}")
    print(f"log: {log_path.relative_to(ROOT)}")
    print(f"status: {(OUT_DIR / f'{args.label}_launcher.md').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
