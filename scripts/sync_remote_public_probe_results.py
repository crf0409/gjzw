#!/usr/bin/env python
"""Pull remote public-corpus probe outputs back into the local repository.

The remote worker only runs fold-level inference probes. Local refresh scripts
remain the source of truth for manuscript figures, source data, and completion
audit, so this sync is intentionally one-way and non-destructive.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], *, check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=check,
    )


def rsync_from_remote(host: str, remote_root: str, rel_path: str) -> subprocess.CompletedProcess[str]:
    local_path = ROOT / rel_path
    local_path.parent.mkdir(parents=True, exist_ok=True)
    return run(
        [
            "rsync",
            "-az",
            f"{host}:{remote_root.rstrip('/')}/{rel_path.rstrip('/')}/",
            f"{local_path}/",
        ]
    )


def remote_pid_status(host: str, remote_root: str, pid_file: str) -> str:
    cmd = (
        f"pid_file='{remote_root.rstrip('/')}/{pid_file}'; "
        "if [ -f \"$pid_file\" ]; then "
        "pid=$(cat \"$pid_file\"); "
        "ps -p \"$pid\" -o pid=,stat=,etime=,cmd= || true; "
        "else echo 'no pid file'; fi"
    )
    proc = run(["ssh", "-o", "BatchMode=yes", host, cmd])
    return proc.stdout.strip()


def sync_once(args: argparse.Namespace) -> int:
    targets = [
        "outputs/public_robustness_attribution",
        "outputs/public_calibration_rotation",
        "outputs/public_followup_probes",
    ]
    failures = 0
    print(f"[remote] {remote_pid_status(args.host, args.remote_root, args.pid_file)}", flush=True)
    for rel_path in targets:
        proc = rsync_from_remote(args.host, args.remote_root, rel_path)
        print(f"[sync] {rel_path}: rc={proc.returncode}", flush=True)
        if proc.returncode != 0:
            failures += 1
            print(proc.stdout.strip().splitlines()[-20:], flush=True)

    if args.summarize and failures == 0:
        proc = run([sys.executable, "scripts/summarize_public_followup_probes.py"])
        print(f"[summarize] rc={proc.returncode}", flush=True)
        if proc.returncode != 0:
            failures += 1
            print(proc.stdout.strip().splitlines()[-40:], flush=True)

    if args.refresh and failures == 0:
        proc = run([sys.executable, "scripts/refresh_three_corpus_public_status.py"])
        print(f"[refresh] rc={proc.returncode}", flush=True)
        if proc.returncode != 0:
            failures += 1
            print(proc.stdout.strip().splitlines()[-80:], flush=True)
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="siton01@172.26.250.16")
    parser.add_argument("--remote-root", default="/home/siton01/gjzw_three_corpus_remote")
    parser.add_argument("--pid-file", default="outputs/public_followup_probes/remote_partial_probe_public_cv_parity_v1.pid")
    parser.add_argument("--interval-sec", type=int, default=300)
    parser.add_argument("--iterations", type=int, default=1, help="Number of sync passes; use 0 to run until interrupted.")
    parser.add_argument("--summarize", action="store_true", help="Rebuild local follow-up source tables after syncing.")
    parser.add_argument("--refresh", action="store_true", help="Run full local three-corpus refresh after syncing.")
    args = parser.parse_args()

    count = 0
    while True:
        failures = sync_once(args)
        count += 1
        if failures:
            raise SystemExit(failures)
        if args.iterations and count >= args.iterations:
            return
        time.sleep(args.interval_sec)


if __name__ == "__main__":
    main()
