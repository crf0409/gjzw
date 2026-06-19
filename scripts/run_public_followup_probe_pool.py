#!/usr/bin/env python
"""Run public-corpus follow-up probes with a resumable device pool.

The older post-CV runner executes the robustness stage and calibration/rotation
stage serially. This pool runner keeps the same per-cell wrappers and output
contracts, but dispatches independent fold cells across a caller-provided list
of devices. Each wrapper remains resumable, so reruns skip completed JSON files.
"""

from __future__ import annotations

import argparse
import csv
import json
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CV_DIR = ROOT / "outputs" / "public_cv_parity"
OUT_DIR = ROOT / "outputs" / "public_followup_probes"
LOG_DIR = OUT_DIR / "logs"
SOURCE_DIR = ROOT / "paper" / "figures" / "nature_source_data"
ROBUST_QUEUE_SOURCE = SOURCE_DIR / "public_robustness_attribution_queue_source.csv"
CALROT_QUEUE_SOURCE = SOURCE_DIR / "public_calibration_rotation_queue_source.csv"


@dataclass(frozen=True)
class Job:
    name: str
    stage: str
    task: str
    dataset: str
    role: str
    seed: str
    fold: str
    command: list[str]


def pid_alive(pid: int) -> bool:
    return subprocess.run(["ps", "-p", str(pid)], check=False, stdout=subprocess.DEVNULL).returncode == 0


def read_pid(run_id: str) -> int | None:
    path = CV_DIR / f"{run_id}.pid"
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def wait_for_cv(run_id: str, poll_sec: int) -> None:
    while True:
        pid = read_pid(run_id)
        if pid is None or not pid_alive(pid):
            return
        print(f"[wait] public CV run {run_id} still alive: pid={pid}; sleeping {poll_sec}s", flush=True)
        time.sleep(poll_sec)


def run_command(name: str, command: list[str], rows: list[dict[str, object]]) -> subprocess.CompletedProcess[str]:
    started = time.time()
    proc = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    rows.append(
        {
            "name": name,
            "command": " ".join(command),
            "returncode": proc.returncode,
            "elapsed_sec": round(time.time() - started, 3),
            "output_tail": "\n".join(proc.stdout.strip().splitlines()[-80:]),
        }
    )
    print(f"{name}: rc={proc.returncode} elapsed={rows[-1]['elapsed_sec']}s", flush=True)
    return proc


def refresh(rows: list[dict[str, object]]) -> dict[str, object]:
    proc = run_command(
        "refresh_three_corpus_public_status",
        [sys.executable, "scripts/refresh_three_corpus_public_status.py"],
        rows,
    )
    if proc.returncode != 0:
        raise RuntimeError("refresh_three_corpus_public_status failed")
    status_path = CV_DIR / "status.json"
    if not status_path.exists():
        raise FileNotFoundError(status_path)
    return json.loads(status_path.read_text(encoding="utf-8"))


def refresh_probe_queues(run_id: str, rows: list[dict[str, object]]) -> dict[str, object]:
    """Refresh only probe queues for lightweight remote workers.

    Full manuscript refresh requires all local paper assets and AL6 evidence.
    Remote worker nodes only need the public-CV checkpoint queues, so keep this
    path independent from figure generation and completion audit.
    """
    robust = run_command(
        "public_robustness_queue",
        [
            sys.executable,
            "scripts/run_public_robustness_attribution.py",
            "--run-id",
            run_id,
            "--dry-run",
            "--limit",
            "0",
        ],
        rows,
    )
    calrot = run_command(
        "public_calibration_rotation_queue",
        [
            sys.executable,
            "scripts/run_public_calibration_rotation.py",
            "--run-id",
            run_id,
            "--dry-run",
            "--limit",
            "0",
        ],
        rows,
    )
    if robust.returncode != 0:
        raise RuntimeError("public robustness queue refresh failed")
    if calrot.returncode != 0:
        raise RuntimeError("public calibration/rotation queue refresh failed")
    robust_rows = read_csv_rows(ROBUST_QUEUE_SOURCE)
    calrot_rows = read_csv_rows(CALROT_QUEUE_SOURCE)
    waiting = sum(row.get("status") == "waiting_for_cv" for row in robust_rows)
    cal_waiting = sum(row.get("status") == "waiting_for_cv" for row in calrot_rows)
    return {
        "all_complete": waiting == 0 and cal_waiting == 0 and bool(robust_rows) and bool(calrot_rows),
        "robust_rows": len(robust_rows),
        "calrot_rows": len(calrot_rows),
        "robust_waiting_for_cv": waiting,
        "calrot_waiting_for_cv": cal_waiting,
    }


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_jobs(args: argparse.Namespace) -> list[Job]:
    jobs: list[Job] = []
    if "robustness" in args.stages:
        for row in read_csv_rows(ROBUST_QUEUE_SOURCE):
            if row.get("status") != "checkpoint_ready":
                continue
            name = f"robust_{row['task']}_s{row['seed']}_f{row['fold']}"
            jobs.append(
                Job(
                    name=name,
                    stage="robustness",
                    task=row["task"],
                    dataset=row["dataset"],
                    role=row["role"],
                    seed=row["seed"],
                    fold=row["fold"],
                    command=[
                        sys.executable,
                        "scripts/run_public_robustness_attribution.py",
                        "--run-id",
                        args.run_id,
                        "--task",
                        row["task"],
                        "--seed",
                        row["seed"],
                        "--fold",
                        row["fold"],
                        "--batch-size",
                        str(args.robust_batch_size),
                    ],
                )
            )
    if "calibration_rotation" in args.stages:
        for row in read_csv_rows(CALROT_QUEUE_SOURCE):
            if row.get("status") == "waiting_for_cv":
                continue
            probes = [probe for probe in ["calibration", "rotation"] if row.get(probe) != "complete"]
            if not probes:
                continue
            name = f"calrot_{row['task']}_s{row['seed']}_f{row['fold']}"
            jobs.append(
                Job(
                    name=name,
                    stage="calibration_rotation",
                    task=row["task"],
                    dataset=row["dataset"],
                    role=row["role"],
                    seed=row["seed"],
                    fold=row["fold"],
                    command=[
                        sys.executable,
                        "scripts/run_public_calibration_rotation.py",
                        "--run-id",
                        args.run_id,
                        "--task",
                        row["task"],
                        "--seed",
                        row["seed"],
                        "--fold",
                        row["fold"],
                        "--batch-size",
                        str(args.calrot_batch_size),
                        "--probes",
                        *probes,
                    ],
                )
            )
    if args.job_shard_count > 1:
        jobs = [
            job
            for idx, job in enumerate(jobs)
            if idx % args.job_shard_count == args.job_shard_index
        ]
    if args.max_jobs is not None:
        jobs = jobs[: args.max_jobs]
    return jobs


def job_log_path(label: str, worker_id: int, job: Job) -> Path:
    safe = job.name.replace("/", "_").replace(":", "_")
    return LOG_DIR / f"{label}_worker{worker_id}_{safe}.log"


def run_pool(jobs: list[Job], devices: list[str], label: str, dry_run: bool) -> list[dict[str, object]]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    job_queue: queue.Queue[Job] = queue.Queue()
    for job in jobs:
        job_queue.put(job)

    results: list[dict[str, object]] = []
    lock = threading.Lock()

    def worker(worker_id: int, device: str) -> None:
        while True:
            try:
                job = job_queue.get_nowait()
            except queue.Empty:
                return
            command = [*job.command, "--device", device]
            log_path = job_log_path(label, worker_id, job)
            started = time.time()
            print(f"[worker {worker_id}] {job.name} on {device}", flush=True)
            if dry_run:
                output = "[dry-run] " + " ".join(command)
                returncode = 0
            else:
                proc = subprocess.run(
                    command,
                    cwd=ROOT,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
                output = proc.stdout
                returncode = proc.returncode
            log_path.write_text(output, encoding="utf-8")
            with lock:
                results.append(
                    {
                        "name": job.name,
                        "stage": job.stage,
                        "dataset": job.dataset,
                        "role": job.role,
                        "seed": job.seed,
                        "fold": job.fold,
                        "device": device,
                        "command": " ".join(command),
                        "returncode": returncode,
                        "elapsed_sec": round(time.time() - started, 3),
                        "log_path": str(log_path.relative_to(ROOT)),
                        "output_tail": "\n".join(output.strip().splitlines()[-80:]),
                    }
                )
            print(f"[worker {worker_id}] {job.name}: rc={returncode}", flush=True)
            job_queue.task_done()

    threads = [
        threading.Thread(target=worker, args=(idx, device), daemon=False)
        for idx, device in enumerate(devices)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    results.sort(key=lambda row: (str(row["stage"]), str(row["dataset"]), str(row["role"]), int(row["seed"]), int(row["fold"])))
    return results


def write_report(
    label: str,
    status_payload: dict[str, object],
    setup_rows: list[dict[str, object]],
    job_rows: list[dict[str, object]],
    devices: list[str],
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    failed_jobs = [row for row in job_rows if row["returncode"] != 0]
    payload = {
        "label": label,
        "generated_at_epoch": time.time(),
        "status_all_complete": bool(status_payload.get("all_complete")),
        "devices": devices,
        "setup_steps": setup_rows,
        "jobs": job_rows,
        "job_count": len(job_rows),
        "failed_job_count": len(failed_jobs),
    }
    json_path = OUT_DIR / f"{label}.json"
    md_path = OUT_DIR / f"{label}.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Public Follow-Up Probe Pool",
        "",
        f"- Label: `{label}`",
        f"- CV all complete: {bool(status_payload.get('all_complete'))}",
        f"- Devices: `{', '.join(devices)}`",
        f"- Jobs: {len(job_rows)}",
        f"- Failed jobs: {len(failed_jobs)}",
        "",
        "## Setup Steps",
        "",
        "| Step | Return code | Seconds |",
        "|---|---:|---:|",
    ]
    for row in setup_rows:
        lines.append(f"| {row['name']} | {row['returncode']} | {row['elapsed_sec']} |")

    lines.extend(["", "## Jobs", "", "| Stage | Cell | Device | Return code | Seconds | Log |", "|---|---|---|---:|---:|---|"])
    for row in job_rows:
        cell = f"{row['dataset']} {row['role']} s{row['seed']}-f{row['fold']}"
        lines.append(
            f"| {row['stage']} | {cell} | {row['device']} | {row['returncode']} | "
            f"{row['elapsed_sec']} | `{row['log_path']}` |"
        )
    if failed_jobs:
        lines.extend(["", "## Failed Job Tails", ""])
        for row in failed_jobs[:12]:
            lines.extend([f"### {row['name']}", "", "```text", str(row["output_tail"]), "```", ""])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote: {md_path.relative_to(ROOT)}", flush=True)


def normalize_devices(args: argparse.Namespace) -> list[str]:
    if args.devices:
        return args.devices
    return [args.device]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="public_cv_parity_v1")
    parser.add_argument("--wait", action="store_true", help="Wait until the recorded public-CV PID exits.")
    parser.add_argument("--poll-sec", type=int, default=300)
    parser.add_argument("--run-probes", action="store_true", help="Launch queued probe jobs.")
    parser.add_argument("--allow-partial", action="store_true", help="Run probes for ready cells even if CV is incomplete.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--devices", nargs="+", default=None, help="Device list for parallel workers, e.g. cuda:0 cuda:1.")
    parser.add_argument("--stages", nargs="+", default=["robustness", "calibration_rotation"], choices=["robustness", "calibration_rotation"])
    parser.add_argument("--robust-batch-size", type=int, default=64)
    parser.add_argument("--calrot-batch-size", type=int, default=128)
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument("--job-shard-count", type=int, default=1)
    parser.add_argument("--job-shard-index", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--label", default=None)
    parser.add_argument(
        "--skip-full-refresh",
        action="store_true",
        help="Refresh only public probe queues; useful for lightweight remote workers without full paper assets.",
    )
    args = parser.parse_args()
    if args.job_shard_count < 1:
        raise SystemExit("--job-shard-count must be >= 1")
    if args.job_shard_index < 0 or args.job_shard_index >= args.job_shard_count:
        raise SystemExit("--job-shard-index must satisfy 0 <= index < count")

    label = args.label or f"post_cv_probe_pool_{time.strftime('%Y%m%d_%H%M%S')}"
    setup_rows: list[dict[str, object]] = []
    job_rows: list[dict[str, object]] = []
    status_payload: dict[str, object] = {}
    devices = normalize_devices(args)
    try:
        if args.wait:
            wait_for_cv(args.run_id, args.poll_sec)
        status_payload = (
            refresh_probe_queues(args.run_id, setup_rows)
            if args.skip_full_refresh
            else refresh(setup_rows)
        )
        all_complete = bool(status_payload.get("all_complete"))
        if not all_complete and not args.allow_partial:
            print("[stop] public CV is not complete; rerun with --allow-partial to probe ready cells.", flush=True)
            return
        if not args.run_probes:
            print("[stop] --run-probes not set; refreshed status only.", flush=True)
            return
        jobs = build_jobs(args)
        print(f"[jobs] queued {len(jobs)} jobs across {len(devices)} device worker(s)", flush=True)
        job_rows = run_pool(jobs, devices, label, args.dry_run)
        run_command("public_followup_summary", [sys.executable, "scripts/summarize_public_followup_probes.py"], setup_rows)
        status_payload = (
            refresh_probe_queues(args.run_id, setup_rows)
            if args.skip_full_refresh
            else refresh(setup_rows)
        )
        failures = [row for row in job_rows if row["returncode"] != 0]
        if failures:
            raise RuntimeError(f"{len(failures)} probe job(s) failed")
    finally:
        write_report(label, status_payload, setup_rows, job_rows, devices)


if __name__ == "__main__":
    main()
