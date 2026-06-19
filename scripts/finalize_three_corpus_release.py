#!/usr/bin/env python
"""Wait for three-corpus evidence completion, then package and upload it.

This finalizer is intentionally conservative. It waits for the public-CV and
post-CV probe processes when requested, refreshes the machine-readable audit,
and only builds/uploads the final ZIP when the completion audit is green.
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
PUBLIC_CV_DIR = ROOT / "outputs" / "public_cv_parity"
FOLLOWUP_DIR = ROOT / "outputs" / "public_followup_probes"
AUDIT_JSON = OUT_DIR / "three_corpus_parity_completion_audit.json"
DEFAULT_CONFIG_DOC = Path("/home/siton02/disk_sdg/md0_backup_2026-04-28/crf/tos.md")


def pid_alive(pid: int) -> bool:
    return subprocess.run(["ps", "-p", str(pid)], check=False, stdout=subprocess.DEVNULL).returncode == 0


def read_pid(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


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
            "output_tail": "\n".join(proc.stdout.strip().splitlines()[-120:]),
        }
    )
    print(f"{name}: rc={proc.returncode} elapsed={rows[-1]['elapsed_sec']}s", flush=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{name} failed with return code {proc.returncode}")
    return proc


def wait_for_pid_file(label: str, path: Path, poll_sec: int, rows: list[dict[str, object]]) -> None:
    pid = read_pid(path)
    if pid is None:
        rows.append(
            {
                "name": f"wait_{label}",
                "command": f"read {path.relative_to(ROOT)}",
                "returncode": 0,
                "elapsed_sec": 0,
                "output_tail": f"pid file missing or invalid; continuing: {path}",
            }
        )
        return
    started = time.time()
    while pid_alive(pid):
        print(f"[wait] {label} pid={pid} still alive; sleeping {poll_sec}s", flush=True)
        time.sleep(poll_sec)
    rows.append(
        {
            "name": f"wait_{label}",
            "command": f"wait for pid {pid}",
            "returncode": 0,
            "elapsed_sec": round(time.time() - started, 3),
            "output_tail": f"{label} pid={pid} has exited",
        }
    )


def load_audit() -> dict[str, object]:
    if not AUDIT_JSON.exists():
        return {"complete": False, "error": f"missing {AUDIT_JSON}"}
    return json.loads(AUDIT_JSON.read_text(encoding="utf-8"))


def latest_package_from_output(output: str) -> Path | None:
    for line in output.splitlines():
        if line.startswith("package: "):
            return ROOT / line.split("package: ", 1)[1].strip()
    return None


def presigned_url_from_output(output: str) -> str:
    for line in output.splitlines():
        if line.startswith("url: "):
            return line.split("url: ", 1)[1].strip()
    return ""


def write_report(label: str, rows: list[dict[str, object]], audit: dict[str, object], package: Path | None, url: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "label": label,
        "generated_at_epoch": time.time(),
        "audit_complete": bool(audit.get("complete")),
        "audit_summary": {
            "pass": audit.get("pass"),
            "warn": audit.get("warn"),
            "fail": audit.get("fail"),
        },
        "package": "" if package is None else str(package.relative_to(ROOT)),
        "url": url,
        "steps": rows,
    }
    json_path = OUT_DIR / f"{label}.json"
    md_path = OUT_DIR / f"{label}.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Three-Corpus Release Finalizer",
        "",
        f"- Label: `{label}`",
        f"- Audit complete: {bool(audit.get('complete'))}",
        f"- Package: `{payload['package'] or 'n/a'}`",
        f"- URL: {url or 'n/a'}",
        "",
        "| Step | Return code | Seconds |",
        "|---|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['name']} | {row['returncode']} | {row['elapsed_sec']} |")
    lines.extend(["", "## Output Tails", ""])
    for row in rows:
        lines.extend([f"### {row['name']}", "", "```text", str(row["output_tail"]), "```", ""])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote: {md_path.relative_to(ROOT)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default=None)
    parser.add_argument("--run-id", default="public_cv_parity_v1")
    parser.add_argument("--followup-label", default="post_cv_followups_public_cv_parity_v1_pool")
    parser.add_argument("--wait", action="store_true", help="Wait for public CV and follow-up probe processes.")
    parser.add_argument("--poll-sec", type=int, default=300)
    parser.add_argument("--config-doc", default=str(DEFAULT_CONFIG_DOC))
    parser.add_argument("--expires-days", type=float, default=3.0)
    parser.add_argument("--include-checkpoints", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Refresh and package dry-run only; do not upload.")
    args = parser.parse_args()

    label = args.label or f"three_corpus_release_finalizer_{time.strftime('%Y%m%d_%H%M%S')}"
    rows: list[dict[str, object]] = []
    package: Path | None = None
    url = ""
    audit: dict[str, object] = {}
    try:
        if args.wait:
            wait_for_pid_file("public_cv", PUBLIC_CV_DIR / f"{args.run_id}.pid", args.poll_sec, rows)
            wait_for_pid_file(
                "public_followup_probes",
                FOLLOWUP_DIR / f"{args.followup_label}.pid",
                args.poll_sec,
                rows,
            )

        run_command("refresh_three_corpus_public_status", [sys.executable, "scripts/refresh_three_corpus_public_status.py"], rows)
        audit = load_audit()
        if not bool(audit.get("complete")):
            print("[stop] completion audit is not complete; final package/upload skipped", flush=True)
            return

        package_cmd = [
            sys.executable,
            "scripts/package_three_corpus_release.py",
            "--label",
            label,
        ]
        if args.include_checkpoints:
            package_cmd.append("--include-checkpoints")
        if args.dry_run:
            package_cmd.append("--dry-run")
        package_proc = run_command("package_three_corpus_release", package_cmd, rows)
        package = latest_package_from_output(package_proc.stdout)
        if args.dry_run:
            return
        if package is None or not package.exists():
            raise RuntimeError("package path was not found in packaging output")

        upload_proc = run_command(
            "upload_release_to_tos",
            [
                sys.executable,
                "scripts/upload_release_to_tos.py",
                str(package),
                "--config-doc",
                args.config_doc,
                "--expires-days",
                str(args.expires_days),
            ],
            rows,
        )
        url = presigned_url_from_output(upload_proc.stdout)
    finally:
        write_report(label, rows, audit, package, url)


if __name__ == "__main__":
    main()
