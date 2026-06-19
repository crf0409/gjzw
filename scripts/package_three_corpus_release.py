#!/usr/bin/env python
"""Build a guarded release ZIP for the three-corpus manuscript evidence.

By default this script refuses to create a final package unless the completion
audit proves that the three-corpus evidence layer is complete. Use
``--allow-incomplete`` only for an explicitly labeled intermediate handoff.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RELEASE_DIR = ROOT / "release_packages"
AUDIT_JSON = ROOT / "outputs" / "three_corpus_completion" / "three_corpus_parity_completion_audit.json"

DEFAULT_INCLUDE_DIRS = [
    "paper",
    "scripts",
    "src",
    "config",
    "outputs/three_corpus_completion",
    "outputs/public_cv_parity",
    "outputs/public_robustness_attribution",
    "outputs/public_calibration_rotation",
    "outputs/public_followup_probes",
    "outputs/three_corpus_parity",
    "outputs/data_audit",
    "outputs/cv_asp_baseline",
    "outputs/cv_asp_aafnet",
    "outputs/cv_as25_baseline",
    "outputs/cv_as25_aafnet",
]

DEFAULT_INCLUDE_FILES = [
    "README.md",
    "requirements.txt",
    "LICENSE",
]

ALWAYS_EXCLUDE_PARTS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "dataset",
    "data",
    "release_packages",
}

HEAVY_EXTENSIONS = {
    ".pth",
    ".pt",
    ".ckpt",
    ".onnx",
    ".zip",
    ".tar",
    ".gz",
    ".bz2",
    ".xz",
    ".7z",
}


@dataclass
class PackagePlan:
    files: list[Path]
    skipped: list[tuple[Path, str]]


def load_audit() -> dict:
    if not AUDIT_JSON.exists():
        return {"complete": False, "error": f"missing {AUDIT_JSON}"}
    try:
        return json.loads(AUDIT_JSON.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"complete": False, "error": f"invalid audit json: {exc}"}


def run_completion_audit() -> dict:
    subprocess.run(
        [sys.executable, "scripts/audit_three_corpus_parity_completion.py"],
        cwd=ROOT,
        check=True,
    )
    return load_audit()


def should_skip(path: Path, include_checkpoints: bool) -> str | None:
    rel_parts = path.relative_to(ROOT).parts
    if any(part in ALWAYS_EXCLUDE_PARTS for part in rel_parts):
        return "excluded directory"
    if not include_checkpoints and path.suffix.lower() in HEAVY_EXTENSIONS:
        return "heavy artifact excluded; use --include-checkpoints to include"
    return None


def collect_files(include_checkpoints: bool) -> PackagePlan:
    files: list[Path] = []
    skipped: list[tuple[Path, str]] = []

    for item in DEFAULT_INCLUDE_FILES:
        path = ROOT / item
        if path.exists() and path.is_file():
            reason = should_skip(path, include_checkpoints)
            if reason:
                skipped.append((path, reason))
            else:
                files.append(path)

    for item in DEFAULT_INCLUDE_DIRS:
        base = ROOT / item
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            reason = should_skip(path, include_checkpoints)
            if reason:
                skipped.append((path, reason))
            else:
                files.append(path)

    unique = sorted(set(files), key=lambda p: str(p.relative_to(ROOT)))
    return PackagePlan(files=unique, skipped=skipped)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(
    package_path: Path,
    plan: PackagePlan,
    audit: dict,
    include_checkpoints: bool,
    allow_incomplete: bool,
) -> Path:
    manifest_path = package_path.with_suffix(package_path.suffix + ".manifest.json")
    payload = {
        "generated_at_epoch": time.time(),
        "package": str(package_path.relative_to(ROOT)),
        "audit_complete": bool(audit.get("complete")),
        "allow_incomplete": allow_incomplete,
        "include_checkpoints": include_checkpoints,
        "file_count": len(plan.files),
        "skipped_count": len(plan.skipped),
        "files": [str(path.relative_to(ROOT)) for path in plan.files],
        "skipped": [
            {"path": str(path.relative_to(ROOT)), "reason": reason}
            for path, reason in plan.skipped[:5000]
        ],
        "audit_summary": {
            "complete": audit.get("complete"),
            "pass": audit.get("pass"),
            "warn": audit.get("warn"),
            "fail": audit.get("fail"),
        },
    }
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def build_zip(package_path: Path, plan: PackagePlan, manifest_path: Path) -> None:
    with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in plan.files:
            zf.write(path, path.relative_to(ROOT))
        zf.write(manifest_path, manifest_path.relative_to(ROOT))


def write_sidecars(package_path: Path) -> None:
    digest = sha256(package_path)
    sidecar = package_path.with_suffix(package_path.suffix + ".sha256")
    sidecar.write_text(f"{digest}  {package_path.name}\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default=None, help="Release label. Defaults to timestamp.")
    parser.add_argument("--allow-incomplete", action="store_true", help="Create an explicitly incomplete package.")
    parser.add_argument("--include-checkpoints", action="store_true", help="Include model weights and other heavy artifacts.")
    parser.add_argument("--dry-run", action="store_true", help="Only write a manifest; do not create the ZIP.")
    args = parser.parse_args()

    RELEASE_DIR.mkdir(parents=True, exist_ok=True)
    audit = run_completion_audit()
    if not bool(audit.get("complete")) and not args.allow_incomplete:
        print("[stop] completion audit is not complete; use --allow-incomplete only for an intermediate package")
        print(f"audit: {AUDIT_JSON.relative_to(ROOT)}")
        raise SystemExit(2)

    label = args.label or time.strftime("three_corpus_release_%Y%m%d_%H%M%S")
    if not bool(audit.get("complete")) and "incomplete" not in label:
        label = f"{label}_incomplete"
    package_path = RELEASE_DIR / f"{label}.zip"
    plan = collect_files(args.include_checkpoints)
    manifest_path = write_manifest(package_path, plan, audit, args.include_checkpoints, args.allow_incomplete)

    if args.dry_run:
        print(f"[dry-run] files={len(plan.files)} skipped={len(plan.skipped)}")
        print(f"manifest: {manifest_path.relative_to(ROOT)}")
        return

    build_zip(package_path, plan, manifest_path)
    write_sidecars(package_path)
    print(f"package: {package_path.relative_to(ROOT)}")
    print(f"sha256: {package_path.with_suffix(package_path.suffix + '.sha256').relative_to(ROOT)}")
    print(f"manifest: {manifest_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
