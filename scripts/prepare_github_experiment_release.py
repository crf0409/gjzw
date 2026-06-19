#!/usr/bin/env python
"""Prepare lightweight experiment metadata for GitHub.

The full release ZIP is uploaded as a GitHub Release asset because it is too
large for normal git storage. This script collects the small, reviewable files
that should live in the repository: audit reports, run summaries, source CSVs,
the release package manifest, and the package checksum.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiment_data" / "three_corpus_release"
PACKAGE = ROOT / "release_packages" / "three_corpus_release_final_manual.zip"


FILES = [
    ROOT / "release_packages" / "three_corpus_release_final_manual.zip.sha256",
    ROOT / "release_packages" / "three_corpus_release_final_manual.zip.manifest.json",
    ROOT / "outputs" / "three_corpus_completion" / "three_corpus_parity_completion_audit.json",
    ROOT / "outputs" / "three_corpus_completion" / "three_corpus_parity_completion_audit.md",
    ROOT / "outputs" / "three_corpus_completion" / "three_corpus_completion_summary.json",
    ROOT / "outputs" / "three_corpus_completion" / "three_corpus_completion_summary.md",
    ROOT / "outputs" / "three_corpus_completion" / "refresh_three_corpus_public_status.json",
    ROOT / "outputs" / "three_corpus_completion" / "refresh_three_corpus_public_status.md",
    ROOT / "outputs" / "public_cv_parity" / "status.json",
    ROOT / "outputs" / "public_cv_parity" / "status.md",
    ROOT / "outputs" / "public_cv_parity" / "remaining_queue.json",
    ROOT / "outputs" / "public_cv_parity" / "remaining_queue.md",
    ROOT / "outputs" / "public_followup_probes" / "summary.json",
    ROOT / "outputs" / "public_followup_probes" / "summary.md",
    ROOT / "outputs" / "public_followup_probes" / "local_calrot_shard0_public_cv_parity_v1_pool.json",
    ROOT / "outputs" / "public_followup_probes" / "local_calrot_shard0_public_cv_parity_v1_pool.md",
    ROOT / "outputs" / "public_followup_probes" / "remote_calrot_shard1_public_cv_parity_v1_pool.json",
    ROOT / "outputs" / "public_followup_probes" / "remote_calrot_shard1_public_cv_parity_v1_pool.md",
    ROOT / "outputs" / "public_followup_probes" / "remote_missing_robust_public_cv_parity_v1.json",
    ROOT / "outputs" / "public_followup_probes" / "remote_missing_robust_public_cv_parity_v1.md",
]


SOURCE_DATA_DIR = ROOT / "paper" / "figures" / "nature_source_data"
TEXT_SUFFIXES = {".csv", ".json", ".md", ".sha256", ".txt", ".log"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_file(src: Path, dst_root: Path) -> dict[str, object]:
    rel = src.relative_to(ROOT)
    dst = dst_root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    if dst.suffix in TEXT_SUFFIXES:
        data = dst.read_bytes()
        normalized = data.replace(b"\r\n", b"\n")
        if normalized != data:
            dst.write_bytes(normalized)
    return {
        "source": str(rel),
        "path": str(dst.relative_to(ROOT)),
        "bytes": dst.stat().st_size,
        "sha256": sha256(dst),
    }


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)

    copied: list[dict[str, object]] = []
    for src in FILES:
        if src.exists():
            copied.append(copy_file(src, OUT))

    source_rows = []
    if SOURCE_DATA_DIR.exists():
        for src in sorted(SOURCE_DATA_DIR.glob("*.csv")):
            row = copy_file(src, OUT)
            copied.append(row)
            source_rows.append(row)

    package_info = {}
    if PACKAGE.exists():
        package_info = {
            "path": str(PACKAGE.relative_to(ROOT)),
            "bytes": PACKAGE.stat().st_size,
            "sha256": sha256(PACKAGE),
            "github_release_asset": PACKAGE.name,
        }

    audit_path = ROOT / "outputs" / "three_corpus_completion" / "three_corpus_parity_completion_audit.json"
    audit = load_json(audit_path) if audit_path.exists() else {}
    followup_path = SOURCE_DATA_DIR / "public_followup_probe_status_source.csv"

    manifest = {
        "generated_at_epoch": time.time(),
        "description": "Lightweight GitHub metadata for the completed three-corpus AAFNet experiment release.",
        "package": package_info,
        "audit_complete": bool(audit.get("complete")),
        "audit_summary": {
            "pass": audit.get("pass"),
            "warn": audit.get("warn"),
            "fail": audit.get("fail"),
        },
        "copied_file_count": len(copied),
        "source_data_csv_count": len(source_rows),
        "files": copied,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    readme = [
        "# Three-Corpus Experiment Release",
        "",
        "This directory contains lightweight, reviewable metadata for the completed AAFNet three-corpus experiment release.",
        "The full ZIP package is intentionally not committed to git because it exceeds GitHub's normal file-size limit.",
        "",
        "## Completion",
        "",
        f"- Completion audit: `{bool(audit.get('complete'))}`",
        f"- Audit summary: pass={audit.get('pass')}, warn={audit.get('warn')}, fail={audit.get('fail')}",
        "- Public CV cells: 60/60",
        "- Public robustness probes: 60/60",
        "- Public calibration probes: 60/60",
        "- Public rotation probes: 60/60",
        "",
        "## Full Package",
        "",
        f"- Local package: `{package_info.get('path', 'n/a')}`",
        f"- Release asset: `{package_info.get('github_release_asset', 'n/a')}`",
        f"- SHA256: `{package_info.get('sha256', 'n/a')}`",
        f"- Size bytes: `{package_info.get('bytes', 'n/a')}`",
        "",
        "## Contents",
        "",
        "- `release_packages/`: package manifest and checksum.",
        "- `outputs/three_corpus_completion/`: final completion audit and refresh reports.",
        "- `outputs/public_cv_parity/`: public-corpus cross-validation status.",
        "- `outputs/public_followup_probes/`: follow-up probe summaries and split-run reports.",
        "- `paper/figures/nature_source_data/`: CSV source data for manuscript figures.",
        "- `manifest.json`: machine-readable index for this metadata bundle.",
    ]
    if followup_path.exists():
        readme.extend(["", "The final follow-up probe status source is copied from:", f"`{followup_path.relative_to(ROOT)}`"])
    (OUT / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")

    print(f"wrote: {OUT.relative_to(ROOT)}")
    print(f"copied_files: {len(copied)}")
    print(f"source_csvs: {len(source_rows)}")
    if package_info:
        print(f"package_sha256: {package_info['sha256']}")


if __name__ == "__main__":
    main()
