#!/usr/bin/env python
"""Audit whether the three-corpus evidence layer is truly complete.

This is a gate, not a training script. It checks the current worktree artifacts
against the manuscript-level requirement that ASP_clean and AS25_clean have the
same figure, source-data and experiment depth as the primary AL6 presentation.
At the current stage the expected result is usually incomplete; the report is
meant to make the remaining gap explicit and machine-readable.
"""

from __future__ import annotations

import csv
import json
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "three_corpus_completion"
REPORT_JSON = OUT_DIR / "three_corpus_parity_completion_audit.json"
REPORT_MD = OUT_DIR / "three_corpus_parity_completion_audit.md"

CV_STATUS = ROOT / "outputs" / "public_cv_parity" / "status.json"
FOLLOWUP_SUMMARY = ROOT / "outputs" / "public_followup_probes" / "summary.json"
RUNBOOK = OUT_DIR / "missing_experiment_runbook.md"
ROTATION_CSV = ROOT / "paper" / "figures" / "nature_source_data" / "figure_rotation_audit.csv"
FIG_DIR = ROOT / "paper" / "figures"
SOURCE_DIR = FIG_DIR / "nature_source_data"
PERFORMANCE_SOURCE = SOURCE_DIR / "F_TC2_three_corpus_performance_source.csv"
COVERAGE_SOURCE = SOURCE_DIR / "F_TC8_public_experiment_coverage_source.csv"
CV_QUEUE_SOURCE = SOURCE_DIR / "public_cv_remaining_queue_source.csv"
ROBUST_QUEUE_SOURCE = SOURCE_DIR / "public_robustness_attribution_queue_source.csv"
CALROT_QUEUE_SOURCE = SOURCE_DIR / "public_calibration_rotation_queue_source.csv"
FOLLOWUP_STATUS_SOURCE = SOURCE_DIR / "public_followup_probe_status_source.csv"
LIVE_STATUS_SOURCE = SOURCE_DIR / "three_corpus_live_status_source.csv"
FINALIZER_LABEL = "three_corpus_release_final_auto"
FINALIZER_LAUNCHER_JSON = OUT_DIR / f"{FINALIZER_LABEL}_launcher.json"
FINALIZER_REPORT_JSON = OUT_DIR / f"{FINALIZER_LABEL}.json"

TASKS = [
    ("ASP_clean", "baseline"),
    ("ASP_clean", "aafnet"),
    ("AS25_clean", "baseline"),
    ("AS25_clean", "aafnet"),
]

FIGURE_STEMS = [
    "F_TC0_three_corpus_samples",
    "F_TC1_three_corpus_audit",
    "F_TC2_three_corpus_performance",
    "F_TC3_three_corpus_confusions",
    "F_TC4_three_corpus_training",
    "F_TC5_three_corpus_probe_matrix",
    "F_TC6_three_corpus_parity",
    "F_TC7_public_interpretability",
    "F_TC8_public_experiment_coverage",
]

SOURCE_FILES = [
    "F_TC0_three_corpus_samples_source.csv",
    "F_TC1_three_corpus_audit_source.csv",
    "F_TC2_three_corpus_performance_source.csv",
    "F_TC3_three_corpus_confusions_source.csv",
    "F_TC4_three_corpus_training_source.csv",
    "F_TC5_three_corpus_probe_matrix_source.csv",
    "F_TC6_three_corpus_parity_calibration_source.csv",
    "F_TC6_three_corpus_parity_delta_source.csv",
    "F_TC6_three_corpus_parity_robustness_source.csv",
    "F_TC6_three_corpus_parity_rotation_source.csv",
    "F_TC7_public_interpretability_predictions_source.csv",
    "F_TC7_public_interpretability_selected_source.csv",
    "F_TC8_public_experiment_coverage_source.csv",
    "F_TC_training_support_audit.csv",
    "figure_rotation_audit.csv",
    "public_cv_parity_status_source.csv",
    "public_cv_remaining_queue_source.csv",
    "public_robustness_attribution_queue_source.csv",
    "public_calibration_rotation_queue_source.csv",
    "public_followup_probe_status_source.csv",
    "public_robustness_fold_metrics_source.csv",
    "public_calibration_fold_metrics_source.csv",
    "public_rotation_fold_metrics_source.csv",
    "public_robustness_summary_source.csv",
    "public_calibration_summary_source.csv",
    "public_rotation_summary_source.csv",
    "three_corpus_live_status_source.csv",
]

SOURCE_COVERAGE_EXPECTATIONS = [
    ("F_TC0_three_corpus_samples_source.csv", "dataset", {"AL6", "ASP_clean", "AS25_clean"}),
    ("F_TC1_three_corpus_audit_source.csv", "raw_dataset", {"AL6", "ASP", "AS25"}),
    ("F_TC2_three_corpus_performance_source.csv", "dataset", {"AL6", "ASP_clean", "AS25_clean"}),
    ("F_TC3_three_corpus_confusions_source.csv", "dataset", {"AL6", "ASP_clean", "AS25_clean"}),
    ("F_TC4_three_corpus_training_source.csv", "dataset", {"AL6", "ASP_clean", "AS25_clean"}),
    ("F_TC5_three_corpus_probe_matrix_source.csv", "dataset", {"AL6", "ASP_clean", "AS25_clean"}),
    ("F_TC6_three_corpus_parity_calibration_source.csv", "dataset", {"AL6", "ASP_clean", "AS25_clean"}),
    ("F_TC6_three_corpus_parity_delta_source.csv", "dataset", {"AL6", "ASP_clean", "AS25_clean"}),
    ("F_TC6_three_corpus_parity_robustness_source.csv", "dataset", {"AL6", "ASP_clean", "AS25_clean"}),
    ("F_TC6_three_corpus_parity_rotation_source.csv", "dataset", {"AL6", "ASP_clean", "AS25_clean"}),
    ("F_TC7_public_interpretability_predictions_source.csv", "dataset", {"ASP_clean", "AS25_clean"}),
    ("F_TC7_public_interpretability_selected_source.csv", "dataset", {"ASP_clean", "AS25_clean"}),
    ("F_TC8_public_experiment_coverage_source.csv", "dataset", {"ASP_clean", "AS25_clean"}),
    ("three_corpus_live_status_source.csv", "dataset", {"ASP_clean", "AS25_clean"}),
]

MODEL_COVERAGE_EXPECTATIONS = [
    ("F_TC2_three_corpus_performance_source.csv", "model", {"baseline", "aafnet"}),
    ("F_TC3_three_corpus_confusions_source.csv", "model", {"baseline", "aafnet"}),
    ("F_TC4_three_corpus_training_source.csv", "model", {"baseline", "aafnet"}),
    ("F_TC6_three_corpus_parity_calibration_source.csv", "model", {"baseline", "aafnet"}),
    ("F_TC6_three_corpus_parity_robustness_source.csv", "model", {"baseline", "aafnet"}),
    ("F_TC6_three_corpus_parity_rotation_source.csv", "model", {"baseline", "aafnet"}),
    ("F_TC7_public_interpretability_predictions_source.csv", "model", {"baseline", "aafnet"}),
    ("F_TC8_public_experiment_coverage_source.csv", "role", {"baseline", "aafnet"}),
]

PAIR_COVERAGE_EXPECTATIONS = [
    ("F_TC2_three_corpus_performance_source.csv", "dataset", "model", {"AL6", "ASP_clean", "AS25_clean"}),
    ("F_TC3_three_corpus_confusions_source.csv", "dataset", "model", {"AL6", "ASP_clean", "AS25_clean"}),
    ("F_TC4_three_corpus_training_source.csv", "dataset", "model", {"AL6", "ASP_clean", "AS25_clean"}),
    ("F_TC6_three_corpus_parity_calibration_source.csv", "dataset", "model", {"AL6", "ASP_clean", "AS25_clean"}),
    ("F_TC6_three_corpus_parity_robustness_source.csv", "dataset", "model", {"AL6", "ASP_clean", "AS25_clean"}),
    ("F_TC6_three_corpus_parity_rotation_source.csv", "dataset", "model", {"AL6", "ASP_clean", "AS25_clean"}),
    ("F_TC7_public_interpretability_predictions_source.csv", "dataset", "model", {"ASP_clean", "AS25_clean"}),
    ("F_TC8_public_experiment_coverage_source.csv", "dataset", "role", {"ASP_clean", "AS25_clean"}),
]


@dataclass
class Check:
    category: str
    requirement: str
    status: str
    evidence: str
    detail: str

    def as_dict(self) -> dict[str, str]:
        return {
            "category": self.category,
            "requirement": self.requirement,
            "status": self.status,
            "evidence": self.evidence,
            "detail": self.detail,
        }


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def file_ok(path: Path, min_bytes: int = 64) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size >= min_bytes


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def workspace_path(value: str) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def pid_alive(pid: int) -> bool:
    return subprocess.run(["ps", "-p", str(pid)], check=False, stdout=subprocess.DEVNULL).returncode == 0


def audit_public_cv() -> list[Check]:
    payload = load_json(CV_STATUS, {})
    rows = payload.get("rows", []) if isinstance(payload, dict) else []
    by_task = {(row.get("dataset"), row.get("role")): row for row in rows}
    checks: list[Check] = []
    completed_total = 0
    target_total = 0
    for dataset, role in TASKS:
        row = by_task.get((dataset, role), {})
        completed = int(row.get("n_folds_completed") or 0)
        target = int(row.get("n_folds_total") or 15)
        completed_total += completed
        target_total += target
        ok = row.get("status") == "complete" and completed == target == 15
        checks.append(
            Check(
                "public_cv",
                f"{dataset} {role} has 5-fold x 3-seed CV",
                "pass" if ok else "fail",
                rel(CV_STATUS),
                f"{completed}/{target} cells; status={row.get('status', 'missing')}",
            )
        )
    checks.append(
        Check(
            "public_cv",
            "All public-corpus CV cells are complete",
            "pass" if completed_total == target_total == 60 else "fail",
            rel(CV_STATUS),
            f"{completed_total}/{target_total} cells complete",
        )
    )
    return checks


def audit_followup_probes() -> list[Check]:
    payload = load_json(FOLLOWUP_SUMMARY, {})
    rows = payload.get("status", []) if isinstance(payload, dict) else []
    by_task = {(row.get("dataset"), row.get("role")): row for row in rows}
    checks: list[Check] = []
    totals = {"robustness": 0, "calibration": 0, "rotation": 0}
    for dataset, role in TASKS:
        row = by_task.get((dataset, role), {})
        target = int(row.get("target_cells") or 15)
        for probe, field in [
            ("robustness", "robustness_complete"),
            ("calibration", "calibration_complete"),
            ("rotation", "rotation_complete"),
        ]:
            completed = int(row.get(field) or 0)
            totals[probe] += completed
            checks.append(
                Check(
                    "followup_probes",
                    f"{dataset} {role} fold-level {probe} probes complete",
                    "pass" if completed == target == 15 else "fail",
                    rel(FOLLOWUP_SUMMARY),
                    f"{completed}/{target} outputs",
                )
            )
    total_complete = sum(totals.values())
    checks.append(
        Check(
            "followup_probes",
            "All public-corpus fold-level probes are complete",
            "pass" if total_complete == 180 else "fail",
            rel(FOLLOWUP_SUMMARY),
            f"{total_complete}/180 outputs; robustness={totals['robustness']}/60, calibration={totals['calibration']}/60, rotation={totals['rotation']}/60",
        )
    )
    return checks


def audit_figures() -> list[Check]:
    checks: list[Check] = []
    for stem in FIGURE_STEMS:
        paths = [
            FIG_DIR / f"{stem}.png",
            FIG_DIR / "nature_exports" / f"{stem}.svg",
            FIG_DIR / "nature_exports" / f"{stem}.pdf",
        ]
        missing = [rel(path) for path in paths if not file_ok(path)]
        checks.append(
            Check(
                "figures",
                f"{stem} has PNG, SVG and PDF exports",
                "pass" if not missing else "fail",
                "; ".join(rel(path) for path in paths),
                "all exports present" if not missing else "missing or too small: " + "; ".join(missing),
            )
        )
    return checks


def audit_export_quality() -> list[Check]:
    checks: list[Check] = []
    for stem in FIGURE_STEMS:
        svg_path = FIG_DIR / "nature_exports" / f"{stem}.svg"
        pdf_path = FIG_DIR / "nature_exports" / f"{stem}.pdf"
        png_path = FIG_DIR / f"{stem}.png"
        if not file_ok(svg_path) or not file_ok(pdf_path) or not file_ok(png_path):
            checks.append(
                Check(
                    "export_quality",
                    f"{stem} export files are non-empty",
                    "fail",
                    f"{rel(png_path)}; {rel(svg_path)}; {rel(pdf_path)}",
                    "one or more export files are missing or too small",
                )
            )
            continue
        svg = svg_path.read_text(encoding="utf-8", errors="replace")
        text_tags = svg.count("<text")
        path_tags = svg.count("<path")
        checks.append(
            Check(
                "export_quality",
                f"{stem} SVG keeps editable text",
                "pass" if text_tags > 0 else "fail",
                rel(svg_path),
                f"text_tags={text_tags}; path_tags={path_tags}; bytes={svg_path.stat().st_size}",
            )
        )
    return checks


def audit_source_data() -> list[Check]:
    checks: list[Check] = []
    for name in SOURCE_FILES:
        path = SOURCE_DIR / name
        checks.append(
            Check(
                "source_data",
                f"{name} is present",
                "pass" if file_ok(path, min_bytes=1) else "fail",
                rel(path),
                f"{path.stat().st_size} bytes" if path.exists() else "missing",
            )
        )
    return checks


def audit_source_dataset_coverage() -> list[Check]:
    checks: list[Check] = []
    for name, field, expected in SOURCE_COVERAGE_EXPECTATIONS:
        path = SOURCE_DIR / name
        if not path.exists():
            checks.append(
                Check(
                    "source_coverage",
                    f"{name} covers expected dataset set",
                    "fail",
                    rel(path),
                    "missing",
                )
            )
            continue
        rows = read_csv_rows(path)
        observed = {row.get(field, "") for row in rows if row.get(field, "")}
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        checks.append(
            Check(
                "source_coverage",
                f"{name} covers expected dataset set",
                "pass" if not missing else "fail",
                rel(path),
                f"field={field}; observed={sorted(observed)}; missing={missing}; extra={extra}",
            )
        )
    return checks


def audit_source_model_coverage() -> list[Check]:
    checks: list[Check] = []
    for name, field, expected in MODEL_COVERAGE_EXPECTATIONS:
        path = SOURCE_DIR / name
        if not path.exists():
            checks.append(
                Check(
                    "source_model_coverage",
                    f"{name} covers baseline and AAFNet",
                    "fail",
                    rel(path),
                    "missing",
                )
            )
            continue
        rows = read_csv_rows(path)
        observed = {row.get(field, "") for row in rows if row.get(field, "")}
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        checks.append(
            Check(
                "source_model_coverage",
                f"{name} covers baseline and AAFNet",
                "pass" if not missing else "fail",
                rel(path),
                f"field={field}; observed={sorted(observed)}; missing={missing}; extra={extra}",
            )
        )
    return checks


def audit_source_pair_coverage() -> list[Check]:
    checks: list[Check] = []
    expected_models = {"baseline", "aafnet"}
    for name, dataset_field, model_field, expected_datasets in PAIR_COVERAGE_EXPECTATIONS:
        path = SOURCE_DIR / name
        if not path.exists():
            checks.append(
                Check(
                    "source_pair_coverage",
                    f"{name} covers each expected dataset-model pair",
                    "fail",
                    rel(path),
                    "missing",
                )
            )
            continue
        rows = read_csv_rows(path)
        observed = {
            (row.get(dataset_field, ""), row.get(model_field, ""))
            for row in rows
            if row.get(dataset_field, "") and row.get(model_field, "")
        }
        expected = {(dataset, model) for dataset in expected_datasets for model in expected_models}
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        checks.append(
            Check(
                "source_pair_coverage",
                f"{name} covers each expected dataset-model pair",
                "pass" if not missing else "fail",
                rel(path),
                f"dataset_field={dataset_field}; model_field={model_field}; observed={len(observed)} pairs; missing={missing}; extra={extra[:6]}",
            )
        )
    return checks


def audit_performance_freshness() -> list[Check]:
    """Ensure F_TC2 follows the live public-CV state instead of stale fallback rows."""
    if not PERFORMANCE_SOURCE.exists():
        return [
            Check(
                "performance_freshness",
                "F_TC2 performance source can be checked against public-CV status",
                "fail",
                rel(PERFORMANCE_SOURCE),
                "missing",
            )
        ]

    cv_payload = load_json(CV_STATUS, {})
    cv_rows = cv_payload.get("rows", []) if isinstance(cv_payload, dict) else []
    f_tc2_rows: dict[tuple[str, str], dict[str, str]] = {}
    with PERFORMANCE_SOURCE.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            f_tc2_rows[(row.get("dataset", ""), row.get("model", ""))] = row

    checks: list[Check] = []
    for dataset, role in TASKS:
        cv_row = next((row for row in cv_rows if row.get("dataset") == dataset and row.get("role") == role), {})
        if not cv_row:
            continue
        completed = int(cv_row.get("n_folds_completed") or 0)
        total = int(cv_row.get("n_folds_total") or 15)
        perf_row = f_tc2_rows.get((dataset, role))
        if perf_row is None:
            checks.append(
                Check(
                    "performance_freshness",
                    f"F_TC2 has a performance row for {dataset} {role}",
                    "fail",
                    rel(PERFORMANCE_SOURCE),
                    "row missing",
                )
            )
            continue

        try:
            repeats = int(float(perf_row.get("n_repeats") or 0))
        except ValueError:
            repeats = -1
        source = perf_row.get("source", "")
        source_path = perf_row.get("source_path", "")
        expected_paths = {rel(CV_STATUS)}
        summary_path = str(cv_row.get("summary_path") or "")
        if summary_path and not summary_path.startswith("partial scan"):
            expected_paths.add(summary_path)

        if completed > 0:
            ok = (
                repeats == completed
                and source_path in expected_paths
                and "fallback" not in source.lower()
                and "cv" in source.lower()
            )
            detail = (
                f"F_TC2 repeats={repeats}; public-CV completed={completed}/{total}; "
                f"source={source or 'n/a'}; source_path={source_path or 'n/a'}"
            )
        else:
            ok = "fallback" in source.lower() and repeats > 0
            detail = (
                "no public-CV cells yet; fallback row is acceptable until the first "
                f"cell exists; F_TC2 repeats={repeats}; source={source or 'n/a'}"
            )

        checks.append(
            Check(
                "performance_freshness",
                f"F_TC2 performance row is fresh for {dataset} {role}",
                "pass" if ok else "fail",
                f"{rel(PERFORMANCE_SOURCE)}; {rel(CV_STATUS)}",
                detail,
            )
        )
    return checks


def audit_coverage_freshness() -> list[Check]:
    """Ensure F_TC8 coverage cells are recomputable from live queue CSV files."""
    required = [COVERAGE_SOURCE, CV_QUEUE_SOURCE, ROBUST_QUEUE_SOURCE, CALROT_QUEUE_SOURCE, FOLLOWUP_STATUS_SOURCE]
    missing = [rel(path) for path in required if not path.exists()]
    if missing:
        return [
            Check(
                "coverage_freshness",
                "F_TC8 coverage source can be checked against live queue sources",
                "fail",
                "; ".join(rel(path) for path in required),
                "missing: " + "; ".join(missing),
            )
        ]

    coverage_rows = read_csv_rows(COVERAGE_SOURCE)
    cv_rows = read_csv_rows(CV_QUEUE_SOURCE)
    robust_rows = read_csv_rows(ROBUST_QUEUE_SOURCE)
    calrot_rows = read_csv_rows(CALROT_QUEUE_SOURCE)
    follow_rows = read_csv_rows(FOLLOWUP_STATUS_SOURCE)
    coverage = {(row.get("dataset", ""), row.get("role", ""), row.get("metric", "")): row for row in coverage_rows}
    follow = {(row.get("dataset", ""), row.get("role", "")): row for row in follow_rows}

    def count(rows: list[dict[str, str]], dataset: str, role: str, predicate) -> int:
        return sum(1 for row in rows if row.get("dataset") == dataset and row.get("role") == role and predicate(row))

    def as_int(row: dict[str, str], field: str) -> int:
        try:
            return int(float(row.get(field) or 0))
        except ValueError:
            return -1

    checks: list[Check] = []
    evidence = "; ".join(rel(path) for path in required)
    for dataset, role in TASKS:
        frow = follow.get((dataset, role), {})
        expected = {
            "cv_done": count(cv_rows, dataset, role, lambda row: row.get("status") == "complete"),
            "robust_ready": count(robust_rows, dataset, role, lambda row: row.get("status") in {"checkpoint_ready", "robustness_complete"}),
            "robust_done": as_int(frow, "robustness_complete"),
            "cal_ready": count(calrot_rows, dataset, role, lambda row: row.get("status") == "checkpoint_ready"),
            "cal_done": as_int(frow, "calibration_complete"),
            "rot_ready": count(calrot_rows, dataset, role, lambda row: row.get("status") == "checkpoint_ready"),
            "rot_done": as_int(frow, "rotation_complete"),
        }
        mismatches: list[str] = []
        for metric, expected_value in expected.items():
            row = coverage.get((dataset, role, metric))
            if row is None:
                mismatches.append(f"{metric}: row missing, expected {expected_value}/15")
                continue
            actual = as_int(row, "completed_cells")
            target = as_int(row, "target_cells")
            try:
                fraction = float(row.get("completion_fraction") or "nan")
            except ValueError:
                fraction = float("nan")
            expected_fraction = expected_value / 15.0
            if actual != expected_value or target != 15 or abs(fraction - expected_fraction) > 1e-6:
                mismatches.append(
                    f"{metric}: coverage={actual}/{target} ({fraction:.4f}), expected={expected_value}/15 ({expected_fraction:.4f})"
                )
        detail = (
            "all coverage cells match queue/follow-up sources"
            if not mismatches
            else "; ".join(mismatches[:5])
        )
        checks.append(
            Check(
                "coverage_freshness",
                f"F_TC8 coverage row set is fresh for {dataset} {role}",
                "pass" if not mismatches else "fail",
                evidence,
                detail,
            )
        )
    return checks


def audit_checkpoint_readiness() -> list[Check]:
    """Verify that completed CV cells are connected to downstream probe queues."""
    required = [CV_QUEUE_SOURCE, ROBUST_QUEUE_SOURCE, CALROT_QUEUE_SOURCE]
    missing = [rel(path) for path in required if not path.exists()]
    if missing:
        return [
            Check(
                "checkpoint_readiness",
                "Public-CV cell queues can be cross-checked",
                "fail",
                "; ".join(rel(path) for path in required),
                "missing: " + "; ".join(missing),
            )
        ]

    cv_rows = read_csv_rows(CV_QUEUE_SOURCE)
    robust_rows = read_csv_rows(ROBUST_QUEUE_SOURCE)
    calrot_rows = read_csv_rows(CALROT_QUEUE_SOURCE)

    def key(row: dict[str, str]) -> tuple[str, str, str, str]:
        return (
            row.get("dataset", ""),
            row.get("role", ""),
            row.get("seed", ""),
            row.get("fold", ""),
        )

    cv_keys = {key(row) for row in cv_rows}
    robust_by_key = {key(row): row for row in robust_rows}
    calrot_by_key = {key(row): row for row in calrot_rows}
    robust_keys = set(robust_by_key)
    calrot_keys = set(calrot_by_key)

    checks: list[Check] = []
    evidence = "; ".join(rel(path) for path in required)
    key_mismatches: list[str] = []
    if cv_keys != robust_keys:
        key_mismatches.append(
            f"robust queue missing={len(cv_keys - robust_keys)}, extra={len(robust_keys - cv_keys)}"
        )
    if cv_keys != calrot_keys:
        key_mismatches.append(
            f"calrot queue missing={len(cv_keys - calrot_keys)}, extra={len(calrot_keys - cv_keys)}"
        )
    checks.append(
        Check(
            "checkpoint_readiness",
            "Public-CV, robustness and calibration/rotation queues enumerate the same cells",
            "pass" if not key_mismatches else "fail",
            evidence,
            f"{len(cv_keys)} CV cells; queue keys match" if not key_mismatches else "; ".join(key_mismatches),
        )
    )

    completed = 0
    waiting = 0
    completed_errors: list[str] = []
    waiting_errors: list[str] = []
    for cv_row in cv_rows:
        cell = key(cv_row)
        dataset, role, seed, fold = cell
        label = f"{dataset} {role} s{seed}-f{fold}"
        robust = robust_by_key.get(cell, {})
        calrot = calrot_by_key.get(cell, {})
        cv_complete = cv_row.get("status") == "complete"
        if cv_complete:
            completed += 1
            metric_path = workspace_path(cv_row.get("metric_path", ""))
            if metric_path is None or not file_ok(metric_path, min_bytes=1):
                completed_errors.append(f"{label}: missing test_metrics.json")

            robust_status = robust.get("status", "")
            robust_ckpt = workspace_path(robust.get("checkpoint", ""))
            if robust_status not in {"checkpoint_ready", "robustness_complete"}:
                completed_errors.append(f"{label}: robust status={robust_status or 'missing'}")
            if robust_ckpt is None or not file_ok(robust_ckpt, min_bytes=1):
                completed_errors.append(f"{label}: robust checkpoint missing")

            calrot_status = calrot.get("status", "")
            cal_status = calrot.get("calibration", "")
            rot_status = calrot.get("rotation", "")
            calrot_ckpt = workspace_path(calrot.get("checkpoint", ""))
            if calrot_status == "waiting_for_cv" or not calrot_status:
                completed_errors.append(f"{label}: calrot status={calrot_status or 'missing'}")
            if cal_status not in {"pending", "complete"}:
                completed_errors.append(f"{label}: calibration={cal_status or 'missing'}")
            if rot_status not in {"pending", "complete"}:
                completed_errors.append(f"{label}: rotation={rot_status or 'missing'}")
            if calrot_ckpt is None or not file_ok(calrot_ckpt, min_bytes=1):
                completed_errors.append(f"{label}: calrot checkpoint missing")
        else:
            waiting += 1
            robust_status = robust.get("status", "")
            calrot_status = calrot.get("status", "")
            cal_status = calrot.get("calibration", "")
            rot_status = calrot.get("rotation", "")
            if robust_status != "waiting_for_cv":
                waiting_errors.append(f"{label}: robust status={robust_status or 'missing'}")
            if calrot_status != "waiting_for_cv":
                waiting_errors.append(f"{label}: calrot status={calrot_status or 'missing'}")
            if cal_status != "waiting_for_cv" or rot_status != "waiting_for_cv":
                waiting_errors.append(
                    f"{label}: calibration={cal_status or 'missing'}, rotation={rot_status or 'missing'}"
                )

    checks.append(
        Check(
            "checkpoint_readiness",
            "Completed public-CV cells expose metrics and probe-ready checkpoints",
            "pass" if not completed_errors else "fail",
            evidence,
            f"{completed} completed cells linked to metrics/checkpoints"
            if not completed_errors
            else "; ".join(completed_errors[:8]),
        )
    )
    checks.append(
        Check(
            "checkpoint_readiness",
            "Incomplete public-CV cells remain blocked in downstream probe queues",
            "pass" if not waiting_errors else "fail",
            evidence,
            f"{waiting} incomplete cells still waiting for CV"
            if not waiting_errors
            else "; ".join(waiting_errors[:8]),
        )
    )
    return checks


def audit_rotation() -> list[Check]:
    if not ROTATION_CSV.exists():
        return [
            Check(
                "rotation_qa",
                "Display-only rotation audit exists",
                "fail",
                rel(ROTATION_CSV),
                "missing",
            )
        ]
    invalid: list[str] = []
    rows = 0
    with ROTATION_CSV.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows += 1
            try:
                value = int(float(row.get("display_rotation_deg", "")))
            except ValueError:
                invalid.append(f"line {row.get('line', '?')}: non-numeric")
                continue
            if value not in {0, 90, 180, 270}:
                invalid.append(f"line {row.get('line', '?')}: {value}")
    return [
        Check(
            "rotation_qa",
            "Display-only rotation entries are valid right-angle corrections",
            "pass" if not invalid else "fail",
            rel(ROTATION_CSV),
            f"{rows} rows; invalid={len(invalid)}" + ("" if not invalid else "; " + "; ".join(invalid[:5])),
        )
    ]


def audit_runbook() -> list[Check]:
    checks: list[Check] = []
    text = RUNBOOK.read_text(encoding="utf-8") if RUNBOOK.exists() else ""
    has_block = "THREE_CORPUS_LIVE_STATUS_START" in text and "THREE_CORPUS_LIVE_STATUS_END" in text
    checks.append(
        Check(
            "runbook",
            "Missing-experiment runbook has auto-synchronized live status",
            "pass" if has_block else "fail",
            rel(RUNBOOK),
            "live status block present" if has_block else "missing live status block",
        )
    )
    if not has_block:
        return checks

    required = [LIVE_STATUS_SOURCE, CV_STATUS, ROBUST_QUEUE_SOURCE, CALROT_QUEUE_SOURCE, FOLLOWUP_STATUS_SOURCE]
    missing = [rel(path) for path in required if not path.exists()]
    if missing:
        checks.append(
            Check(
                "runbook",
                "Runbook live source data can be checked against machine status",
                "fail",
                "; ".join(rel(path) for path in required),
                "missing: " + "; ".join(missing),
            )
        )
        return checks

    live_rows = {(row.get("dataset", ""), row.get("role", "")): row for row in read_csv_rows(LIVE_STATUS_SOURCE)}
    cv_payload = load_json(CV_STATUS, {})
    cv_rows = cv_payload.get("rows", []) if isinstance(cv_payload, dict) else []
    cv_by_task = {(row.get("dataset", ""), row.get("role", "")): row for row in cv_rows}
    robust_rows = read_csv_rows(ROBUST_QUEUE_SOURCE)
    calrot_rows = read_csv_rows(CALROT_QUEUE_SOURCE)
    follow_rows = {(row.get("dataset", ""), row.get("role", "")): row for row in read_csv_rows(FOLLOWUP_STATUS_SOURCE)}

    def count(rows: list[dict[str, str]], dataset: str, role: str, predicate) -> int:
        return sum(1 for row in rows if row.get("dataset") == dataset and row.get("role") == role and predicate(row))

    def as_int(row: dict[str, str], field: str) -> int:
        try:
            return int(float(row.get(field) or 0))
        except ValueError:
            return -1

    mismatches: list[str] = []
    totals = {
        "cv_completed": 0,
        "cv_total": 0,
        "cv_missing": 0,
        "probe_complete": 0,
        "probe_ready": 0,
        "probe_waiting": 0,
    }
    for dataset, role in TASKS:
        live = live_rows.get((dataset, role))
        if live is None:
            mismatches.append(f"{dataset} {role}: live status row missing")
            continue
        cv = cv_by_task.get((dataset, role), {})
        follow = follow_rows.get((dataset, role), {})
        completed = int(cv.get("n_folds_completed") or 0)
        total = int(cv.get("n_folds_total") or 15)
        missing_cv = max(total - completed, 0)
        expected = {
            "cv_completed": completed,
            "cv_total": total,
            "cv_missing": missing_cv,
            "robustness_ready": count(robust_rows, dataset, role, lambda row: row.get("status") in {"checkpoint_ready", "robustness_complete"}),
            "robustness_complete": as_int(follow, "robustness_complete"),
            "robustness_pending_ready": count(robust_rows, dataset, role, lambda row: row.get("status") == "checkpoint_ready"),
            "calibration_ready": count(calrot_rows, dataset, role, lambda row: row.get("status") != "waiting_for_cv"),
            "calibration_complete": as_int(follow, "calibration_complete"),
            "calibration_pending_ready": count(
                calrot_rows,
                dataset,
                role,
                lambda row: row.get("status") != "waiting_for_cv" and row.get("calibration") != "complete",
            ),
            "rotation_ready": count(calrot_rows, dataset, role, lambda row: row.get("status") != "waiting_for_cv"),
            "rotation_complete": as_int(follow, "rotation_complete"),
            "rotation_pending_ready": count(
                calrot_rows,
                dataset,
                role,
                lambda row: row.get("status") != "waiting_for_cv" and row.get("rotation") != "complete",
            ),
            "waiting_for_cv": missing_cv,
        }
        for field, expected_value in expected.items():
            actual = as_int(live, field)
            if actual != expected_value:
                mismatches.append(f"{dataset} {role} {field}: live={actual}, expected={expected_value}")
        totals["cv_completed"] += expected["cv_completed"]
        totals["cv_total"] += expected["cv_total"]
        totals["cv_missing"] += expected["cv_missing"]
        totals["probe_complete"] += (
            expected["robustness_complete"] + expected["calibration_complete"] + expected["rotation_complete"]
        )
        totals["probe_ready"] += (
            expected["robustness_pending_ready"]
            + expected["calibration_pending_ready"]
            + expected["rotation_pending_ready"]
        )
        totals["probe_waiting"] += expected["cv_missing"] * 3

    expected_snippets = [
        f"Public-corpus CV: {totals['cv_completed']}/{totals['cv_total']} fold-seed cells complete; {totals['cv_missing']} cells remain.",
        f"Fold-level follow-up probes: {totals['probe_complete']}/{totals['cv_total'] * 3} outputs complete; {totals['probe_ready']} outputs are ready to run from existing checkpoints; {totals['probe_waiting']} outputs are still waiting for CV checkpoints.",
    ]
    for snippet in expected_snippets:
        if snippet not in text:
            mismatches.append(f"runbook summary line missing or stale: {snippet}")

    checks.append(
        Check(
            "runbook",
            "Runbook live status matches current machine-readable queues",
            "pass" if not mismatches else "fail",
            "; ".join(rel(path) for path in required),
            "live CSV and Markdown totals match current status" if not mismatches else "; ".join(mismatches[:6]),
        )
    )
    return checks


def audit_launchers() -> list[Check]:
    follow_dir = ROOT / "outputs" / "public_followup_probes"
    launcher_paths = sorted(follow_dir.glob("*_launcher.json"))
    if not launcher_paths:
        return [
            Check(
                "automation",
                "Post-CV follow-up launcher is recorded",
                "warn",
                rel(follow_dir),
                "no launcher status found; this is not a final-evidence failure but requires manual post-CV handoff",
            )
        ]
    infos: list[dict[str, Any]] = []
    for path in launcher_paths:
        payload = load_json(path, {})
        pid = payload.get("pid")
        alive = isinstance(pid, int) and pid_alive(pid)
        command = str(payload.get("command") or "")
        try:
            tokens = shlex.split(command)
        except ValueError:
            tokens = command.split()
        runner_ok = any(
            runner in tokens
            for runner in [
                "scripts/run_public_post_cv_followups.py",
                "scripts/run_public_followup_probe_pool.py",
            ]
        )
        expected_flags = ["--wait", "--run-probes", "--run-id", "public_cv_parity_v1"]
        missing_flags = [flag for flag in expected_flags if flag not in tokens]
        if not runner_ok:
            missing_flags.append("post-CV runner script")
        forbidden_flags = [flag for flag in ["--allow-partial"] if flag in tokens]
        command_ok = not missing_flags and not forbidden_flags
        infos.append(
            {
                "path": path,
                "payload": payload,
                "pid": pid,
                "alive": alive,
                "command_ok": command_ok,
                "missing_flags": missing_flags,
                "forbidden_flags": forbidden_flags,
                "command": command,
            }
        )

    active_valid_launcher = any(info["alive"] and info["command_ok"] for info in infos)
    checks: list[Check] = []
    for info in infos:
        path = info["path"]
        payload = info["payload"]
        pid = info["pid"]
        alive = bool(info["alive"])
        command_ok = bool(info["command_ok"])
        tracked_ok = alive or active_valid_launcher
        if alive:
            tracked_detail = f"pid={pid}; alive=True; log={payload.get('log_path', 'n/a')}"
        elif active_valid_launcher:
            tracked_detail = f"pid={pid}; alive=False; superseded by another active valid launcher"
        else:
            tracked_detail = f"pid={pid}; alive=False; log={payload.get('log_path', 'n/a')}"
        checks.append(
            Check(
                "automation",
                f"Post-CV launcher {payload.get('label', path.stem)} is tracked",
                "pass" if tracked_ok else "warn",
                rel(path),
                tracked_detail,
            )
        )
        checks.append(
            Check(
                "automation",
                f"Post-CV launcher {payload.get('label', path.stem)} waits for full CV and runs probes",
                "pass" if command_ok else "fail",
                rel(path),
                "command has --wait and --run-probes without --allow-partial"
                if command_ok
                else (
                    f"missing={info['missing_flags']}; forbidden={info['forbidden_flags']}; "
                    f"command={info['command'] or 'n/a'}"
                ),
            )
        )
    return checks


def audit_release_finalizer() -> list[Check]:
    """Track final package/upload automation without blocking experiment completion."""
    checks: list[Check] = []
    report = load_json(FINALIZER_REPORT_JSON, {})
    if isinstance(report, dict) and report.get("url") and report.get("package"):
        checks.append(
            Check(
                "release_automation",
                "Final package upload report exists",
                "pass",
                rel(FINALIZER_REPORT_JSON),
                f"package={report.get('package')}; url_present=True",
            )
        )
        return checks

    launcher = load_json(FINALIZER_LAUNCHER_JSON, {})
    if not isinstance(launcher, dict) or not launcher:
        return [
            Check(
                "release_automation",
                "Final package/upload finalizer is tracked",
                "warn",
                rel(FINALIZER_LAUNCHER_JSON),
                "missing launcher status; final upload may require manual start after audit completion",
            )
        ]

    pid = launcher.get("pid")
    alive = isinstance(pid, int) and pid_alive(pid)
    command = str(launcher.get("command") or "")
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()
    required = [
        "scripts/finalize_three_corpus_release.py",
        "--wait",
        "--run-id",
        "public_cv_parity_v1",
        "--followup-label",
        "post_cv_followups_public_cv_parity_v1_pool",
        "--config-doc",
        "--expires-days",
    ]
    missing = [token for token in required if token not in tokens]
    command_ok = not missing
    checks.append(
        Check(
            "release_automation",
            "Final package/upload finalizer is tracked",
            "pass" if alive else "warn",
            rel(FINALIZER_LAUNCHER_JSON),
            f"pid={pid}; alive={alive}; log={launcher.get('log_path', 'n/a')}",
        )
    )
    checks.append(
        Check(
            "release_automation",
            "Final package/upload finalizer waits for full evidence and creates a 3-day link",
            "pass" if command_ok else "warn",
            rel(FINALIZER_LAUNCHER_JSON),
            "command waits for CV/probes and uses config-doc plus expires-days"
            if command_ok
            else f"missing={missing}; command={command or 'n/a'}",
        )
    )
    return checks


def collect_checks() -> list[Check]:
    checks: list[Check] = []
    checks.extend(audit_public_cv())
    checks.extend(audit_followup_probes())
    checks.extend(audit_figures())
    checks.extend(audit_export_quality())
    checks.extend(audit_source_data())
    checks.extend(audit_source_dataset_coverage())
    checks.extend(audit_source_model_coverage())
    checks.extend(audit_source_pair_coverage())
    checks.extend(audit_performance_freshness())
    checks.extend(audit_coverage_freshness())
    checks.extend(audit_checkpoint_readiness())
    checks.extend(audit_rotation())
    checks.extend(audit_runbook())
    checks.extend(audit_launchers())
    checks.extend(audit_release_finalizer())
    return checks


def write_reports(checks: list[Check]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fail_count = sum(check.status == "fail" for check in checks)
    warn_count = sum(check.status == "warn" for check in checks)
    pass_count = sum(check.status == "pass" for check in checks)
    payload = {
        "generated_at_epoch": time.time(),
        "complete": fail_count == 0,
        "pass": pass_count,
        "warn": warn_count,
        "fail": fail_count,
        "checks": [check.as_dict() for check in checks],
    }
    REPORT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Three-Corpus Parity Completion Audit",
        "",
        f"- Complete: {payload['complete']}",
        f"- Pass: {pass_count}",
        f"- Warn: {warn_count}",
        f"- Fail: {fail_count}",
        "",
        "| Category | Requirement | Status | Detail | Evidence |",
        "|---|---|---:|---|---|",
    ]
    for check in checks:
        lines.append(
            f"| {check.category} | {check.requirement} | {check.status} | "
            f"{check.detail.replace('|', '/')} | `{check.evidence}` |"
        )
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    checks = collect_checks()
    write_reports(checks)
    print(f"wrote: {REPORT_MD.relative_to(ROOT)}")
    print(f"complete: {all(check.status != 'fail' for check in checks)}")
    failures = [check for check in checks if check.status == "fail"]
    for check in failures[:12]:
        print(f"[fail] {check.category}: {check.requirement} ({check.detail})")


if __name__ == "__main__":
    main()
