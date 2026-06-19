#!/usr/bin/env python
"""Summarize public-corpus follow-up probes into audit/source-data tables.

This script does not run models. It scans fold-level outputs from
``run_public_robustness_attribution.py`` and
``run_public_calibration_rotation.py`` and writes compact Markdown/CSV status
tables for manuscript bookkeeping.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, pstdev


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "public_followup_probes"
SOURCE_DIR = ROOT / "paper" / "figures" / "nature_source_data"
RUN_ID = "public_cv_parity_v1"
SEEDS = [42, 1337, 2024]
FOLDS = list(range(5))

TASKS = [
    ("ASP_clean", "baseline", "asp_baseline"),
    ("ASP_clean", "aafnet", "asp_aafnet"),
    ("AS25_clean", "baseline", "as25_baseline"),
    ("AS25_clean", "aafnet", "as25_aafnet"),
]


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def robustness_result(task: str, seed: int, fold: int) -> Path:
    return (
        ROOT
        / "outputs"
        / "public_robustness_attribution"
        / task
        / RUN_ID
        / f"seed{seed}_fold{fold}"
        / "eval"
        / "resnet50"
        / "results.json"
    )


def calrot_result(task: str, seed: int, fold: int) -> Path:
    return ROOT / "outputs" / "public_calibration_rotation" / task / RUN_ID / f"seed{seed}_fold{fold}" / "results.json"


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    if fieldnames is not None:
        keys = fieldnames
    else:
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
    if not keys:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


STATUS_FIELDS = ["dataset", "role", "task", "robustness_complete", "calibration_complete", "rotation_complete", "target_cells"]
ROBUST_FIELDS = ["dataset", "role", "task", "seed", "fold", "metric", "value", "clean_accuracy", "result_path"]
CALIBRATION_FIELDS = ["dataset", "role", "task", "seed", "fold", "condition", "stage", "metric", "value", "temperature", "result_path"]
ROTATION_FIELDS = ["dataset", "role", "task", "seed", "fold", "metric", "angle_deg", "value", "result_path"]


def summary_fields(key_fields: list[str]) -> list[str]:
    return key_fields + ["n", "mean", "std"]


def ensure_ordered(row: dict[str, object], keys: list[str]) -> dict[str, object]:
    out = {key: row.get(key, "") for key in keys}
    for key, value in row.items():
        if key not in keys:
            out[key] = value
    return out


def summarize_values(values: list[float]) -> tuple[str, str, str]:
    if not values:
        return "0", "", ""
    return str(len(values)), f"{mean(values):.8f}", f"{pstdev(values):.8f}" if len(values) > 1 else "0.00000000"


def collect() -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    status_rows: list[dict[str, object]] = []
    robust_rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    rotation_rows: list[dict[str, object]] = []

    for dataset, role, task in TASKS:
        robust_complete = 0
        cal_complete = 0
        rot_complete = 0
        for seed in SEEDS:
            for fold in FOLDS:
                r_payload = load_json(robustness_result(task, seed, fold))
                if r_payload is not None and "robustness" in r_payload:
                    robust_complete += 1
                    clean = r_payload.get("clean_accuracy")
                    all_acc: list[float] = []
                    for family, values in r_payload["robustness"].items():
                        accs = [float(item["accuracy"]) for item in values]
                        all_acc.extend(accs)
                        robust_rows.append(
                            {
                                "dataset": dataset,
                                "role": role,
                                "task": task,
                                "seed": seed,
                                "fold": fold,
                                "metric": f"{family}_mean",
                                "value": mean(accs),
                                "clean_accuracy": clean,
                                "result_path": str(robustness_result(task, seed, fold).relative_to(ROOT)),
                            }
                        )
                    if all_acc:
                        robust_rows.append(
                            {
                                "dataset": dataset,
                                "role": role,
                                "task": task,
                                "seed": seed,
                                "fold": fold,
                                "metric": "all_corruption_mean",
                                "value": mean(all_acc),
                                "clean_accuracy": clean,
                                "result_path": str(robustness_result(task, seed, fold).relative_to(ROOT)),
                            }
                        )

                c_payload = load_json(calrot_result(task, seed, fold))
                if c_payload is not None and "calibration" in c_payload:
                    cal_complete += 1
                    cal = c_payload["calibration"]
                    for condition, cond_payload in cal.get("conditions", {}).items():
                        for stage in ["pre", "post"]:
                            metrics = cond_payload.get(stage, {})
                            for metric in ["accuracy", "ece", "nll", "brier"]:
                                if metric in metrics:
                                    calibration_rows.append(
                                        {
                                            "dataset": dataset,
                                            "role": role,
                                            "task": task,
                                            "seed": seed,
                                            "fold": fold,
                                            "condition": condition,
                                            "stage": stage,
                                            "metric": metric,
                                            "value": metrics[metric],
                                            "temperature": cal.get("temperature"),
                                            "result_path": str(calrot_result(task, seed, fold).relative_to(ROOT)),
                                        }
                                    )
                if c_payload is not None and "rotation" in c_payload:
                    rot_complete += 1
                    rot = c_payload["rotation"]
                    for metric, value in rot.get("summary", {}).items():
                        rotation_rows.append(
                            {
                                "dataset": dataset,
                                "role": role,
                                "task": task,
                                "seed": seed,
                                "fold": fold,
                                "metric": metric,
                                "value": value,
                                "result_path": str(calrot_result(task, seed, fold).relative_to(ROOT)),
                            }
                        )
                    for angle, value in zip(rot.get("angles", []), rot.get("accuracies", [])):
                        rotation_rows.append(
                            {
                                "dataset": dataset,
                                "role": role,
                                "task": task,
                                "seed": seed,
                                "fold": fold,
                                "metric": "angle_accuracy",
                                "angle_deg": angle,
                                "value": value,
                                "result_path": str(calrot_result(task, seed, fold).relative_to(ROOT)),
                            }
                        )

        status_rows.append(
            {
                "dataset": dataset,
                "role": role,
                "task": task,
                "robustness_complete": robust_complete,
                "calibration_complete": cal_complete,
                "rotation_complete": rot_complete,
                "target_cells": 15,
            }
        )
    return status_rows, robust_rows, calibration_rows, rotation_rows


def aggregate_metric(rows: list[dict[str, object]], key_fields: list[str]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[float]] = {}
    for row in rows:
        key = tuple(row.get(field, "") for field in key_fields)
        value = row.get("value")
        if value == "" or value is None:
            continue
        groups.setdefault(key, []).append(float(value))

    out: list[dict[str, object]] = []
    for key, values in sorted(groups.items()):
        n, avg, std = summarize_values(values)
        out.append({field: key[i] for i, field in enumerate(key_fields)} | {"n": n, "mean": avg, "std": std})
    return out


def write_outputs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    status_rows, robust_rows, calibration_rows, rotation_rows = collect()

    robust_summary = aggregate_metric(robust_rows, ["dataset", "role", "metric"])
    calibration_summary = aggregate_metric(calibration_rows, ["dataset", "role", "condition", "stage", "metric"])
    rotation_summary = aggregate_metric(rotation_rows, ["dataset", "role", "metric"])

    write_csv(SOURCE_DIR / "public_followup_probe_status_source.csv", status_rows, STATUS_FIELDS)
    write_csv(SOURCE_DIR / "public_robustness_fold_metrics_source.csv", robust_rows, ROBUST_FIELDS)
    write_csv(SOURCE_DIR / "public_calibration_fold_metrics_source.csv", calibration_rows, CALIBRATION_FIELDS)
    write_csv(SOURCE_DIR / "public_rotation_fold_metrics_source.csv", rotation_rows, ROTATION_FIELDS)
    write_csv(SOURCE_DIR / "public_robustness_summary_source.csv", robust_summary, summary_fields(["dataset", "role", "metric"]))
    write_csv(SOURCE_DIR / "public_calibration_summary_source.csv", calibration_summary, summary_fields(["dataset", "role", "condition", "stage", "metric"]))
    write_csv(SOURCE_DIR / "public_rotation_summary_source.csv", rotation_summary, summary_fields(["dataset", "role", "metric"]))

    payload = {
        "status": status_rows,
        "robustness_summary": robust_summary,
        "calibration_summary": calibration_summary,
        "rotation_summary": rotation_summary,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Public Follow-Up Probe Summary",
        "",
        f"- CV run id: `{RUN_ID}`",
        "- This summary only aggregates completed fold-level probe JSON files.",
        "",
        "| Dataset | Role | Robustness | Calibration | Rotation |",
        "|---|---|---:|---:|---:|",
    ]
    for row in status_rows:
        lines.append(
            f"| {row['dataset']} | {row['role']} | {row['robustness_complete']}/15 | "
            f"{row['calibration_complete']}/15 | {row['rotation_complete']}/15 |"
        )
    lines += [
        "",
        "## Source Data",
        "",
        "- `public_followup_probe_status_source.csv`",
        "- `public_robustness_fold_metrics_source.csv`",
        "- `public_calibration_fold_metrics_source.csv`",
        "- `public_rotation_fold_metrics_source.csv`",
        "- `public_robustness_summary_source.csv`",
        "- `public_calibration_summary_source.csv`",
        "- `public_rotation_summary_source.csv`",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((OUT_DIR / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    write_outputs()
