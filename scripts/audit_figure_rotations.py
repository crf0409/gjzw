#!/usr/bin/env python
"""Audit display-only rotations recorded in paper figure source data.

The inference figures keep model inputs unchanged and rotate only rendered
thumbnails/overlays. This audit makes those display corrections explicit and
checks that all recorded rotations use right-angle values.
"""

from __future__ import annotations

import csv
from pathlib import Path

from figure_display_rotation import display_rotation, iter_display_rotation_rules, normalize_rotation


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "paper" / "figures" / "nature_source_data"
OUT_CSV = SOURCE_DIR / "figure_rotation_audit.csv"
OUT_MD = ROOT / "paper" / "figures" / "figure_rotation_audit.md"


def split_rotation_values(value: str) -> list[int]:
    out: list[int] = []
    for part in str(value).replace(",", ";").split(";"):
        part = part.strip()
        if not part:
            continue
        out.append(int(float(part)))
    return out


def split_index_values(value: str) -> list[str]:
    return [part.strip() for part in str(value).replace(",", ";").split(";") if part.strip()]


def infer_dataset(path: Path, row: dict[str, str]) -> str:
    dataset = row.get("dataset") or row.get("sample_dataset") or row.get("source")
    if dataset:
        return str(dataset)
    al6_only_figures = {
        "F_Q2a_retrieval_effect",
        "F_Q2d_selective_rejection_effect",
        "F_Q2e_augmentation_invariance_effect",
    }
    if path.stem in al6_only_figures or row.get("figure") in al6_only_figures:
        return "AL6"
    return "-"


def samples_for_key(key: str, row: dict[str, str]) -> list[str]:
    if key == "retrieved_display_rotation_deg":
        return [row.get("retrieved_index", "-")]
    if key == "comparison_display_rotation_deg":
        return [row.get("comparison_index", "-")]
    if key == "query_display_rotation_deg":
        return [row.get("query_index", "-")]
    if key == "displayed_support_rotation_deg":
        return [row.get("displayed_support_index", "-")]
    if key == "support_display_rotation_degs":
        return split_index_values(row.get("support_indices", "-"))
    return [row.get("sample_index") or row.get("index") or row.get("displayed_support_index") or "-"]


def validate_against_rule(
    path: Path,
    line_no: int,
    row: dict[str, str],
    key: str,
    sample: str,
    value: int,
    errors: list[str],
) -> None:
    dataset = infer_dataset(path, row)
    if dataset == "-" or sample == "-":
        return
    try:
        expected = display_rotation(dataset, int(sample))
    except (TypeError, ValueError):
        return
    if value != expected:
        errors.append(
            f"{path.name}:{line_no}:{key} sample {dataset}/{sample} has {value}, expected {expected}"
        )


def collect_rows() -> tuple[list[dict[str, object]], list[str]]:
    rows: list[dict[str, object]] = []
    errors: list[str] = []
    for path in sorted(SOURCE_DIR.glob("*.csv")):
        if path.name == OUT_CSV.name:
            continue
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames:
                    continue
                rotation_keys = [
                    key
                    for key in reader.fieldnames
                    if key.endswith("rotation_deg") or key.endswith("rotation_degs")
                ]
                if not rotation_keys:
                    continue
                for line_no, row in enumerate(reader, start=2):
                    for key in rotation_keys:
                        raw = row.get(key, "")
                        if raw in ("", None):
                            continue
                        try:
                            values = split_rotation_values(str(raw))
                        except ValueError:
                            errors.append(f"{path.name}:{line_no}:{key} is not numeric: {raw!r}")
                            continue
                        samples = samples_for_key(key, row)
                        if len(samples) == 1 and len(values) > 1:
                            samples = samples * len(values)
                        if len(samples) != len(values):
                            errors.append(
                                f"{path.name}:{line_no}:{key} has {len(values)} rotations but {len(samples)} sample ids"
                            )
                            continue
                        for sample, value in zip(samples, values, strict=True):
                            try:
                                value = normalize_rotation(value)
                            except ValueError:
                                errors.append(f"{path.name}:{line_no}:{key} has invalid angle {value}")
                                continue
                            validate_against_rule(path, line_no, row, key, sample, value, errors)
                            if value == 0:
                                continue
                            rows.append(
                                {
                                    "source_csv": path.name,
                                    "line": line_no,
                                    "rotation_field": key,
                                    "display_rotation_deg": value,
                                    "figure": row.get("figure", path.stem),
                                    "dataset": infer_dataset(path, row),
                                    "sample_index": sample,
                                    "note": "display-only; model logits/features/GradCAM targets use unmodified cached tensors",
                                }
                            )
        except csv.Error as exc:
            errors.append(f"{path.name}: CSV parse error: {exc}")
    return rows, errors


def write_csv(rows: list[dict[str, object]]) -> None:
    keys = [
        "source_csv",
        "line",
        "rotation_field",
        "display_rotation_deg",
        "figure",
        "dataset",
        "sample_index",
        "note",
    ]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, object]], errors: list[str]) -> None:
    rule_rows = list(iter_display_rotation_rules())
    lines = [
        "# Figure Rotation Audit",
        "",
        "This audit covers display-only rotations recorded in figure source-data CSV files.",
        "Model logits, features, confidence scores and GradCAM targets are computed from the unmodified cached tensors; rotations are applied only to rendered thumbnails or input/heatmap overlays.",
        "",
        f"- Source directory: `paper/figures/nature_source_data`",
        f"- Declared display-correction rules: {len(rule_rows)}",
        f"- Non-zero display rotations recorded: {len(rows)}",
        f"- Invalid rotation entries: {len(errors)}",
        "",
    ]
    if rule_rows:
        lines.extend(
            [
                "## Declared Rules",
                "",
                "| Dataset | Sample | Rotation |",
                "|---|---:|---:|",
            ]
        )
        for row in rule_rows:
            lines.append(f"| {row['dataset']} | {row['sample_index']} | {row['display_rotation_deg']} |")
        lines.append("")
    if rows:
        lines.extend(
            [
                "## Recorded Non-Zero Rotations",
                "",
                "| Source CSV | Figure | Dataset | Sample | Field | Rotation |",
                "|---|---|---|---:|---|---:|",
            ]
        )
        for row in rows:
            lines.append(
                f"| `{row['source_csv']}` | {row['figure']} | {row['dataset']} | "
                f"{row['sample_index']} | `{row['rotation_field']}` | {row['display_rotation_deg']} |"
            )
        lines.append("")
    if errors:
        lines.extend(["## Errors", ""])
        lines.extend(f"- {err}" for err in errors)
        lines.append("")
    else:
        lines.extend(
            [
                "## Validation",
                "",
                "- All recorded non-zero rotations use right-angle values: 90, 180 or 270 degrees.",
                "- Recorded rotations match the shared display-correction rule table.",
                "",
            ]
        )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows, errors = collect_rows()
    write_csv(rows)
    write_markdown(rows, errors)
    print(f"rotation rows: {len(rows)}")
    print(f"errors: {len(errors)}")
    print(f"wrote: {OUT_MD.relative_to(ROOT)}")
    print(f"wrote: {OUT_CSV.relative_to(ROOT)}")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
