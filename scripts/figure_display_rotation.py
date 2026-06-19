#!/usr/bin/env python
"""Shared display-only orientation corrections for manuscript image panels.

The angles here are used only immediately before plotting thumbnails or
GradCAM overlays. Cached tensors and model inputs must remain unchanged so the
reported logits, features and confidences stay traceable to the original data.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np


ALLOWED_ROTATIONS = {0, 90, 180, 270}

DISPLAY_ROTATIONS: dict[str, dict[int, int]] = {
    "AL6": {
        1: 270,
        15: 90,
        109: 270,
        110: 90,
        155: 90,
        204: 90,
        237: 270,
        264: 270,
        270: 270,
        322: 90,
        336: 90,
        412: 90,
    },
    "ASP_clean": {
        2078: 270,
    },
    "AS25_clean": {
        449: 270,
        464: 270,
        859: 270,
        1106: 270,
    },
}


def normalize_rotation(rotation_deg: int | float | str) -> int:
    angle = int(float(rotation_deg)) % 360
    if angle not in ALLOWED_ROTATIONS:
        raise ValueError(f"display rotation must be one of {sorted(ALLOWED_ROTATIONS)}; got {rotation_deg!r}")
    return angle


def display_rotation(dataset: str, index: int | str) -> int:
    """Return the display-only correction angle for a dataset sample."""
    return normalize_rotation(DISPLAY_ROTATIONS.get(str(dataset), {}).get(int(index), 0))


def rotate_display_array(arr: np.ndarray, rotation_deg: int | float | str) -> np.ndarray:
    """Rotate an already-rendered image array for display, not inference."""
    angle = normalize_rotation(rotation_deg)
    if angle == 0:
        return arr
    return np.ascontiguousarray(np.rot90(arr, angle // 90))


def iter_display_rotation_rules() -> Iterator[dict[str, object]]:
    for dataset in sorted(DISPLAY_ROTATIONS):
        for index, angle in sorted(DISPLAY_ROTATIONS[dataset].items()):
            yield {
                "dataset": dataset,
                "sample_index": index,
                "display_rotation_deg": normalize_rotation(angle),
            }
