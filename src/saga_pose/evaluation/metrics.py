"""Pose metrics with explicit units and aggregation definitions."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.typing import NDArray

TranslationUnit = Literal["millimetres", "metres"]


def rotation_error_degrees(
    ground_truth: NDArray[np.float64],
    prediction: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute the geodesic angle on SO(3) for matrices shaped ``(..., 3, 3)``."""
    if ground_truth.shape != prediction.shape or ground_truth.shape[-2:] != (3, 3):
        raise ValueError("Rotation arrays must share shape (..., 3, 3)")
    relative = prediction @ np.swapaxes(ground_truth, -1, -2)
    trace = np.trace(relative, axis1=-2, axis2=-1)
    cosine = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return np.asarray(np.degrees(np.arccos(cosine)), dtype=np.float64)


def translation_error_meters(
    ground_truth: NDArray[np.float64],
    prediction: NDArray[np.float64],
    *,
    input_unit: TranslationUnit = "millimetres",
) -> NDArray[np.float64]:
    """Return absolute Euclidean translation error in metres."""
    if ground_truth.shape != prediction.shape or ground_truth.shape[-1] != 3:
        raise ValueError("Translation arrays must share shape (..., 3)")
    scale = 0.001 if input_unit == "millimetres" else 1.0
    errors = np.linalg.norm(ground_truth - prediction, axis=-1) * scale
    return np.asarray(errors, dtype=np.float64)


def range_normalized_translation_error(
    ground_truth: NDArray[np.float64],
    prediction: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return dimensionless translation error divided by ground-truth range."""
    if ground_truth.shape != prediction.shape or ground_truth.shape[-1] != 3:
        raise ValueError("Translation arrays must share shape (..., 3)")
    ranges = np.linalg.norm(ground_truth, axis=-1)
    if np.any(ranges <= 0.0):
        raise ValueError("Ground-truth translation range must be positive")
    errors = np.linalg.norm(ground_truth - prediction, axis=-1) / ranges
    return np.asarray(errors, dtype=np.float64)


def threshold_success_rate(errors: NDArray[np.float64], threshold: float) -> float:
    """Return the fraction of finite errors strictly below a positive threshold."""
    if errors.size == 0:
        raise ValueError("errors must not be empty")
    if threshold <= 0.0:
        raise ValueError("threshold must be positive")
    if not np.all(np.isfinite(errors)):
        raise ValueError("errors must be finite")
    return float(np.mean(errors < threshold))


def aggregate_by_sequence(
    values: NDArray[np.float64],
    sequence_ids: Sequence[str],
) -> dict[str, float]:
    """Average frame metrics within each sequence, the project's statistical unit."""
    flat_values = np.ravel(values)
    if flat_values.size != len(sequence_ids) or flat_values.size == 0:
        raise ValueError("values and sequence_ids must have the same nonzero length")
    if not np.all(np.isfinite(flat_values)):
        raise ValueError("values must be finite")
    grouped: dict[str, list[float]] = defaultdict(list)
    for sequence_id, value in zip(sequence_ids, flat_values, strict=True):
        grouped[sequence_id].append(float(value))
    return {key: float(np.mean(grouped[key])) for key in sorted(grouped)}
