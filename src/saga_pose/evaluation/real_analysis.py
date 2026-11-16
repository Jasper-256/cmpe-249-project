"""Per-frame real-test analysis shared by every evaluated method.

Learned models (``scripts/evaluate_real.py``) and the geometric baseline
(``scripts/geometric_fusion.py``) are scored by this one function, so their
overall, per-sequence, and author-slice numbers are directly comparable.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from saga_pose.data.real import slice_mask
from saga_pose.evaluation.metrics import aggregate_by_sequence, threshold_success_rate

SLICE_NAMES = ("harsh", "low-motion")


def summarize_errors(
    rotation: np.ndarray, translation: np.ndarray, valid: np.ndarray
) -> dict[str, float]:
    """Frame-level summary with PnP failures already counted in the errors."""
    return {
        "frames": int(rotation.size),
        "are_degrees_mean": float(np.mean(rotation)),
        "are_degrees_median": float(np.median(rotation)),
        "ate_m_mean": float(np.mean(translation)),
        "ate_m_median": float(np.median(translation)),
        "success_rotation_below_10_deg": threshold_success_rate(rotation, 10.0),
        "success_translation_below_10_mm": threshold_success_rate(translation, 0.01),
        "pnp_valid_fraction": float(np.mean(valid)),
    }


def analyze_real_frames(
    frames: Mapping[str, Any], slices: Mapping[str, Mapping[str, tuple[int, int]]]
) -> dict[str, Any]:
    """Overall, per-sequence, and harsh/low-motion slice metrics for one method."""
    rotation = np.asarray(frames["rotation_error_deg"])
    translation = np.asarray(frames["translation_error_m"])
    valid = np.asarray(frames["pnp_valid"])
    sequences = [str(sequence) for sequence in frames["sequence"]]
    numbers = np.asarray(frames["frame_number"])
    result: dict[str, Any] = {
        "overall_frames": summarize_errors(rotation, translation, valid),
        "sequence_mean_are_deg": aggregate_by_sequence(rotation, sequences),
        "sequence_mean_ate_m": aggregate_by_sequence(translation, sequences),
        "slices": {},
    }
    result["overall_sequence_mean"] = {
        "are_degrees": float(np.mean(list(result["sequence_mean_are_deg"].values()))),
        "ate_m": float(np.mean(list(result["sequence_mean_ate_m"].values()))),
    }
    sequence_array = np.asarray(sequences)
    unknown = set(slices) - set(sequences)
    if unknown:
        raise ValueError(f"slice sequences missing from predictions: {sorted(unknown)}")
    for slice_name in SLICE_NAMES:
        mask = np.zeros(rotation.size, dtype=bool)
        for sequence, ranges in slices.items():
            mask |= (sequence_array == sequence) & slice_mask(numbers, ranges[slice_name])
        if not mask.any():
            raise ValueError(f"empty {slice_name} slice")
        summary: dict[str, Any] = dict(
            summarize_errors(rotation[mask], translation[mask], valid[mask])
        )
        summary["sequence_mean_are_deg"] = aggregate_by_sequence(
            rotation[mask], list(sequence_array[mask])
        )
        summary["sequence_mean_ate_m"] = aggregate_by_sequence(
            translation[mask], list(sequence_array[mask])
        )
        result["slices"][slice_name] = summary
    return result
