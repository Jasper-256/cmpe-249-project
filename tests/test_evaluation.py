from __future__ import annotations

import numpy as np
import pytest

from saga_pose.evaluation.bootstrap import paired_sequence_bootstrap
from saga_pose.evaluation.metrics import (
    aggregate_by_sequence,
    range_normalized_translation_error,
    rotation_error_degrees,
    threshold_success_rate,
    translation_error_meters,
)


def _rotation_z(degrees: float) -> np.ndarray:
    radians = np.radians(degrees)
    cosine, sine = np.cos(radians), np.sin(radians)
    return np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def test_rotation_error_known_angles_and_clamping() -> None:
    ground_truth = np.stack([np.eye(3), np.eye(3), np.eye(3)]).astype(np.float64)
    prediction = np.stack([np.eye(3), _rotation_z(90.0), _rotation_z(180.0)])
    np.testing.assert_allclose(
        rotation_error_degrees(ground_truth, prediction),
        [0.0, 90.0, 180.0],
        atol=1e-7,
    )


def test_translation_metrics_keep_absolute_and_normalized_names_distinct() -> None:
    ground_truth = np.array([[0.0, 0.0, 1000.0]], dtype=np.float64)
    prediction = np.array([[6.0, 8.0, 1000.0]], dtype=np.float64)
    np.testing.assert_allclose(translation_error_meters(ground_truth, prediction), [0.01])
    np.testing.assert_allclose(
        range_normalized_translation_error(ground_truth, prediction),
        [0.01],
    )
    assert threshold_success_rate(np.array([0.009, 0.01]), 0.01) == 0.5


def test_sequence_aggregation_and_paired_bootstrap_are_deterministic() -> None:
    candidate = aggregate_by_sequence(
        np.array([1.0, 3.0, 2.0, 4.0]),
        ["a", "a", "b", "b"],
    )
    assert candidate == {"a": 2.0, "b": 3.0}
    baseline = {"a": 3.0, "b": 4.0}
    first = paired_sequence_bootstrap(candidate, baseline, resamples=1000, seed=7)
    second = paired_sequence_bootstrap(candidate, baseline, resamples=1000, seed=7)
    assert first == second
    assert first.observed_mean_difference == -1.0
    assert first.confidence_interval == pytest.approx((-1.0, -1.0))
    assert first.probability_candidate_better == 1.0


def test_paired_bootstrap_rejects_mismatched_sequences() -> None:
    with pytest.raises(ValueError, match="same nonempty sequence keys"):
        paired_sequence_bootstrap({"a": 1.0}, {"b": 1.0})
