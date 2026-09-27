"""Verified pose metrics, aggregation, and uncertainty estimates."""

from saga_pose.evaluation.bootstrap import BootstrapResult, paired_sequence_bootstrap
from saga_pose.evaluation.metrics import (
    aggregate_by_sequence,
    range_normalized_translation_error,
    rotation_error_degrees,
    threshold_success_rate,
    translation_error_meters,
)

__all__ = [
    "BootstrapResult",
    "aggregate_by_sequence",
    "paired_sequence_bootstrap",
    "range_normalized_translation_error",
    "rotation_error_degrees",
    "threshold_success_rate",
    "translation_error_meters",
]
