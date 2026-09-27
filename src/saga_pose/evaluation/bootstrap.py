"""Paired sequence-level bootstrap for lower-is-better pose metrics."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BootstrapResult:
    """Summary of candidate-minus-baseline sequence effects."""

    observed_mean_difference: float
    confidence_interval: tuple[float, float]
    probability_candidate_better: float
    confidence_level: float
    resamples: int
    seed: int
    sequence_count: int


def paired_sequence_bootstrap(
    candidate: Mapping[str, float],
    baseline: Mapping[str, float],
    *,
    resamples: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 249,
) -> BootstrapResult:
    """Bootstrap the mean paired effect over complete sequences.

    The reported effect is ``candidate - baseline``. For error metrics, negative
    values favor the candidate.
    """
    if set(candidate) != set(baseline) or not candidate:
        raise ValueError("candidate and baseline must have the same nonempty sequence keys")
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie in (0, 1)")

    keys = sorted(candidate)
    differences = np.asarray(
        [candidate[key] - baseline[key] for key in keys],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(differences)):
        raise ValueError("sequence metrics must be finite")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(keys), size=(resamples, len(keys)))
    bootstrap_means = np.mean(differences[indices], axis=1)
    tail = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(bootstrap_means, [tail, 1.0 - tail])
    return BootstrapResult(
        observed_mean_difference=float(np.mean(differences)),
        confidence_interval=(float(lower), float(upper)),
        probability_candidate_better=float(np.mean(bootstrap_means < 0.0)),
        confidence_level=confidence_level,
        resamples=resamples,
        seed=seed,
        sequence_count=len(keys),
    )
