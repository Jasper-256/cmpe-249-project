"""Keypoint-to-pose evaluation shared by every learned variant.

PnP failures are never silently dropped. Following the FRESH paper, a frame
whose solver fails (or whose keypoints are non-finite) receives the default
pose: a zero rotation vector (identity rotation) and a zero translation. Those
frames therefore contribute large errors to the primary ``all_frames`` metrics.
Valid-only metrics are reported separately as a solver diagnostic.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from saga_pose.evaluation.metrics import (
    range_normalized_translation_error,
    rotation_error_degrees,
    threshold_success_rate,
    translation_error_meters,
)
from saga_pose.geometry.pnp import solve_pose_epnp

FAILURE_POLICY = "FRESH default pose: identity rotation (zero rotation vector), zero translation"
ROTATION_SUCCESS_DEGREES = 10.0
TRANSLATION_SUCCESS_METRES = 0.01


def default_failure_pose() -> NDArray[np.float64]:
    """Return the FRESH default pose assigned to failed PnP solves."""
    return np.eye(4, dtype=np.float64)


def recover_poses(
    keypoints_xy: NDArray[np.float64],
    landmarks_object_mm: NDArray[np.float64],
    intrinsics: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.bool_], NDArray[np.float64]]:
    """Solve EPnP+LM per frame and substitute the default pose on failure.

    Returns ``(poses, valid, reprojection_rmse_px)``; reprojection is NaN for
    failed frames.
    """
    keypoints = np.asarray(keypoints_xy, dtype=np.float64)
    if keypoints.ndim != 3 or keypoints.shape[-1] != 2:
        raise ValueError("keypoints_xy must have shape (N, J, 2)")
    count = keypoints.shape[0]
    landmarks = np.asarray(landmarks_object_mm, dtype=np.float64)
    if landmarks.ndim == 2:
        landmarks = np.broadcast_to(landmarks, (count, *landmarks.shape))
    camera = np.asarray(intrinsics, dtype=np.float64)
    if camera.ndim == 2:
        camera = np.broadcast_to(camera, (count, 3, 3))
    if landmarks.shape != (count, keypoints.shape[1], 3) or camera.shape != (count, 3, 3):
        raise ValueError("landmarks and intrinsics must match the keypoint batch")

    poses = np.repeat(default_failure_pose()[None], count, axis=0)
    valid = np.zeros(count, dtype=np.bool_)
    reprojection = np.full(count, np.nan, dtype=np.float64)
    for index in range(count):
        if not all(
            np.all(np.isfinite(array))
            for array in (keypoints[index], landmarks[index], camera[index])
        ):
            continue
        result = solve_pose_epnp(landmarks[index], keypoints[index], camera[index])
        if result is None or not np.all(np.isfinite(result.object_to_camera)):
            continue
        poses[index] = result.object_to_camera
        valid[index] = True
        reprojection[index] = result.reprojection_rmse_px
    return poses, valid, reprojection


def _summaries(
    rotation_errors: NDArray[np.float64],
    translation_errors: NDArray[np.float64],
    normalized_errors: NDArray[np.float64],
) -> dict[str, float]:
    return {
        "are_degrees_mean": float(np.mean(rotation_errors)),
        "are_degrees_median": float(np.median(rotation_errors)),
        "absolute_translation_error_m_mean": float(np.mean(translation_errors)),
        "absolute_translation_error_m_median": float(np.median(translation_errors)),
        "range_normalized_translation_error_mean": float(np.mean(normalized_errors)),
        "success_rotation_below_10_deg": threshold_success_rate(
            rotation_errors, ROTATION_SUCCESS_DEGREES
        ),
        "success_translation_below_10_mm": threshold_success_rate(
            translation_errors, TRANSLATION_SUCCESS_METRES
        ),
        "success_joint_10_deg_10_mm": float(
            np.mean(
                (rotation_errors < ROTATION_SUCCESS_DEGREES)
                & (translation_errors < TRANSLATION_SUCCESS_METRES)
            )
        ),
    }


def pose_error_arrays(
    ground_truth_poses_mm: NDArray[np.float64],
    predicted_poses_mm: NDArray[np.float64],
) -> dict[str, NDArray[np.float64]]:
    """Return per-frame ARE (deg), ATE (m), and range-normalized translation error."""
    ground_truth = np.asarray(ground_truth_poses_mm, dtype=np.float64)
    prediction = np.asarray(predicted_poses_mm, dtype=np.float64)
    return {
        "rotation_error_deg": rotation_error_degrees(
            ground_truth[:, :3, :3], prediction[:, :3, :3]
        ),
        "translation_error_m": translation_error_meters(
            ground_truth[:, :3, 3], prediction[:, :3, 3]
        ),
        "range_normalized_translation_error": range_normalized_translation_error(
            ground_truth[:, :3, 3], prediction[:, :3, 3]
        ),
    }


def summarize_pose_errors(
    ground_truth_poses_mm: NDArray[np.float64],
    predicted_poses_mm: NDArray[np.float64],
    valid: NDArray[np.bool_],
    reprojection_rmse_px: NDArray[np.float64] | None = None,
) -> dict[str, Any]:
    """Summarize pose errors with failures included (primary) and valid-only (diagnostic)."""
    valid_mask = np.asarray(valid, dtype=np.bool_)
    count = valid_mask.size
    if count == 0:
        raise ValueError("at least one frame is required")
    if np.shape(ground_truth_poses_mm) != (count, 4, 4) or np.shape(predicted_poses_mm) != (
        count,
        4,
        4,
    ):
        raise ValueError("poses must have shape (N, 4, 4) matching the validity mask")
    errors = pose_error_arrays(ground_truth_poses_mm, predicted_poses_mm)
    summary: dict[str, Any] = {
        "frame_count": count,
        "pnp_valid_count": int(valid_mask.sum()),
        "pnp_valid_fraction": float(valid_mask.mean()),
        "failure_policy": FAILURE_POLICY,
        "all_frames": _summaries(
            errors["rotation_error_deg"],
            errors["translation_error_m"],
            errors["range_normalized_translation_error"],
        ),
        "valid_only": None,
    }
    if valid_mask.any():
        summary["valid_only"] = _summaries(
            errors["rotation_error_deg"][valid_mask],
            errors["translation_error_m"][valid_mask],
            errors["range_normalized_translation_error"][valid_mask],
        )
        if reprojection_rmse_px is not None:
            summary["valid_only"]["reprojection_rmse_px_mean"] = float(
                np.mean(np.asarray(reprojection_rmse_px)[valid_mask])
            )
    return summary
