"""PnP failures must count against pose metrics under the FRESH default policy."""

from __future__ import annotations

import numpy as np
import pytest

from saga_pose.evaluation.pose import (
    default_failure_pose,
    recover_poses,
    summarize_pose_errors,
)
from saga_pose.geometry.projection import project_points

POINTS = np.array(
    [
        [-120.0, -80.0, -50.0],
        [120.0, -80.0, -50.0],
        [120.0, 80.0, -50.0],
        [-120.0, 80.0, -50.0],
        [-90.0, -60.0, 70.0],
        [90.0, -60.0, 70.0],
        [90.0, 60.0, 70.0],
        [-90.0, 60.0, 70.0],
    ],
    dtype=np.float64,
)
INTRINSICS = np.array([[620.0, 0.0, 128.0], [0.0, 615.0, 128.0], [0.0, 0.0, 1.0]])


def _pose() -> np.ndarray:
    angle = np.deg2rad(18.0)
    pose = np.eye(4)
    pose[:3, :3] = np.array(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ]
    )
    pose[:3, 3] = [35.0, -20.0, 1800.0]
    return pose


def test_failed_frames_receive_default_pose_and_count_in_primary_metrics() -> None:
    ground_truth = np.stack([_pose(), _pose()])
    keypoints = np.stack([project_points(POINTS, _pose(), INTRINSICS)] * 2)
    keypoints[1, 0, 0] = np.nan  # Non-finite prediction is a failure, not a skip.

    poses, valid, reprojection = recover_poses(keypoints, POINTS, INTRINSICS)

    assert valid.tolist() == [True, False]
    np.testing.assert_allclose(poses[1], default_failure_pose())
    assert np.isnan(reprojection[1])

    summary = summarize_pose_errors(ground_truth, poses, valid, reprojection)
    assert summary["pnp_valid_fraction"] == 0.5
    # The valid frame is exact; the failed frame is scored against identity/zero.
    assert summary["valid_only"]["are_degrees_median"] == pytest.approx(0.0, abs=1e-5)
    assert summary["all_frames"]["are_degrees_mean"] == pytest.approx(9.0, abs=1e-4)
    expected_failed_ate = np.linalg.norm(_pose()[:3, 3]) / 1000.0
    assert summary["all_frames"]["absolute_translation_error_m_mean"] == pytest.approx(
        expected_failed_ate / 2.0, rel=1e-6
    )
    assert summary["all_frames"]["success_joint_10_deg_10_mm"] == 0.5
    assert summary["valid_only"]["success_joint_10_deg_10_mm"] == 1.0


def test_all_failures_leave_valid_only_empty() -> None:
    keypoints = np.full((1, POINTS.shape[0], 2), np.nan)
    poses, valid, _ = recover_poses(keypoints, POINTS, INTRINSICS)
    summary = summarize_pose_errors(_pose()[None], poses, valid)
    assert summary["pnp_valid_count"] == 0
    assert summary["valid_only"] is None
    assert summary["all_frames"]["success_rotation_below_10_deg"] == 0.0
