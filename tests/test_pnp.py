"""Tests for the shared PnP pose-recovery contract."""

from __future__ import annotations

import numpy as np
import pytest

from saga_pose.evaluation.metrics import rotation_error_degrees
from saga_pose.geometry.pnp import solve_pose_epnp
from saga_pose.geometry.projection import project_points


def test_epnp_recovers_object_to_camera_pose_and_units() -> None:
    points = np.array(
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
    angle = np.deg2rad(18.0)
    rotation = np.array(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ]
    )
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = np.array([35.0, -20.0, 1800.0])
    intrinsics = np.array([[620.0, 0.0, 128.0], [0.0, 615.0, 128.0], [0.0, 0.0, 1.0]])
    image_points = project_points(points, pose, intrinsics)

    result = solve_pose_epnp(points, image_points, intrinsics)

    assert result is not None
    assert result.reprojection_rmse_px < 1e-6
    assert (
        rotation_error_degrees(pose[None, :3, :3], result.object_to_camera[None, :3, :3])[0] < 1e-5
    )
    assert result.object_to_camera[:3, 3] == pytest.approx(pose[:3, 3], abs=1e-5)


def test_epnp_rejects_too_few_correspondences() -> None:
    with pytest.raises(ValueError, match="four"):
        solve_pose_epnp(np.zeros((3, 3)), np.zeros((3, 2)), np.eye(3))
