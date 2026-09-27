from __future__ import annotations

import numpy as np

from saga_pose.geometry.projection import (
    clamp_bbox_xyxy,
    crop_resize_intrinsics,
    crop_resize_points,
    padded_square_bbox_xyxy,
    project_points,
    scale_intrinsics,
)


def test_padded_square_bbox_preserves_center_and_allows_border_padding() -> None:
    crop = padded_square_bbox_xyxy((-10.0, 20.0, 90.0, 70.0), padding_factor=1.2)

    assert crop == (-20, -15, 100, 105)
    assert crop[2] - crop[0] == crop[3] - crop[1]


def test_project_points_object_to_camera() -> None:
    points = np.array([[0.0, 0.0, 2.0], [1.0, -1.0, 2.0]], dtype=np.float64)
    pose = np.eye(4, dtype=np.float64)
    intrinsics = np.array(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    projected = project_points(points, pose, intrinsics)
    np.testing.assert_allclose(projected, [[50.0, 40.0], [100.0, -10.0]])


def test_scale_intrinsics_preserves_resized_projection() -> None:
    intrinsics = np.array(
        [[100.0, 0.0, 50.0], [0.0, 120.0, 40.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    scaled = scale_intrinsics(intrinsics, (200, 100), (100, 25))
    np.testing.assert_allclose(
        scaled,
        [[50.0, 0.0, 24.75], [0.0, 30.0, 9.625], [0.0, 0.0, 1.0]],
    )


def test_crop_resize_intrinsics_matches_transformed_projection() -> None:
    points = np.array([[0.0, 0.0, 2.0], [1.0, -1.0, 2.0]], dtype=np.float64)
    pose = np.eye(4, dtype=np.float64)
    intrinsics = np.array(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    crop = (10, 20, 110, 70)
    target_size = (200, 200)
    source_projection = project_points(points, pose, intrinsics)
    expected = crop_resize_points(source_projection, crop, target_size)
    transformed_intrinsics = crop_resize_intrinsics(intrinsics, crop, target_size)
    actual = project_points(points, pose, transformed_intrinsics)
    np.testing.assert_allclose(actual, expected)


def test_clamp_bbox_accepts_off_frame_fresh_box() -> None:
    assert clamp_bbox_xyxy((-5.2, 10.1, 810.4, 719.2), (800, 720)) == (0, 10, 800, 720)
