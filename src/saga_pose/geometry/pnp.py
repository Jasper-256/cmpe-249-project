"""Shared Perspective-n-Point pose recovery for keypoint experiments."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from saga_pose.geometry.projection import project_points


@dataclass(frozen=True)
class PnPResult:
    """Object-to-camera pose and its reprojection residual."""

    object_to_camera: NDArray[np.float64]
    reprojection_rmse_px: float


def solve_pose_epnp(
    points_object: NDArray[np.float64],
    image_points: NDArray[np.float64],
    intrinsics: NDArray[np.float64],
    *,
    refine: bool = True,
) -> PnPResult | None:
    """Solve object-to-camera pose with EPnP and optional LM refinement.

    FRESH annotations and OpenCV use the same convention:
    ``X_camera = R @ X_object + t``. Translation retains the unit of
    ``points_object``; FRESH landmarks therefore produce millimetres.
    """
    object_points = np.asarray(points_object, dtype=np.float64)
    observations = np.asarray(image_points, dtype=np.float64)
    camera_matrix = np.asarray(intrinsics, dtype=np.float64)
    if object_points.ndim != 2 or object_points.shape[1] != 3:
        raise ValueError("points_object must have shape (N, 3)")
    if observations.shape != (object_points.shape[0], 2):
        raise ValueError("image_points must have shape (N, 2)")
    if object_points.shape[0] < 4:
        raise ValueError("At least four correspondences are required")
    if camera_matrix.shape != (3, 3):
        raise ValueError("intrinsics must have shape (3, 3)")
    if not all(
        np.all(np.isfinite(array)) for array in (object_points, observations, camera_matrix)
    ):
        raise ValueError("PnP inputs must be finite")

    success, rotation_vector, translation = cv2.solvePnP(
        object_points,
        observations,
        camera_matrix,
        None,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not success:
        return None
    if refine:
        rotation_vector, translation = cv2.solvePnPRefineLM(
            object_points,
            observations,
            camera_matrix,
            None,
            rotation_vector,
            translation,
        )
    rotation, _ = cv2.Rodrigues(rotation_vector)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation
    pose[:3, 3] = np.ravel(translation)
    try:
        reprojection = project_points(object_points, pose, camera_matrix)
    except ValueError:
        return None
    rmse = float(np.sqrt(np.mean((reprojection - observations) ** 2)))
    if not np.isfinite(rmse):
        return None
    return PnPResult(object_to_camera=pose, reprojection_rmse_px=rmse)
