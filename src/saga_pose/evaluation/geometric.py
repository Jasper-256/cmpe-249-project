"""FRESH-style learning-free geometric fusion of RGB and event keypoints.

Each landmark contributes two candidate 2D observations, one from the RGB-only
model and one from the event-only model. A single RANSAC PnP over all 2J
correspondences keeps the geometrically consistent subset, so a modality whose
keypoints are corrupted (e.g. by RGB saturation) is rejected per point; the
inlier set is refined with Levenberg-Marquardt.

This follows the FRESH paper's idea of learning-free test-time fusion of
separately trained keypoint networks, but it is not the published pipeline
(which adds a detector, temporal box logic, dropout uncertainty, and distance
checks); results are labelled "FRESH-style" accordingly. Failures receive the
FRESH default pose, exactly like every learned model.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from saga_pose.evaluation.pose import default_failure_pose
from saga_pose.geometry.projection import project_points

RANSAC_ITERATIONS = 500
RANSAC_CONFIDENCE = 0.999
RANSAC_SEED = 249
MIN_INLIERS = 6  # distinct landmarks
MAX_DEPTH_MM = 100_000.0


def ransac_pose_from_sets(
    keypoint_sets: Sequence[NDArray[np.float64]],
    landmarks: NDArray[np.float64],
    intrinsics: NDArray[np.float64],
    *,
    threshold_px: float,
) -> tuple[NDArray[np.float64], NDArray[np.bool_], NDArray[np.float64]]:
    """RANSAC EPnP over one or more (N, J, 2) keypoint sets for the same landmarks.

    One set is the single-model robust-solver control; two sets (RGB-only and
    event-only models) are the FRESH-style geometric fusion. Returns
    ``(poses_mm, valid, inlier_fraction)``.
    """
    sets = [np.asarray(keypoints, dtype=np.float64) for keypoints in keypoint_sets]
    if not sets or any(item.shape != sets[0].shape or item.ndim != 3 for item in sets):
        raise ValueError("keypoint sets must share shape (N, J, 2)")
    count, joints = sets[0].shape[:2]
    points = np.broadcast_to(np.asarray(landmarks, dtype=np.float64), (count, joints, 3))
    cameras = np.broadcast_to(np.asarray(intrinsics, dtype=np.float64), (count, 3, 3))
    landmark_ids = np.tile(np.arange(joints), len(sets))
    poses = np.repeat(default_failure_pose()[None], count, axis=0)
    valid = np.zeros(count, dtype=np.bool_)
    inlier_fraction = np.zeros(count, dtype=np.float64)
    for index in range(count):
        object_points = np.concatenate([points[index]] * len(sets))
        image_points = np.concatenate([item[index] for item in sets])
        finite = np.all(np.isfinite(image_points), axis=1)
        if finite.sum() < 4:
            continue
        cv2.setRNGSeed(RANSAC_SEED)  # deterministic hypothesis sampling per frame
        success, rotation_vector, translation, inliers = cv2.solvePnPRansac(
            object_points[finite],
            image_points[finite],
            cameras[index],
            None,
            iterationsCount=RANSAC_ITERATIONS,
            reprojectionError=threshold_px,
            confidence=RANSAC_CONFIDENCE,
            flags=cv2.SOLVEPNP_EPNP,
        )
        if not success or inliers is None:
            continue
        selected = inliers.ravel()
        # Count distinct landmarks: the same landmark seen by both models is one
        # geometric constraint, not two.
        if np.unique(landmark_ids[finite][selected]).size < MIN_INLIERS:
            continue
        rotation_vector, translation = cv2.solvePnPRefineLM(
            object_points[finite][selected],
            image_points[finite][selected],
            cameras[index],
            None,
            rotation_vector,
            translation,
        )
        pose = np.eye(4)
        pose[:3, :3], _ = cv2.Rodrigues(rotation_vector)
        pose[:3, 3] = np.ravel(translation)
        # Degenerate inlier sets can let LM diverge to absurd poses; such solutions
        # are failures (default pose), exactly like a failed PnP for learned models.
        if not np.all(np.isfinite(pose)) or not 0.0 < pose[2, 3] < MAX_DEPTH_MM:
            continue
        try:
            reprojected = project_points(
                object_points[finite][selected], pose, cameras[index]
            )  # also rejects behind-camera poses
        except ValueError:
            continue
        residual = np.sqrt(
            np.mean(np.sum((reprojected - image_points[finite][selected]) ** 2, axis=1))
        )
        if residual > threshold_px:
            continue
        poses[index] = pose
        valid[index] = True
        inlier_fraction[index] = len(selected) / (len(sets) * joints)
    return poses, valid, inlier_fraction


def fuse_keypoints_ransac(
    rgb_keypoints: NDArray[np.float64],
    event_keypoints: NDArray[np.float64],
    landmarks: NDArray[np.float64],
    intrinsics: NDArray[np.float64],
    *,
    threshold_px: float,
) -> tuple[NDArray[np.float64], NDArray[np.bool_], NDArray[np.float64]]:
    """FRESH-style fusion: RANSAC over both unimodal models' keypoints."""
    return ransac_pose_from_sets(
        [rgb_keypoints, event_keypoints], landmarks, intrinsics, threshold_px=threshold_px
    )
