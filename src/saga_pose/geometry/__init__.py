"""Geometry utilities for pose contracts and evaluation."""

from saga_pose.geometry.pnp import PnPResult, solve_pose_epnp
from saga_pose.geometry.projection import (
    clamp_bbox_xyxy,
    crop_resize_intrinsics,
    crop_resize_points,
    padded_square_bbox_xyxy,
    project_points,
    rotation_matrix_diagnostics,
    scale_intrinsics,
)

__all__ = [
    "PnPResult",
    "clamp_bbox_xyxy",
    "crop_resize_intrinsics",
    "crop_resize_points",
    "padded_square_bbox_xyxy",
    "project_points",
    "rotation_matrix_diagnostics",
    "scale_intrinsics",
    "solve_pose_epnp",
]
