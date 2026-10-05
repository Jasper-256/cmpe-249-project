"""Camera projection helpers with explicit object-to-camera conventions."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

BBoxXYXY = tuple[float, float, float, float]
IntegerBBoxXYXY = tuple[int, int, int, int]


def project_points(
    points_object: NDArray[np.float64],
    object_to_camera: NDArray[np.float64],
    intrinsics: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Project object-frame points using ``X_camera = R X_object + t``."""
    if points_object.ndim != 2 or points_object.shape[1] != 3:
        raise ValueError("points_object must have shape (N, 3)")
    if object_to_camera.shape != (4, 4):
        raise ValueError("object_to_camera must have shape (4, 4)")
    if intrinsics.shape != (3, 3):
        raise ValueError("intrinsics must have shape (3, 3)")

    camera_points = points_object @ object_to_camera[:3, :3].T + object_to_camera[:3, 3]
    if np.any(camera_points[:, 2] <= 0.0):
        raise ValueError("All projected points must have positive camera-frame depth")
    homogeneous = camera_points @ intrinsics.T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def rotation_matrix_diagnostics(rotation: NDArray[np.float64]) -> tuple[float, float]:
    """Return maximum orthogonality residual and determinant."""
    if rotation.shape != (3, 3):
        raise ValueError("rotation must have shape (3, 3)")
    orthogonality_error = float(np.max(np.abs(rotation.T @ rotation - np.eye(3))))
    return orthogonality_error, float(np.linalg.det(rotation))


def scale_intrinsics(
    intrinsics: NDArray[np.float64],
    source_size_wh: tuple[int, int],
    target_size_wh: tuple[int, int],
) -> NDArray[np.float64]:
    """Scale pinhole intrinsics for a direct, uncropped image resize."""
    if intrinsics.shape != (3, 3):
        raise ValueError("intrinsics must have shape (3, 3)")
    source_width, source_height = source_size_wh
    target_width, target_height = target_size_wh
    if min(source_width, source_height, target_width, target_height) <= 0:
        raise ValueError("Image dimensions must be positive")
    return crop_resize_intrinsics(
        intrinsics,
        (0, 0, source_width, source_height),
        target_size_wh,
    )


def clamp_bbox_xyxy(
    bbox_xyxy: BBoxXYXY,
    image_size_wh: tuple[int, int],
) -> IntegerBBoxXYXY:
    """Convert a possibly off-frame box into a nonempty integer image crop."""
    width, height = image_size_wh
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    x1, y1, x2, y2 = bbox_xyxy
    if not np.all(np.isfinite(np.asarray(bbox_xyxy))) or x2 <= x1 or y2 <= y1:
        raise ValueError("bbox_xyxy must be finite and have positive area")
    left = max(0, min(width - 1, int(np.floor(x1))))
    top = max(0, min(height - 1, int(np.floor(y1))))
    right = max(left + 1, min(width, int(np.ceil(x2))))
    bottom = max(top + 1, min(height, int(np.ceil(y2))))
    return left, top, right, bottom


def padded_square_bbox_xyxy(
    bbox_xyxy: BBoxXYXY,
    *,
    padding_factor: float = 1.2,
) -> IntegerBBoxXYXY:
    """Return a center-preserving padded square crop, allowing image padding.

    Coordinates are intentionally not clamped to an image boundary. Callers can
    pad outside-image regions while retaining a square field of view and a
    geometrically consistent pixel transform.
    """
    x1, y1, x2, y2 = bbox_xyxy
    if not np.all(np.isfinite(np.asarray(bbox_xyxy))) or x2 <= x1 or y2 <= y1:
        raise ValueError("bbox_xyxy must be finite and have positive area")
    if not np.isfinite(padding_factor) or padding_factor < 1.0:
        raise ValueError("padding_factor must be finite and at least 1")
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    side = max(x2 - x1, y2 - y1) * padding_factor
    left = int(np.floor(center_x - side / 2.0))
    top = int(np.floor(center_y - side / 2.0))
    right = int(np.ceil(center_x + side / 2.0))
    bottom = int(np.ceil(center_y + side / 2.0))
    return left, top, right, bottom


def crop_resize_points(
    points_xy: NDArray[np.float64],
    crop_xyxy: IntegerBBoxXYXY,
    target_size_wh: tuple[int, int],
) -> NDArray[np.float64]:
    """Map pixel centers through an integer crop and a half-pixel-center resize.

    Pillow BILINEAR and OpenCV INTER_LINEAR share this geometric mapping even
    though Pillow additionally antialiases when downsampling.
    """
    if points_xy.ndim != 2 or points_xy.shape[1] != 2:
        raise ValueError("points_xy must have shape (N, 2)")
    left, top, right, bottom = crop_xyxy
    target_width, target_height = target_size_wh
    if right <= left or bottom <= top or min(target_width, target_height) <= 0:
        raise ValueError("Crop and target dimensions must be positive")
    scale_x = target_width / (right - left)
    scale_y = target_height / (bottom - top)
    transformed = points_xy.copy()
    transformed[:, 0] = (points_xy[:, 0] - left + 0.5) * scale_x - 0.5
    transformed[:, 1] = (points_xy[:, 1] - top + 0.5) * scale_y - 0.5
    return transformed


def crop_resize_intrinsics(
    intrinsics: NDArray[np.float64],
    crop_xyxy: IntegerBBoxXYXY,
    target_size_wh: tuple[int, int],
) -> NDArray[np.float64]:
    """Update intrinsics for crop + resize using pixel-center coordinates."""
    if intrinsics.shape != (3, 3):
        raise ValueError("intrinsics must have shape (3, 3)")
    left, top, right, bottom = crop_xyxy
    target_width, target_height = target_size_wh
    if right <= left or bottom <= top or min(target_width, target_height) <= 0:
        raise ValueError("Crop and target dimensions must be positive")
    scale_x = target_width / (right - left)
    scale_y = target_height / (bottom - top)
    pixel_transform = np.array(
        [
            [scale_x, 0.0, scale_x * (0.5 - left) - 0.5],
            [0.0, scale_y, scale_y * (0.5 - top) - 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return pixel_transform @ intrinsics
