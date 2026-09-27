"""Definitions for the pre-training RGB clipping feasibility screen."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray


def clipped_pixel_mask(rgb: NDArray[np.uint8], threshold: int) -> NDArray[np.bool_]:
    """Return pixels whose maximum RGB channel reaches an 8-bit threshold."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must have shape (H, W, 3)")
    if not 0 <= threshold <= 255:
        raise ValueError("threshold must lie in [0, 255]")
    return np.max(rgb, axis=2) >= threshold


def patch_fraction_map(mask: NDArray[np.bool_], grid_size: int = 16) -> NDArray[np.float64]:
    """Area-average a binary mask onto a square patch grid."""
    if mask.ndim != 2:
        raise ValueError("mask must have shape (H, W)")
    if grid_size <= 0:
        raise ValueError("grid_size must be positive")
    return cv2.resize(
        mask.astype(np.float32),
        (grid_size, grid_size),
        interpolation=cv2.INTER_AREA,
    ).astype(np.float64)


def crop_xyxy(
    image: NDArray[np.uint8],
    bbox_xyxy: tuple[float, float, float, float],
) -> NDArray[np.uint8]:
    """Crop an image to a clamped XYXY annotation box."""
    height, width = image.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    left = max(0, min(width - 1, int(np.floor(x1))))
    top = max(0, min(height - 1, int(np.floor(y1))))
    right = max(left + 1, min(width, int(np.ceil(x2))))
    bottom = max(top + 1, min(height, int(np.ceil(y2))))
    return image[top:bottom, left:right]


def resize_rgb(rgb: NDArray[np.uint8], size_wh: tuple[int, int]) -> NDArray[np.uint8]:
    """Resize an RGB array with the bilinear interpolation used by the audit."""
    resized = cv2.resize(rgb, size_wh, interpolation=cv2.INTER_LINEAR)
    return np.asarray(resized, dtype=np.uint8)


def event_activity_mask(
    event_rgb: NDArray[np.uint8], neutral_value: int = 127
) -> NDArray[np.bool_]:
    """Detect non-neutral pixels in FRESH's three-level event-frame encoding."""
    if event_rgb.ndim != 3 or event_rgb.shape[2] != 3:
        raise ValueError("event_rgb must have shape (H, W, 3)")
    return np.any(event_rgb != neutral_value, axis=2)
