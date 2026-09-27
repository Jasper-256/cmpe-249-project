from __future__ import annotations

import numpy as np

from saga_pose.analysis.saturation import (
    clipped_pixel_mask,
    crop_xyxy,
    event_activity_mask,
    patch_fraction_map,
)


def test_clipped_pixel_mask_uses_max_channel() -> None:
    rgb = np.array([[[253, 0, 0], [0, 254, 0]], [[0, 0, 255], [1, 2, 3]]], dtype=np.uint8)
    np.testing.assert_array_equal(
        clipped_pixel_mask(rgb, 254),
        [[False, True], [True, False]],
    )


def test_patch_fraction_map_area_averages() -> None:
    mask = np.zeros((4, 4), dtype=np.bool_)
    mask[:2, :2] = True
    np.testing.assert_allclose(patch_fraction_map(mask, 2), [[1.0, 0.0], [0.0, 0.0]])


def test_crop_xyxy_clamps_to_image() -> None:
    image = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    crop = crop_xyxy(image, (-2.0, 1.0, 9.0, 3.0))
    assert crop.shape == (2, 5, 3)


def test_event_activity_uses_neutral_gray() -> None:
    event = np.full((2, 2, 3), 127, dtype=np.uint8)
    event[0, 1] = 255
    np.testing.assert_array_equal(
        event_activity_mask(event),
        [[False, True], [False, False]],
    )
