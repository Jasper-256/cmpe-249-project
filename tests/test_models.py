from __future__ import annotations

import pytest
import torch

from saga_pose.models.heatmaps import (
    HRNetHeatmapRegressor,
    decode_heatmap_argmax,
    gaussian_keypoint_heatmaps,
)
from saga_pose.models.keypoints import PilotKeypointRegressor


def test_pilot_keypoint_regressor_shape_and_backward() -> None:
    model = PilotKeypointRegressor(num_keypoints=18)
    inputs = torch.rand(2, 3, 256, 256)
    output = model(inputs)
    assert output.shape == (2, 18, 2)
    output.square().mean().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_pilot_keypoint_regressor_rejects_bad_shape() -> None:
    model = PilotKeypointRegressor(num_keypoints=4)
    with pytest.raises(ValueError, match="shape"):
        model(torch.rand(2, 1, 64, 64))


def test_gaussian_heatmap_decode_tracks_keypoints() -> None:
    keypoints = torch.tensor([[[40.0, 80.0], [201.0, 159.0]]])

    heatmaps, visibility = gaussian_keypoint_heatmaps(keypoints)
    decoded, confidence = decode_heatmap_argmax(heatmaps)

    assert heatmaps.shape == (1, 2, 64, 64)
    assert torch.all(visibility == 1)
    assert torch.max(torch.abs(decoded - keypoints)) <= 1.1
    assert torch.all(confidence > 0.9)


def test_hrnet_heatmap_regressor_shape_and_backward() -> None:
    model = HRNetHeatmapRegressor(num_keypoints=18, pretrained=False)
    inputs = torch.rand(1, 3, 256, 256)

    output = model(inputs)
    output.mean().backward()

    assert output.shape == (1, 18, 64, 64)
    assert all(parameter.grad is not None for parameter in model.head.parameters())
