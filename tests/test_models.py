from __future__ import annotations

import pytest
import torch
from transformers import SegformerConfig

from saga_pose.models.heatmaps import (
    HRNetHeatmapRegressor,
    SegFormerHeatmapRegressor,
    TimmPyramidHeatmapRegressor,
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


def test_generic_pyramid_heatmap_regressor_shape() -> None:
    model = TimmPyramidHeatmapRegressor(
        num_keypoints=4,
        backbone_name="efficientnet_b0",
        pretrained=False,
    )
    output = model(torch.rand(1, 3, 256, 256))
    assert output.shape == (1, 4, 64, 64)


def test_segformer_heatmap_regressor_shape() -> None:
    config = SegformerConfig(
        depths=[1, 1, 1, 1],
        hidden_sizes=[8, 16, 32, 64],
        num_attention_heads=[1, 1, 2, 4],
    )
    model = SegFormerHeatmapRegressor(num_keypoints=4, pretrained=False, config=config)
    output = model(torch.rand(1, 3, 256, 256))
    assert output.shape == (1, 4, 64, 64)


def test_decoder_levels_use_frozen_nearest_resampling() -> None:
    from saga_pose.models.heatmaps import resize_to_heatmap_grid

    feature = torch.arange(128 * 128, dtype=torch.float32).reshape(1, 1, 128, 128)
    resized = resize_to_heatmap_grid(feature)
    assert resized.shape == (1, 1, 64, 64)
    assert torch.equal(resized, feature[:, :, ::2, ::2])
    assert resize_to_heatmap_grid(torch.ones(1, 1, 16, 16)).shape == (1, 1, 64, 64)


def test_backbones_receive_their_own_pretraining_normalization() -> None:
    hrnet = HRNetHeatmapRegressor(num_keypoints=4, pretrained=False)
    assert hrnet.input_mean == pytest.approx((0.485, 0.456, 0.406))
    mobilevit = TimmPyramidHeatmapRegressor(
        num_keypoints=4, backbone_name="mobilevitv2_100.cvnets_in1k", pretrained=False
    )
    assert mobilevit.input_mean == (0.0, 0.0, 0.0)
    assert mobilevit.input_std == (1.0, 1.0, 1.0)
    inputs = torch.rand(1, 3, 8, 8)
    assert torch.equal(mobilevit.normalize(inputs), inputs)
