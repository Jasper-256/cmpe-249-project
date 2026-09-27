"""Spatial keypoint heatmaps for the reportable model family."""

from __future__ import annotations

from typing import Any, cast

import timm
import torch
from torch import Tensor, nn
from torch.nn import functional as F


def gaussian_keypoint_heatmaps(
    keypoints_xy: Tensor,
    *,
    image_size_wh: tuple[int, int] = (256, 256),
    heatmap_size_wh: tuple[int, int] = (64, 64),
    sigma: float = 1.5,
) -> tuple[Tensor, Tensor]:
    """Render continuous-center Gaussian targets and in-frame keypoint weights."""
    if keypoints_xy.ndim != 3 or keypoints_xy.shape[-1] != 2:
        raise ValueError("keypoints_xy must have shape (B, J, 2)")
    image_width, image_height = image_size_wh
    heatmap_width, heatmap_height = heatmap_size_wh
    if min(image_width, image_height, heatmap_width, heatmap_height) <= 0 or sigma <= 0:
        raise ValueError("image sizes and sigma must be positive")
    scale = keypoints_xy.new_tensor([heatmap_width / image_width, heatmap_height / image_height])
    centers = (keypoints_xy + 0.5) * scale - 0.5
    visible = (
        (centers[..., 0] >= 0.0)
        & (centers[..., 0] <= heatmap_width - 1)
        & (centers[..., 1] >= 0.0)
        & (centers[..., 1] <= heatmap_height - 1)
    )
    grid_y, grid_x = torch.meshgrid(
        torch.arange(heatmap_height, device=keypoints_xy.device, dtype=keypoints_xy.dtype),
        torch.arange(heatmap_width, device=keypoints_xy.device, dtype=keypoints_xy.dtype),
        indexing="ij",
    )
    squared_distance = (grid_x[None, None] - centers[..., 0, None, None]).square() + (
        grid_y[None, None] - centers[..., 1, None, None]
    ).square()
    heatmaps = torch.exp(-squared_distance / (2.0 * sigma**2))
    return heatmaps * visible[..., None, None], visible.to(keypoints_xy.dtype)


def decode_heatmap_argmax(
    heatmaps: Tensor,
    *,
    image_size_wh: tuple[int, int] = (256, 256),
) -> tuple[Tensor, Tensor]:
    """Decode heatmaps with the conventional local quarter-pixel adjustment."""
    if heatmaps.ndim != 4:
        raise ValueError("heatmaps must have shape (B, J, H, W)")
    batch_size, num_keypoints, height, width = heatmaps.shape
    flat = heatmaps.reshape(batch_size, num_keypoints, -1)
    confidence, indices = flat.max(dim=-1)
    x = indices.remainder(width)
    y = torch.div(indices, width, rounding_mode="floor")
    coordinates = torch.stack((x, y), dim=-1).to(heatmaps.dtype)

    interior = (x > 0) & (x < width - 1) & (y > 0) & (y < height - 1)
    batch_indices = torch.arange(batch_size, device=heatmaps.device)[:, None]
    joint_indices = torch.arange(num_keypoints, device=heatmaps.device)[None, :]
    x_minus = heatmaps[batch_indices, joint_indices, y, (x - 1).clamp_min(0)]
    x_plus = heatmaps[batch_indices, joint_indices, y, (x + 1).clamp_max(width - 1)]
    y_minus = heatmaps[batch_indices, joint_indices, (y - 1).clamp_min(0), x]
    y_plus = heatmaps[batch_indices, joint_indices, (y + 1).clamp_max(height - 1), x]
    adjustment = torch.stack((torch.sign(x_plus - x_minus), torch.sign(y_plus - y_minus)), dim=-1)
    coordinates = coordinates + 0.25 * adjustment * interior[..., None]

    image_width, image_height = image_size_wh
    scale = coordinates.new_tensor([image_width / width, image_height / height])
    coordinates = (coordinates + 0.5) * scale - 0.5
    return coordinates, confidence


class HRNetHeatmapRegressor(nn.Module):
    """Compact HRNet-W18-Small-v2 feature pyramid with a 64x64 heatmap head."""

    def __init__(
        self,
        num_keypoints: int,
        *,
        pretrained: bool = False,
        decoder_channels: int = 64,
    ) -> None:
        super().__init__()
        if num_keypoints <= 0 or decoder_channels <= 0:
            raise ValueError("num_keypoints and decoder_channels must be positive")
        self.num_keypoints = num_keypoints
        self.backbone = timm.create_model(
            "hrnet_w18_small_v2.ms_in1k",
            pretrained=pretrained,
            features_only=True,
        )
        channels = cast(Any, self.backbone).feature_info.channels()
        self.lateral = nn.ModuleList(
            nn.Conv2d(channel, decoder_channels, kernel_size=1) for channel in channels
        )
        self.head = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, decoder_channels),
            nn.GELU(),
            nn.Conv2d(decoder_channels, num_keypoints, kernel_size=1),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 4 or inputs.shape[1:] != (3, 256, 256):
            raise ValueError("inputs must have the frozen shape (B, 3, 256, 256)")
        features = cast(list[Tensor], self.backbone(inputs))
        target_size = (64, 64)
        fused = F.interpolate(self.lateral[0](features[0]), size=target_size, mode="nearest")
        for feature, projection in zip(features[1:], self.lateral[1:], strict=True):
            level = F.interpolate(projection(feature), size=target_size, mode="nearest")
            fused = fused + level
        return cast(Tensor, self.head(fused))
