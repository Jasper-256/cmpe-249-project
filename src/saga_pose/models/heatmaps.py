"""Spatial keypoint heatmaps for the reportable model family."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import timm
import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _local_pretrained_file(model_name: str) -> Path | None:
    """Return a cached timm weight file, bypassing the Hugging Face hub client.

    On this Windows host Application Control blocks the unsigned OpenSSL DLL that
    ``huggingface_hub`` imports, so weights already in the local hub cache are
    loaded directly from disk. The file is byte-identical to the hub download.
    """
    hub = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache" / "huggingface" / "hub"))
    for filename in ("model.safetensors", "pytorch_model.bin"):
        matches = sorted(
            hub.glob(f"models--timm--{model_name}/snapshots/*/{filename}"),
            key=lambda path: path.stat().st_mtime,
        )
        if matches:
            return matches[-1]  # newest snapshot
    return None


def create_timm_backbone(model_name: str, *, pretrained: bool, **kwargs: Any) -> nn.Module:
    """Create a ``features_only`` timm backbone, preferring locally cached weights."""
    if pretrained:
        local = _local_pretrained_file(model_name)
        if local is not None:
            kwargs["pretrained_cfg_overlay"] = {"file": str(local)}
    model: nn.Module = timm.create_model(
        model_name, pretrained=pretrained, features_only=True, **kwargs
    )
    return model


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def normalize_inputs(inputs: Tensor, mean: Sequence[float], std: Sequence[float]) -> Tensor:
    """Apply per-channel normalization to ``[0, 1]`` images shaped ``(B, 3, H, W)``."""
    mean_tensor = inputs.new_tensor(tuple(mean))[None, :, None, None]
    std_tensor = inputs.new_tensor(tuple(std))[None, :, None, None]
    return (inputs - mean_tensor) / std_tensor


HEATMAP_SIZE = 64


def resize_to_heatmap_grid(feature: Tensor, size: int = HEATMAP_SIZE) -> Tensor:
    """Bring a pyramid level to ``size x size`` with nearest-neighbour resampling.

    This is the frozen HRNet-reference decoder. Average-pooling the stride-2
    level instead was tested on 2026-10-05 and slowed the fixed 16-sample RGB
    overfit from 1.70 px to 6.26 px final RMSE (same seed/LR), so the sharper
    nearest sampling of the 1x1-projected level is retained for every variant.
    """
    if feature.shape[-2:] == (size, size):
        return feature
    return F.interpolate(feature, size=(size, size), mode="nearest")


def fuse_pyramid(features: Sequence[Tensor], lateral: nn.ModuleList) -> Tensor:
    """Sum 1x1-projected pyramid levels on the shared heatmap grid."""
    fused: Tensor | None = None
    for feature, projection in zip(features, lateral, strict=True):
        level = resize_to_heatmap_grid(projection(feature))
        fused = level if fused is None else fused + level
    if fused is None:
        raise ValueError("at least one pyramid level is required")
    return fused


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


class TimmPyramidHeatmapRegressor(nn.Module):
    """A timm feature pyramid with the project's shared 64x64 heatmap decoder."""

    def __init__(
        self,
        num_keypoints: int,
        *,
        backbone_name: str,
        pretrained: bool = False,
        decoder_channels: int = 64,
        backbone_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if num_keypoints <= 0 or decoder_channels <= 0:
            raise ValueError("num_keypoints and decoder_channels must be positive")
        self.num_keypoints = num_keypoints
        self.backbone_name = backbone_name
        self.backbone = create_timm_backbone(
            backbone_name, pretrained=pretrained, **(backbone_kwargs or {})
        )
        channels = tuple(cast(Any, self.backbone).feature_info.channels())
        self.feature_channels = channels
        # Each pretrained package is fed the normalization it was trained with
        # (for example, MobileViTv2/cvnets expects unnormalized [0, 1] inputs).
        pretrained_cfg = cast(dict[str, Any], getattr(self.backbone, "pretrained_cfg", {}))
        self.input_mean: tuple[float, ...] = tuple(pretrained_cfg.get("mean", IMAGENET_MEAN))
        self.input_std: tuple[float, ...] = tuple(pretrained_cfg.get("std", IMAGENET_STD))
        self.lateral = nn.ModuleList(
            nn.Conv2d(channel, decoder_channels, kernel_size=1) for channel in channels
        )
        self.head = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, decoder_channels),
            nn.GELU(),
            nn.Conv2d(decoder_channels, num_keypoints, kernel_size=1),
        )

    def pyramid_features(self, inputs: Tensor) -> list[Tensor]:
        """Return pyramid levels in channels-first layout."""
        features = cast(list[Tensor], self.backbone(inputs))
        channels_first: list[Tensor] = []
        for feature, channels in zip(features, self.feature_channels, strict=True):
            if feature.shape[1] != channels:
                if feature.shape[-1] != channels:
                    raise ValueError(
                        "backbone feature channel layout does not match feature metadata"
                    )
                feature = feature.permute(0, 3, 1, 2).contiguous()
            channels_first.append(feature)
        return channels_first

    def normalize(self, inputs: Tensor) -> Tensor:
        """Normalize ``[0, 1]`` inputs with this backbone's pretraining statistics."""
        return normalize_inputs(inputs, self.input_mean, self.input_std)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 4 or inputs.shape[1:] != (3, 256, 256):
            raise ValueError("inputs must have the frozen shape (B, 3, 256, 256)")
        return cast(Tensor, self.head(fuse_pyramid(self.pyramid_features(inputs), self.lateral)))


class SegFormerHeatmapRegressor(nn.Module):
    """SegFormer MiT pyramid with the same project heatmap decoder."""

    def __init__(
        self,
        num_keypoints: int,
        *,
        pretrained: bool = False,
        decoder_channels: int = 64,
        model_id: str = "nvidia/mit-b0",
        config: Any = None,
    ) -> None:
        super().__init__()
        if num_keypoints <= 0 or decoder_channels <= 0:
            raise ValueError("num_keypoints and decoder_channels must be positive")
        self.num_keypoints = num_keypoints
        self.model_id = model_id
        if pretrained and config is not None:
            raise ValueError("a custom SegFormer config cannot be combined with pretrained weights")
        if pretrained:
            # Imported lazily: transformers pulls in SSL, which this host may block.
            from transformers import SegformerConfig, SegformerModel

            self.backbone = SegformerModel.from_pretrained(model_id)
        else:
            from transformers import SegformerConfig, SegformerModel

            self.backbone = SegformerModel(config or SegformerConfig())  # type: ignore[no-untyped-call]
        self.feature_channels = tuple(self.backbone.config.hidden_sizes)
        # The Hugging Face SegFormer image processor uses ImageNet statistics.
        self.input_mean = IMAGENET_MEAN
        self.input_std = IMAGENET_STD
        self.lateral = nn.ModuleList(
            nn.Conv2d(channel, decoder_channels, kernel_size=1) for channel in self.feature_channels
        )
        self.head = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, decoder_channels),
            nn.GELU(),
            nn.Conv2d(decoder_channels, num_keypoints, kernel_size=1),
        )

    def pyramid_features(self, inputs: Tensor) -> list[Tensor]:
        """Return the four native MiT feature levels."""
        outputs = self.backbone(
            pixel_values=inputs,
            output_hidden_states=True,
            return_dict=True,
        )
        if outputs.hidden_states is None:
            raise RuntimeError("SegFormer did not return its feature pyramid")
        features = list(outputs.hidden_states)
        if len(features) != len(self.feature_channels):
            raise RuntimeError("SegFormer returned an unexpected number of feature levels")
        return features

    def normalize(self, inputs: Tensor) -> Tensor:
        """Normalize ``[0, 1]`` inputs with this backbone's pretraining statistics."""
        return normalize_inputs(inputs, self.input_mean, self.input_std)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 4 or inputs.shape[1:] != (3, 256, 256):
            raise ValueError("inputs must have the frozen shape (B, 3, 256, 256)")
        return cast(Tensor, self.head(fuse_pyramid(self.pyramid_features(inputs), self.lateral)))


class HRNetHeatmapRegressor(TimmPyramidHeatmapRegressor):
    """Compact HRNet-W18-Small-v2 reference with the shared heatmap decoder."""

    def __init__(
        self,
        num_keypoints: int,
        *,
        pretrained: bool = False,
        decoder_channels: int = 64,
    ) -> None:
        super().__init__(
            num_keypoints,
            backbone_name="hrnet_w18_small_v2.ms_in1k",
            pretrained=pretrained,
            decoder_channels=decoder_channels,
        )
