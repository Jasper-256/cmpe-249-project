"""sim2real-v1: batch augmentation that closes known synthetic-to-real gaps.

Two gaps were measured without any model selection on real data:

- **Pose coverage.** Every synthetic FRESH pose has zero in-plane camera roll,
  while real sequences need a median 42-83 degrees of roll to reach the nearest
  training pose (labels only; see ``docs/research-log.md``, 2026-11-15).
- **Appearance.** Synthetic RGB has a pure black background, no blur, and no
  sensor noise; synthetic event frames have no noise events.

The recipe follows the augmentation families the FRESH authors report for
their own keypoint networks (random rotation and translation, blur, colour
jitter and noise on RGB, noise events on the event channel). All magnitudes are
fixed here and versioned; none was tuned on real data.

The geometric transform is shared by RGB, the event map, and the clipping
grid, so RGB-event pairing is preserved. Keypoints are transformed analytically
and :func:`invert_points` maps predictions back to the original crop frame, so
pose recovery always uses the untouched intrinsics.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

CLIP_LEVEL = 254.0 / 255.0


@dataclass(frozen=True)
class Sim2RealConfig:
    """Frozen magnitudes of sim2real-v1."""

    version: str = "sim2real-v1"
    rotation_deg: float = 180.0
    scale_range: tuple[float, float] = (0.8, 1.15)
    translate_fraction: float = 0.08
    photometric_probability: float = 0.8
    brightness_range: tuple[float, float] = (0.7, 1.4)
    contrast_range: tuple[float, float] = (0.75, 1.3)
    channel_gain: float = 0.08
    ambient_max: float = 0.3
    blur_probability: float = 0.5
    blur_sigma_range: tuple[float, float] = (0.4, 1.8)
    rgb_noise_std_max: float = 0.04
    event_noise_density_max: float = 0.04
    event_dropout_max: float = 0.3

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _uniform(shape: tuple[int, ...], low: float, high: float, generator: torch.Generator) -> Tensor:
    return torch.rand(shape, generator=generator) * (high - low) + low


def sample_affine(
    batch_size: int, config: Sim2RealConfig, generator: torch.Generator, size: int = 256
) -> Tensor:
    """Sample forward pixel-space affines ``(B, 2, 3)`` about the crop centre."""
    angle = torch.deg2rad(
        _uniform((batch_size,), -config.rotation_deg, config.rotation_deg, generator)
    )
    scale = _uniform((batch_size,), *config.scale_range, generator)
    shift = _uniform(
        (batch_size, 2), -config.translate_fraction, config.translate_fraction, generator
    ) * float(size)
    cos, sin = torch.cos(angle) * scale, torch.sin(angle) * scale
    linear = torch.stack([torch.stack([cos, -sin], -1), torch.stack([sin, cos], -1)], -2)
    centre = (size - 1) / 2.0
    centre_vec = torch.full((batch_size, 2, 1), centre)
    offset = centre_vec - linear @ centre_vec + shift[:, :, None]
    return torch.cat([linear, offset], dim=-1)


def transform_points(points_xy: Tensor, affine: Tensor) -> Tensor:
    """Apply forward affines ``(B, 2, 3)`` to points ``(B, J, 2)``."""
    affine = affine.to(points_xy)
    return points_xy @ affine[:, :, :2].transpose(1, 2) + affine[:, None, :, 2]


def invert_points(points_xy: Tensor, affine: Tensor) -> Tensor:
    """Map points from the augmented frame back to the original crop frame."""
    affine = affine.to(points_xy)
    inverse: Tensor = torch.linalg.inv(affine[:, :, :2])
    restored: Tensor = (points_xy - affine[:, None, :, 2]) @ inverse.transpose(1, 2)
    return restored


def warp(images: Tensor, affine: Tensor, *, fill: float | Tensor = 0.0) -> Tensor:
    """Warp ``(B, C, H, W)`` images with forward pixel affines; ``fill`` outside the source."""
    batch, _, height, width = images.shape
    if height != width:
        raise ValueError("sim2real warps expect square inputs")
    affine = affine.to(images)
    # affine_grid wants output->input in normalized coordinates. Pixel p maps to
    # n = (2p + 1) / W - 1, so the linear part is unchanged and offsets scale by 2 / W
    # after re-expressing the affine about the (normalized) origin.
    inverse_linear = torch.linalg.inv(affine[:, :, :2])
    inverse_offset = -(inverse_linear @ affine[:, :, 2:])
    centre = (width - 1) / 2.0
    ones = torch.full((batch, 2, 1), centre, dtype=images.dtype, device=images.device)
    normalized_offset = (inverse_linear @ ones + inverse_offset - ones) * (2.0 / width)
    theta = torch.cat([inverse_linear, normalized_offset], dim=-1)
    grid = F.affine_grid(theta, list(images.shape), align_corners=False)
    fill_tensor = torch.as_tensor(fill, dtype=images.dtype, device=images.device)
    if fill_tensor.ndim == 1:
        fill_tensor = fill_tensor[:, None, None, None]
    shifted = images - fill_tensor
    return (
        F.grid_sample(shifted, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        + fill_tensor
    )


def _gaussian_blur(images: Tensor, sigma: Tensor) -> Tensor:
    """Per-sample separable Gaussian blur; ``sigma <= 0`` leaves a sample unchanged."""
    radius = 5
    offsets = torch.arange(-radius, radius + 1, dtype=images.dtype, device=images.device)
    safe = sigma.clamp_min(1e-3).to(images)[:, None]
    kernel = torch.exp(-(offsets[None] ** 2) / (2.0 * safe**2))
    kernel = kernel / kernel.sum(dim=1, keepdim=True)
    batch, channels, height, width = images.shape
    flat = images.reshape(1, batch * channels, height, width)
    weight = kernel.repeat_interleave(channels, dim=0)
    blurred = F.conv2d(
        F.pad(flat, (radius, radius, 0, 0), mode="replicate"),
        weight[:, None, None, :],
        groups=batch * channels,
    )
    blurred = F.conv2d(
        F.pad(blurred, (0, 0, radius, radius), mode="replicate"),
        weight[:, None, :, None],
        groups=batch * channels,
    ).reshape(batch, channels, height, width)
    keep = (sigma > 0).to(images)[:, None, None, None]
    return keep * blurred + (1.0 - keep) * images


def clipping_from_rgb(rgb: Tensor, grid: int = 16) -> Tensor:
    """Clipping grid ``(B, grid, grid)`` of an RGB batch in ``[0, 1]`` (max channel >= 254)."""
    clipped = (rgb.amax(dim=1, keepdim=True) >= CLIP_LEVEL).to(rgb.dtype)
    return F.adaptive_avg_pool2d(clipped, grid)[:, 0]


def augment_batch(
    inputs: dict[str, Tensor],
    keypoints_xy: Tensor,
    config: Sim2RealConfig,
    generator: torch.Generator,
    *,
    event_neutral: float = 0.0,
) -> tuple[dict[str, Tensor], Tensor, Tensor]:
    """Augment unnormalized protocol inputs; return inputs, keypoints, and the affines.

    ``event_neutral`` is the event map's background level in ``[0, 1]`` (0 for
    the activity and magnitude maps, 127/255 for raw synthetic frames).
    """
    rgb, event, clipping = inputs["rgb"], inputs["event"], inputs["clipping"]
    batch = rgb.shape[0]
    affine = sample_affine(batch, config, generator, size=rgb.shape[-1])
    rgb = warp(rgb, affine)
    event = warp(event, affine, fill=event_neutral)
    clipping = warp(clipping[:, None], affine)[:, 0].clamp(0.0, 1.0)
    keypoints = transform_points(keypoints_xy, affine)

    def per_sample(low: float, high: float) -> Tensor:
        return _uniform((batch, 1, 1, 1), low, high, generator)

    photometric = (
        torch.rand((batch, 1, 1, 1), generator=generator) < config.photometric_probability
    ).to(rgb.dtype)
    gain = per_sample(*config.brightness_range) * (
        1.0 + _uniform((batch, 3, 1, 1), -config.channel_gain, config.channel_gain, generator)
    )
    contrast = per_sample(*config.contrast_range)
    ambient = per_sample(0.0, config.ambient_max)
    ambient_colour = _uniform((batch, 3, 1, 1), 0.2, 0.9, generator)
    mean = rgb.mean(dim=(1, 2, 3), keepdim=True)
    changed = ((rgb - mean) * contrast + mean) * gain
    changed = changed * (1.0 - ambient) + ambient * ambient_colour
    blur_on = torch.rand((batch,), generator=generator) < config.blur_probability
    sigma = torch.where(
        blur_on, _uniform((batch,), *config.blur_sigma_range, generator), torch.zeros(batch)
    )
    changed = _gaussian_blur(changed, sigma)
    noise_std = per_sample(0.0, config.rgb_noise_std_max)
    changed = changed + torch.randn(rgb.shape, generator=generator) * noise_std
    rgb = (photometric * changed + (1.0 - photometric) * rgb).clamp(0.0, 1.0)
    # Photometric gain can newly saturate pixels; keep the clipping grid truthful.
    clipping = torch.maximum(clipping, clipping_from_rgb(rgb, clipping.shape[-1]))

    # Event channel: drop a share of true events, add isolated noise events, mild blur.
    deviation = event - event_neutral
    keep = (
        torch.rand(event[:, :1].shape, generator=generator)
        >= per_sample(0.0, config.event_dropout_max)
    ).to(event.dtype)
    density = per_sample(0.0, config.event_noise_density_max)
    noise_events = (torch.rand(event[:, :1].shape, generator=generator) < density).to(event.dtype)
    peak = deviation.abs().amax(dim=(1, 2, 3), keepdim=True).clamp_min(0.25)
    deviation = deviation * keep + noise_events * peak * (deviation.abs() < 1e-3)
    event_sigma = torch.where(
        torch.rand((batch,), generator=generator) < config.blur_probability,
        _uniform((batch,), 0.3, 1.0, generator),
        torch.zeros(batch),
    )
    event = (_gaussian_blur(deviation, event_sigma) + event_neutral).clamp(0.0, 1.0)
    return {"rgb": rgb, "event": event, "clipping": clipping}, keypoints, affine


def roll_only_config(rotation_deg: float = 180.0) -> Sim2RealConfig:
    """Geometric roll alone (no scale, shift, or photometric change): a diagnostic."""
    return Sim2RealConfig(
        version="roll-only",
        rotation_deg=rotation_deg,
        scale_range=(1.0, 1.0),
        translate_fraction=0.0,
        photometric_probability=0.0,
        blur_probability=0.0,
        event_noise_density_max=0.0,
        event_dropout_max=0.0,
    )


def rotation_matrix_deg(angle_deg: float) -> Tensor:
    """2x2 rotation used by tests."""
    angle = math.radians(angle_deg)
    return torch.tensor([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
