"""Targeted specular saturation (TSS): the protocol-2 synthetic clipping corruption.

FRESH deliberately rendered its synthetic data without harsh-lighting effects,
so natural synthetic crops never clip (G1A) and a generic sun-flare augmentation
rarely reaches the target (S1A). TSS models the case the hypothesis is about:
specular glare *on the spacecraft* that saturates RGB while the co-registered
event frame is unchanged. Each highlight is an anisotropic Gaussian core with a
wider bloom, added as white light and clipped at 255, centred on object pixels.

The corruption is a deterministic function of ``(sample_id, seed)``, so every
learned variant receives byte-identical corrupted samples. It is applied before
resize, at native crop resolution, matching where clipping is measured. Event
frames are deliberately left untouched; real glare can also create spurious
events, which is a documented limitation of this protocol.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class TargetedSaturationConfig:
    """Frozen protocol-2 parameters; changing any value is a new protocol version."""

    name: str = "tss-v1"
    probability: float = 0.5
    min_highlights: int = 1
    max_highlights: int = 3
    sigma_fraction_range: tuple[float, float] = (0.03, 0.10)
    aspect_ratio_range: tuple[float, float] = (0.5, 2.0)
    core_gain_range: tuple[float, float] = (1.5, 3.0)
    bloom_sigma_scale: float = 3.0
    bloom_gain_fraction: float = 0.15
    object_luminance_threshold: int = 24

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def sample_rng(sample_id: str, seed: int) -> np.random.Generator:
    """Return a generator determined only by the sample identity and protocol seed."""
    digest = hashlib.sha256(f"{seed}:{sample_id}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def targeted_specular_saturation(
    rgb_crop: NDArray[np.uint8],
    sample_id: str,
    *,
    seed: int = 249,
    config: TargetedSaturationConfig | None = None,
) -> tuple[NDArray[np.uint8], dict[str, Any]]:
    """Apply TSS to a native-resolution RGB crop.

    Returns the corrupted crop (or an unchanged copy) and a metadata record of the
    applied highlights for provenance.
    """
    if rgb_crop.ndim != 3 or rgb_crop.shape[2] != 3 or rgb_crop.dtype != np.uint8:
        raise ValueError("rgb_crop must be uint8 with shape (H, W, 3)")
    settings = config or TargetedSaturationConfig()
    rng = sample_rng(sample_id, seed)
    applied = bool(rng.random() < settings.probability)
    metadata: dict[str, Any] = {"protocol": settings.name, "applied": applied, "highlights": []}
    if not applied:
        return rgb_crop.copy(), metadata

    height, width = rgb_crop.shape[:2]
    side = float(max(height, width))
    luminance = rgb_crop.max(axis=2)
    object_rows, object_columns = np.nonzero(luminance > settings.object_luminance_threshold)
    rows, columns = np.mgrid[0:height, 0:width].astype(np.float32)
    gain_field = np.zeros((height, width), dtype=np.float32)
    count = int(rng.integers(settings.min_highlights, settings.max_highlights + 1))
    for _ in range(count):
        if object_rows.size:
            pick = int(rng.integers(object_rows.size))
            center_y, center_x = float(object_rows[pick]), float(object_columns[pick])
        else:
            center_y, center_x = (height - 1) / 2.0, (width - 1) / 2.0
        sigma = float(rng.uniform(*settings.sigma_fraction_range)) * side
        aspect = float(rng.uniform(*settings.aspect_ratio_range))
        sigma_u, sigma_v = sigma * np.sqrt(aspect), sigma / np.sqrt(aspect)
        angle = float(rng.uniform(0.0, np.pi))
        gain = float(rng.uniform(*settings.core_gain_range)) * 255.0
        cosine, sine = np.cos(angle), np.sin(angle)
        offset_x, offset_y = columns - center_x, rows - center_y
        u = cosine * offset_x + sine * offset_y
        v = -sine * offset_x + cosine * offset_y
        core = np.exp(-0.5 * ((u / sigma_u) ** 2 + (v / sigma_v) ** 2))
        bloom_scale = settings.bloom_sigma_scale
        bloom = np.exp(
            -0.5 * ((u / (sigma_u * bloom_scale)) ** 2 + (v / (sigma_v * bloom_scale)) ** 2)
        )
        gain_field += gain * core + settings.bloom_gain_fraction * gain * bloom
        metadata["highlights"].append(
            {
                "center_xy": [center_x, center_y],
                "sigma_px": sigma,
                "aspect": aspect,
                "angle_rad": angle,
                "gain_8bit": gain,
            }
        )
    corrupted = np.clip(rgb_crop.astype(np.float32) + gain_field[..., None], 0.0, 255.0)
    return np.asarray(np.rint(corrupted), dtype=np.uint8), metadata
