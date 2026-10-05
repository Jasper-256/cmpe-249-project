"""Event-frame representations that bridge FRESH's synthetic and real encodings.

FRESH synthetic event frames are grey with three levels: neutral 127 and two
event levels (191, 255), both brighter than neutral. Real event frames are
continuous, neutral 128, with one polarity darker and one brighter. A model
trained on raw synthetic frames therefore never sees darker-than-neutral
pixels. Three representations are compared as separately named study arms:

- ``raw``: the author-faithful grey frame (what FRESH and UE-Pose consume).
- ``activity``: polarity-agnostic binary activity, ``|pixel - neutral| >= 32``.
- ``magnitude``: polarity-agnostic soft activity, ``min(1, |pixel - neutral| / 64)``.

Both constants are format-defined from the synthetic encoding, never tuned on
real data: 64 is the smallest synthetic event deviation (191 - 127), so both
synthetic polarities map to magnitude 1 and the synthetic levels carry no
polarity; 32 is the midpoint between neutral and that level, so faint real
deviations (normalization or compression residue) do not count as events. All
maps are computed at native crop resolution, before resizing.
"""

from __future__ import annotations

from typing import Final, Literal

import numpy as np
from numpy.typing import NDArray
from PIL import Image

EventRepresentation = Literal["raw", "activity", "magnitude"]
EVENT_REPRESENTATIONS: Final[tuple[EventRepresentation, ...]] = ("raw", "activity", "magnitude")
SYNTHETIC_EVENT_NEUTRAL: Final[int] = 127
REAL_EVENT_NEUTRAL: Final[int] = 128
SYNTHETIC_EVENT_DEVIATION: Final[float] = 64.0
ACTIVITY_THRESHOLD: Final[float] = SYNTHETIC_EVENT_DEVIATION / 2.0


def _grey(event_crop: NDArray[np.uint8]) -> NDArray[np.uint8]:
    if event_crop.ndim == 3:
        # FRESH event frames are grey stored as RGB; a colour-coded polarity
        # would be silently lost by keeping one channel, so verify the format.
        if not (
            np.array_equal(event_crop[..., 0], event_crop[..., 1])
            and np.array_equal(event_crop[..., 1], event_crop[..., 2])
        ):
            raise ValueError("event frame channels differ; expected a grey encoding")
        return np.asarray(event_crop[..., 0], dtype=np.uint8)
    if event_crop.ndim == 2:
        return np.asarray(event_crop, dtype=np.uint8)
    raise ValueError("event_crop must have shape (H, W) or (H, W, C)")


def native_event_representations(
    event_crop: NDArray[np.uint8],
    neutral: int,
) -> dict[EventRepresentation, NDArray[np.float32]]:
    """Return each representation in ``[0, 1]`` at the crop's native resolution."""
    grey = _grey(event_crop).astype(np.float32)
    deviation = np.abs(grey - float(neutral))
    return {
        "raw": grey / 255.0,
        "activity": (deviation >= ACTIVITY_THRESHOLD).astype(np.float32),
        "magnitude": np.minimum(deviation / SYNTHETIC_EVENT_DEVIATION, 1.0).astype(np.float32),
    }


def resize_unit_map(values: NDArray[np.float32], size_wh: tuple[int, int]) -> NDArray[np.uint8]:
    """Resize a ``[0, 1]`` map with Pillow bilinear (the image contract) and quantize."""
    resized = Image.fromarray(values.astype(np.float32), mode="F").resize(
        size_wh, resample=Image.Resampling.BILINEAR
    )
    return np.asarray(np.clip(np.rint(np.asarray(resized) * 255.0), 0, 255), dtype=np.uint8)


def event_representation_maps(
    event_crop: NDArray[np.uint8],
    neutral: int,
    size_wh: tuple[int, int],
) -> dict[EventRepresentation, NDArray[np.uint8]]:
    """Return all representations as resized single-channel uint8 maps."""
    native = native_event_representations(event_crop, neutral)
    return {name: resize_unit_map(native[name], size_wh) for name in EVENT_REPRESENTATIONS}
