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

Two event-only study arms (events-v1) add representations that live in a
separate supplement cache (``event-supp-v1``), not in crop-v2:

- ``activity64``: ``|pixel - neutral| >= 64``. 64 is the synthetic event level
  (191 - 127), so synthetic maps equal ``activity`` by construction; real
  residue between 32 and 63 is removed.
- ``activity_support``: the ``activity`` map keeping only pixels with at least
  one active 8-neighbour (the standard background-activity filter), applied at
  native resolution before resizing.
"""

from __future__ import annotations

from typing import Final, Literal

import numpy as np
from numpy.typing import NDArray
from PIL import Image

EventRepresentation = Literal["raw", "activity", "magnitude", "activity64", "activity_support"]
# Stored in the crop-v2 cache (built by every archive item).
CROP_V2_REPRESENTATIONS: Final[tuple[EventRepresentation, ...]] = ("raw", "activity", "magnitude")
# Stored in the separate event-supp-v1 supplement cache.
SUPPLEMENT_REPRESENTATIONS: Final[tuple[EventRepresentation, ...]] = (
    "activity64",
    "activity_support",
)
EVENT_REPRESENTATIONS: Final[tuple[EventRepresentation, ...]] = (
    *CROP_V2_REPRESENTATIONS,
    *SUPPLEMENT_REPRESENTATIONS,
)
SUPPLEMENT_VERSION: Final[str] = "event-supp-v1"
SYNTHETIC_EVENT_NEUTRAL: Final[int] = 127
REAL_EVENT_NEUTRAL: Final[int] = 128
SYNTHETIC_EVENT_DEVIATION: Final[float] = 64.0
ACTIVITY_THRESHOLD: Final[float] = SYNTHETIC_EVENT_DEVIATION / 2.0
ACTIVITY64_THRESHOLD: Final[float] = SYNTHETIC_EVENT_DEVIATION


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
    """Return each crop-v2 representation in ``[0, 1]`` at the crop's native resolution."""
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
    """Return the crop-v2 representations as resized single-channel uint8 maps."""
    native = native_event_representations(event_crop, neutral)
    return {name: resize_unit_map(native[name], size_wh) for name in CROP_V2_REPRESENTATIONS}


def support_filter(active: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Keep active pixels with at least one active 8-neighbour (zero outside the map)."""
    padded = np.pad(active, 1, mode="constant", constant_values=False)
    height, width = active.shape
    neighbours = np.zeros(active.shape, dtype=np.int32)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            neighbours += padded[1 + dy : 1 + dy + height, 1 + dx : 1 + dx + width]
    return np.asarray(active & (neighbours > 0), dtype=np.bool_)


def native_supplement_representations(
    event_crop: NDArray[np.uint8],
    neutral: int,
) -> dict[EventRepresentation, NDArray[np.float32]]:
    """Return the event-supp-v1 representations in ``[0, 1]`` at native resolution."""
    deviation = np.abs(_grey(event_crop).astype(np.float32) - float(neutral))
    activity = deviation >= ACTIVITY_THRESHOLD
    return {
        "activity64": (deviation >= ACTIVITY64_THRESHOLD).astype(np.float32),
        "activity_support": support_filter(activity).astype(np.float32),
    }


def supplement_representation_maps(
    event_crop: NDArray[np.uint8],
    neutral: int,
    size_wh: tuple[int, int],
) -> dict[EventRepresentation, NDArray[np.uint8]]:
    """Return the supplement representations as resized single-channel uint8 maps.

    Resizing and quantization are identical to :func:`event_representation_maps`.
    """
    native = native_supplement_representations(event_crop, neutral)
    return {name: resize_unit_map(native[name], size_wh) for name in SUPPLEMENT_REPRESENTATIONS}
