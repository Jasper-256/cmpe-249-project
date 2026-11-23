"""Label-free comparison of synthetic and real FRESH event inputs.

Measures how the event input differs between domains at three levels, without
running any trained model:

1. **Crop representation** (what the networks see): the 256 px ``activity`` map
   from the crop-v2 cache, plus the cached grey ``raw`` map for polarity. Seeded
   samples of synthetic train/validation crops, the same train crops after the
   frozen sim2real-v1 batch augmentation, and every real crop.
2. **Native event frames** (before our representation): the full 720x800 FRESH
   event PNGs read zero-copy from the archives; signed deviation histograms and a
   per-pixel hot-pixel census per real sequence.
3. **Raw event streams** (real only): the first seconds of each ``events.txt``
   after the first RGB frame; event rate, polarity, per-pixel hot-pixel census in
   sensor coordinates, and an event-rate spectrum for mains-flicker peaks.

Real annotation boxes and projected keypoints are used only to define regions
(object box, background ring, near-object hull), the same box the frozen crop
already uses. No pose prediction or error is computed here. The only model
outputs read are the existing ``outputs/real-v2`` event-only frame errors, to
relate per-sequence input statistics to the already-reported failure.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import subprocess
import sys
import time
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from PIL import Image
from scipy import stats as scipy_stats

from saga_pose.data.archives import open_fresh_archive
from saga_pose.data.real import load_real_slices, slice_mask
from saga_pose.data.representations import (
    ACTIVITY_THRESHOLD,
    REAL_EVENT_NEUTRAL,
    SYNTHETIC_EVENT_NEUTRAL,
)
from saga_pose.data.sim2real import Sim2RealConfig, augment_batch, warp

OBJECTS = ("cassini", "satty", "soho")
ACTIVE_LEVEL = 0.5  # activity maps are bilinear-resized binary maps; >= 0.5 counts as active
BOX_MARGIN_CROP = 4  # px at 256: background ring starts this far outside the object box
NEAR_MARGIN_CROP = 6  # px at 256: dilation of the keypoint hull for "near object"
BOX_MARGIN_NATIVE = 20  # px at native resolution
HOT_FRACTIONS = (0.5, 0.9)
CONTRAST_EPS = 1e-3  # added to both mean activities in the object-contrast ratio
WEAK_OBJECT_CONTRAST = (
    2.0  # frames below this box/background activity ratio carry little object signal
)
FRAME_WH = (800, 720)  # native FRESH frame size (synthetic and real)
MAINS_BANDS_HZ = (50.0, 100.0, 120.0, 150.0, 200.0, 300.0)

CROP_STATS = (
    "density",
    "mean_activity",
    "partial_density",
    "box_mean_activity",
    "background_mean_activity",
    "object_contrast",
    "crop_native_px",
    "box_density",
    "background_density",
    "object_share",
    "isolated_fraction",
    "largest_component_share",
    "components_per_kpx",
    "thickness_px",
)
RAW_CROP_STATS = ("positive_density", "negative_density", "negative_share", "weak_density")


# --------------------------------------------------------------------------- helpers


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def skeleton_pixels(mask: NDArray[np.uint8]) -> int:
    """Pixel count of the morphological (Lantuejoul) skeleton of a binary mask."""
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    image = mask.copy()
    skeleton = np.zeros_like(mask)
    for _ in range(64):
        if not image.any():
            break
        opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        skeleton |= image & ~opened
        image = cv2.erode(image, kernel)
    return int(np.count_nonzero(skeleton))


def mask_stats(active: NDArray[np.bool_]) -> dict[str, float]:
    """Spatial structure of a binary event mask (isolation, components, thickness)."""
    count = int(active.sum())
    if count == 0:
        return {
            "isolated_fraction": np.nan,
            "largest_component_share": np.nan,
            "components_per_kpx": 0.0,
            "thickness_px": np.nan,
        }
    as_u8 = active.astype(np.uint8)
    neighbours = cv2.filter2D(as_u8, cv2.CV_16S, np.ones((3, 3), np.float32)) - as_u8
    isolated = float(np.count_nonzero(active & (neighbours == 0)) / count)
    n_labels, _, stats_cc, _ = cv2.connectedComponentsWithStats(as_u8, connectivity=8)
    areas = stats_cc[1:, cv2.CC_STAT_AREA]
    skeleton = skeleton_pixels(as_u8)
    return {
        "isolated_fraction": isolated,
        "largest_component_share": float(areas.max() / count),
        "components_per_kpx": float((n_labels - 1) / (active.size / 1000.0)),
        "thickness_px": float(count / max(skeleton, 1)),
    }


def describe(values: NDArray[np.floating[Any]] | list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"n": 0}
    p1, p5, p10, p25, p50, p75, p90, p95, p99 = np.percentile(
        array, [1, 5, 10, 25, 50, 75, 90, 95, 99]
    )
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "p01": float(p1),
        "p05": float(p5),
        "p10": float(p10),
        "p25": float(p25),
        "median": float(p50),
        "p75": float(p75),
        "p90": float(p90),
        "p95": float(p95),
        "p99": float(p99),
    }


def box_masks(
    box_xyxy: NDArray[np.floating[Any]], shape: tuple[int, int], margin: int
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Return (inside-box, background = outside box dilated by ``margin``)."""
    height, width = shape
    left, top, right, bottom = box_xyxy
    inside = np.zeros(shape, dtype=bool)
    x0, y0 = int(np.floor(max(left, 0))), int(np.floor(max(top, 0)))
    x1, y1 = int(np.ceil(min(right, width))), int(np.ceil(min(bottom, height)))
    inside[y0:y1, x0:x1] = True
    dilated = np.zeros(shape, dtype=bool)
    dl, dt = max(x0 - margin, 0), max(y0 - margin, 0)
    dr, db = min(x1 + margin, width), min(y1 + margin, height)
    dilated[dt:db, dl:dr] = True
    return inside, ~dilated


def hull_mask(
    points_xy: NDArray[np.floating[Any]], shape: tuple[int, int], margin: int
) -> NDArray[np.bool_]:
    canvas = np.zeros(shape, dtype=np.uint8)
    hull = cv2.convexHull(np.round(points_xy).astype(np.int32).reshape(-1, 1, 2))
    cv2.fillConvexPoly(canvas, hull, 1)
    if margin > 0:
        canvas = cv2.dilate(canvas, np.ones((2 * margin + 1, 2 * margin + 1), np.uint8))
    return canvas.astype(bool)


# --------------------------------------------------------------------------- level 1: crops


def crop_frame_stats(
    activity: NDArray[np.floating[Any]],
    raw: NDArray[np.uint8] | None,
    neutral: int,
    inside: NDArray[np.bool_],
    background: NDArray[np.bool_],
    near: NDArray[np.bool_],
) -> dict[str, float]:
    active = activity >= ACTIVE_LEVEL
    count = int(active.sum())
    box_mean = float(activity[inside].mean()) if inside.any() else np.nan
    background_mean = float(activity[background].mean()) if background.any() else np.nan
    out: dict[str, float] = {
        "density": float(active.mean()),
        "mean_activity": float(activity.mean()),
        "partial_density": float(((activity > 0.02) & (activity < ACTIVE_LEVEL)).mean()),
        "box_mean_activity": box_mean,
        "background_mean_activity": background_mean,
        "object_contrast": (box_mean + CONTRAST_EPS) / (background_mean + CONTRAST_EPS),
        "box_density": float(active[inside].mean()) if inside.any() else np.nan,
        "background_density": float(active[background].mean()) if background.any() else np.nan,
        "object_share": float(active[near].sum() / count) if count else np.nan,
    }
    out.update(mask_stats(active))
    if raw is not None:
        deviation = raw.astype(np.int16) - neutral
        positive = deviation >= ACTIVITY_THRESHOLD
        negative = deviation <= -ACTIVITY_THRESHOLD
        n_pos, n_neg = int(positive.sum()), int(negative.sum())
        magnitude = np.abs(deviation)
        out.update(
            {
                "positive_density": float(positive.mean()),
                "negative_density": float(negative.mean()),
                "negative_share": float(n_neg / (n_pos + n_neg)) if n_pos + n_neg else np.nan,
                "weak_density": float(((magnitude >= 2) & (magnitude < ACTIVITY_THRESHOLD)).mean()),
                "background_mean_abs_deviation": float(magnitude[background].mean())
                if background.any()
                else np.nan,
            }
        )
    return out


class CropCache:
    """Minimal read-only view of one crop-v2 cache directory."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        self.sample_ids: list[str] = self.manifest["sample_ids"]
        self.sequences = np.asarray(self.manifest["sequences"])

    def array(self, name: str) -> NDArray[Any]:
        return np.load(self.root / f"{name}.npy", mmap_mode="r")

    def array_hashes(self) -> dict[str, str]:
        return {k: v["sha256"] for k, v in self.manifest["arrays"].items() if "sha256" in v}


def crop_regions(
    cache: CropCache, index: NDArray[np.int64]
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """Object boxes, crop keypoints and the in-frame rectangle, all in 256 px crop coordinates."""
    bbox = np.asarray(cache.array("source_bbox_xyxy")[index], dtype=np.float64)
    crop = np.asarray(cache.array("crop_xyxy")[index], dtype=np.float64)
    scale = 256.0 / (crop[:, 2] - crop[:, 0])
    boxes = (bbox - np.concatenate([crop[:, :2], crop[:, :2]], axis=1)) * scale[:, None]
    keypoints = np.asarray(cache.array("keypoints_xy")[index], dtype=np.float32)
    frame = np.asarray([0.0, 0.0, *FRAME_WH])[None] - np.concatenate([crop[:, :2], crop[:, :2]], 1)
    return boxes.astype(np.float32), keypoints, (frame * scale[:, None]).astype(np.float32)


def region_masks(
    box: NDArray[np.float32], keypoints: NDArray[np.float32], frame: NDArray[np.float32]
) -> tuple[NDArray[np.bool_], NDArray[np.bool_], NDArray[np.bool_]]:
    """Inside-box, in-frame background ring, and near-object hull masks at 256 px.

    Crop pixels outside the native frame are constant neutral fill, so they are
    excluded from the background ring.
    """
    inside, background = box_masks(box, (256, 256), BOX_MARGIN_CROP)
    in_frame, _ = box_masks(frame, (256, 256), 0)
    near = hull_mask(keypoints, (256, 256), NEAR_MARGIN_CROP)
    return inside, background & in_frame, near


def crop_group_stats(
    cache: CropCache, index: NDArray[np.int64], neutral: int
) -> dict[str, NDArray[np.float64]]:
    activity = cache.array("event_activity_uint8")
    raw = cache.array("event_raw_uint8")
    boxes, keypoints, frames = crop_regions(cache, index)
    rows: list[dict[str, float]] = []
    for position, item in enumerate(index):
        inside, background, near = region_masks(
            boxes[position], keypoints[position], frames[position]
        )
        rows.append(
            crop_frame_stats(
                activity[item, 0].astype(np.float32) / 255.0,
                np.asarray(raw[item, 0]),
                neutral,
                inside,
                background,
                near,
            )
        )
    stats = {key: np.asarray([row[key] for row in rows], dtype=np.float64) for key in rows[0]}
    stats["crop_native_px"] = crop_native_px(cache, index)
    return stats


def crop_native_px(cache: CropCache, index: NDArray[np.int64]) -> NDArray[np.float64]:
    """Native side length of each square crop (before the 256 px resize)."""
    crop = np.asarray(cache.array("crop_xyxy")[index], dtype=np.float64)
    return crop[:, 2] - crop[:, 0]


def augmented_group_stats(
    cache: CropCache, index: NDArray[np.int64], seed: int, batch_size: int = 64
) -> tuple[dict[str, NDArray[np.float64]], NDArray[np.float32]]:
    """Stats of the activity map after the frozen sim2real-v1 batch augmentation."""
    config = Sim2RealConfig()
    generator = torch.Generator().manual_seed(seed)
    activity = cache.array("event_activity_uint8")
    rgb = cache.array("rgb_uint8")
    clipping = cache.array("clipping_ratio")
    boxes, keypoints, frames = crop_regions(cache, index)
    rows: list[dict[str, float]] = []
    examples: list[NDArray[np.float32]] = []
    for start in range(0, len(index), batch_size):
        chunk = index[start : start + batch_size]
        local = np.arange(start, start + len(chunk))
        inputs = {
            "rgb": torch.from_numpy(np.asarray(rgb[chunk], dtype=np.float32) / 255.0),
            "event": torch.from_numpy(np.asarray(activity[chunk], dtype=np.float32) / 255.0),
            "clipping": torch.from_numpy(np.asarray(clipping[chunk], dtype=np.float32)),
        }
        regions = []
        for position in local:
            inside, background, near = region_masks(
                boxes[position], keypoints[position], frames[position]
            )
            regions.append(np.stack([inside, background, near]).astype(np.float32))
        region_tensor = torch.from_numpy(np.stack(regions))
        out, _, affine = augment_batch(
            inputs, torch.from_numpy(keypoints[local]), config, generator, event_neutral=0.0
        )
        warped = (warp(region_tensor, affine) >= 0.5).numpy()
        event = out["event"][:, 0].numpy()
        for row in range(len(chunk)):
            inside, background, near = warped[row]
            rows.append(crop_frame_stats(event[row], None, 0, inside, background, near))
        if start == 0:
            examples = [event[row] for row in range(min(8, len(chunk)))]
    stats = {key: np.asarray([row[key] for row in rows], dtype=np.float64) for key in rows[0]}
    stats["crop_native_px"] = crop_native_px(cache, index)
    return stats, np.stack(examples)


# --------------------------------------------------------------------------- level 2: native frames


def _decode_grey(payload: bytes) -> NDArray[np.uint8]:
    with Image.open(io.BytesIO(payload)) as image:
        array = np.asarray(image)
    return array[..., 0] if array.ndim == 3 else array


def native_worker(job: dict[str, Any]) -> dict[str, Any]:
    """Per-frame native stats, signed-deviation histogram and hot-pixel census."""
    kind, members, neutral = job["kind"], job["members"], job["neutral"]
    boxes = np.asarray(job["boxes"], dtype=np.float64)
    histogram = np.zeros(511, dtype=np.int64)  # deviation -255..255
    active_counts: NDArray[np.int32] | None = None
    rows: list[dict[str, float]] = []
    with open_fresh_archive(kind, Path(job["raw_root"])) as archive:
        for member, box in zip(members, boxes, strict=True):
            grey = _decode_grey(archive.read(member))
            deviation = grey.astype(np.int16) - neutral
            histogram += np.bincount((deviation + 255).ravel(), minlength=511)
            active = np.abs(deviation) >= ACTIVITY_THRESHOLD
            if active_counts is None:
                active_counts = np.zeros(active.shape, dtype=np.int32)
            active_counts += active
            inside, background = box_masks(box, active.shape, BOX_MARGIN_NATIVE)
            positive = int((deviation >= ACTIVITY_THRESHOLD).sum())
            negative = int((deviation <= -ACTIVITY_THRESHOLD).sum())
            magnitude = np.abs(deviation)
            row = {
                "density": float(active.mean()),
                "box_density": float(active[inside].mean()),
                "background_density": float(active[background].mean()),
                "nonneutral_density": float((magnitude >= 1).mean()),
                "weak_density": float(((magnitude >= 2) & (magnitude < ACTIVITY_THRESHOLD)).mean()),
                "background_mean_abs_deviation": float(magnitude[background].mean()),
                "negative_share": negative / (positive + negative)
                if positive + negative
                else np.nan,
            }
            row.update(mask_stats(active))
            rows.append(row)
    assert active_counts is not None
    frames = len(members)
    fraction = active_counts / frames
    hot = {
        f"pixels_active_ge_{int(level * 100)}pct_of_frames": int((fraction >= level).sum())
        for level in HOT_FRACTIONS
    }
    hot["frames"] = frames
    hot["sensor_pixels"] = int(fraction.size)
    hot["mean_pixel_active_fraction"] = float(fraction.mean())
    return {
        "name": job["name"],
        "rows": {key: [row[key] for row in rows] for key in rows[0]},
        "histogram": histogram.tolist(),
        "hot": hot,
    }


# --------------------------------------------------------------------------- level 3: raw streams


def _iter_event_chunks(stream: Any, chunk_bytes: int) -> Iterator[pd.DataFrame]:
    tail = b""
    while True:
        block = stream.read(chunk_bytes)
        if not block:
            if tail.strip():
                yield _parse(tail)
            return
        block = tail + block
        cut = block.rfind(b"\n") + 1
        tail = block[cut:]
        if cut:
            yield _parse(block[:cut])


def _parse(payload: bytes) -> pd.DataFrame:
    return pd.read_csv(
        io.BytesIO(payload),
        header=None,
        names=["t", "x", "y", "p"],
        dtype=np.int64,
        skipinitialspace=True,
    )


def stream_worker(job: dict[str, Any]) -> dict[str, Any]:
    """Event rate, polarity, hot pixels and flicker spectrum from a raw stream window."""
    sequence, seconds = job["sequence"], float(job["seconds"])
    bin_us = int(job["bin_us"])
    with open_fresh_archive("real", Path(job["raw_root"])) as archive:
        lines = archive.read(f"real/{sequence}/timestamps.txt").decode().split()
        frame_times = np.asarray([int(v) for v in lines[1::2]], dtype=np.int64)
        start = int(frame_times[0])
        stop = start + int(seconds * 1e6)
        pieces: list[pd.DataFrame] = []
        with archive.open(f"real/{sequence}/events/events.txt") as stream:
            for chunk in _iter_event_chunks(stream, 32 * 1024 * 1024):
                keep = chunk[(chunk.t >= start) & (chunk.t < stop)]
                if len(keep):
                    pieces.append(keep)
                if chunk.t.iloc[-1] >= stop:
                    break
    events = pd.concat(pieces, ignore_index=True)
    t = events.t.to_numpy()
    x = events.x.to_numpy()
    y = events.y.to_numpy()
    p = events.p.to_numpy()
    span = (t[-1] - start) / 1e6
    width, height = 1280, 720
    counts = np.bincount(y * width + x, minlength=width * height)
    # per-pixel share of 33.33 ms frame intervals with >= 1 event
    frame_bins = np.clip(((t - start) / (1e6 / 30.0)).astype(np.int64), 0, None)
    n_frames = int(frame_bins.max()) + 1
    pixel_frame = np.unique(frame_bins * (width * height) + y * width + x)
    per_pixel_frames = np.bincount(pixel_frame % (width * height), minlength=width * height)
    fraction = per_pixel_frames / n_frames
    order = np.sort(counts)[::-1]
    # event-rate spectrum
    rate = np.bincount(((t - start) // bin_us).astype(np.int64)).astype(np.float64)
    rate = rate[: int(seconds * 1e6 // bin_us)]
    rate -= rate.mean()
    window = np.hanning(rate.size)
    power = np.abs(np.fft.rfft(rate * window)) ** 2
    freqs = np.fft.rfftfreq(rate.size, d=bin_us / 1e6)
    analysis = (freqs >= 20.0) & (freqs <= 1000.0)
    floor = float(np.median(power[analysis]))
    peaks: dict[str, float] = {}
    for band in MAINS_BANDS_HZ:
        sel = np.abs(freqs - band) <= 2.0
        peaks[f"{int(band)}Hz_peak_over_median"] = float(power[sel].max() / floor)
    top = np.argsort(power * analysis)[::-1]
    top_peaks: list[dict[str, float]] = []
    for index in top:
        if len(top_peaks) == 5:
            break
        if any(abs(freqs[index] - peak["hz"]) < 5.0 for peak in top_peaks):
            continue
        top_peaks.append(
            {"hz": float(freqs[index]), "power_over_median": float(power[index] / floor)}
        )
    # downsampled spectrum for the figure: 1 Hz resolution up to 500 Hz
    grid = np.arange(1.0, 501.0)
    spectrum = np.interp(grid, freqs, power / floor)
    return {
        "sequence": sequence,
        "window_seconds": float(span),
        "events": int(t.size),
        "event_rate_mev_per_s": float(t.size / span / 1e6),
        "events_per_frame_interval": float(t.size / span / 30.0),
        "on_share": float(p.mean()),
        "pixels_with_any_event_share": float((counts > 0).mean()),
        "mean_pixel_frame_active_share": float(fraction.mean()),
        "pixels_active_ge_50pct_of_frames": int((fraction >= 0.5).sum()),
        "pixels_active_ge_90pct_of_frames": int((fraction >= 0.9).sum()),
        "event_share_top_0p1pct_pixels": float(order[: (width * height) // 1000].sum() / t.size),
        "event_share_top_1pct_pixels": float(order[: (width * height) // 100].sum() / t.size),
        "spectrum_bin_us": bin_us,
        "mains_peaks": peaks,
        "top_peaks_20_1000Hz": top_peaks,
        "spectrum_1_500Hz_over_median": spectrum.tolist(),
    }


# --------------------------------------------------------------------------- figures


COLOURS = {"synthetic": "#2a78d6", "sim2real": "#1baf7a", "real": "#eb6834"}
INK, MUTED = "#0b0b0b", "#52514e"


def _style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.facecolor": "#fcfcfb",
            "axes.facecolor": "#fcfcfb",
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#e4e3df",
            "grid.linewidth": 0.6,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "font.size": 9,
            "axes.titlesize": 10,
            "legend.frameon": False,
        }
    )


def figure_examples(
    path: Path,
    synthetic: dict[str, NDArray[np.uint8]],
    augmented: dict[str, NDArray[np.float32]],
    real: dict[str, NDArray[np.uint8]],
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 6, figsize=(10.5, 5.6))
    rows = (
        ("Synthetic (train)", synthetic),
        ("Synthetic + sim2real-v1", augmented),
        ("Real", real),
    )
    for r, (label, maps) in enumerate(rows):
        for o, obj in enumerate(OBJECTS):
            for k in range(2):
                ax = axes[r, 2 * o + k]
                ax.imshow(maps[obj][k], cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
                ax.grid(False)
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_color("#c3c2b7")
                if r == 0:
                    ax.set_title(obj, color=INK)
            axes[r, 0].set_ylabel(label, color=INK)
    fig.suptitle(
        "Activity maps the event network sees (256 px crops; dark = active)",
        color=INK,
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def figure_distributions(path: Path, groups: dict[str, dict[str, NDArray[np.float64]]]) -> None:
    import matplotlib.pyplot as plt

    panels = (
        ("mean_activity", "Mean activity over the crop", True),
        ("background_mean_activity", "Mean activity outside the object box", True),
        ("partial_density", "Share of faint pixels (0.02 < activity < 0.5)", True),
        ("object_contrast", "Object contrast (box / background mean activity)", True),
        ("isolated_fraction", "Isolated active pixels (no active 8-neighbour)", False),
        ("crop_native_px", "Native crop side before 256 px resize (px)", False),
    )
    floor = 1e-4
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.4))
    for ax, (key, title, log) in zip(axes.ravel(), panels, strict=True):
        for name, values in groups.items():
            finite = np.sort(values[key][np.isfinite(values[key])])
            if log:
                finite = np.maximum(finite, floor)
            share = np.arange(1, finite.size + 1) / finite.size
            ax.step(finite, share, where="post", color=COLOURS[name], linewidth=2)
        if log:
            ax.set_xscale("log")
        ax.set_ylim(0, 1)
        ax.set_title(title, loc="left", color=INK)
    for ax in axes[:, 0]:
        ax.set_ylabel("Cumulative share of frames")
    labels = {
        "synthetic": "Synthetic train",
        "sim2real": "Same frames + sim2real-v1",
        "real": "Real (all 24 sequences)",
    }
    handles = [
        plt.Line2D([0], [0], color=COLOURS[name], linewidth=2, label=labels[name])
        for name in groups
    ]
    fig.legend(handles=handles, loc="upper left", ncol=3, bbox_to_anchor=(0.005, 0.955))
    fig.suptitle(
        f"Per-frame statistics of the activity map (ECDF; values below {floor:g} "
        f"drawn at {floor:g})",
        color=INK,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def figure_sequences(
    path: Path,
    per_sequence: dict[str, dict[str, dict[str, float]]],
    bands: dict[str, dict[str, dict[str, dict[str, float]]]],
    errors: dict[str, float],
) -> None:
    import matplotlib.pyplot as plt

    keys = (
        ("density", "Active-pixel density"),
        ("background_mean_activity", "Mean activity outside the object box"),
        ("object_contrast", "Object contrast (box / background mean activity)"),
    )
    sequences = sorted(per_sequence)
    fig, axes = plt.subplots(len(keys) + 1, 1, figsize=(11, 9.5), sharex=True)
    x = np.arange(len(sequences))
    for ax, (key, title) in zip(axes, keys, strict=False):
        medians = [per_sequence[s][key]["median"] for s in sequences]
        low = [per_sequence[s][key]["p10"] for s in sequences]
        high = [per_sequence[s][key]["p90"] for s in sequences]
        for obj in OBJECTS:
            idx = [i for i, s in enumerate(sequences) if s.startswith(obj)]
            lo, hi = min(idx) - 0.45, max(idx) + 0.45
            for name, alpha in (("synthetic", 0.16), ("sim2real", 0.16)):
                band = bands[name][obj][key]
                ax.fill_between(
                    [lo, hi],
                    band["p05"],
                    band["p95"],
                    color=COLOURS[name],
                    alpha=alpha,
                    linewidth=0,
                )
                ax.hlines(band["median"], lo, hi, color=COLOURS[name], linewidth=1.5)
        if key == "object_contrast":
            ax.set_yscale("log")
            ax.axhline(WEAK_OBJECT_CONTRAST, color=INK, linewidth=0.8, linestyle="--")
        ax.bar(x, medians, width=0.7, color=COLOURS["real"], zorder=3)
        ax.errorbar(
            x,
            medians,
            yerr=[np.subtract(medians, low), np.subtract(high, medians)],
            fmt="none",
            ecolor=INK,
            elinewidth=0.8,
            zorder=4,
        )
        ax.set_title(title, loc="left", color=INK)
    err_ax = axes[-1]
    err_ax.bar(x, [errors.get(s, np.nan) for s in sequences], width=0.7, color="#8a8984", zorder=3)
    err_ax.axhline(126.5, color=INK, linewidth=1, linestyle="--")
    err_ax.text(-0.5, 128, "chance 126.5", ha="left", va="bottom", color=MUTED)
    err_ax.set_title(
        "Existing result (outputs/real-v2): event-only median rotation error, deg",
        loc="left",
        color=INK,
    )
    err_ax.set_xticks(x)
    err_ax.set_xticklabels(sequences, rotation=60, ha="right")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLOURS["real"], label="Real sequence median (p10-p90)"),
        plt.Rectangle(
            (0, 0), 1, 1, color=COLOURS["synthetic"], alpha=0.35, label="Synthetic p5-p95"
        ),
        plt.Rectangle(
            (0, 0), 1, 1, color=COLOURS["sim2real"], alpha=0.35, label="Sim2real-v1 p5-p95"
        ),
    ]
    fig.legend(handles=handles, loc="upper right", ncol=3, bbox_to_anchor=(0.99, 1.0))
    fig.suptitle(
        "Real sequences against synthetic ranges (crop activity map)", color=INK, x=0.01, ha="left"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def figure_native(
    path: Path, histograms: dict[str, NDArray[np.float64]], streams: dict[str, dict[str, Any]]
) -> None:
    import matplotlib.pyplot as plt

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))
    deviation = np.arange(-255, 256)
    styles = {"synthetic": "synthetic", "real": "real"}
    for name, histogram in histograms.items():
        colour = styles[name]
        share = histogram / histogram.sum()
        nonzero = share > 0
        left.plot(
            deviation[nonzero],
            share[nonzero],
            linestyle="none" if name == "synthetic" else "-",
            color=COLOURS[colour],
            linewidth=1.6,
            marker="o" if name == "synthetic" else None,
            markersize=6,
            label=name,
        )
    left.axvspan(-ACTIVITY_THRESHOLD, ACTIVITY_THRESHOLD, color="#e4e3df", alpha=0.6, linewidth=0)
    left.text(0, 2e-7, "below activity\nthreshold", ha="center", color=MUTED, fontsize=8)
    left.set_yscale("log")
    left.set_ylim(1e-7, 1.5)
    left.set_xlabel("Pixel value minus neutral (native 720x800 event frame)")
    left.set_title("Signed deviation, all pixels (share)", loc="left", color=INK)
    left.legend(["Synthetic (127 neutral)", "Real (128 neutral)"], loc="upper right")

    grid = np.arange(1.0, 501.0)
    for obj, colour in zip(OBJECTS, ("#2a78d6", "#1baf7a", "#eb6834"), strict=True):
        spectra = np.asarray(
            [v["spectrum_1_500Hz_over_median"] for s, v in streams.items() if s.startswith(obj)]
        )
        right.plot(grid, np.median(spectra, axis=0), color=colour, linewidth=1.4, label=obj)
    for band in (100, 200, 300, 400):
        right.axvline(band, color=MUTED, linewidth=0.8, linestyle=":")
    right.set_yscale("log")
    right.set_xlabel("Frequency (Hz)")
    right.set_title("Raw event-rate power / median (median over sequences)", loc="left", color=INK)
    right.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- main


def out_of_support(real: NDArray[np.float64], reference: NDArray[np.float64]) -> float:
    """Share of real values outside the reference distribution's [p1, p99]."""
    real = real[np.isfinite(real)]
    reference = reference[np.isfinite(reference)]
    if real.size == 0 or reference.size == 0:
        return float("nan")
    low, high = np.percentile(reference, [1, 99])
    return float(((real < low) | (real > high)).mean())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/fresh/cache/crop-v2"))
    parser.add_argument("--raw-root", type=Path, default=Path("data/fresh/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/diagnostics/event_stats"))
    parser.add_argument("--real-results", type=Path, default=Path("outputs/real-v2"))
    parser.add_argument("--seed", type=int, default=249)
    parser.add_argument("--train-per-object", type=int, default=1000)
    parser.add_argument("--validation-per-object", type=int, default=300)
    parser.add_argument("--native-synthetic-per-object", type=int, default=200)
    parser.add_argument("--native-real-stride", type=int, default=1)
    parser.add_argument("--stream-seconds", type=float, default=2.0)
    parser.add_argument("--stream-bin-us", type=int, default=250)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--skip-native", action="store_true")
    parser.add_argument("--skip-stream", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _style()
    rng = np.random.default_rng(args.seed)

    # ---- level 1: crop representation
    groups: dict[str, dict[str, dict[str, NDArray[np.float64]]]] = {
        "synthetic": {},
        "validation": {},
        "sim2real": {},
        "real": {},
    }
    caches: dict[str, CropCache] = {}
    selections: dict[str, NDArray[np.int64]] = {}
    examples: dict[str, dict[str, Any]] = {"synthetic": {}, "sim2real": {}, "real": {}}
    real_sequences: dict[str, NDArray[np.str_]] = {}
    for obj in OBJECTS:
        for role, count, group in (
            ("train", args.train_per_object, "synthetic"),
            ("validation", args.validation_per_object, "validation"),
        ):
            cache = CropCache(args.cache_dir / f"{obj}-{role}")
            caches[f"{obj}-{role}"] = cache
            index = np.sort(rng.choice(len(cache.sample_ids), size=count, replace=False))
            selections[f"{obj}-{role}"] = index
            groups[group][obj] = crop_group_stats(cache, index, SYNTHETIC_EVENT_NEUTRAL)
            print(f"[crop] {obj}-{role}: {len(index)} frames", flush=True)
        train = caches[f"{obj}-train"]
        train_index = selections[f"{obj}-train"]
        groups["sim2real"][obj], augmented_examples = augmented_group_stats(
            train, train_index, seed=args.seed + OBJECTS.index(obj)
        )
        print(f"[crop] {obj}-train + sim2real-v1", flush=True)
        real = CropCache(args.cache_dir / f"real-{obj}")
        caches[f"real-{obj}"] = real
        real_index = np.arange(len(real.sample_ids))
        groups["real"][obj] = crop_group_stats(real, real_index, REAL_EVENT_NEUTRAL)
        real_sequences[obj] = real.sequences
        print(f"[crop] real-{obj}: {len(real_index)} frames", flush=True)
        activity = train.array("event_activity_uint8")
        examples["synthetic"][obj] = [activity[train_index[k], 0] / 255.0 for k in range(2)]
        examples["sim2real"][obj] = list(augmented_examples[:2])
        real_activity = real.array("event_activity_uint8")
        picks = [len(real.sample_ids) // 4, (3 * len(real.sample_ids)) // 4]
        examples["real"][obj] = [real_activity[k, 0] / 255.0 for k in picks]

    keys = (
        list(CROP_STATS)
        + list(RAW_CROP_STATS)
        + ["background_mean_abs_deviation", "box_mean_activity", "background_mean_activity"]
    )

    def summarise(group: dict[str, dict[str, NDArray[np.float64]]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for obj, values in group.items():
            out[obj] = {k: describe(values[k]) for k in keys if k in values}
            out[obj]["weak_object_frame_share"] = float(
                np.mean(values["object_contrast"] < WEAK_OBJECT_CONTRAST)
            )
        pooled = {
            k: np.concatenate([group[o][k] for o in group]) for k in keys if k in group[OBJECTS[0]]
        }
        out["pooled"] = {k: describe(v) for k, v in pooled.items()}
        out["pooled"]["weak_object_frame_share"] = float(
            np.mean(pooled["object_contrast"] < WEAK_OBJECT_CONTRAST)
        )
        return out

    crop_summary = {name: summarise(group) for name, group in groups.items()}

    with open_fresh_archive("real", args.raw_root) as archive:
        slices = load_real_slices(archive)
    per_sequence: dict[str, dict[str, dict[str, float]]] = {}
    per_slice: dict[str, dict[str, dict[str, float]]] = {}
    for obj in OBJECTS:
        sequences = real_sequences[obj]
        frames = np.asarray(caches[f"real-{obj}"].array("frame_number"))
        for sequence in sorted(set(sequences.tolist())):
            mask = sequences == sequence
            per_sequence[sequence] = {
                k: describe(v[mask]) for k, v in groups["real"][obj].items() if k in keys
            }
            per_sequence[sequence]["weak_object_frame_share"] = {
                "value": float(
                    np.mean(groups["real"][obj]["object_contrast"][mask] < WEAK_OBJECT_CONTRAST)
                )
            }
        for slice_name in ("harsh", "low-motion"):
            mask = np.zeros(len(sequences), dtype=bool)
            for sequence, ranges in slices.items():
                mask |= (sequences == sequence) & slice_mask(frames, ranges[slice_name])
            rest = ~mask
            per_slice[f"{obj}/{slice_name}"] = {
                k: {
                    "in_slice_median": float(np.nanmedian(v[mask])) if mask.any() else np.nan,
                    "rest_median": float(np.nanmedian(v[rest])),
                    "in_slice_frames": int(mask.sum()),
                }
                for k, v in groups["real"][obj].items()
                if k
                in (
                    "density",
                    "object_contrast",
                    "background_density",
                    "isolated_fraction",
                    "object_share",
                )
            }

    # coverage: share of real frames outside the synthetic / augmented [p1, p99]
    coverage: dict[str, dict[str, dict[str, float]]] = {}
    for obj in OBJECTS:
        coverage[obj] = {}
        for k in CROP_STATS:
            real_values = groups["real"][obj][k]
            coverage[obj][k] = {
                "real_median": float(np.nanmedian(real_values)),
                "synthetic_median": float(np.nanmedian(groups["synthetic"][obj][k])),
                "sim2real_median": float(np.nanmedian(groups["sim2real"][obj][k])),
                "real_outside_synthetic_p1_p99": out_of_support(
                    real_values, groups["synthetic"][obj][k]
                ),
                "real_outside_sim2real_p1_p99": out_of_support(
                    real_values, groups["sim2real"][obj][k]
                ),
            }
    ranking = sorted(
        (
            {
                "statistic": k,
                "mean_real_outside_sim2real_p1_p99": float(
                    np.mean([coverage[o][k]["real_outside_sim2real_p1_p99"] for o in OBJECTS])
                ),
                "mean_real_outside_synthetic_p1_p99": float(
                    np.mean([coverage[o][k]["real_outside_synthetic_p1_p99"] for o in OBJECTS])
                ),
                "real_over_sim2real_median_ratio": {
                    o: coverage[o][k]["real_median"] / coverage[o][k]["sim2real_median"]
                    if coverage[o][k]["sim2real_median"]
                    else None
                    for o in OBJECTS
                },
            }
            for k in CROP_STATS
        ),
        key=lambda row: -row["mean_real_outside_sim2real_p1_p99"],
    )

    # link to the existing event-only result (read only)
    errors: dict[str, float] = {}
    for obj in OBJECTS:
        path = args.real_results / f"s2r-u-event-activity-{obj}" / "frames_primary.npz"
        if path.is_file():
            data = np.load(path)
            for sequence in sorted(set(data["sequence"].tolist())):
                mask = data["sequence"] == sequence
                errors[sequence] = float(np.nanmedian(data["rotation_error_deg"][mask]))
    link: dict[str, Any] = {"per_sequence_median_rotation_error_deg": errors}
    if errors:
        sequences = sorted(errors)
        link["spearman_vs_sequence_median"] = {}
        for k in (
            "density",
            "object_contrast",
            "background_density",
            "isolated_fraction",
            "object_share",
            "thickness_px",
            "negative_share",
            "largest_component_share",
        ):
            rho, p_value = scipy_stats.spearmanr(
                [per_sequence[s][k]["median"] for s in sequences], [errors[s] for s in sequences]
            )
            link["spearman_vs_sequence_median"][k] = {"rho": float(rho), "p": float(p_value)}
        link["note"] = (
            "24 sequences, all near chance; correlations are descriptive only and "
            "not a basis for any model or threshold choice."
        )

    # ---- level 2: native frames
    native: dict[str, Any] = {}
    histograms: dict[str, NDArray[np.float64]] = {}
    if not args.skip_native:
        jobs: list[dict[str, Any]] = []
        for obj in OBJECTS:
            cache = caches[f"{obj}-train"]
            index = np.sort(
                rng.choice(
                    len(cache.sample_ids), size=args.native_synthetic_per_object, replace=False
                )
            )
            members = [
                "synthetic/"
                + cache.sample_ids[i]
                .replace("_rgb.png", "_event.png")
                .replace(f"{obj}/", f"{obj}/frames/", 1)
                for i in index
            ]
            jobs.append(
                {
                    "name": f"synthetic/{obj}",
                    "kind": "synthetic",
                    "members": members,
                    "boxes": np.asarray(cache.array("source_bbox_xyxy")[index]).tolist(),
                    "neutral": SYNTHETIC_EVENT_NEUTRAL,
                    "raw_root": str(args.raw_root),
                }
            )
            real = caches[f"real-{obj}"]
            bboxes = np.asarray(real.array("source_bbox_xyxy"))
            for sequence in sorted(set(real.sequences.tolist())):
                index = np.where(real.sequences == sequence)[0][:: args.native_real_stride]
                members = [
                    f"real/{sequence}/frames/"
                    + Path(real.sample_ids[i]).name.replace("_rgb.png", "_event.png")
                    for i in index
                ]
                jobs.append(
                    {
                        "name": f"real/{sequence}",
                        "kind": "real",
                        "members": members,
                        "boxes": bboxes[index].tolist(),
                        "neutral": REAL_EVENT_NEUTRAL,
                        "raw_root": str(args.raw_root),
                    }
                )
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(native_worker, jobs))
        histograms = {"synthetic": np.zeros(511), "real": np.zeros(511)}
        native_rows: dict[str, dict[str, list[float]]] = {}
        for result in results:
            domain = result["name"].split("/")[0]
            histograms[domain] += np.asarray(result["histogram"], dtype=np.float64)
            native_rows[result["name"]] = result["rows"]
            native[result["name"]] = {
                "hot_pixels": result["hot"],
                **{k: describe(v) for k, v in result["rows"].items()},
            }
        for domain in ("synthetic", "real"):
            share = histograms[domain] / histograms[domain].sum()
            native[f"{domain}_signed_deviation_share"] = {
                "neutral": float(share[255]),
                "abs_1_to_31": float(share[255 - 31 : 255].sum() + share[256 : 256 + 31].sum()),
                "positive_ge_32": float(share[255 + 32 :].sum()),
                "negative_le_minus_32": float(share[: 255 - 31].sum()),
                "distinct_values_present": int((histograms[domain] > 0).sum()),
            }
        for domain in ("synthetic", "real"):
            pooled_rows = [v for k, v in native_rows.items() if k.startswith(domain)]
            native[f"{domain}_pooled"] = {
                k: describe(
                    np.concatenate([np.asarray(r[k], dtype=np.float64) for r in pooled_rows])
                )
                for k in pooled_rows[0]
            }
        print("[native] done", flush=True)

    # ---- level 3: raw streams
    streams: dict[str, dict[str, Any]] = {}
    if not args.skip_stream:
        stream_jobs = [
            {
                "sequence": s,
                "seconds": args.stream_seconds,
                "bin_us": args.stream_bin_us,
                "raw_root": str(args.raw_root),
            }
            for s in sorted(per_sequence)
        ]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for result in pool.map(stream_worker, stream_jobs):
                streams[result["sequence"]] = result
                print(f"[stream] {result['sequence']}", flush=True)

    # ---- figures
    figure_examples(
        args.output_dir / "fig1_activity_examples.png",
        examples["synthetic"],
        examples["sim2real"],
        examples["real"],
    )
    pooled_groups = {
        name: {k: np.concatenate([groups[name][o][k] for o in OBJECTS]) for k in CROP_STATS}
        for name in ("synthetic", "sim2real", "real")
    }
    figure_distributions(args.output_dir / "fig2_crop_distributions.png", pooled_groups)
    bands = {
        name: {obj: {k: describe(groups[name][obj][k]) for k in CROP_STATS} for obj in OBJECTS}
        for name in ("synthetic", "sim2real")
    }
    figure_sequences(args.output_dir / "fig3_per_sequence.png", per_sequence, bands, errors)
    if histograms and streams:
        figure_native(args.output_dir / "fig4_native_and_stream.png", histograms, streams)

    stream_summary = {
        s: {k: v for k, v in r.items() if k != "spectrum_1_500Hz_over_median"}
        for s, r in streams.items()
    }
    summary = {
        "schema": "event-gap-stats-v1",
        "provenance": {
            "script": "scripts/event_gap_stats.py",
            "script_sha256": _sha256_file(Path(__file__)),
            "git_commit": _git("rev-parse", "HEAD"),
            "working_tree_dirty": bool(_git("status", "--porcelain")),
            "uv_lock_sha256": _sha256_file(Path("uv.lock")) if Path("uv.lock").is_file() else None,
            "cache_dir": str(args.cache_dir).replace("\\", "/"),
            "cache_array_sha256": {
                name: {
                    k: v
                    for k, v in cache.array_hashes().items()
                    if k.startswith("event")
                    or k in ("source_bbox_xyxy", "crop_xyxy", "keypoints_xy")
                }
                for name, cache in caches.items()
            },
            "seed": args.seed,
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "torch": torch.__version__,
            "opencv": cv2.__version__,
            "platform": platform.platform(),
            "device": "cpu",
            "runtime_seconds": None,
        },
        "protocol": {
            "samples": {
                "synthetic_train_per_object": args.train_per_object,
                "synthetic_validation_per_object": args.validation_per_object,
                "sim2real": "the same synthetic-train frames after Sim2RealConfig() "
                "(sim2real-v1) augment_batch, seeded generator seed+object_index",
                "real": "all crops of all 24 sequences",
                "native_synthetic_per_object": args.native_synthetic_per_object,
                "native_real_stride": args.native_real_stride,
                "stream_window": f"first {args.stream_seconds} s after the first RGB "
                "frame timestamp",
            },
            "definitions": {
                "active": f"activity map >= {ACTIVE_LEVEL} (crop) or |pixel - neutral| >= "
                f"{ACTIVITY_THRESHOLD:g} (native)",
                "box_density": "active share inside the annotation box mapped into the crop",
                "background_density": f"active share outside the box dilated by "
                f"{BOX_MARGIN_CROP} px (crop) / {BOX_MARGIN_NATIVE} px (native)",
                "object_share": f"share of active pixels inside the convex hull of projected "
                f"keypoints dilated by {NEAR_MARGIN_CROP} px",
                "isolated_fraction": "share of active pixels with no active 8-neighbour",
                "largest_component_share": "share of active pixels in the largest "
                "8-connected component",
                "components_per_kpx": "8-connected components per 1000 pixels",
                "thickness_px": "active pixels / morphological-skeleton pixels "
                "(approximate line width)",
                "negative_share": "darker-than-neutral events / all events in the grey raw map",
                "weak_density": "share of pixels with 2 <= |pixel - neutral| < 32 "
                "(sub-threshold residue)",
                "hot_pixels": "pixels active in >= 50 % / >= 90 % of frames (native frames) "
                "or of 33.3 ms intervals (raw stream)",
                "out_of_support": "share of real frames outside the reference group's [p1, p99]",
            },
        },
        "crop_representation": crop_summary,
        "real_per_sequence": per_sequence,
        "real_author_slices": per_slice,
        "coverage": coverage,
        "ranking_by_real_outside_sim2real": ranking,
        "native_frames": native,
        "raw_streams": stream_summary,
        "event_only_result_link": link,
    }
    summary["provenance"]["runtime_seconds"] = round(time.perf_counter() - started, 1)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.output_dir / 'summary.json'} in {time.perf_counter() - started:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
