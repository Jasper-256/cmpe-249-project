"""Visual audit of the event pipeline: event maps as fed to the network vs RGB edges.

For two frames of each real FRESH sequence (and a few synthetic validation
samples) it overlays the crop-v2 event map (the ``activity`` map the event
branch receives, before backbone normalization) on the paired RGB crop's edge
map, with ground-truth keypoints projected into the crop. Where an existing
round-2 event-only prediction file is present, its saved keypoints are drawn
too (read only; nothing is re-evaluated).

A label-free alignment statistic is printed per sequence: the integer shift
(within +/-24 px) maximizing the normalized cross-correlation between the
event activity map and the top-decile RGB edge map, and the correlation at
zero shift, for the unflipped and left-right / up-down flipped event map.

Usage: uv run python scripts/audit_event_pipeline.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

OBJECTS = ("cassini", "satty", "soho")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/fresh/cache/crop-v2"))
    parser.add_argument("--real-outputs", type=Path, default=Path("outputs/real-v2"))
    parser.add_argument("--event-run", default="s2r-u-event-activity")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/diagnostics/event_audit"))
    parser.add_argument("--synthetic-per-object", type=int, default=4)
    return parser.parse_args()


def _open(cache: Path) -> tuple[dict[str, Any], dict[str, NDArray[Any]]]:
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    names = ("rgb_uint8", "event_activity_uint8", "event_raw_uint8", "keypoints_xy")
    arrays = {name: np.load(cache / f"{name}.npy", mmap_mode="r") for name in names}
    return manifest, arrays


def edge_map(rgb_chw: NDArray[np.uint8]) -> NDArray[np.float32]:
    grey = rgb_chw.astype(np.float32).mean(axis=0)
    return np.hypot(ndimage.sobel(grey, 1), ndimage.sobel(grey, 0)).astype(np.float32)


def _ncc_shift(
    a: NDArray[np.float32], b: NDArray[np.float32], radius: int = 24
) -> tuple[int, int, float, float]:
    a = a - a.mean()
    b = b - b.mean()
    corr = np.fft.fftshift(np.fft.ifft2(np.fft.fft2(a) * np.conj(np.fft.fft2(b))).real)
    centre = np.array(corr.shape) // 2
    window = corr[
        centre[0] - radius : centre[0] + radius + 1, centre[1] - radius : centre[1] + radius + 1
    ]
    iy, ix = np.unravel_index(int(np.argmax(window)), window.shape)
    norm = float(np.sqrt((a**2).sum() * (b**2).sum())) + 1e-9
    return (
        int(iy - radius),
        int(ix - radius),
        float(window.max()) / norm,
        float(corr[centre[0], centre[1]]) / norm,
    )


def alignment_stats(rgb: NDArray[np.uint8], activity: NDArray[np.uint8]) -> dict[str, float]:
    edges = edge_map(rgb)
    edges = (edges > np.percentile(edges, 90)).astype(np.float32)
    event = activity.astype(np.float32) / 255.0
    stats: dict[str, float] = {}
    for label, variant in (("none", event), ("lr", event[:, ::-1]), ("ud", event[::-1])):
        dy, dx, peak, zero = _ncc_shift(edges, ndimage.gaussian_filter(variant.copy(), 1.5))
        stats[f"{label}_dy"], stats[f"{label}_dx"] = dy, dx
        stats[f"{label}_peak"], stats[f"{label}_zero"] = peak, zero
    return stats


def overlay(
    ax: Any,
    rgb: NDArray[np.uint8],
    activity: NDArray[np.uint8],
    truth: NDArray[np.float32],
    predicted: NDArray[np.float32] | None,
    title: str,
) -> None:
    edges = edge_map(rgb)
    edges = np.clip(edges / (np.percentile(edges, 99) + 1e-6), 0, 1)
    event = activity.astype(np.float32) / 255.0
    image = np.stack([event, 0.75 * edges, 0.75 * edges], axis=-1)  # events red, edges cyan
    ax.imshow(np.clip(image, 0, 1), interpolation="nearest")
    ax.scatter(truth[:, 0], truth[:, 1], s=7, c="yellow", marker="o", linewidths=0)
    if predicted is not None:
        ax.scatter(predicted[:, 0], predicted[:, 1], s=9, c="lime", marker="x", linewidths=0.8)
    ax.set_title(title, fontsize=6)
    ax.set_xlim(0, rgb.shape[-1])
    ax.set_ylim(rgb.shape[-2], 0)
    ax.axis("off")


def event_panel(
    ax: Any, activity: NDArray[np.uint8], truth: NDArray[np.float32], title: str
) -> None:
    ax.imshow(activity, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ax.scatter(truth[:, 0], truth[:, 1], s=5, c="yellow", marker="o", linewidths=0, alpha=0.6)
    ax.set_title(title, fontsize=6)
    ax.axis("off")


def rgb_panel(ax: Any, rgb: NDArray[np.uint8], truth: NDArray[np.float32], title: str) -> None:
    ax.imshow(np.moveaxis(rgb, 0, -1))
    ax.scatter(truth[:, 0], truth[:, 1], s=7, c="yellow", marker="o", linewidths=0)
    ax.set_title(title, fontsize=6)
    ax.axis("off")


def _predictions(path: Path) -> dict[str, NDArray[np.float32]]:
    if not path.is_file():
        return {}
    frames = np.load(path)
    return {
        str(sample): np.asarray(points, dtype=np.float32)
        for sample, points in zip(
            frames["sample_id"], frames["predicted_keypoints_xy"], strict=True
        )
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"event_map": "event_activity_uint8 (crop-v2)", "sequences": {}}
    for object_name in OBJECTS:
        manifest, arrays = _open(args.cache_dir / f"real-{object_name}")
        predicted = _predictions(
            args.real_outputs / f"{args.event_run}-{object_name}" / "frames_primary.npz"
        )
        sequences = np.asarray(manifest["sequences"])
        names = sorted(set(sequences.tolist()))
        fig, axes = plt.subplots(len(names), 6, figsize=(9.0, 1.55 * len(names)), dpi=100)
        for row, sequence in enumerate(names):
            indices = np.flatnonzero(sequences == sequence)
            stats = [
                alignment_stats(arrays["rgb_uint8"][i], arrays["event_activity_uint8"][i, 0])
                for i in indices[np.linspace(0, len(indices) - 1, 12).astype(int)]
            ]
            summary["sequences"][sequence] = {
                key: float(np.median([s[key] for s in stats])) for key in stats[0]
            }
            summary["sequences"][sequence]["active_fraction_median"] = float(
                np.median(
                    [
                        (arrays["event_activity_uint8"][i, 0] > 127).mean()
                        for i in indices[np.linspace(0, len(indices) - 1, 12).astype(int)]
                    ]
                )
            )
            for column, position in enumerate((len(indices) // 3, (2 * len(indices)) // 3)):
                index = int(indices[position])
                sample = manifest["sample_ids"][index]
                rgb = np.asarray(arrays["rgb_uint8"][index])
                truth = np.asarray(arrays["keypoints_xy"][index])
                label = f"{sample.replace('_rgb.png', '')}"
                activity = np.asarray(arrays["event_activity_uint8"][index, 0])
                rgb_panel(axes[row, 3 * column], rgb, truth, label + " RGB")
                event_panel(axes[row, 3 * column + 1], activity, truth, "activity map (as fed)")
                overlay(
                    axes[row, 3 * column + 2],
                    rgb,
                    np.asarray(arrays["event_activity_uint8"][index, 0]),
                    truth,
                    predicted.get(sample),
                    "activity (red) / RGB edges (cyan)",
                )
        fig.suptitle(
            f"real {object_name}: yellow = GT keypoints, "
            f"green x = saved {args.event_run} prediction",
            fontsize=7,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.985))
        fig.savefig(args.output_dir / f"real_{object_name}.png")
        plt.close(fig)

    count = args.synthetic_per_object
    fig, axes = plt.subplots(
        len(OBJECTS), 3 * count, figsize=(1.5 * 3 * count, 1.7 * len(OBJECTS)), dpi=100
    )
    for row, object_name in enumerate(OBJECTS):
        manifest, arrays = _open(args.cache_dir / f"{object_name}-validation")
        total = len(manifest["sample_ids"])
        for column, index in enumerate(np.linspace(0, total - 1, count).astype(int)):
            rgb = np.asarray(arrays["rgb_uint8"][index])
            truth = np.asarray(arrays["keypoints_xy"][index])
            sample = manifest["sample_ids"][index].replace("_rgb.png", "")
            rgb_panel(axes[row, 3 * column], rgb, truth, f"{sample} RGB")
            event_panel(
                axes[row, 3 * column + 1],
                np.asarray(arrays["event_activity_uint8"][index, 0]),
                truth,
                "activity map",
            )
            overlay(
                axes[row, 3 * column + 2],
                rgb,
                np.asarray(arrays["event_activity_uint8"][index, 0]),
                truth,
                None,
                "activity / edges",
            )
    fig.suptitle("synthetic validation (unaugmented cache maps): yellow = GT keypoints", fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(args.output_dir / "synthetic.png")
    plt.close(fig)

    (args.output_dir / "alignment_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    for sequence, stats in summary["sequences"].items():
        print(
            f"{sequence:16s} shift=({stats['none_dy']:+.0f},{stats['none_dx']:+.0f}) "
            f"ncc0={stats['none_zero']:.3f} lr={stats['lr_peak']:.3f} ud={stats['ud_peak']:.3f} "
            f"active={stats['active_fraction_median']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
