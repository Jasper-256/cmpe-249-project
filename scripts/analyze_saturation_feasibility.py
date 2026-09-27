"""Run the predeclared synthetic-only RGB clipping viability screen."""

from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from saga_pose.analysis.saturation import (
    clipped_pixel_mask,
    crop_xyxy,
    event_activity_mask,
    patch_fraction_map,
    resize_rgb,
)
from saga_pose.data.archives import open_fresh_archive
from saga_pose.data.fresh import frame_member, load_metadata, metadata_members

THRESHOLDS = (250, 254, 255)
GRID_SIZE = 16
RESIZE_WH = (256, 256)
PATCH_ACTIVATION_FRACTION = 0.01
GO_PREVALENCE = 0.10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("data/fresh/raw"))
    parser.add_argument("--samples-per-object-split", type=int, default=64)
    parser.add_argument("--seed", type=int, default=249)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/feasibility/saturation-signal.json"),
    )
    return parser.parse_args()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _decode_rgb(payload: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(payload)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "p25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "p75": float(np.quantile(array, 0.75)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def _sample_indices(length: int, count: int, rng: np.random.Generator) -> np.ndarray:
    if count <= 0:
        raise ValueError("samples-per-object-split must be positive")
    count = min(count, length)
    return np.sort(rng.choice(length, size=count, replace=False))


def analyze(raw_root: Path, samples_per_split: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    patch_clip_254: list[np.ndarray] = []
    patch_event_activity: list[np.ndarray] = []
    compressed_bytes = 0
    decoded_bytes = 0
    started = time.perf_counter()

    with open_fresh_archive("synthetic", raw_root) as archive:
        for member in metadata_members(archive, real=False):
            metadata = load_metadata(archive, member)
            indices = _sample_indices(len(metadata.annotations), samples_per_split, rng)
            for annotation_index in indices:
                annotation = metadata.annotations[int(annotation_index)]
                rgb_member = frame_member(member, annotation.filename_rgb)
                event_member = frame_member(member, annotation.filename_event)
                rgb_bytes = archive.read(rgb_member)
                event_bytes = archive.read(event_member)
                compressed_bytes += archive.getinfo(rgb_member).compress_size
                compressed_bytes += archive.getinfo(event_member).compress_size
                rgb = _decode_rgb(rgb_bytes)
                event = _decode_rgb(event_bytes)
                decoded_bytes += rgb.nbytes + event.nbytes

                rgb_crop = crop_xyxy(rgb, annotation.bbox_xyxy)
                event_crop = crop_xyxy(event, annotation.bbox_xyxy)
                rgb_resized = resize_rgb(rgb_crop, RESIZE_WH)
                event_patch = patch_fraction_map(event_activity_mask(event_crop), GRID_SIZE)
                patch_event_activity.append(event_patch.reshape(-1))

                threshold_stats: dict[str, Any] = {}
                for threshold in THRESHOLDS:
                    raw_mask = clipped_pixel_mask(rgb_crop, threshold)
                    resized_mask = clipped_pixel_mask(rgb_resized, threshold)
                    raw_patch = patch_fraction_map(raw_mask, GRID_SIZE)
                    resized_patch = patch_fraction_map(resized_mask, GRID_SIZE)
                    if threshold == 254:
                        patch_clip_254.append(raw_patch.reshape(-1))
                    threshold_stats[str(threshold)] = {
                        "raw_crop_fraction": float(np.mean(raw_mask)),
                        "raw_max_patch_fraction": float(np.max(raw_patch)),
                        "raw_patch_fraction_ge_1pct": float(
                            np.mean(raw_patch >= PATCH_ACTIVATION_FRACTION)
                        ),
                        "raw_has_patch_ge_1pct": bool(
                            np.any(raw_patch >= PATCH_ACTIVATION_FRACTION)
                        ),
                        "resized_crop_fraction": float(np.mean(resized_mask)),
                        "resized_max_patch_fraction": float(np.max(resized_patch)),
                        "resized_patch_fraction_ge_1pct": float(
                            np.mean(resized_patch >= PATCH_ACTIVATION_FRACTION)
                        ),
                        "resized_has_patch_ge_1pct": bool(
                            np.any(resized_patch >= PATCH_ACTIVATION_FRACTION)
                        ),
                    }

                rows.append(
                    {
                        "metadata_member": member,
                        "object": metadata.object_name,
                        "role": metadata.role,
                        "annotation_offset": int(annotation_index),
                        "rgb_member": rgb_member,
                        "event_member": event_member,
                        "bbox_xyxy": list(annotation.bbox_xyxy),
                        "crop_size_wh": [int(rgb_crop.shape[1]), int(rgb_crop.shape[0])],
                        "rgb_crop_max_value": int(np.max(rgb_crop)),
                        "event_activity_fraction": float(np.mean(event_activity_mask(event_crop))),
                        "event_max_patch_activity": float(np.max(event_patch)),
                        "thresholds": threshold_stats,
                    }
                )

    elapsed = time.perf_counter() - started
    threshold_summaries: dict[str, Any] = {}
    for threshold in THRESHOLDS:
        key = str(threshold)
        raw_global = [float(row["thresholds"][key]["raw_crop_fraction"]) for row in rows]
        raw_max_patch = [float(row["thresholds"][key]["raw_max_patch_fraction"]) for row in rows]
        resized_global = [float(row["thresholds"][key]["resized_crop_fraction"]) for row in rows]
        raw_prevalence = float(
            np.mean([row["thresholds"][key]["raw_has_patch_ge_1pct"] for row in rows])
        )
        resized_prevalence = float(
            np.mean([row["thresholds"][key]["resized_has_patch_ge_1pct"] for row in rows])
        )
        threshold_summaries[key] = {
            "raw_crop_clipped_fraction": _quantiles(raw_global),
            "raw_max_patch_clipped_fraction": _quantiles(raw_max_patch),
            "raw_crop_prevalence_with_patch_ge_1pct": raw_prevalence,
            "resized_crop_clipped_fraction": _quantiles(resized_global),
            "resized_crop_prevalence_with_patch_ge_1pct": resized_prevalence,
            "prevalence_change_after_resize": resized_prevalence - raw_prevalence,
        }

    clip_vector = np.concatenate(patch_clip_254)
    event_vector = np.concatenate(patch_event_activity)
    if np.std(clip_vector) == 0.0 or np.std(event_vector) == 0.0:
        clip_event_correlation: float | None = None
    else:
        clip_event_correlation = float(np.corrcoef(clip_vector, event_vector)[0, 1])

    primary_prevalence = float(threshold_summaries["254"]["raw_crop_prevalence_with_patch_ge_1pct"])
    proceed = primary_prevalence >= GO_PREVALENCE
    manifest = json.loads(Path("docs/fresh-dataset-manifest.json").read_text(encoding="utf-8"))
    synthetic_manifest = next(item for item in manifest["files"] if item["name"] == "synthetic.zip")

    return {
        "schema_version": 1,
        "status": "go" if proceed else "stop",
        "decision": {
            "proceed_with_natural_synthetic_learned_gate": proceed,
            "observed_prevalence": primary_prevalence,
            "required_prevalence": GO_PREVALENCE,
            "rule": (
                "At threshold 254 before resize, at least 10% of sampled target crops "
                "must contain a 16x16-grid patch with at least 1% clipped pixels."
            ),
        },
        "protocol": {
            "dataset_scope": "synthetic train and synthetic validation only",
            "sample_method": "stratified uniform random without replacement",
            "samples_per_object_split": samples_per_split,
            "seed": seed,
            "thresholds_8bit": list(THRESHOLDS),
            "grid_size": GRID_SIZE,
            "resize_wh": list(RESIZE_WH),
            "resize_interpolation": "OpenCV INTER_LINEAR",
            "crop_source": "annotation XYXY target box; shared/oracle diagnostic",
            "clipping_definition": "max RGB channel >= threshold before normalization",
            "event_activity_definition": "any channel differs from neutral value 127",
            "synthetic_archive_md5": synthetic_manifest["md5"],
        },
        "sample_count": len(rows),
        "threshold_summaries": threshold_summaries,
        "event_activity_fraction": _quantiles(
            [float(row["event_activity_fraction"]) for row in rows]
        ),
        "rgb_crop_max_value": _quantiles([float(row["rgb_crop_max_value"]) for row in rows]),
        "zero_observation_prevalence_upper_95_rule_of_three": (
            3.0 / len(rows) if primary_prevalence == 0.0 else None
        ),
        "patch_clip_event_pearson_at_254": clip_event_correlation,
        "io": {
            "elapsed_seconds": elapsed,
            "compressed_member_bytes": compressed_bytes,
            "decoded_array_bytes": decoded_bytes,
            "samples_per_second": len(rows) / elapsed,
            "decoded_megabytes_per_second": decoded_bytes / elapsed / 1_000_000,
        },
        "samples": rows,
    }


def main() -> int:
    args = parse_args()
    report = analyze(args.raw_root, args.samples_per_object_split, args.seed)
    _write_json(args.output, report)
    decision = report["decision"]
    print(f"Synthetic clipping viability: {report['status'].upper()}")
    print(f"  sampled pairs: {report['sample_count']:,}")
    print(
        "  threshold-254 crop prevalence: "
        f"{100 * decision['observed_prevalence']:.2f}% "
        f"(required {100 * decision['required_prevalence']:.1f}%)"
    )
    print(f"  decode + analysis: {report['io']['samples_per_second']:.1f} pairs/s")
    print(f"  report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
