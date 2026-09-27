"""Evaluate two bounded synthetic clipping augmentations without real data."""

from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import albumentations as A
import numpy as np
from PIL import Image

from saga_pose.analysis.saturation import (
    clipped_pixel_mask,
    crop_xyxy,
    patch_fraction_map,
    resize_rgb,
)
from saga_pose.data.archives import open_fresh_archive
from saga_pose.data.fresh import frame_member, load_metadata, metadata_members

GRID_SIZE = 16
RESIZE_WH = (256, 256)
THRESHOLD = 254
PATCH_ACTIVATION_FRACTION = 0.01
TRAINING_APPLICATION_PROBABILITY = 0.5

PROFILES: dict[str, dict[str, Any]] = {
    "light": {
        "src_radius": 24,
        "num_flare_circles_range": (1, 3),
    },
    "moderate": {
        "src_radius": 48,
        "num_flare_circles_range": (2, 4),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("data/fresh/raw"))
    parser.add_argument("--samples-per-object-split", type=int, default=16)
    parser.add_argument("--seed", type=int, default=249)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/feasibility/flare-augmentation.json"),
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


def _sample_indices(length: int, count: int, rng: np.random.Generator) -> np.ndarray:
    if count <= 0:
        raise ValueError("samples-per-object-split must be positive")
    return np.sort(rng.choice(length, size=min(count, length), replace=False))


def _make_transform(profile: dict[str, Any], seed: int) -> A.Compose:
    return A.Compose(
        [
            A.RandomSunFlare(
                flare_roi=(0.0, 0.0, 1.0, 1.0),
                angle_range=(0.0, 1.0),
                src_color=(255, 255, 255),
                method="physics_based",
                p=1.0,
                **profile,
            )
        ],
        seed=seed,
        strict=True,
    )


def _quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def analyze(raw_root: Path, samples_per_split: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    samples: list[tuple[str, str, np.ndarray]] = []
    with open_fresh_archive("synthetic", raw_root) as archive:
        for member in metadata_members(archive, real=False):
            metadata = load_metadata(archive, member)
            indices = _sample_indices(len(metadata.annotations), samples_per_split, rng)
            for annotation_index in indices:
                annotation = metadata.annotations[int(annotation_index)]
                rgb_member = frame_member(member, annotation.filename_rgb)
                rgb = _decode_rgb(archive.read(rgb_member))
                crop = crop_xyxy(rgb, annotation.bbox_xyxy)
                resized = resize_rgb(crop, RESIZE_WH)
                samples.append((member, rgb_member, resized))

    profile_reports: dict[str, Any] = {}
    for profile_name, profile in PROFILES.items():
        first_transform = _make_transform(profile, seed)
        repeated_transform = _make_transform(profile, seed)
        rows: list[dict[str, Any]] = []
        deterministic = True
        shapes_preserved = True
        for member, rgb_member, image in samples:
            augmented = np.asarray(first_transform(image=image)["image"], dtype=np.uint8)
            repeated = np.asarray(repeated_transform(image=image)["image"], dtype=np.uint8)
            deterministic = deterministic and np.array_equal(augmented, repeated)
            shapes_preserved = shapes_preserved and augmented.shape == image.shape
            mask = clipped_pixel_mask(augmented, THRESHOLD)
            patch_map = patch_fraction_map(mask, GRID_SIZE)
            rows.append(
                {
                    "metadata_member": member,
                    "rgb_member": rgb_member,
                    "clipped_fraction": float(np.mean(mask)),
                    "max_patch_clipped_fraction": float(np.max(patch_map)),
                    "active_patch_fraction": float(np.mean(patch_map >= PATCH_ACTIVATION_FRACTION)),
                    "has_active_patch": bool(np.any(patch_map >= PATCH_ACTIVATION_FRACTION)),
                }
            )

        crop_prevalence = float(np.mean([row["has_active_patch"] for row in rows]))
        active_patch_values = [float(row["active_patch_fraction"]) for row in rows]
        clipped_values = [float(row["clipped_fraction"]) for row in rows]
        median_active_patch_fraction = float(np.median(active_patch_values))
        median_clipped_fraction = float(np.median(clipped_values))
        accepted = (
            deterministic
            and shapes_preserved
            and crop_prevalence >= 0.90
            and 0.02 <= median_active_patch_fraction <= 0.50
            and median_clipped_fraction <= 0.25
        )
        profile_reports[profile_name] = {
            "parameters": profile,
            "accepted": accepted,
            "deterministic_replay": deterministic,
            "shapes_preserved": shapes_preserved,
            "augmented_crop_prevalence_with_patch_ge_1pct": crop_prevalence,
            "expected_training_batch_prevalence_at_p_0_5": (
                TRAINING_APPLICATION_PROBABILITY * crop_prevalence
            ),
            "active_patch_fraction": _quantiles(active_patch_values),
            "clipped_pixel_fraction": _quantiles(clipped_values),
            "samples": rows,
        }

    selected = next(
        (name for name in PROFILES if profile_reports[name]["accepted"]),
        None,
    )
    manifest = json.loads(Path("docs/fresh-dataset-manifest.json").read_text(encoding="utf-8"))
    synthetic_manifest = next(item for item in manifest["files"] if item["name"] == "synthetic.zip")
    return {
        "schema_version": 1,
        "status": "go" if selected is not None else "stop",
        "selected_profile": selected,
        "acceptance_rule": {
            "deterministic_replay": True,
            "shape_preserved": True,
            "augmented_crop_prevalence_min": 0.90,
            "median_active_patch_fraction_range": [0.02, 0.50],
            "median_clipped_pixel_fraction_max": 0.25,
        },
        "protocol": {
            "dataset_scope": "synthetic train and synthetic validation only",
            "sample_count": len(samples),
            "samples_per_object_split": samples_per_split,
            "seed": seed,
            "input_size_wh": list(RESIZE_WH),
            "threshold_8bit": THRESHOLD,
            "grid_size": GRID_SIZE,
            "training_application_probability": TRAINING_APPLICATION_PROBABILITY,
            "library": f"albumentations {A.__version__}",
            "transform": "RandomSunFlare(method='physics_based')",
            "synthetic_archive_md5": synthetic_manifest["md5"],
            "limitation": (
                "Framework-provided optical approximation; not calibrated to FRESH real "
                "camera statistics and not evidence of physical fidelity."
            ),
        },
        "profiles": profile_reports,
    }


def main() -> int:
    args = parse_args()
    report = analyze(args.raw_root, args.samples_per_object_split, args.seed)
    _write_json(args.output, report)
    print(f"Synthetic flare augmentation viability: {report['status'].upper()}")
    print(f"  selected profile: {report['selected_profile']}")
    for name, profile in report["profiles"].items():
        print(
            f"  {name}: accepted={profile['accepted']}, "
            "crop prevalence="
            f"{100 * profile['augmented_crop_prevalence_with_patch_ge_1pct']:.1f}%, "
            f"median active patches={100 * profile['active_patch_fraction']['median']:.1f}%"
        )
    print(f"  report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
