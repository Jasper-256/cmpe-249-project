"""Create a leakage-resistant grouped synthetic train/validation split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from saga_pose.data.archives import open_fresh_archive
from saga_pose.data.fresh import FreshAnnotation, load_metadata

BLOCK_COUNT = 8
BLOCK_LENGTH = 80
ROTATION_GUARD_DEGREES = 5.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("data/fresh/raw"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/fresh/index/synthetic-split-v1.json"),
    )
    return parser.parse_args()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sample_id(object_name: str, annotation: FreshAnnotation) -> str:
    return f"{object_name}/{annotation.filename_rgb}"


def _membership_hash(sample_ids: list[str]) -> str:
    canonical = "\n".join(sorted(sample_ids)).encode()
    return hashlib.sha256(canonical).hexdigest()


def _minimum_rotation_distance_degrees(
    query_quaternions: np.ndarray,
    reference_quaternions: np.ndarray,
) -> np.ndarray:
    minimum = np.full(query_quaternions.shape[0], np.inf, dtype=np.float64)
    for start in range(0, query_quaternions.shape[0], 2048):
        chunk = query_quaternions[start : start + 2048]
        absolute_dots = np.clip(np.abs(chunk @ reference_quaternions.T), 0.0, 1.0)
        distances = np.degrees(2.0 * np.arccos(absolute_dots))
        minimum[start : start + len(chunk)] = np.min(distances, axis=1)
    return minimum


def create_split(raw_root: Path) -> dict[str, Any]:
    object_reports: dict[str, Any] = {}
    with open_fresh_archive("synthetic", raw_root) as archive:
        for object_name in ("cassini", "satty", "soho"):
            annotations_with_source: list[tuple[FreshAnnotation, str]] = []
            for source_role, filename in (
                ("author_train", "train.json"),
                ("author_test", "test.json"),
            ):
                member = f"synthetic/{object_name}/{filename}"
                metadata = load_metadata(archive, member)
                annotations_with_source.extend(
                    (annotation, source_role) for annotation in metadata.annotations
                )
            annotations_with_source.sort(
                key=lambda item: int(item[0].filename_rgb.split("_", maxsplit=1)[0])
            )
            annotations = [item[0] for item in annotations_with_source]
            sample_count = len(annotations)

            centers = np.linspace(0, sample_count - 1, BLOCK_COUNT + 2)[1:-1]
            validation_positions: set[int] = set()
            block_ranges: list[list[int]] = []
            for center in centers:
                start = round(center - BLOCK_LENGTH / 2)
                start = max(0, min(sample_count - BLOCK_LENGTH, start))
                stop = start + BLOCK_LENGTH
                validation_positions.update(range(start, stop))
                block_ranges.append([start, stop])

            rotations = np.stack([annotation.pose[:3, :3] for annotation in annotations])
            quaternions = Rotation.from_matrix(rotations).as_quat()
            validation_indices = np.asarray(sorted(validation_positions), dtype=np.int64)
            candidate_indices = np.asarray(
                [index for index in range(sample_count) if index not in validation_positions],
                dtype=np.int64,
            )
            distances = _minimum_rotation_distance_degrees(
                quaternions[candidate_indices],
                quaternions[validation_indices],
            )
            train_indices = candidate_indices[distances > ROTATION_GUARD_DEGREES]
            guard_indices = candidate_indices[distances <= ROTATION_GUARD_DEGREES]
            retained_distances = distances[distances > ROTATION_GUARD_DEGREES]

            train_ids = [
                _sample_id(object_name, annotations[int(index)]) for index in train_indices
            ]
            validation_ids = [
                _sample_id(object_name, annotations[int(index)]) for index in validation_indices
            ]
            guard_ids = [
                _sample_id(object_name, annotations[int(index)]) for index in guard_indices
            ]
            source_counts: dict[str, dict[str, int]] = {}
            for role, indices in (
                ("train", train_indices),
                ("validation", validation_indices),
                ("guard_excluded", guard_indices),
            ):
                source_counts[role] = {
                    source: sum(
                        annotations_with_source[int(index)][1] == source for index in indices
                    )
                    for source in ("author_train", "author_test")
                }

            object_reports[object_name] = {
                "total": sample_count,
                "train_count": len(train_ids),
                "validation_count": len(validation_ids),
                "guard_excluded_count": len(guard_ids),
                "minimum_train_validation_rotation_degrees": float(np.min(retained_distances)),
                "validation_position_blocks_half_open": block_ranges,
                "source_counts": source_counts,
                "train_membership_sha256": _membership_hash(train_ids),
                "validation_membership_sha256": _membership_hash(validation_ids),
                "guard_membership_sha256": _membership_hash(guard_ids),
                "train_ids": train_ids,
                "validation_ids": validation_ids,
                "guard_excluded_ids": guard_ids,
            }

    minimum_distance = min(
        report["minimum_train_validation_rotation_degrees"] for report in object_reports.values()
    )
    return {
        "schema_version": 1,
        "split_name": "fresh-synthetic-grouped-v1",
        "status": "pass" if minimum_distance > ROTATION_GUARD_DEGREES else "fail",
        "source": "union of author train.json and test.json; real data is never included",
        "method": {
            "ordering": "numeric RGB filename within each object",
            "validation_blocks": BLOCK_COUNT,
            "frames_per_validation_block": BLOCK_LENGTH,
            "rotation_guard_degrees": ROTATION_GUARD_DEGREES,
            "guard_rule": (
                "Exclude every non-validation frame within 5 geodesic SO(3) degrees of "
                "any validation pose."
            ),
        },
        "minimum_train_validation_rotation_degrees": minimum_distance,
        "objects": object_reports,
    }


def main() -> int:
    args = parse_args()
    report = create_split(args.raw_root)
    _write_json(args.output, report)
    print(f"Grouped synthetic split: {report['status'].upper()}")
    for object_name, object_report in report["objects"].items():
        print(
            f"  {object_name}: train={object_report['train_count']:,}, "
            f"validation={object_report['validation_count']:,}, "
            f"guard={object_report['guard_excluded_count']:,}, "
            "min rotation="
            f"{object_report['minimum_train_validation_rotation_degrees']:.3f} deg"
        )
    print(f"  report: {args.output}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
