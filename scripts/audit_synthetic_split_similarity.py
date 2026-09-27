"""Check synthetic validation frames against nearest numbered training neighbors."""

from __future__ import annotations

import argparse
import io
import json
from bisect import bisect_left
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from saga_pose.analysis.saturation import crop_xyxy
from saga_pose.data.archives import open_fresh_archive
from saga_pose.data.fresh import FreshAnnotation, frame_member, load_metadata
from saga_pose.evaluation.metrics import rotation_error_degrees

COMPARISON_SIZE_WH = (64, 64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("data/fresh/raw"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/fresh/index/synthetic-split-similarity.json"),
    )
    return parser.parse_args()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _numeric_stem(filename: str) -> int:
    return int(filename.split("_", maxsplit=1)[0])


def _decode_crop(archive: Any, metadata_member: str, annotation: FreshAnnotation) -> np.ndarray:
    payload = archive.read(frame_member(metadata_member, annotation.filename_rgb))
    with Image.open(io.BytesIO(payload)) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    crop = crop_xyxy(rgb, annotation.bbox_xyxy)
    return cv2.resize(crop, COMPARISON_SIZE_WH, interpolation=cv2.INTER_AREA)


def _difference_hash(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    return resized[:, 1:] > resized[:, :-1]


def _quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "p05": float(np.quantile(array, 0.05)),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def audit(raw_root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    with open_fresh_archive("synthetic", raw_root) as archive:
        for object_name in ("cassini", "satty", "soho"):
            train_member = f"synthetic/{object_name}/train.json"
            validation_member = f"synthetic/{object_name}/test.json"
            train = load_metadata(archive, train_member)
            validation = load_metadata(archive, validation_member)
            train_by_number = {
                _numeric_stem(annotation.filename_rgb): annotation
                for annotation in train.annotations
            }
            train_numbers = sorted(train_by_number)

            for validation_annotation in validation.annotations:
                number = _numeric_stem(validation_annotation.filename_rgb)
                insertion = bisect_left(train_numbers, number)
                candidate_numbers = train_numbers[max(0, insertion - 1) : insertion + 1]
                validation_crop = _decode_crop(
                    archive,
                    validation_member,
                    validation_annotation,
                )
                validation_hash = _difference_hash(validation_crop)
                comparisons: list[dict[str, Any]] = []
                for candidate_number in candidate_numbers:
                    train_annotation = train_by_number[candidate_number]
                    train_crop = _decode_crop(archive, train_member, train_annotation)
                    hamming = int(np.count_nonzero(validation_hash != _difference_hash(train_crop)))
                    mean_absolute_difference = float(
                        np.mean(
                            np.abs(
                                validation_crop.astype(np.float32) - train_crop.astype(np.float32)
                            )
                        )
                        / 255.0
                    )
                    rotation_difference = float(
                        rotation_error_degrees(
                            validation_annotation.pose[None, :3, :3],
                            train_annotation.pose[None, :3, :3],
                        )[0]
                    )
                    translation_difference = float(
                        np.linalg.norm(
                            validation_annotation.pose[:3, 3] - train_annotation.pose[:3, 3]
                        )
                    )
                    comparisons.append(
                        {
                            "train_rgb": train_annotation.filename_rgb,
                            "number_distance": abs(candidate_number - number),
                            "dhash_hamming_64bit": hamming,
                            "crop_mean_absolute_difference_0_1": mean_absolute_difference,
                            "rotation_difference_degrees": rotation_difference,
                            "translation_difference_mm": translation_difference,
                        }
                    )
                closest = min(
                    comparisons,
                    key=lambda item: (
                        item["dhash_hamming_64bit"],
                        item["crop_mean_absolute_difference_0_1"],
                    ),
                )
                near_duplicate_proxy = bool(
                    closest["dhash_hamming_64bit"] <= 4
                    and closest["crop_mean_absolute_difference_0_1"] <= 0.02
                    and closest["rotation_difference_degrees"] <= 1.0
                )
                rows.append(
                    {
                        "object": object_name,
                        "validation_rgb": validation_annotation.filename_rgb,
                        "closest_adjacent_train": closest,
                        "near_duplicate_proxy": near_duplicate_proxy,
                    }
                )

    hamming_values = [float(row["closest_adjacent_train"]["dhash_hamming_64bit"]) for row in rows]
    image_differences = [
        float(row["closest_adjacent_train"]["crop_mean_absolute_difference_0_1"]) for row in rows
    ]
    rotation_differences = [
        float(row["closest_adjacent_train"]["rotation_difference_degrees"]) for row in rows
    ]
    proxy_count = sum(bool(row["near_duplicate_proxy"]) for row in rows)
    return {
        "schema_version": 1,
        "status": "pass" if proxy_count == 0 else "review",
        "scope": (
            "Every synthetic validation RGB crop versus the nearest lower/upper numbered "
            "training frames; diagnostic for the systematic holdout scheme."
        ),
        "near_duplicate_proxy_definition": (
            "64-bit dHash distance <= 4, normalized 64x64 crop mean absolute difference "
            "<= 0.02, and rotation difference <= 1 degree."
        ),
        "comparisons": len(rows),
        "near_duplicate_proxy_count": proxy_count,
        "nearest_dhash_hamming_64bit": _quantiles(hamming_values),
        "nearest_crop_mean_absolute_difference_0_1": _quantiles(image_differences),
        "nearest_rotation_difference_degrees": _quantiles(rotation_differences),
        "rows": rows,
    }


def main() -> int:
    args = parse_args()
    report = audit(args.raw_root)
    _write_json(args.output, report)
    print(f"Synthetic split similarity: {report['status'].upper()}")
    print(f"  validation frames checked: {report['comparisons']}")
    print(f"  near-duplicate proxies: {report['near_duplicate_proxy_count']}")
    print(
        "  nearest dHash median/min: "
        f"{report['nearest_dhash_hamming_64bit']['median']:.1f}/"
        f"{report['nearest_dhash_hamming_64bit']['min']:.0f}"
    )
    print(f"  report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
