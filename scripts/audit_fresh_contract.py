"""Audit FRESH pair, geometry, unit, and split contracts without extraction."""

from __future__ import annotations

import argparse
import io
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from saga_pose.data.archives import open_fresh_archive
from saga_pose.data.fresh import frame_member, load_metadata, metadata_members
from saga_pose.geometry.projection import project_points, rotation_matrix_diagnostics

IMAGE_SIZE_WH = (800, 720)
PROJECTION_TOLERANCE_PX = 1e-6
ROTATION_TOLERANCE = 1e-5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("data/fresh/raw"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/fresh/index/contract-audit.json"),
    )
    return parser.parse_args()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _pose_key(pose: np.ndarray) -> bytes:
    return np.round(pose, decimals=7).tobytes()


def audit(raw_root: Path) -> dict[str, Any]:
    failures: list[str] = []
    warnings: list[str] = []
    metadata_reports: list[dict[str, Any]] = []
    split_pairs: dict[str, dict[str, set[tuple[str, str]]]] = defaultdict(dict)
    split_poses: dict[str, dict[str, set[bytes]]] = defaultdict(dict)

    total_annotations = 0
    total_missing_members = 0
    total_out_of_frame_bboxes = 0
    global_max_projection_error = 0.0
    global_max_rotation_error = 0.0
    translation_norms_mm: list[float] = []

    for kind, is_real in (("synthetic", False), ("real", True)):
        with open_fresh_archive(kind, raw_root) as archive:
            names = set(archive.namelist())
            for member in metadata_members(archive, real=is_real):
                metadata = load_metadata(archive, member)
                pair_keys: set[tuple[str, str]] = set()
                pose_keys: set[bytes] = set()
                member_missing = 0
                member_max_projection = 0.0
                member_max_rotation = 0.0
                invalid_bboxes = 0
                out_of_frame_bboxes = 0
                invalid_bottom_rows = 0

                for annotation in metadata.annotations:
                    total_annotations += 1
                    pair = (annotation.filename_rgb, annotation.filename_event)
                    if pair in pair_keys:
                        failures.append(f"{member}: duplicate pair {pair}")
                    pair_keys.add(pair)
                    pose_keys.add(_pose_key(annotation.pose))

                    for filename in pair:
                        resolved = frame_member(member, filename)
                        if resolved not in names:
                            member_missing += 1

                    x1, y1, x2, y2 = annotation.bbox_xyxy
                    coordinates = np.asarray(annotation.bbox_xyxy, dtype=np.float64)
                    has_positive_area = x2 > x1 and y2 > y1
                    intersects_image = (
                        x2 > 0 and y2 > 0 and x1 < IMAGE_SIZE_WH[0] and y1 < IMAGE_SIZE_WH[1]
                    )
                    if (
                        not np.all(np.isfinite(coordinates))
                        or not has_positive_area
                        or not intersects_image
                    ):
                        invalid_bboxes += 1
                    elif not (
                        x1 >= 0 and x2 <= IMAGE_SIZE_WH[0] and y1 >= 0 and y2 <= IMAGE_SIZE_WH[1]
                    ):
                        out_of_frame_bboxes += 1

                    bottom_error = float(
                        np.max(np.abs(annotation.pose[3] - np.array([0.0, 0.0, 0.0, 1.0])))
                    )
                    if bottom_error > ROTATION_TOLERANCE:
                        invalid_bottom_rows += 1

                    rotation_error, determinant = rotation_matrix_diagnostics(
                        annotation.pose[:3, :3]
                    )
                    member_max_rotation = max(member_max_rotation, rotation_error)
                    if (
                        rotation_error > ROTATION_TOLERANCE
                        or abs(determinant - 1.0) > ROTATION_TOLERANCE
                    ):
                        failures.append(
                            f"{member} {annotation.filename_rgb}: invalid rotation "
                            f"(orth={rotation_error:.3g}, det={determinant:.9g})"
                        )

                    projected = project_points(
                        metadata.landmarks_3d,
                        annotation.pose,
                        metadata.intrinsics,
                    )
                    projection_error = float(
                        np.max(np.linalg.norm(projected - annotation.keypoints, axis=1))
                    )
                    member_max_projection = max(member_max_projection, projection_error)
                    translation_norms_mm.append(float(np.linalg.norm(annotation.pose[:3, 3])))

                total_missing_members += member_missing
                total_out_of_frame_bboxes += out_of_frame_bboxes
                global_max_projection_error = max(
                    global_max_projection_error, member_max_projection
                )
                global_max_rotation_error = max(global_max_rotation_error, member_max_rotation)
                if member_missing:
                    failures.append(
                        f"{member}: {member_missing} referenced frame members are missing"
                    )
                if invalid_bboxes:
                    failures.append(f"{member}: {invalid_bboxes} invalid XYXY boxes")
                if invalid_bottom_rows:
                    failures.append(f"{member}: {invalid_bottom_rows} invalid pose bottom rows")
                if member_max_projection > PROJECTION_TOLERANCE_PX:
                    failures.append(
                        f"{member}: projection residual {member_max_projection:.6g} px exceeds "
                        f"{PROJECTION_TOLERANCE_PX:g} px"
                    )

                first = metadata.annotations[0]
                sample_shapes: dict[str, list[int]] = {}
                sample_modes: dict[str, str] = {}
                for modality, filename in (
                    ("rgb", first.filename_rgb),
                    ("event", first.filename_event),
                ):
                    with Image.open(
                        io.BytesIO(archive.read(frame_member(member, filename)))
                    ) as image:
                        sample_shapes[modality] = [image.width, image.height]
                        sample_modes[modality] = image.mode
                        if image.size != IMAGE_SIZE_WH:
                            failures.append(
                                f"{member}: sample {modality} size {image.size} != {IMAGE_SIZE_WH}"
                            )

                if not is_real:
                    object_name = metadata.object_name
                    role = metadata.role
                    split_pairs[object_name][role] = pair_keys
                    split_poses[object_name][role] = pose_keys

                metadata_reports.append(
                    {
                        "member": member,
                        "role": metadata.role,
                        "object": metadata.object_name,
                        "annotations": len(metadata.annotations),
                        "landmarks": int(metadata.landmarks_3d.shape[0]),
                        "missing_frame_members": member_missing,
                        "out_of_frame_bboxes_requiring_clamp": out_of_frame_bboxes,
                        "max_projection_error_px": member_max_projection,
                        "max_rotation_orthogonality_error": member_max_rotation,
                        "sample_image_size_wh": sample_shapes,
                        "sample_image_modes": sample_modes,
                    }
                )

    split_reports: dict[str, dict[str, int]] = {}
    for object_name in sorted(split_pairs):
        train_pairs = split_pairs[object_name]["train"]
        validation_pairs = split_pairs[object_name]["validation"]
        train_poses = split_poses[object_name]["train"]
        validation_poses = split_poses[object_name]["validation"]
        pair_overlap = len(train_pairs & validation_pairs)
        exact_pose_overlap = len(train_poses & validation_poses)
        split_reports[object_name] = {
            "train_pairs": len(train_pairs),
            "validation_pairs": len(validation_pairs),
            "exact_filename_pair_overlap": pair_overlap,
            "exact_pose_overlap_at_7_decimals": exact_pose_overlap,
        }
        if pair_overlap:
            failures.append(
                f"{object_name}: {pair_overlap} train/validation filename-pair overlaps"
            )
        if exact_pose_overlap:
            warnings.append(
                f"{object_name}: {exact_pose_overlap} train/validation poses match at 7 decimals"
            )

    norms = np.asarray(translation_norms_mm, dtype=np.float64)
    return {
        "schema_version": 1,
        "status": "pass" if not failures else "fail",
        "definitions": {
            "pose": "object-to-camera homogeneous transform; Xc = R Xo + t",
            "annotation_translation_unit": "millimetres",
            "reported_translation_unit": "metres after exactly one division by 1000",
            "bbox": "XYXY in the 800x720 source frame",
            "synthetic_test_role": "synthetic validation; never the real held-out test",
        },
        "totals": {
            "metadata_files": len(metadata_reports),
            "annotations": total_annotations,
            "missing_frame_members": total_missing_members,
            "out_of_frame_bboxes_requiring_clamp": total_out_of_frame_bboxes,
            "max_projection_error_px": global_max_projection_error,
            "max_rotation_orthogonality_error": global_max_rotation_error,
            "translation_norm_mm": {
                "min": float(np.min(norms)),
                "median": float(np.median(norms)),
                "max": float(np.max(norms)),
            },
        },
        "synthetic_split_overlap": split_reports,
        "metadata": metadata_reports,
        "warnings": warnings,
        "failures": failures,
    }


def main() -> int:
    args = parse_args()
    report = audit(args.raw_root)
    _write_json(args.output, report)
    totals = report["totals"]
    print(f"FRESH contract audit: {report['status'].upper()}")
    print(f"  annotations: {totals['annotations']:,}")
    print(f"  max projection residual: {totals['max_projection_error_px']:.3g} px")
    print(f"  missing referenced frames: {totals['missing_frame_members']}")
    print(f"  report: {args.output}")
    for failure in report["failures"][:10]:
        print(f"  FAILURE: {failure}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
