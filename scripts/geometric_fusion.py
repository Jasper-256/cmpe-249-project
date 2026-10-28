"""FRESH-style geometric fusion baseline: select on synthetic, apply frozen to real.

``select``: for each predeclared unimodal pair, fuse the saved synthetic-validation
keypoints of the RGB-only and event-only runs with RANSAC PnP at each candidate
inlier threshold, and freeze the threshold with the lowest pooled mean rotation
error. Real data are never read in this mode.

``real``: after ``evaluate_real.py`` has produced real-test keypoints for the
paired unimodal runs, fuse them with the frozen threshold and write
``real_report.json`` files in the same schema, so ``analyze_real.py`` treats the
baseline like any other method. Runs once per pair and object.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from saga_pose.data.archives import open_fresh_archive
from saga_pose.data.cache import FreshCropCacheDataset
from saga_pose.data.real import load_real_slices
from saga_pose.evaluation.geometric import fuse_keypoints_ransac
from saga_pose.evaluation.pose import pose_error_arrays
from saga_pose.evaluation.real_analysis import analyze_real_frames

OBJECTS = ("cassini", "satty", "soho")
THRESHOLDS_PX = (1.0, 2.0, 4.0, 8.0, 16.0)
# Predeclared pairs: FRESH-faithful (natural-trained RGB) and TSS-trained RGB,
# each with the activity-encoded event-only model.
PAIRS = {
    "g-ransac-natural": ("u-rgb-natural", "u-event-activity"),
    "g-ransac-tss": ("u-rgb-tss", "u-event-activity"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("select", "real"))
    parser.add_argument("--runs-dir", type=Path, default=Path("outputs/runs/main-hrnet"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/fresh/cache/crop-v2"))
    parser.add_argument("--real-dir", type=Path, default=Path("outputs/real"))
    parser.add_argument("--raw-root", type=Path, default=Path("data/fresh/raw"))
    parser.add_argument("--selection", type=Path, default=Path("outputs/geometric/selection.json"))
    return parser.parse_args()


def _geometry(cache: FreshCropCacheDataset, sample_ids: np.ndarray) -> dict[str, np.ndarray]:
    index = {sample: position for position, sample in enumerate(cache.sample_ids)}
    rows = np.asarray([index[str(sample)] for sample in sample_ids])
    return {
        "landmarks": np.asarray(cache.array("landmarks_object_mm"))[rows].astype(np.float64),
        "intrinsics": np.asarray(cache.array("intrinsics"))[rows].astype(np.float64),
        "truth": np.asarray(cache.array("pose_object_to_camera_mm"))[rows].astype(np.float64),
    }


def _paired_frames(rgb_path: Path, event_path: Path) -> tuple[Any, Any]:
    rgb, event = np.load(rgb_path), np.load(event_path)
    if not np.array_equal(rgb["sample_id"], event["sample_id"]):
        raise ValueError(f"{rgb_path} and {event_path} cover different frames")
    return rgb, event


def select(args: argparse.Namespace) -> int:
    report: dict[str, Any] = {"thresholds_px": list(THRESHOLDS_PX), "pairs": {}}
    for pair, (rgb_method, event_method) in PAIRS.items():
        scores: dict[str, dict[str, float]] = {}
        for threshold in THRESHOLDS_PX:
            rotation: list[np.ndarray] = []
            translation: list[np.ndarray] = []
            valid: list[np.ndarray] = []
            for object_name in OBJECTS:
                rgb, event = _paired_frames(
                    args.runs_dir / f"{rgb_method}-{object_name}" / "validation_frames.npz",
                    args.runs_dir / f"{event_method}-{object_name}" / "validation_frames.npz",
                )
                cache = FreshCropCacheDataset(args.cache_dir / f"{object_name}-validation")
                geometry = _geometry(cache, rgb["sample_id"])
                poses, ok, _ = fuse_keypoints_ransac(
                    rgb["predicted_keypoints_xy"],
                    event["predicted_keypoints_xy"],
                    geometry["landmarks"],
                    geometry["intrinsics"],
                    threshold_px=threshold,
                )
                errors = pose_error_arrays(geometry["truth"], poses)
                rotation.append(errors["rotation_error_deg"])
                translation.append(errors["translation_error_m"])
                valid.append(ok)
            all_rotation, all_translation = np.concatenate(rotation), np.concatenate(translation)
            scores[str(threshold)] = {
                "are_degrees_mean": float(all_rotation.mean()),
                "ate_m_mean": float(all_translation.mean()),
                "pnp_valid_fraction": float(np.concatenate(valid).mean()),
            }
            print(f"{pair} threshold {threshold:>4} px: {scores[str(threshold)]}", flush=True)
        chosen = min(scores, key=lambda key: scores[key]["are_degrees_mean"])
        report["pairs"][pair] = {
            "rgb_method": rgb_method,
            "event_method": event_method,
            "selection_rule": "lowest pooled synthetic-validation mean ARE (own-protocol frames)",
            "scores": scores,
            "threshold_px": float(chosen),
        }
    args.selection.parent.mkdir(parents=True, exist_ok=True)
    args.selection.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"selection frozen: {args.selection}")
    return 0


def real(args: argparse.Namespace) -> int:
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    with open_fresh_archive("real", args.raw_root) as archive:
        slices = load_real_slices(archive)
    for pair, spec in selection["pairs"].items():
        for object_name in OBJECTS:
            out_dir = args.real_dir / f"{pair}-{object_name}"
            if (out_dir / "real_report.json").is_file():
                print(f"[{pair}-{object_name}] already evaluated; refusing to re-evaluate")
                continue
            rgb, event = _paired_frames(
                args.real_dir / f"{spec['rgb_method']}-{object_name}" / "frames_primary.npz",
                args.real_dir / f"{spec['event_method']}-{object_name}" / "frames_primary.npz",
            )
            cache = FreshCropCacheDataset(args.cache_dir / f"real-{object_name}")
            geometry = _geometry(cache, rgb["sample_id"])
            poses, valid, inliers = fuse_keypoints_ransac(
                rgb["predicted_keypoints_xy"],
                event["predicted_keypoints_xy"],
                geometry["landmarks"],
                geometry["intrinsics"],
                threshold_px=float(spec["threshold_px"]),
            )
            errors = pose_error_arrays(geometry["truth"], poses)
            frames = {
                "sample_id": rgb["sample_id"],
                "sequence": rgb["sequence"],
                "frame_number": rgb["frame_number"],
                "pnp_valid": valid,
                "rotation_error_deg": errors["rotation_error_deg"],
                "translation_error_m": errors["translation_error_m"],
                "inlier_fraction": inliers,
            }
            object_slices = {k: v for k, v in slices.items() if k in set(map(str, rgb["sequence"]))}
            out_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out_dir / "frames_primary.npz", **frames)
            payload = {
                "schema_version": 1,
                "run": f"{pair}-{object_name}",
                "method": (
                    "FRESH-style RANSAC fusion of unimodal keypoints (not the published pipeline)"
                ),
                "pair": spec,
                "object": object_name,
                "results": {"primary": {"analysis": analyze_real_frames(frames, object_slices)}},
            }
            (out_dir / "real_report.json").write_text(
                json.dumps(payload, indent=2, default=float) + "\n", encoding="utf-8"
            )
            print(f"[{pair}-{object_name}] done", flush=True)
    return 0


def main() -> int:
    args = parse_args()
    return select(args) if args.mode == "select" else real(args)


if __name__ == "__main__":
    raise SystemExit(main())
