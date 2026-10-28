"""Solver-confound control: score every method under EPnP and under RANSAC EPnP.

The FRESH-style geometric baseline uses a robust solver (RANSAC, distinct-landmark
inlier minimum, depth and residual checks) while learned models use plain EPnP+LM
on all keypoints. This control re-solves the saved synthetic-validation keypoints
of every unimodal and learned-fusion run with the *same* robust solver and
threshold, so "fusion" and "robust solving" can be separated. Synthetic only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from saga_pose.data.cache import FreshCropCacheDataset
from saga_pose.evaluation.geometric import ransac_pose_from_sets
from saga_pose.evaluation.pose import pose_error_arrays

OBJECTS = ("cassini", "satty", "soho")


def _geometry(cache: FreshCropCacheDataset, sample_ids: np.ndarray) -> dict[str, np.ndarray]:
    index = {sample: position for position, sample in enumerate(cache.sample_ids)}
    rows = np.asarray([index[str(sample)] for sample in sample_ids])
    return {
        "landmarks": np.asarray(cache.array("landmarks_object_mm"))[rows].astype(np.float64),
        "intrinsics": np.asarray(cache.array("intrinsics"))[rows].astype(np.float64),
        "truth": np.asarray(cache.array("pose_object_to_camera_mm"))[rows].astype(np.float64),
    }


def _score(errors: list[dict[str, np.ndarray]]) -> dict[str, float]:
    rotation = np.concatenate([e["rotation_error_deg"] for e in errors])
    translation = np.concatenate([e["translation_error_m"] for e in errors])
    return {
        "are_mean": float(rotation.mean()),
        "are_median": float(np.median(rotation)),
        "ate_mean_mm": float(1000 * translation.mean()),
        "ate_median_mm": float(1000 * np.median(translation)),
        "success_10deg_10mm": float(np.mean((rotation < 10.0) & (translation < 0.01))),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=Path("outputs/runs/main-hrnet"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/fresh/cache/crop-v2"))
    parser.add_argument("--selection", type=Path, default=Path("outputs/geometric/selection.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/geometric/solver_control"))
    args = parser.parse_args()
    threshold = float(
        json.loads(args.selection.read_text(encoding="utf-8"))["pairs"]["g-ransac-tss"][
            "threshold_px"
        ]
    )
    caches = {o: FreshCropCacheDataset(args.cache_dir / f"{o}-validation") for o in OBJECTS}
    results: dict[str, dict[str, dict[str, float]]] = {}
    for protocol in ("natural", "tss"):
        fusion_methods = sorted(
            path.name.removesuffix("-cassini")
            for path in args.runs_dir.glob(f"f-*-{protocol}-activity-cassini")
        )
        methods = [f"u-rgb-{protocol}", "u-event-activity", *fusion_methods]
        for method in methods:
            epnp, ransac = [], []
            for object_name in OBJECTS:
                frames = np.load(
                    args.runs_dir / f"{method}-{object_name}" / "validation_frames.npz"
                )
                geometry = _geometry(caches[object_name], frames["sample_id"])
                epnp.append(
                    {
                        "rotation_error_deg": frames["rotation_error_deg"],
                        "translation_error_m": frames["translation_error_m"],
                    }
                )
                poses, _, _ = ransac_pose_from_sets(
                    [frames["predicted_keypoints_xy"]],
                    geometry["landmarks"],
                    geometry["intrinsics"],
                    threshold_px=threshold,
                )
                ransac.append(pose_error_arrays(geometry["truth"], poses))
            label = f"{method} [{protocol} val]"
            results[label] = {"epnp": _score(epnp), "ransac": _score(ransac)}
            print(label, results[label], flush=True)
        # Two-model geometric fusion under the identical robust solver.
        fused = []
        for object_name in OBJECTS:
            rgb = np.load(
                args.runs_dir / f"u-rgb-{protocol}-{object_name}" / "validation_frames.npz"
            )
            event = np.load(
                args.runs_dir / f"u-event-activity-{object_name}" / "validation_frames.npz"
            )
            assert np.array_equal(rgb["sample_id"], event["sample_id"])
            geometry = _geometry(caches[object_name], rgb["sample_id"])
            poses, _, _ = ransac_pose_from_sets(
                [rgb["predicted_keypoints_xy"], event["predicted_keypoints_xy"]],
                geometry["landmarks"],
                geometry["intrinsics"],
                threshold_px=threshold,
            )
            fused.append(pose_error_arrays(geometry["truth"], poses))
        label = f"g-ransac-{protocol} (RGB+event keypoints) [{protocol} val]"
        results[label] = {"ransac": _score(fused)}
        print(label, results[label], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        f"RANSAC threshold {threshold:g} px (frozen). Pooled over three objects; "
        "synthetic validation.",
        "",
        "| method [validation] | EPnP ARE mean / med | EPnP ATE mean mm | RANSAC ARE mean / med "
        "| RANSAC ATE mean mm | RANSAC 10°/10mm |",
        "|---|---|---|---|---|---|",
    ]
    for label, scores in results.items():
        e, r = scores.get("epnp"), scores["ransac"]
        epnp_cells = (
            f"{e['are_mean']:.2f} / {e['are_median']:.2f} | {e['ate_mean_mm']:.1f}"
            if e
            else "— | —"
        )
        lines.append(
            f"| {label} | {epnp_cells} | {r['are_mean']:.2f} / {r['are_median']:.2f} "
            f"| {r['ate_mean_mm']:.1f} | {r['success_10deg_10mm']:.3f} |"
        )
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
