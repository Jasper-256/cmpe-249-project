"""Re-score round-2 SOHO predictions with the soho-4 half-turn treated as equivalent.

The SOHO forensics (scripts/soho_forensics.py) found that on trajectory 4 the
models predict the true pose turned 180 degrees about one fixed body axis.
Here each frame's rotation error becomes min(error to GT, error to GT composed
with that half-turn), for every SOHO frame. Translation error is unchanged.

Caveat: the axis was fitted from round-2 predictions on the real soho-4
sequences, so this is a secondary, post-hoc number reported next to the
original metric, never instead of it. Nothing is tuned.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

from saga_pose.data.cache import FreshCropCacheDataset

AXIS = np.array(
    json.loads(Path("outputs/diagnostics/soho/soho_forensics.json").read_text())[
        "fitted_flip_axis_body_soho4"
    ]
)
OUT = Path("outputs/diagnostics/soho/symmetry_rescore.json")


def half_turn(axis: np.ndarray) -> np.ndarray:
    a = axis / np.linalg.norm(axis)
    return 2.0 * np.outer(a, a) - np.eye(3)


def angle_deg(r_pred: np.ndarray, r_ref: np.ndarray) -> np.ndarray:
    rel = np.einsum("nji,njk->nik", r_pred, r_ref)
    cos = (np.trace(rel, axis1=1, axis2=2) - 1.0) / 2.0
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def summary(rot: np.ndarray, trans_m: np.ndarray) -> dict[str, float]:
    return {
        "are_mean_deg": float(rot.mean()),
        "are_median_deg": float(np.median(rot)),
        "rot10_pct": float(100 * (rot < 10).mean()),
        "joint_pct": float(100 * ((rot < 10) & (trans_m < 0.010)).mean()),
    }


def main() -> int:
    cache = FreshCropCacheDataset(Path("data/fresh/cache/crop-v2/real-soho"))
    gt = np.asarray(cache.array("pose_object_to_camera_mm"))[:, :3, :3].astype(np.float64)
    gt_frames = np.asarray(cache.array("frame_number"))
    flip = half_turn(AXIS)
    gt_flipped = gt @ flip
    result: dict[str, object] = {
        "axis_body": AXIS.tolist(),
        "note": "post-hoc; axis fitted on round-2 soho-4 predictions; original metric is primary",
        "runs": {},
    }
    runs: dict[str, object] = {}
    for path in sorted(glob.glob("outputs/real-v2/s2r-*-soho/frames_primary.npz")):
        z = np.load(path)
        if not np.array_equal(z["frame_number"], gt_frames):
            raise ValueError(f"frame order mismatch in {path}")
        pred = z["predicted_pose_mm"][:, :3, :3].astype(np.float64)
        original = angle_deg(pred, gt)
        if np.abs(original - z["rotation_error_deg"]).max() > 0.05:
            raise ValueError(f"recomputed error disagrees with stored error in {path}")
        symmetric = np.minimum(original, angle_deg(pred, gt_flipped))
        trans = z["translation_error_m"]
        sequences = z["sequence"]
        per_sequence = {
            str(seq): {
                "original_mean": float(original[sequences == seq].mean()),
                "symmetric_mean": float(symmetric[sequences == seq].mean()),
                "symmetric_median": float(np.median(symmetric[sequences == seq])),
            }
            for seq in sorted(set(sequences.tolist()))
        }
        runs[Path(path).parent.name] = {
            "original": summary(original, trans),
            "symmetric": summary(symmetric, trans),
            "per_sequence": per_sequence,
        }
    result["runs"] = runs
    OUT.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
