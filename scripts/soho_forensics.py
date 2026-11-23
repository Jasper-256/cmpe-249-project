"""Forensic audit of why every model fails on SOHO trajectory 4 (soho-4-close/far).

Competing explanations tested (no model is trained, evaluated, or tuned here):

* H-label: the soho-4 ground-truth pose (or its convention) is wrong.
* H-sym: SOHO is nearly 2-fold symmetric, so a flipped pose explains the image
  almost as well as the true pose.
* H-cov: soho-4 viewpoints lie outside the synthetic training pose coverage.

Inputs are read-only: the crop-v2 cache (real and synthetic), the SOHO dense CAD
point cloud inside ``models.zip`` (zero-copy), and the already-saved round-2 real
predictions under ``outputs/real-v2``. Real labels are used only as an audit.
Outputs go to ``outputs/diagnostics/soho/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

from saga_pose.data.archives import read_fresh_member
from saga_pose.data.cache import FreshCropCacheDataset

CACHE = Path("data/fresh/cache/crop-v2")
REAL_V2 = Path("outputs/real-v2")
OUT = Path("outputs/diagnostics/soho")
PRIMARY = "s2r-u-rgb-tss-soho"
SOHO_SEQS = tuple(f"soho-{k}-{d}" for k in (1, 2, 3, 4) for d in ("close", "far"))
FLIP_DEG = 150.0  # a frame counts as a "near-flip" above this rotation error
SIZE = 256
Array = NDArray[np.float64]

GT_C, PRED_C, FLIP_C, MUTED = "#1b9e77", "#d95f02", "#7570b3", "#888888"


# ----------------------------------------------------------------------------- geometry
def project(points: Array, pose: Array, intrinsics: Array) -> Array:
    cam = points @ pose[:3, :3].T + pose[:3, 3]
    hom = cam @ intrinsics.T
    return hom[:, :2] / hom[:, 2:3]


def geodesic_pairs(a: Array, b: Array) -> Array:
    """Row-wise rotation angle of a_i^T b_i in degrees."""
    trace = np.einsum("nij,nij->n", a, b)
    return np.degrees(np.arccos(np.clip((trace - 1) / 2, -1, 1)))


def geodesic_all(a: Array, b: Array) -> Array:
    trace = np.einsum("nij,mij->nm", a, b)
    return np.degrees(np.arccos(np.clip((trace - 1) / 2, -1, 1)))


def roll_free_all(a: Array, b: Array) -> Array:
    """Distance allowing a free in-plane (camera-z) roll, as theory_figures.pose_coverage."""
    m = np.einsum("mij,nkj->nmik", b, a)
    trace = np.hypot(m[..., 0, 0] + m[..., 1, 1], m[..., 1, 0] - m[..., 0, 1]) + m[..., 2, 2]
    return np.degrees(np.arccos(np.clip((trace - 1) / 2, -1, 1)))


def half_turn(axis: Array) -> Array:
    axis = axis / np.linalg.norm(axis)
    return 2.0 * np.outer(axis, axis) - np.eye(3)


def rotvec(rel: Array) -> tuple[Array, Array]:
    """Angle (deg) and unit axis of each rotation matrix, robust near 180 degrees."""
    angles = np.degrees(np.arccos(np.clip((np.einsum("nii->n", rel) - 1) / 2, -1, 1)))
    axes = np.stack(
        [rel[:, 2, 1] - rel[:, 1, 2], rel[:, 0, 2] - rel[:, 2, 0], rel[:, 1, 0] - rel[:, 0, 1]], 1
    )
    near_pi = angles > 150
    if near_pi.any():
        # For angles near pi, the axis is the dominant eigenvector of (R + R^T)/2 + I.
        sym = (rel[near_pi] + rel[near_pi].transpose(0, 2, 1)) / 2 + np.eye(3)
        _, vecs = np.linalg.eigh(sym)
        ax = vecs[:, :, -1]
        sign = np.sign(np.einsum("ni,ni->n", ax, axes[near_pi]))
        sign[sign == 0] = 1
        axes[near_pi] = ax * sign[:, None]
    axes /= np.linalg.norm(axes, axis=1, keepdims=True) + 1e-12
    return angles, axes


def fibonacci_hemisphere(n: int) -> Array:
    k = np.arange(n) + 0.5
    z = k / n
    phi = np.pi * (1 + 5**0.5) * k
    r = np.sqrt(1 - z**2)
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], 1)


def fit_flip_axis(rel: Array, axes: Array) -> tuple[Array, Array]:
    """Median-over-frames residual angle of S_a^T rel for every candidate half-turn axis a."""
    quad = np.einsum("mi,nij,mj->nm", axes, rel, axes)
    trace = 2 * quad - np.einsum("nii->n", rel)[:, None]
    resid = np.degrees(np.arccos(np.clip((trace - 1) / 2, -1, 1)))
    return np.median(resid, 0), resid


def axis_consistency(axes: Array) -> tuple[list[float], float]:
    """Sign-invariant mean axis (principal eigenvector of sum a a^T) and its concentration."""
    scatter = axes.T @ axes / len(axes)
    vals, vecs = np.linalg.eigh(scatter)
    mean = vecs[:, -1]
    if mean[np.argmax(np.abs(mean))] < 0:
        mean = -mean
    return [round(float(v), 4) for v in mean], round(float(vals[-1]), 4)


# --------------------------------------------------------------------------- silhouettes
def render_mask(dense: Array, pose: Array, intrinsics: Array) -> NDArray[np.bool_]:
    xy = np.rint(project(dense, pose, intrinsics)).astype(int)
    keep = (xy[:, 0] >= 0) & (xy[:, 0] < SIZE) & (xy[:, 1] >= 0) & (xy[:, 1] < SIZE)
    mask = np.zeros((SIZE, SIZE), bool)
    mask[xy[keep, 1], xy[keep, 0]] = True
    mask = ndimage.binary_closing(mask, iterations=2)
    return ndimage.binary_fill_holes(mask)


def otsu(values: NDArray[np.uint8]) -> float:
    hist = np.bincount(values.ravel(), minlength=256).astype(float)
    p = hist / hist.sum()
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    between = (mu[-1] * omega - mu) ** 2 / (omega * (1 - omega) + 1e-12)
    return float(np.argmax(between))


def rgb_foreground(rgb: NDArray[np.uint8]) -> NDArray[np.bool_]:
    """Simple label-free foreground: Otsu on grey level inside the non-padded crop."""
    grey = rgb.mean(0).astype(np.uint8)
    valid = grey > 0
    if valid.sum() < 100:
        return np.zeros_like(valid)
    mask = (grey > otsu(grey[valid])) & valid
    mask = ndimage.binary_opening(mask, iterations=1)
    return ndimage.binary_fill_holes(mask)


def event_edges(activity: NDArray[np.uint8]) -> NDArray[np.bool_]:
    on = (activity >= 128).astype(float)
    return (ndimage.uniform_filter(on, 3) >= 3 / 9) & (on > 0)


def iou(a: NDArray[np.bool_], b: NDArray[np.bool_]) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else float("nan")


def contour(mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
    return mask & ~ndimage.binary_erosion(mask)


def chamfer(mask: NDArray[np.bool_], edge_distance: Array) -> float:
    c = contour(mask)
    return float(np.median(edge_distance[c])) if c.any() else float("nan")


# --------------------------------------------------------------------------------- data
def load_cache(split: str) -> dict[str, Any]:
    cache = FreshCropCacheDataset(CACHE / split)
    return {
        "sample_id": np.asarray(cache.sample_ids),
        "sequence": np.asarray(cache.sequences),
        "frame": np.asarray(cache.array("frame_number")),
        "pose": np.asarray(cache.array("pose_object_to_camera_mm"), dtype=np.float64),
        "K": np.asarray(cache.array("intrinsics"), dtype=np.float64),
        "landmarks": np.asarray(cache.array("landmarks_object_mm")[0], dtype=np.float64),
        "rgb": cache.array("rgb_uint8"),
        "activity": cache.array("event_activity_uint8"),
        "manifest_sha256": hashlib.sha256(
            (CACHE / split / "manifest.json").read_bytes()
        ).hexdigest(),
    }


def load_frames(run: str, cache: dict[str, Any]) -> dict[str, Any]:
    frames = dict(np.load(REAL_V2 / run / "frames_primary.npz"))
    index = {s: i for i, s in enumerate(cache["sample_id"])}
    rows = np.asarray([index[str(s)] for s in frames["sample_id"]])
    frames["rows"] = rows
    return frames


def dense_soho(stride: int) -> Array:
    raw = read_fresh_member("models", "models/soho/dense.json")
    return np.asarray(json.loads(raw)["dense_points"], dtype=np.float64)[::stride]


# ------------------------------------------------------------------------------ analyses
def relative_rotation(cache: dict[str, Any], frames: dict[str, Any]) -> dict[str, Any]:
    rows = frames["rows"]
    rg = cache["pose"][rows, :3, :3]
    rp = frames["predicted_pose_mm"].astype(np.float64)[:, :3, :3]
    rel_body = np.einsum("nji,njk->nik", rg, rp)  # R_gt^T R_pred, object frame
    rel_cam = np.einsum("nij,nkj->nik", rp, rg)  # R_pred R_gt^T, camera frame
    return {"rel_body": rel_body, "rel_cam": rel_cam, "rg": rg, "rp": rp}


def symmetry_scores(
    poses: Array, intrinsics: Array, flip: Array, landmarks: Array, dense: Array
) -> dict[str, Array]:
    """Keypoint-set distance (after optimal relabelling) and silhouette IoU for R vs R*flip."""
    kp_dist, sil_iou, view_z = [], [], []
    for pose, k in zip(poses, intrinsics, strict=True):
        flipped = pose.copy()
        flipped[:3, :3] = pose[:3, :3] @ flip
        a, b = project(landmarks, pose, k), project(landmarks, flipped, k)
        diag = float(np.linalg.norm(a.max(0) - a.min(0)))
        cost = np.linalg.norm(a[:, None] - b[None], axis=2)
        r, c = linear_sum_assignment(cost)
        kp_dist.append(cost[r, c].mean() / diag)
        sil_iou.append(iou(render_mask(dense, pose, k), render_mask(dense, flipped, k)))
        # angle between the camera optical axis and the body +/-z (bus) axis
        view_z.append(np.degrees(np.arccos(abs(pose[2, 2]))))
    return {
        "kp_dist": np.asarray(kp_dist),
        "iou": np.asarray(sil_iou),
        "view_z": np.asarray(view_z),
    }


def view_angle_to_axis(rot: Array, axis: Array) -> Array:
    """Angle (deg, 0-90) between the camera optical axis and a body axis (sign-free)."""
    return np.degrees(np.arccos(np.clip(np.abs(rot[:, 2, :] @ axis), 0, 1)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-stride", type=int, default=6)
    parser.add_argument("--sym-samples", type=int, default=400)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    real = load_cache("real-soho")
    landmarks = real["landmarks"]
    dense = dense_soho(args.dense_stride)
    primary = load_frames(PRIMARY, real)
    rel = relative_rotation(real, primary)
    seq = primary["sequence"]
    err = primary["rotation_error_deg"]
    summary: dict[str, Any] = {
        "provenance": {
            "script": "scripts/soho_forensics.py",
            "git_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
            ).stdout.strip(),
            "uv_lock_sha256": hashlib.sha256(Path("uv.lock").read_bytes()).hexdigest(),
            "real_cache_manifest_sha256": real["manifest_sha256"],
            "primary_predictions": f"{REAL_V2 / PRIMARY / 'frames_primary.npz'}",
            "dense_model": "models.zip:models/soho/dense.json",
            "dense_stride": args.dense_stride,
            "seed": 0,
            "note": "audit only: no training, no new real evaluation, nothing tuned",
        }
    }

    # ---- (0) internal label consistency: stored 2D keypoints vs projected 3D landmarks at GT
    kp_cache = np.asarray(FreshCropCacheDataset(CACHE / "real-soho").array("keypoints_xy"))
    reproj = np.array(
        [
            np.abs(project(landmarks, real["pose"][i], real["K"][i]) - kp_cache[i]).max()
            for i in range(len(real["pose"]))
        ]
    )
    summary["label_internal_consistency_max_px"] = {
        s: round(float(reproj[real["sequence"] == s].max()), 4) for s in SOHO_SEQS
    }

    # ---- (1) relative rotation predicted vs GT
    angles, axes_body = rotvec(rel["rel_body"])
    _, axes_cam = rotvec(rel["rel_cam"])
    candidates = fibonacci_hemisphere(4000)
    per_seq: dict[str, Any] = {}
    soho4 = np.isin(seq, ["soho-4-close", "soho-4-far"])
    med_resid, _ = fit_flip_axis(rel["rel_body"][soho4], candidates)
    flip_axis = candidates[int(np.argmin(med_resid))]
    if flip_axis[np.argmax(np.abs(flip_axis))] < 0:
        flip_axis = -flip_axis
    flip = half_turn(flip_axis)
    body_axes = {"x": np.eye(3)[0], "y": np.eye(3)[1], "z": np.eye(3)[2]}
    for s in SOHO_SEQS:
        m = seq == s
        big = m & (angles > 120)
        seq_resid, _ = fit_flip_axis(rel["rel_body"][m], candidates)
        best = candidates[int(np.argmin(seq_resid))]
        sym_err = {
            name: round(
                float(np.median(geodesic_pairs(rel["rg"][m] @ half_turn(ax), rel["rp"][m]))), 2
            )
            for name, ax in {**body_axes, "fitted_soho4": flip_axis}.items()
        }
        per_seq[s] = {
            "frames": int(m.sum()),
            "mean_rotation_error_deg": round(float(err[m].mean()), 2),
            "median_rotation_error_deg": round(float(np.median(err[m])), 2),
            "frac_error_gt_120": round(float((angles[m] > 120).mean()), 3),
            "frac_error_gt_150": round(float((angles[m] > FLIP_DEG).mean()), 3),
            "body_axis_of_flip_frames": axis_consistency(axes_body[big])[0]
            if big.sum() > 2
            else None,
            "body_axis_concentration": axis_consistency(axes_body[big])[1]
            if big.sum() > 2
            else None,
            "camera_axis_concentration": axis_consistency(axes_cam[big])[1]
            if big.sum() > 2
            else None,
            "best_single_half_turn_axis_body": [round(float(v), 3) for v in best],
            "median_error_after_half_turn_deg": sym_err,
            "view_angle_to_body_z_deg_median": round(
                float(np.median(view_angle_to_axis(rel["rg"][m], body_axes["z"]))), 1
            ),
            "view_angle_to_flip_axis_deg_median": round(
                float(np.median(view_angle_to_axis(rel["rg"][m], flip_axis))), 1
            ),
            "pred_depth_over_gt_depth_median": round(
                float(
                    np.median(
                        primary["predicted_pose_mm"][m, 2, 3]
                        / real["pose"][primary["rows"][m], 2, 3]
                    )
                ),
                3,
            ),
        }
    summary["primary_model"] = PRIMARY
    summary["fitted_flip_axis_body_soho4"] = [round(float(v), 4) for v in flip_axis]
    summary["fitted_flip_axis_angle_to_body_axes_deg"] = {
        k: round(float(np.degrees(np.arccos(abs(flip_axis @ v)))), 1) for k, v in body_axes.items()
    }
    summary["relative_rotation_by_sequence"] = per_seq

    # ---- (2) image evidence: silhouettes vs label-free RGB foreground and event edges
    image_rows: list[dict[str, Any]] = []
    for i, row in enumerate(primary["rows"]):
        if not (seq[i].startswith("soho-4") or seq[i].startswith("soho-3")) and i % 3:
            continue
        k = real["K"][row]
        gt = real["pose"][row]
        pred = primary["predicted_pose_mm"][i].astype(np.float64)
        gt_flip = gt.copy()
        gt_flip[:3, :3] = gt[:3, :3] @ flip
        fg = rgb_foreground(np.asarray(real["rgb"][row]))
        edges = event_edges(np.asarray(real["activity"][row, 0]))
        edist = (
            ndimage.distance_transform_edt(~edges) if edges.any() else np.full(edges.shape, 99.0)
        )
        rec: dict[str, Any] = {"sequence": str(seq[i]), "frame": int(primary["frame_number"][i])}
        for name, pose in (("gt", gt), ("gt_flipped", gt_flip), ("pred", pred)):
            try:
                mask = render_mask(dense, pose, k)
            except FloatingPointError:
                mask = np.zeros((SIZE, SIZE), bool)
            rec[f"iou_rgb_{name}"] = iou(mask, fg)
            rec[f"chamfer_event_{name}"] = chamfer(mask, edist)
        image_rows.append(rec)
    image: dict[str, Any] = {}
    for s in SOHO_SEQS:
        rs = [r for r in image_rows if r["sequence"] == s]
        entry: dict[str, Any] = {"frames": len(rs)}
        for key in ("iou_rgb", "chamfer_event"):
            for name in ("gt", "gt_flipped", "pred"):
                v = np.asarray([r[f"{key}_{name}"] for r in rs], float)
                entry[f"{key}_{name}_median"] = round(float(np.nanmedian(v)), 3)
            g = np.asarray([r[f"{key}_gt"] for r in rs], float)
            p = np.asarray([r[f"{key}_pred"] for r in rs], float)
            better = (g > p) if key == "iou_rgb" else (g < p)
            entry[f"{key}_frac_gt_better_than_pred"] = round(float(np.nanmean(better)), 3)
        image[s] = entry
    summary["image_evidence"] = {
        "method": (
            "rendered dense-CAD silhouette (closing+fill) at GT, GT*flip and predicted pose; "
            "IoU against an Otsu grey-level RGB foreground; median distance (px, 256 crop) from "
            "the silhouette contour to denoised event-activity edges. soho-3/4 all frames, "
            "other sequences every third frame."
        ),
        "by_sequence": image,
    }

    # ---- (3) symmetry: how alike R and R*flip look, as a function of viewpoint
    syn = load_cache("soho-validation")
    pick = rng.choice(len(syn["pose"]), size=min(args.sym_samples, len(syn["pose"])), replace=False)
    random_axis = rng.normal(size=3)
    sym_flip = symmetry_scores(syn["pose"][pick], syn["K"][pick], flip, landmarks, dense)
    sym_rand = symmetry_scores(
        syn["pose"][pick], syn["K"][pick], half_turn(random_axis), landmarks, dense
    )
    sym_z = symmetry_scores(
        syn["pose"][pick], syn["K"][pick], half_turn(np.eye(3)[2]), landmarks, dense
    )
    view_flip_syn = view_angle_to_axis(syn["pose"][pick, :3, :3], flip_axis)
    real_sym: dict[str, Any] = {}
    real_sym_iou: dict[str, Array] = {}
    for s in SOHO_SEQS:
        rows = np.where(real["sequence"] == s)[0]
        rows = rows[np.linspace(0, len(rows) - 1, 40).astype(int)]
        sc = symmetry_scores(real["pose"][rows], real["K"][rows], flip, landmarks, dense)
        real_sym_iou[s] = sc["iou"]
        real_sym[s] = {
            "flip_silhouette_iou_median": round(float(np.median(sc["iou"])), 3),
            "flip_keypoint_set_distance_median": round(float(np.median(sc["kp_dist"])), 4),
        }
    summary["symmetry"] = {
        "synthetic_validation_samples": len(pick),
        "fitted_axis": {
            "silhouette_iou_median": round(float(np.median(sym_flip["iou"])), 3),
            "keypoint_set_distance_median": round(float(np.median(sym_flip["kp_dist"])), 4),
        },
        "body_z_axis": {
            "silhouette_iou_median": round(float(np.median(sym_z["iou"])), 3),
            "keypoint_set_distance_median": round(float(np.median(sym_z["kp_dist"])), 4),
        },
        "random_axis_control": {
            "axis": [round(float(v), 3) for v in random_axis / np.linalg.norm(random_axis)],
            "silhouette_iou_median": round(float(np.median(sym_rand["iou"])), 3),
            "keypoint_set_distance_median": round(float(np.median(sym_rand["kp_dist"])), 4),
        },
        "fitted_axis_iou_by_view_angle_bin": {
            f"{lo}-{lo + 15}": round(
                float(
                    np.median(sym_flip["iou"][(view_flip_syn >= lo) & (view_flip_syn < lo + 15)])
                ),
                3,
            )
            for lo in range(0, 90, 15)
            if ((view_flip_syn >= lo) & (view_flip_syn < lo + 15)).any()
        },
        "keypoint_set_distance_note": "mean matched distance / GT keypoint bbox diagonal",
        "real_gt_poses_by_sequence": real_sym,
    }

    # ---- (4) coverage of synthetic training rotations
    train = load_cache("soho-train")["pose"][::3, :3, :3]
    coverage: dict[str, Any] = {}
    cov_samples: dict[str, Array] = {}
    for s in SOHO_SEQS:
        m = np.where(real["sequence"] == s)[0]
        m = m[np.linspace(0, len(m) - 1, min(150, len(m))).astype(int)]
        r = real["pose"][m, :3, :3]
        free = roll_free_all(r, train).min(1)
        free_flip = roll_free_all(r @ flip, train).min(1)
        plain = geodesic_all(r, train).min(1)
        cov_samples[s] = free
        coverage[s] = {
            "roll_free_nearest_train_deg_median": round(float(np.median(free)), 2),
            "roll_free_nearest_train_deg_p90": round(float(np.percentile(free, 90)), 2),
            "plain_nearest_train_deg_median": round(float(np.median(plain)), 2),
            "flipped_gt_roll_free_nearest_train_deg_median": round(float(np.median(free_flip)), 2),
        }
    val = syn["pose"][::4, :3, :3]
    coverage["synthetic_validation_reference_roll_free_median"] = round(
        float(np.median(roll_free_all(val[:300], train).min(1))), 2
    )
    summary["coverage"] = coverage

    # ---- (5) flips across all round-2 runs and objects
    flips: dict[str, Any] = {}
    other_caches = {o: load_cache(f"real-{o}") for o in ("cassini", "satty")}
    other_caches["soho"] = real
    for run_dir in sorted(REAL_V2.iterdir()):
        if not (run_dir / "frames_primary.npz").is_file():
            continue
        obj = run_dir.name.rsplit("-", 1)[1]
        cache = other_caches[obj]
        fr = load_frames(run_dir.name, cache)
        rr = relative_rotation(cache, fr)
        ang, ax = rotvec(rr["rel_body"])
        per: dict[str, Any] = {}
        for s in sorted(set(fr["sequence"])):
            m = fr["sequence"] == s
            big = m & (ang > FLIP_DEG)
            per[s] = {
                "frac_gt_150": round(float(big.sum() / m.sum()), 3),
                "flip_axis_concentration": axis_consistency(ax[big])[1] if big.sum() > 5 else None,
                "flip_axis_body": axis_consistency(ax[big])[0] if big.sum() > 5 else None,
            }
        flips[run_dir.name] = per
    summary["near_flip_rates_all_runs"] = flips
    by_obj: dict[str, Any] = {}
    for obj in ("cassini", "satty", "soho"):
        rates: dict[str, list[float]] = {}
        for run, per in flips.items():
            if run.endswith(obj):
                for s, v in per.items():
                    rates.setdefault(s, []).append(v["frac_gt_150"])
        by_obj[obj] = {s: round(float(np.mean(v)), 3) for s, v in sorted(rates.items())}
    summary["near_flip_rate_mean_over_runs"] = by_obj

    (OUT / "soho_forensics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    make_figures(
        real,
        primary,
        dense,
        flip,
        angles,
        axes_body,
        image_rows,
        sym_flip,
        sym_rand,
        real_sym_iou,
        cov_samples,
        landmarks,
    )
    print(
        json.dumps(
            {
                k: summary[k]
                for k in ("fitted_flip_axis_body_soho4", "fitted_flip_axis_angle_to_body_axes_deg")
            },
            indent=1,
        )
    )
    return 0


# ------------------------------------------------------------------------------ figures
def make_figures(
    real: dict[str, Any],
    primary: dict[str, Any],
    dense: Array,
    flip: Array,
    angles: Array,
    axes_body: Array,
    image_rows: list[dict[str, Any]],
    sym_flip: dict[str, Array],
    sym_rand: dict[str, Array],
    real_sym_iou: dict[str, Array],
    cov: dict[str, Array],
    landmarks: Array,
) -> None:
    plt.rcParams.update({"font.size": 8})
    seq = primary["sequence"]
    # Overlay figure: rows = frames, columns = image, GT, prediction.
    picks = [
        ("soho-4-close", 0),
        ("soho-4-close", 94),
        ("soho-4-close", 282),
        ("soho-4-far", 150),
        ("soho-3-close", 163),
        ("soho-3-far", 150),
    ]
    fig, axs = plt.subplots(len(picks), 3, figsize=(7.2, 2.45 * len(picks)))
    sub = dense[::3]
    for r, (s, frame) in enumerate(picks):
        cand = np.where(seq == s)[0]
        i = cand[np.argmin(np.abs(primary["frame_number"][cand] - frame))]
        row = primary["rows"][i]
        img = np.asarray(real["rgb"][row]).transpose(1, 2, 0)
        k = real["K"][row]
        poses = (
            ("ground truth", real["pose"][row], GT_C),
            ("prediction", primary["predicted_pose_mm"][i].astype(np.float64), PRED_C),
        )
        axs[r, 0].imshow(img)
        axs[r, 0].set_title(f"{s}  frame {primary['frame_number'][i]}", loc="left")
        for c, (name, pose, col) in enumerate(poses, start=1):
            a = axs[r, c]
            a.imshow(img)
            d = project(sub, pose, k)
            a.scatter(d[:, 0], d[:, 1], s=0.15, c=col, alpha=0.12, linewidths=0)
            kp = project(landmarks, pose, k)
            a.scatter(kp[:, 0], kp[:, 1], s=6, c=col, edgecolors="white", linewidths=0.4)
            a.set_title(f"{name}  ({angles[i]:.0f} deg apart)" if c == 2 else name, loc="left")
        for a in axs[r]:
            a.set_xlim(0, SIZE)
            a.set_ylim(SIZE, 0)
            a.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "soho_overlays.png", dpi=170, facecolor="white")
    plt.close(fig)

    # Summary figure: 4 panels.
    fig, axs = plt.subplots(2, 2, figsize=(8.6, 6.6))
    a = axs[0, 0]
    colors = {
        "soho-4-close": "#d95f02",
        "soho-4-far": "#e6ab02",
        "soho-2-close": "#7570b3",
        "soho-2-far": "#a6a2d6",
        "soho-1-close": "#1b9e77",
    }
    for s, col in colors.items():
        m = (seq == s) & (angles > 120)
        if m.sum() < 3:
            continue
        ax = axes_body[m] * np.sign(axes_body[m] @ np.array([0.0, 0.0, 1.0]))[:, None]
        a.scatter(ax[:, 0], ax[:, 1], s=5, c=col, alpha=0.6, label=f"{s} ({m.sum()})")
    t = np.linspace(0, 2 * np.pi, 200)
    a.plot(np.cos(t), np.sin(t), c=MUTED, lw=0.6)
    a.scatter([0], [0], marker="+", c="k", s=60)
    a.set_aspect("equal")
    a.set_title(
        "axis of pred-vs-GT rotation (frames >120 deg),\n"
        "body frame viewed down +z (centre = bus axis)",
        loc="left",
    )
    a.set_xlabel("body x (panel axis)")
    a.set_ylabel("body y")
    a.legend(frameon=False, fontsize=6, loc="lower left")

    a = axs[0, 1]
    order = [
        "soho-1-close",
        "soho-1-far",
        "soho-2-close",
        "soho-2-far",
        "soho-3-close",
        "soho-3-far",
        "soho-4-close",
        "soho-4-far",
    ]
    g = [np.array([r["iou_rgb_gt"] for r in image_rows if r["sequence"] == s]) for s in order]
    p = [np.array([r["iou_rgb_pred"] for r in image_rows if r["sequence"] == s]) for s in order]
    f = [
        np.array([r["iou_rgb_gt_flipped"] for r in image_rows if r["sequence"] == s]) for s in order
    ]
    x = np.arange(len(order))
    for off, vals, col, lab in (
        (-0.25, g, GT_C, "ground truth"),
        (0.0, f, FLIP_C, "GT half-turned"),
        (0.25, p, PRED_C, "prediction"),
    ):
        med = [np.nanmedian(v) for v in vals]
        lo = [np.nanpercentile(v, 25) for v in vals]
        hi = [np.nanpercentile(v, 75) for v in vals]
        a.errorbar(
            x + off,
            med,
            yerr=[np.subtract(med, lo), np.subtract(hi, med)],
            fmt="o",
            ms=3.5,
            c=col,
            lw=1,
            label=lab,
        )
    a.set_xticks(x, [s.replace("soho-", "") for s in order], rotation=40)
    a.set_ylabel("silhouette IoU vs RGB foreground")
    a.set_title("image evidence (median, IQR)", loc="left")
    a.legend(frameon=False, fontsize=6)

    a = axs[1, 0]
    bins = np.linspace(0, 1, 41)
    a.hist(
        sym_flip["iou"],
        bins=bins,
        density=True,
        color=FLIP_C,
        alpha=0.35,
        label="synthetic validation poses",
    )
    a.hist(
        sym_rand["iou"],
        bins=bins,
        density=True,
        histtype="step",
        color=MUTED,
        lw=1,
        label="synthetic, random axis (control)",
    )
    a.hist(
        np.concatenate([real_sym_iou[s] for s in ("soho-3-close", "soho-3-far")]),
        bins=bins,
        density=True,
        histtype="step",
        color=GT_C,
        lw=1.5,
        label="real soho-3 GT poses",
    )
    a.hist(
        np.concatenate([real_sym_iou[s] for s in ("soho-4-close", "soho-4-far")]),
        bins=bins,
        density=True,
        histtype="step",
        color=PRED_C,
        lw=1.5,
        label="real soho-4 GT poses",
    )
    a.set_xlabel("IoU(silhouette at R, silhouette at R * half-turn)")
    a.set_yticks([])
    a.set_title("how alike the fitted half-turn looks", loc="left")
    a.legend(frameon=False, fontsize=6)

    a = axs[1, 1]
    bins = np.linspace(0, 60, 31)
    for s in order:
        col = (
            "#d95f02"
            if s.startswith("soho-4")
            else ("#1b9e77" if s.startswith("soho-3") else MUTED)
        )
        a.hist(
            cov[s],
            bins=bins,
            histtype="step",
            color=col,
            lw=1.3 if col != MUTED else 0.8,
            density=True,
            label=s if col != MUTED else None,
        )
    a.set_xlabel("roll-free distance to nearest training rotation, deg")
    a.set_yticks([])
    a.set_title("coverage (grey: soho-1/2)", loc="left")
    a.legend(frameon=False, fontsize=6)
    for a in axs.ravel():
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "soho_forensics_summary.png", dpi=170, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
