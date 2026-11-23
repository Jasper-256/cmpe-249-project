"""Frozen real-test evaluation of synthetic-selected checkpoints (G6).

Run only after the experiment matrix, checkpoints, and analysis are frozen. For
each run it evaluates the selected checkpoint once on all 24 real sequences
(annotation-corrected, shared annotation-box crop), saves per-frame predictions
before any aggregation, and reports overall, per-sequence, and author-slice
(harsh, low-motion) metrics with PnP failures counted. Fusion runs also receive
the predeclared clipping interventions (shuffle, zero) and gate bypass. Runs
whose event representation is ``activity`` are additionally evaluated with the
``magnitude`` real encoding (identical synthetic training data). Runs whose
representation lives in the event-supp-v1 supplement read it from
``--supplement-dir``, the same order-checked maps used in training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, cast

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.utils.data import DataLoader

from saga_pose.data.archives import open_fresh_archive
from saga_pose.data.cache import FreshCropCacheDataset
from saga_pose.data.real import load_real_slices
from saga_pose.data.representations import SUPPLEMENT_REPRESENTATIONS
from saga_pose.evaluation.geometric import ransac_pose_from_sets
from saga_pose.evaluation.pose import pose_error_arrays
from saga_pose.evaluation.real_analysis import analyze_real_frames
from saga_pose.models.fusion import ClipMode
from saga_pose.training.engine import ModelSpec, Protocol, build_model, evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path, help="run directories with report.json")
    parser.add_argument("--real-cache-dir", type=Path, default=Path("data/fresh/cache/crop-v2"))
    parser.add_argument("--raw-root", type=Path, default=Path("data/fresh/raw"))
    parser.add_argument(
        "--supplement-dir", type=Path, default=Path("data/fresh/cache/event-supp-v1")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/real"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/real/freeze_manifest.json"),
        help="freeze manifest that every evaluated checkpoint must match",
    )
    parser.add_argument("--confirm-frozen", action="store_true", required=True)
    return parser.parse_args()


def _check_supplement_code(report: dict[str, Any], supplement: Path, supplement_dir: Path) -> None:
    """Require the real supplement to share the training supplements' code hash.

    Each training supplement manifest is first matched against the hash the run
    report recorded, so the comparison is against the maps the model saw.
    """
    recorded = cast(dict[str, str], report["data"].get("supplement_manifests_sha256") or {})
    if not recorded:
        raise SystemExit(f"{report['name']}: run report records no training supplement")
    real_manifest = json.loads((supplement / "manifest.json").read_text(encoding="utf-8"))
    for split, manifest_sha in recorded.items():
        train_manifest_path = supplement_dir / split / "manifest.json"
        if _sha256(train_manifest_path) != manifest_sha:
            raise SystemExit(f"{report['name']}: {split} supplement changed after training")
        train_manifest = json.loads(train_manifest_path.read_text(encoding="utf-8"))
        if train_manifest["code_sha256"] != real_manifest["code_sha256"]:
            raise SystemExit(
                f"{report['name']}: real supplement code differs from the {split} supplement"
            )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ransac_frames(
    frames: dict[str, Any], cache: FreshCropCacheDataset, threshold_px: float
) -> dict[str, Any]:
    """Re-solve saved keypoints with the frozen RANSAC solver (same failure policy)."""
    index = {sample: position for position, sample in enumerate(cache.sample_ids)}
    rows = np.asarray([index[str(sample)] for sample in frames["sample_id"]])
    landmarks = np.asarray(cache.array("landmarks_object_mm"))[rows].astype(np.float64)
    intrinsics = np.asarray(cache.array("intrinsics"))[rows].astype(np.float64)
    truth = np.asarray(cache.array("pose_object_to_camera_mm"))[rows].astype(np.float64)
    poses, valid, _ = ransac_pose_from_sets(
        [np.asarray(frames["predicted_keypoints_xy"], dtype=np.float64)],
        landmarks,
        intrinsics,
        threshold_px=threshold_px,
    )
    errors = pose_error_arrays(truth, poses)
    return {
        "sequence": frames["sequence"],
        "frame_number": frames["frame_number"],
        "pnp_valid": valid,
        "rotation_error_deg": errors["rotation_error_deg"],
        "translation_error_m": errors["translation_error_m"],
    }


def main() -> int:
    args = parse_args()
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = False
    # The freeze manifest is mandatory: only frozen checkpoints, byte-identical to the
    # ones recorded before any real read, may be evaluated.
    manifest_path = args.manifest
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selection_path = Path("outputs/geometric/selection.json")
    if _sha256(selection_path) != manifest["geometric_selection_sha256"]:
        raise SystemExit("geometric selection changed after the freeze")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    provenance = {
        "freeze_manifest_sha256": _sha256(manifest_path),
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip(),
        "uv_lock_sha256": _sha256(Path("uv.lock")),
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(device),
        "precision": "float32",
        "batch_size": 32,
    }
    ransac_threshold = float(selection["pairs"]["g-ransac-tss"]["threshold_px"])
    with open_fresh_archive("real", args.raw_root) as archive:
        slices = cast(dict[str, dict[str, tuple[int, int]]], load_real_slices(archive))
    for run_dir in args.runs:
        report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
        objects = report["config"]["objects"]
        if len(objects) != 1:
            raise ValueError("real evaluation expects one per-object model per run")
        real_cache = args.real_cache_dir / f"real-{objects[0]}"
        report_spec = report["spec"]
        supplement = (
            args.supplement_dir / f"real-{objects[0]}"
            if report_spec["event_representation"] in SUPPLEMENT_REPRESENTATIONS
            and report_spec["modality"] != "rgb"
            else None
        )
        if supplement is not None:
            _check_supplement_code(report, supplement, args.supplement_dir)
        real = FreshCropCacheDataset(real_cache, supplement)
        loader: DataLoader[Any] = DataLoader(real, batch_size=32, shuffle=False, num_workers=2)
        object_slices = {
            name: ranges for name, ranges in slices.items() if name in set(real.sequences)
        }
        spec = ModelSpec(**report["spec"])
        checkpoint = Path(report["provenance"]["checkpoint"])
        frozen = manifest["runs"].get(report["name"])
        if frozen is None or _sha256(checkpoint) != frozen["checkpoint_sha256"]:
            raise SystemExit(f"{report['name']}: checkpoint is not the frozen one")
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        num_keypoints = int(real[0]["keypoints_xy"].shape[0])
        model = build_model(spec, num_keypoints, pretrained=False).to(device)
        model.load_state_dict(state["model"])
        out_dir = args.output_dir / report["name"]
        if (out_dir / "real_report.json").is_file():
            # The real test set is evaluated once per frozen checkpoint.
            print(f"[{report['name']}] already evaluated; refusing to re-evaluate", flush=True)
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        conditions: list[tuple[str, ModelSpec, ClipMode, float | None]] = [
            ("primary", spec, "normal", None)
        ]
        if spec.event_representation == "activity" and spec.modality != "rgb":
            conditions.append(
                (
                    "magnitude_encoding",
                    ModelSpec(**{**report["spec"], "event_representation": "magnitude"}),
                    "normal",
                    None,
                )
            )
        if spec.modality == "fusion":
            conditions += [
                ("clip_shuffle", spec, "shuffle", None),
                ("clip_zero", spec, "zero", None),
            ]
            if spec.fusion not in ("concat", "mean", "attention"):
                conditions.append(("gate_bypass_0.5", spec, "normal", 0.5))
        results: dict[str, Any] = {}
        for label, condition_spec, clip_mode, override in conditions:
            # Real data are never corrupted: the natural RGB path is used for every run.
            evaluation = evaluate(
                model,
                loader,
                condition_spec,
                cast(Protocol, "natural"),
                device,
                clip_mode=clip_mode,
                gate_override=override,
                keep_frames=True,
            )
            frames = evaluation.pop("frames")
            np.savez_compressed(out_dir / f"frames_{label}.npz", **frames)
            results[label] = {**evaluation, "analysis": analyze_real_frames(frames, object_slices)}
            if label == "primary":
                # Pre-registered secondary analysis (2026-10-28): the same keypoints
                # re-solved with the frozen RANSAC solver used by the geometric baseline.
                results["primary_ransac"] = {
                    "analysis": analyze_real_frames(
                        _ransac_frames(frames, real, ransac_threshold), object_slices
                    ),
                    "ransac_threshold_px": ransac_threshold,
                }
            overall = results[label]["analysis"]["overall_sequence_mean"]
            print(
                f"[{report['name']}] {label}: seq-mean ARE={overall['are_degrees']:.2f}deg "
                f"ATE={overall['ate_m']:.4f}m",
                flush=True,
            )
        payload = {
            "schema_version": 1,
            "run": report["name"],
            "spec": report["spec"],
            "training_protocol": report["protocol"],
            "checkpoint_sha256": _sha256(checkpoint),
            "provenance": provenance,
            "ransac_threshold_note": (
                "primary_ransac uses the TSS-pair threshold (4 px) for every model, "
                "the identical solver of the 2026-10-28 control"
            ),
            "real_cache_manifest_sha256": _sha256(real_cache / "manifest.json"),
            "real_supplement_manifest_sha256": _sha256(supplement / "manifest.json")
            if supplement is not None
            else None,
            "object": objects[0],
            "selected_epoch": report["outcome"].get("selected_epoch"),
            "results": results,
        }
        (out_dir / "real_report.json").write_text(
            json.dumps(payload, indent=2, default=float) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
