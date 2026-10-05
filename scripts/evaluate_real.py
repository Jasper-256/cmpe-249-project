"""Frozen real-test evaluation of synthetic-selected checkpoints (G6).

Run only after the experiment matrix, checkpoints, and analysis are frozen. For
each run it evaluates the selected checkpoint once on all 24 real sequences
(annotation-corrected, shared annotation-box crop), saves per-frame predictions
before any aggregation, and reports overall, per-sequence, and author-slice
(harsh, low-motion) metrics with PnP failures counted. Fusion runs also receive
the predeclared clipping interventions (shuffle, zero) and gate bypass. Runs
whose event representation is ``activity`` are additionally evaluated with the
``magnitude`` real encoding (identical synthetic training data).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, cast

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.utils.data import DataLoader

from saga_pose.data.archives import open_fresh_archive
from saga_pose.data.cache import FreshCropCacheDataset
from saga_pose.data.real import load_real_slices, slice_mask
from saga_pose.evaluation.metrics import aggregate_by_sequence, threshold_success_rate
from saga_pose.models.fusion import ClipMode
from saga_pose.training.engine import ModelSpec, Protocol, build_model, evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path, help="run directories with report.json")
    parser.add_argument("--real-cache-dir", type=Path, default=Path("data/fresh/cache/crop-v2"))
    parser.add_argument("--raw-root", type=Path, default=Path("data/fresh/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/real"))
    parser.add_argument("--confirm-frozen", action="store_true", required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summaries(
    rotation: np.ndarray, translation: np.ndarray, valid: np.ndarray
) -> dict[str, float]:
    return {
        "frames": int(rotation.size),
        "are_degrees_mean": float(np.mean(rotation)),
        "are_degrees_median": float(np.median(rotation)),
        "ate_m_mean": float(np.mean(translation)),
        "ate_m_median": float(np.median(translation)),
        "success_rotation_below_10_deg": threshold_success_rate(rotation, 10.0),
        "success_translation_below_10_mm": threshold_success_rate(translation, 0.01),
        "pnp_valid_fraction": float(np.mean(valid)),
    }


def _analyze(
    frames: dict[str, Any], slices: dict[str, dict[str, tuple[int, int]]]
) -> dict[str, Any]:
    rotation = np.asarray(frames["rotation_error_deg"])
    translation = np.asarray(frames["translation_error_m"])
    valid = np.asarray(frames["pnp_valid"])
    sequences = list(frames["sequence"])
    numbers = np.asarray(frames["frame_number"])
    result: dict[str, Any] = {
        "overall_frames": _summaries(rotation, translation, valid),
        "sequence_mean_are_deg": aggregate_by_sequence(rotation, sequences),
        "sequence_mean_ate_m": aggregate_by_sequence(translation, sequences),
        "slices": {},
    }
    result["overall_sequence_mean"] = {
        "are_degrees": float(np.mean(list(result["sequence_mean_are_deg"].values()))),
        "ate_m": float(np.mean(list(result["sequence_mean_ate_m"].values()))),
    }
    sequence_array = np.asarray(sequences)
    for slice_name in ("harsh", "low-motion"):
        mask = np.zeros(rotation.size, dtype=bool)
        unknown = set(slices) - set(sequences)
        if unknown:
            raise ValueError(f"slice sequences missing from predictions: {sorted(unknown)}")
        for sequence, ranges in slices.items():
            in_sequence = sequence_array == sequence
            mask |= in_sequence & slice_mask(numbers, ranges[slice_name])  # type: ignore[index]
        if not mask.any():
            raise ValueError(f"empty {slice_name} slice")
        result["slices"][slice_name] = _summaries(rotation[mask], translation[mask], valid[mask])
        result["slices"][slice_name]["sequence_mean_are_deg"] = aggregate_by_sequence(
            rotation[mask], list(sequence_array[mask])
        )
        result["slices"][slice_name]["sequence_mean_ate_m"] = aggregate_by_sequence(
            translation[mask], list(sequence_array[mask])
        )
    return result


def main() -> int:
    args = parse_args()
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = False
    with open_fresh_archive("real", args.raw_root) as archive:
        slices = cast(dict[str, dict[str, tuple[int, int]]], load_real_slices(archive))
    for run_dir in args.runs:
        report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
        objects = report["config"]["objects"]
        if len(objects) != 1:
            raise ValueError("real evaluation expects one per-object model per run")
        real_cache = args.real_cache_dir / f"real-{objects[0]}"
        real = FreshCropCacheDataset(real_cache)
        loader: DataLoader[Any] = DataLoader(real, batch_size=32, shuffle=False, num_workers=2)
        object_slices = {
            name: ranges for name, ranges in slices.items() if name in set(real.sequences)
        }
        spec = ModelSpec(**report["spec"])
        checkpoint = Path(report["provenance"]["checkpoint"])
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
            if spec.fusion not in ("concat", "mean"):
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
            results[label] = {**evaluation, "analysis": _analyze(frames, object_slices)}
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
            "real_cache_manifest_sha256": _sha256(real_cache / "manifest.json"),
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
