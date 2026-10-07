"""Run the frozen short Cassini train/validation pilot for RGB and event inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset

from saga_pose.data.dataset import FreshArchiveDataset, FreshItem
from saga_pose.evaluation.pose import recover_poses, summarize_pose_errors
from saga_pose.geometry.projection import rotation_matrix_diagnostics
from saga_pose.models.heatmaps import (
    HRNetHeatmapRegressor,
    decode_heatmap_argmax,
    gaussian_keypoint_heatmaps,
)
from saga_pose.models.keypoints import PilotKeypointRegressor

Modality = Literal["rgb", "event"]
Architecture = Literal["direct_coordinate", "hrnet_heatmap"]
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("data/fresh/raw"))
    parser.add_argument(
        "--split-index",
        type=Path,
        default=Path("data/fresh/index/synthetic-split-v1.json"),
    )
    parser.add_argument("--object", default="cassini", choices=("cassini", "satty", "soho"))
    parser.add_argument("--train-samples", type=int, default=1024)
    parser.add_argument("--validation-samples", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--minimum-validation-improvement-px", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=249)
    parser.add_argument(
        "--architecture",
        choices=("direct_coordinate", "hrnet_heatmap"),
        default="direct_coordinate",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/feasibility/short-unimodal-pilot.json"),
    )
    return parser.parse_args()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _evenly_spaced_indices(length: int, count: int) -> list[int]:
    if count <= 0 or count > length:
        raise ValueError(f"sample count must be in [1, {length}]")
    return np.linspace(0, length - 1, count, dtype=np.int64).tolist()


def _channel_statistics(
    dataset: FreshArchiveDataset,
    train_indices: Sequence[int],
    count: int = 64,
) -> dict[Modality, tuple[Tensor, Tensor]]:
    selected = np.linspace(0, len(train_indices) - 1, min(count, len(train_indices)), dtype=int)
    modalities: tuple[Modality, ...] = ("rgb", "event")
    sums: dict[Modality, Tensor] = {key: torch.zeros(3, dtype=torch.float64) for key in modalities}
    squared_sums: dict[Modality, Tensor] = {
        key: torch.zeros(3, dtype=torch.float64) for key in modalities
    }
    pixels = 0
    for position in selected:
        item = dataset[train_indices[int(position)]]
        for modality in modalities:
            values = item[modality].double()
            sums[modality] += values.sum(dim=(1, 2))
            squared_sums[modality] += (values * values).sum(dim=(1, 2))
        pixels += item["rgb"].shape[1] * item["rgb"].shape[2]
    result: dict[Modality, tuple[Tensor, Tensor]] = {}
    for modality in modalities:
        mean = sums[modality] / pixels
        variance = squared_sums[modality] / pixels - mean.square()
        result[modality] = (mean.float(), variance.clamp_min(1e-6).sqrt().float())
    return result


def _normalize(inputs: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    return (inputs - mean[None, :, None, None]) / std[None, :, None, None]


def _predict_keypoints_px(
    model: nn.Module,
    inputs: Tensor,
    architecture: Architecture,
) -> Tensor:
    output: Tensor = model(inputs)
    if architecture == "hrnet_heatmap":
        coordinates, _ = decode_heatmap_argmax(output)
        return coordinates
    return output * 255.0


def _make_loader(
    dataset: FreshArchiveDataset,
    indices: Sequence[int],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader[FreshItem]:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        Subset(dataset, list(indices)),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


@torch.inference_mode()
def _evaluate(
    model: nn.Module,
    loader: DataLoader[FreshItem],
    modality: Modality,
    mean: Tensor,
    std: Tensor,
    device: torch.device,
    architecture: Architecture,
    *,
    recover_pose: bool,
) -> dict[str, Any]:
    model.eval()
    squared_error_sum = 0.0
    coordinate_count = 0
    sample_count = 0
    predicted_keypoints: list[np.ndarray[Any, np.dtype[np.float64]]] = []
    landmarks: list[np.ndarray[Any, np.dtype[np.float64]]] = []
    intrinsics: list[np.ndarray[Any, np.dtype[np.float64]]] = []
    ground_truth_poses: list[np.ndarray[Any, np.dtype[np.float64]]] = []
    for batch in loader:
        inputs = _normalize(batch[modality], mean, std).to(device, non_blocking=True)
        predictions_px = _predict_keypoints_px(model, inputs, architecture).cpu()
        targets_px = batch["keypoints_xy"]
        squared_error_sum += float(torch.sum((predictions_px - targets_px) ** 2))
        coordinate_count += targets_px.numel()
        sample_count += targets_px.shape[0]
        if recover_pose:
            predicted_keypoints.append(predictions_px.double().numpy())
            landmarks.append(batch["landmarks_object_mm"].double().numpy())
            intrinsics.append(batch["intrinsics"].double().numpy())
            ground_truth_poses.append(batch["pose_object_to_camera_mm"].double().numpy())
    evaluation: dict[str, Any] = {
        "sample_count": sample_count,
        "coordinate_rmse_px": float(np.sqrt(squared_error_sum / coordinate_count)),
    }
    if not recover_pose:
        return evaluation
    poses, valid, reprojection = recover_poses(
        np.concatenate(predicted_keypoints),
        np.concatenate(landmarks),
        np.concatenate(intrinsics),
    )
    max_orthogonality_residual = 0.0
    max_determinant_error = 0.0
    for pose in poses[valid]:
        orthogonality, determinant = rotation_matrix_diagnostics(pose[:3, :3])
        max_orthogonality_residual = max(max_orthogonality_residual, orthogonality)
        max_determinant_error = max(max_determinant_error, abs(determinant - 1.0))
    summary = summarize_pose_errors(np.concatenate(ground_truth_poses), poses, valid, reprojection)
    # Top-level pose metrics include PnP failures under the FRESH default-pose
    # policy; valid-only values are retained as a nested solver diagnostic.
    evaluation.update(
        {
            "pnp_valid_count": summary["pnp_valid_count"],
            "pnp_valid_fraction": summary["pnp_valid_fraction"],
            "pnp_failure_policy": summary["failure_policy"],
            "max_rotation_orthogonality_residual": max_orthogonality_residual,
            "max_rotation_determinant_error": max_determinant_error,
            **summary["all_frames"],
            # Kept at top level for schema continuity with frozen reports.
            "reprojection_rmse_px_mean": (
                summary["valid_only"]["reprojection_rmse_px_mean"]
                if summary["valid_only"] is not None
                else None
            ),
            "valid_only": summary["valid_only"],
        }
    )
    return evaluation


@torch.inference_mode()
def _profile_forward(
    model: nn.Module,
    example: Tensor,
    device: torch.device,
    *,
    warmup_steps: int = 20,
    timed_steps: int = 100,
) -> dict[str, float | int]:
    model.eval()
    inputs = example.to(device)
    for _ in range(warmup_steps):
        model(inputs)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for _ in range(timed_steps):
        model(inputs)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return {
        "batch_size": inputs.shape[0],
        "warmup_steps": warmup_steps,
        "timed_steps": timed_steps,
        "batch_latency_ms": 1000.0 * elapsed / timed_steps,
        "throughput_samples_per_second": inputs.shape[0] * timed_steps / elapsed,
    }


def _train_modality(
    modality: Modality,
    train_dataset: FreshArchiveDataset,
    validation_dataset: FreshArchiveDataset,
    train_indices: Sequence[int],
    validation_indices: Sequence[int],
    statistics: tuple[Tensor, Tensor],
    architecture: Architecture,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    early_stopping_patience: int,
    minimum_validation_improvement_px: float,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)
    mean, std = statistics
    first_item = train_dataset[train_indices[0]]
    num_keypoints = first_item["keypoints_xy"].shape[0]
    model: nn.Module
    if architecture == "hrnet_heatmap":
        model = HRNetHeatmapRegressor(num_keypoints, pretrained=True).to(device)
        weight_decay = 1e-4
    else:
        model = PilotKeypointRegressor(num_keypoints).to(device)
        weight_decay = 0.0
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_loader = _make_loader(
        train_dataset,
        train_indices,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )
    validation_loader = _make_loader(
        validation_dataset,
        validation_indices,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
    )
    initial = _evaluate(
        model,
        validation_loader,
        modality,
        mean,
        std,
        device,
        architecture,
        recover_pose=False,
    )
    history: list[dict[str, float | int]] = []
    best_validation_rmse = float("inf")
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    epochs_without_improvement = 0
    stopped_early = False
    started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        loss_sum = 0.0
        samples = 0
        for batch in train_loader:
            inputs = _normalize(batch[modality], mean, std).to(device, non_blocking=True)
            targets_px = batch["keypoints_xy"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(inputs)
            if architecture == "hrnet_heatmap":
                targets, visibility = gaussian_keypoint_heatmaps(targets_px)
                weights = visibility[:, :, None, None]
                loss = torch.sum((prediction - targets).square() * weights) / (
                    weights.sum() * prediction.shape[-2] * prediction.shape[-1]
                )
            else:
                loss = torch.mean((prediction - targets_px / 255.0) ** 2)
            if not torch.isfinite(loss):
                raise RuntimeError(f"{modality} loss became non-finite at epoch {epoch + 1}")
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * targets_px.shape[0]
            samples += targets_px.shape[0]
        validation = _evaluate(
            model,
            validation_loader,
            modality,
            mean,
            std,
            device,
            architecture,
            recover_pose=False,
        )
        validation_rmse = float(validation["coordinate_rmse_px"])
        checkpoint_improved = (
            best_state is None
            or validation_rmse <= best_validation_rmse - minimum_validation_improvement_px
        )
        if checkpoint_improved:
            best_validation_rmse = validation_rmse
            best_epoch = epoch + 1
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": loss_sum / samples,
                "validation_coordinate_rmse_px": validation_rmse,
                "checkpoint_improved": int(checkpoint_improved),
            }
        )
        print(
            f"  {modality} epoch {epoch + 1}/{epochs}: "
            f"loss={loss_sum / samples:.6f}, validation RMSE={validation_rmse:.2f}px, "
            f"checkpoint={checkpoint_improved}",
            flush=True,
        )
        if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
            stopped_early = True
            break
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    if best_state is None:
        raise RuntimeError("training completed without a checkpoint")
    model.load_state_dict(best_state)
    final = _evaluate(
        model,
        validation_loader,
        modality,
        mean,
        std,
        device,
        architecture,
        recover_pose=True,
    )
    example_batch = next(iter(validation_loader))[modality]
    profile = _profile_forward(model, _normalize(example_batch, mean, std), device)
    criteria = {
        "finite_training": all(np.isfinite(row["train_loss"]) for row in history),
        "pnp_valid_fraction_at_least_0_95": final["pnp_valid_fraction"] >= 0.95,
        "validation_coordinate_rmse_at_most_20_px": final["coordinate_rmse_px"] <= 20.0,
        "median_are_at_most_20_deg": final.get("are_degrees_median", float("inf")) <= 20.0,
        "median_ate_at_most_0_25_m": final.get("absolute_translation_error_m_median", float("inf"))
        <= 0.25,
    }
    return {
        "modality": modality,
        "architecture": architecture,
        "architecture_competent": all(criteria.values()),
        "predeclared_criteria": criteria,
        "parameters_total": sum(parameter.numel() for parameter in model.parameters()),
        "parameters_trainable": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "input_channel_mean": mean.tolist(),
        "input_channel_std": std.tolist(),
        "initial_validation": initial,
        "final_validation": final,
        "checkpoint_selection": {
            "metric": "synthetic_validation_coordinate_rmse_px",
            "mode": "minimum",
            "selected_epoch": best_epoch,
            "selected_metric_value": best_validation_rmse,
            "epochs_completed": len(history),
            "stopped_early": stopped_early,
            "patience": early_stopping_patience,
            "minimum_improvement_px": minimum_validation_improvement_px,
        },
        "history": history,
        "training_elapsed_seconds": elapsed,
        "training_samples_per_second": len(history) * len(train_indices) / elapsed,
        "peak_vram_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        ),
        "forward_profile": profile,
    }


def main() -> int:
    args = parse_args()
    if min(args.train_samples, args.validation_samples, args.epochs, args.batch_size) <= 0:
        raise ValueError("sample counts, epochs, and batch size must be positive")
    if args.early_stopping_patience < 0:
        raise ValueError("early stopping patience must be nonnegative")
    if args.minimum_validation_improvement_px < 0.0:
        raise ValueError("minimum validation improvement must be nonnegative")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = FreshArchiveDataset(
        args.raw_root, args.split_index, role="train", objects=(args.object,)
    )
    validation_dataset = FreshArchiveDataset(
        args.raw_root, args.split_index, role="validation", objects=(args.object,)
    )
    train_indices = _evenly_spaced_indices(len(train_dataset), args.train_samples)
    validation_indices = _evenly_spaced_indices(len(validation_dataset), args.validation_samples)
    modalities: tuple[Modality, ...] = ("rgb", "event")
    statistics: dict[Modality, tuple[Tensor, Tensor]]
    if args.architecture == "hrnet_heatmap":
        statistics = {
            modality: (IMAGENET_MEAN.clone(), IMAGENET_STD.clone()) for modality in modalities
        }
    else:
        statistics = _channel_statistics(train_dataset, train_indices)
    results = [
        _train_modality(
            modality,
            train_dataset,
            validation_dataset,
            train_indices,
            validation_indices,
            statistics[modality],
            args.architecture,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            early_stopping_patience=args.early_stopping_patience,
            minimum_validation_improvement_px=args.minimum_validation_improvement_px,
            seed=args.seed,
            device=device,
        )
        for modality in modalities
    ]
    train_dataset.close()
    validation_dataset.close()
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": (
            "pass" if all(result["architecture_competent"] for result in results) else "fail"
        ),
        "scope": "bounded synthetic one-object architecture screen; not a final baseline",
        "config": {
            "dataset": "fresh-synthetic-grouped-v1",
            "object": args.object,
            "train_samples": args.train_samples,
            "validation_samples": args.validation_samples,
            "selection": "evenly spaced within each frozen split role",
            "crop_source": "shared annotation box",
            "crop_transform": "center-preserving square, 1.2x padding, zero border",
            "target_size_wh": [256, 256],
            "architecture": args.architecture,
            "pretraining": (
                "timm/hrnet_w18_small_v2.ms_in1k" if args.architecture == "hrnet_heatmap" else None
            ),
            "heatmap_size_wh": [64, 64] if args.architecture == "hrnet_heatmap" else None,
            "heatmap_sigma": 1.5 if args.architecture == "hrnet_heatmap" else None,
            "normalization": (
                "ImageNet mean/std for both modalities"
                if args.architecture == "hrnet_heatmap"
                else "estimated independently from 64 grouped training samples"
            ),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "checkpoint_selection": {
                "metric": "synthetic_validation_coordinate_rmse_px",
                "mode": "minimum",
                "maximum_epochs": args.epochs,
                "early_stopping_patience": args.early_stopping_patience,
                "minimum_improvement_px": args.minimum_validation_improvement_px,
            },
            "optimizer": (
                "AdamW(weight_decay=1e-4)"
                if args.architecture == "hrnet_heatmap"
                else "AdamW(weight_decay=0)"
            ),
            "augmentation": None,
            "seed": args.seed,
            "pose_solver": "OpenCV EPnP followed by solvePnPRefineLM",
        },
        "provenance": {
            "git_commit": _git_commit(),
            "uv_lock_sha256": _sha256(Path("uv.lock")),
            "split_index_sha256": _sha256(args.split_index),
            "source_sha256": {
                path: _sha256(Path(path))
                for path in (
                    "scripts/run_short_unimodal_pilot.py",
                    "src/saga_pose/data/dataset.py",
                    "src/saga_pose/models/heatmaps.py",
                    "src/saga_pose/models/keypoints.py",
                    "src/saga_pose/geometry/pnp.py",
                    "src/saga_pose/evaluation/metrics.py",
                )
            },
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "precision": "float32",
        },
        "results": results,
    }
    _write_json(args.output, report)
    print(f"Short unimodal pilot: {report['status'].upper()}")
    for result in results:
        final = result["final_validation"]
        print(
            f"  {result['modality']}: RMSE={final['coordinate_rmse_px']:.2f}px, "
            f"PnP={final['pnp_valid_fraction']:.1%}, "
            f"median ARE={final.get('are_degrees_median', float('nan')):.2f}deg, "
            f"median ATE={final.get('absolute_translation_error_m_median', float('nan')):.3f}m, "
            f"competent={result['architecture_competent']}"
        )
    print(f"  report: {args.output}")
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
