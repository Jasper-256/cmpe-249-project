"""Profile complete HRNet heatmap training steps on the frozen FRESH path."""

from __future__ import annotations

import argparse
import gc
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
from torch.utils.data import DataLoader, Subset

from saga_pose.data.dataset import FreshArchiveDataset, FreshItem
from saga_pose.models.heatmaps import HRNetHeatmapRegressor, gaussian_keypoint_heatmaps

Modality = Literal["rgb", "event"]
Precision = Literal["float32", "bfloat16"]
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
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--timed-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=249)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/feasibility/training-compute-profile.json"),
    )
    return parser.parse_args()


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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _evenly_spaced_indices(length: int, count: int) -> list[int]:
    if count <= 0 or count > length:
        raise ValueError(f"sample count must be in [1, {length}]")
    return np.linspace(0, length - 1, count, dtype=np.int64).tolist()


def _loader(
    dataset: FreshArchiveDataset,
    indices: Sequence[int],
    batch_size: int,
    seed: int,
) -> DataLoader[FreshItem]:
    return DataLoader(
        Subset(dataset, list(indices)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(seed),
    )


def _distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def _profile_trial(
    dataset: FreshArchiveDataset,
    indices: Sequence[int],
    modality: Modality,
    precision: Precision,
    batch_size: int,
    warmup_steps: int,
    timed_steps: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    gc.collect()
    torch.cuda.empty_cache()

    num_keypoints = dataset[indices[0]]["keypoints_xy"].shape[0]
    model = HRNetHeatmapRegressor(num_keypoints, pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = _loader(dataset, indices, batch_size, seed)
    iterator = iter(loader)
    use_bfloat16 = precision == "bfloat16"
    data_wait_seconds: list[float] = []
    compute_seconds: list[float] = []
    losses: list[float] = []
    torch.cuda.reset_peak_memory_stats(device)
    model.train()

    for step in range(warmup_steps + timed_steps):
        data_started = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        data_elapsed = time.perf_counter() - data_started

        torch.cuda.synchronize(device)
        compute_started = time.perf_counter()
        inputs = (
            (batch[modality] - IMAGENET_MEAN[None, :, None, None])
            / IMAGENET_STD[None, :, None, None]
        ).to(device, non_blocking=True)
        targets_px = batch["keypoints_xy"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=use_bfloat16,
        ):
            prediction = model(inputs)
            targets, visibility = gaussian_keypoint_heatmaps(targets_px)
            weights = visibility[:, :, None, None]
            loss = torch.sum((prediction - targets).square() * weights) / (
                weights.sum() * prediction.shape[-2] * prediction.shape[-1]
            )
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"non-finite {precision} loss for {modality}, batch size {batch_size}"
            )
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize(device)
        compute_elapsed = time.perf_counter() - compute_started

        if step >= warmup_steps:
            data_wait_seconds.append(data_elapsed)
            compute_seconds.append(compute_elapsed)
            losses.append(float(loss.detach()))

    gradients_finite = all(
        parameter.grad is None or bool(torch.all(torch.isfinite(parameter.grad)))
        for parameter in model.parameters()
    )
    measured_seconds = sum(data_wait_seconds) + sum(compute_seconds)
    result = {
        "modality": modality,
        "precision": precision,
        "batch_size": batch_size,
        "warmup_steps": warmup_steps,
        "timed_steps": timed_steps,
        "finite_losses": all(np.isfinite(losses)),
        "finite_gradients": gradients_finite,
        "first_measured_loss": losses[0],
        "last_measured_loss": losses[-1],
        "data_wait_seconds": _distribution(data_wait_seconds),
        "compute_seconds": _distribution(compute_seconds),
        "complete_step_seconds": _distribution(
            [
                data_wait + compute
                for data_wait, compute in zip(data_wait_seconds, compute_seconds, strict=True)
            ]
        ),
        "end_to_end_samples_per_second": timed_steps * batch_size / measured_seconds,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
        "parameters_total": sum(parameter.numel() for parameter in model.parameters()),
        "parameters_trainable": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }
    del optimizer, model, loader, iterator
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> int:
    args = parse_args()
    if min(args.samples, args.warmup_steps, args.timed_steps) <= 0:
        raise ValueError("samples, warm-up steps, and timed steps must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("this compute profile requires CUDA")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda")
    dataset = FreshArchiveDataset(
        args.raw_root,
        args.split_index,
        role="train",
        objects=(args.object,),
    )
    indices = _evenly_spaced_indices(len(dataset), args.samples)
    bfloat16_supported = torch.cuda.is_bf16_supported()
    modalities: tuple[Modality, ...] = ("rgb", "event")
    precisions: tuple[Precision, ...] = ("float32", "bfloat16")
    results: list[dict[str, Any]] = []
    for modality in modalities:
        for precision in precisions:
            if precision == "bfloat16" and not bfloat16_supported:
                continue
            for batch_size in (16, 32):
                print(
                    f"Profiling {modality}, {precision}, batch={batch_size}...",
                    flush=True,
                )
                result = _profile_trial(
                    dataset,
                    indices,
                    modality,
                    precision,
                    batch_size,
                    args.warmup_steps,
                    args.timed_steps,
                    args.seed,
                    device,
                )
                results.append(result)
                print(
                    f"  {result['end_to_end_samples_per_second']:.2f} samples/s, "
                    f"peak={result['peak_vram_bytes'] / 2**30:.2f} GiB",
                    flush=True,
                )
    dataset.close()
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": (
            "pass"
            if results and all(row["finite_losses"] and row["finite_gradients"] for row in results)
            else "fail"
        ),
        "scope": "complete training-step compute feasibility; not an accuracy result",
        "config": {
            "dataset": "fresh-synthetic-grouped-v1",
            "object": args.object,
            "samples": args.samples,
            "selection": "evenly spaced within frozen train role",
            "architecture": "timm/hrnet_w18_small_v2.ms_in1k plus 64-channel heatmap decoder",
            "optimizer": "AdamW(lr=1e-3, weight_decay=1e-4)",
            "warmup_steps": args.warmup_steps,
            "timed_steps": args.timed_steps,
            "seed": args.seed,
            "deterministic_algorithms": True,
        },
        "provenance": {
            "git_commit": _git_commit(),
            "uv_lock_sha256": _sha256(Path("uv.lock")),
            "split_index_sha256": _sha256(args.split_index),
            "source_sha256": {
                path: _sha256(Path(path))
                for path in (
                    "scripts/profile_training_compute.py",
                    "src/saga_pose/data/dataset.py",
                    "src/saga_pose/models/heatmaps.py",
                )
            },
            "python": platform.python_version(),
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "bfloat16_supported": bfloat16_supported,
        },
        "results": results,
    }
    _write_json(args.output, report)
    print(f"Training compute profile: {report['status'].upper()}")
    print(f"  report: {args.output}")
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
