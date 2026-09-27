"""Overfit the compact pretrained heatmap model on 16 RGB/event samples."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
import timm
import torch
from torch import Tensor

from saga_pose.data.dataset import FreshArchiveDataset
from saga_pose.models.heatmaps import (
    HRNetHeatmapRegressor,
    decode_heatmap_argmax,
    gaussian_keypoint_heatmaps,
)

Modality = Literal["rgb", "event"]
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
    parser.add_argument("--examples", type=int, default=16)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=249)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/feasibility/heatmap-tiny-overfit.json"),
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


def _normalize(inputs: Tensor) -> Tensor:
    return (inputs - IMAGENET_MEAN[None, :, None, None]) / IMAGENET_STD[None, :, None, None]


@torch.inference_mode()
def _rmse(model: HRNetHeatmapRegressor, inputs: Tensor, targets: Tensor) -> float:
    model.eval()
    coordinates, _ = decode_heatmap_argmax(model(inputs))
    return float(torch.sqrt(torch.mean((coordinates - targets) ** 2)))


def _train_modality(
    modality: Modality,
    inputs_cpu: Tensor,
    keypoints_cpu: Tensor,
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)
    inputs = _normalize(inputs_cpu).to(device)
    keypoints = keypoints_cpu.to(device)
    targets, visibility = gaussian_keypoint_heatmaps(keypoints)
    model = HRNetHeatmapRegressor(keypoints.shape[1], pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    initial_rmse = _rmse(model, inputs, keypoints)
    generator = torch.Generator().manual_seed(seed)
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    model.train()
    for step in range(steps):
        indices = torch.randperm(inputs.shape[0], generator=generator)[:batch_size].to(device)
        prediction = model(inputs[indices])
        weights = visibility[indices, :, None, None]
        loss = torch.sum((prediction - targets[indices]).square() * weights) / (
            weights.sum() * prediction.shape[-2] * prediction.shape[-1]
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"{modality} loss became non-finite at step {step + 1}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % 25 == 0 or step + 1 == steps:
            history.append({"step": step + 1, "heatmap_mse": float(loss.detach())})
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    final_rmse = _rmse(model, inputs, keypoints)
    reduction = 1.0 - final_rmse / initial_rmse
    passed = final_rmse <= 5.0 and reduction >= 0.90
    return {
        "modality": modality,
        "passed": passed,
        "criterion": "final coordinate RMSE <= 5 px and >= 90% reduction from initialization",
        "initial_coordinate_rmse_px": initial_rmse,
        "final_coordinate_rmse_px": final_rmse,
        "rmse_reduction_fraction": reduction,
        "parameters_total": sum(parameter.numel() for parameter in model.parameters()),
        "parameters_trainable": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "steps": steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "elapsed_seconds": elapsed,
        "steps_per_second": steps / elapsed,
        "peak_vram_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        ),
        "history": history,
    }


def main() -> int:
    args = parse_args()
    if min(args.examples, args.steps, args.batch_size) <= 0 or args.batch_size > args.examples:
        raise ValueError("positive counts and batch_size <= examples are required")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = FreshArchiveDataset(
        args.raw_root, args.split_index, role="train", objects=(args.object,)
    )
    indices = np.linspace(0, len(dataset) - 1, args.examples, dtype=np.int64)
    items = [dataset[int(index)] for index in indices]
    dataset.close()
    keypoints = torch.stack([item["keypoints_xy"] for item in items])
    inputs = {
        modality: torch.stack([item[modality] for item in items]) for modality in ("rgb", "event")
    }
    results = [
        _train_modality(
            modality,
            inputs[modality],
            keypoints,
            steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            device=device,
        )
        for modality in ("rgb", "event")
    ]
    report = {
        "schema_version": 1,
        "status": "pass" if all(result["passed"] for result in results) else "fail",
        "scope": "heatmap data-path competence only; not a reportable baseline",
        "config": {
            "dataset": "fresh-synthetic-grouped-v1 train",
            "object": args.object,
            "examples": args.examples,
            "sample_ids": [item["sample_id"] for item in items],
            "selection": "evenly spaced over grouped-v1 training membership",
            "crop": "center-preserving square, 1.2x padding, zero border",
            "architecture": "timm HRNet-W18-Small-v2 features + 64-channel FPN heatmap head",
            "pretraining": "timm/hrnet_w18_small_v2.ms_in1k",
            "heatmap_size_wh": [64, 64],
            "heatmap_sigma": 1.5,
            "normalization": "ImageNet mean/std for both modalities",
            "optimizer": "AdamW(weight_decay=1e-4)",
            "augmentation": None,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
        },
        "provenance": {
            "git_commit": _git_commit(),
            "uv_lock_sha256": _sha256(Path("uv.lock")),
            "split_index_sha256": _sha256(args.split_index),
            "model_source_sha256": _sha256(Path("src/saga_pose/models/heatmaps.py")),
            "script_sha256": _sha256(Path(__file__)),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "timm": timm.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "precision": "float32",
        },
        "results": results,
    }
    _write_json(args.output, report)
    print(f"Heatmap tiny overfit: {report['status'].upper()}")
    for result in results:
        print(
            f"  {result['modality']}: {result['initial_coordinate_rmse_px']:.2f} -> "
            f"{result['final_coordinate_rmse_px']:.2f}px, "
            f"{result['steps_per_second']:.1f} steps/s, pass={result['passed']}"
        )
    print(f"  report: {args.output}")
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
