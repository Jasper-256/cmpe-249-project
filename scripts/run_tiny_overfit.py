"""Overfit 16 synthetic pairs independently with RGB and event pilot models."""

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
import torch
from torch import Tensor

from saga_pose.data.dataset import FreshArchiveDataset
from saga_pose.models.keypoints import PilotKeypointRegressor

Modality = Literal["rgb", "event"]


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
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=249)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/feasibility/tiny-overfit.json"),
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


def _standardize(inputs: Tensor) -> tuple[Tensor, list[float], list[float]]:
    mean = inputs.mean(dim=(0, 2, 3), keepdim=True)
    standard_deviation = inputs.std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-3)
    normalized = (inputs - mean) / standard_deviation
    return (
        normalized,
        [float(value) for value in mean.flatten()],
        [float(value) for value in standard_deviation.flatten()],
    )


def _train_modality(
    modality: Modality,
    inputs_cpu: Tensor,
    targets_cpu: Tensor,
    *,
    steps: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)
    inputs, channel_mean, channel_std = _standardize(inputs_cpu)
    inputs = inputs.to(device)
    targets = targets_cpu.to(device)
    model = PilotKeypointRegressor(num_keypoints=targets.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    model.train()
    with torch.no_grad():
        initial_prediction = model(inputs)
        initial_rmse_px = float(torch.sqrt(torch.mean((initial_prediction - targets) ** 2)) * 255)
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(inputs)
        loss = torch.mean((prediction - targets) ** 2)
        if not torch.isfinite(loss):
            raise RuntimeError(f"{modality} loss became non-finite at step {step}")
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % 25 == 0 or step + 1 == steps:
            rmse_px = float(torch.sqrt(loss.detach()) * 255)
            history.append({"step": step + 1, "loss": float(loss.detach()), "rmse_px": rmse_px})
    torch.cuda.synchronize(device) if device.type == "cuda" else None
    elapsed = time.perf_counter() - started
    model.eval()
    with torch.no_grad():
        final_prediction = model(inputs)
        final_rmse_px = float(torch.sqrt(torch.mean((final_prediction - targets) ** 2)) * 255)
    reduction = 1.0 - final_rmse_px / initial_rmse_px
    passed = final_rmse_px <= 5.0 and reduction >= 0.90
    return {
        "modality": modality,
        "passed": passed,
        "criterion": "final coordinate RMSE <= 5 px and >= 90% reduction from initialization",
        "initial_rmse_px": initial_rmse_px,
        "final_rmse_px": final_rmse_px,
        "rmse_reduction_fraction": reduction,
        "steps": steps,
        "learning_rate": learning_rate,
        "parameters": parameter_count,
        "elapsed_seconds": elapsed,
        "steps_per_second": steps / elapsed,
        "peak_vram_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        ),
        "input_channel_mean": channel_mean,
        "input_channel_std": channel_std,
        "history": history,
    }


def main() -> int:
    args = parse_args()
    if args.examples <= 0 or args.steps <= 0:
        raise ValueError("examples and steps must be positive")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = FreshArchiveDataset(
        args.raw_root,
        args.split_index,
        role="train",
        objects=(args.object,),
    )
    indices = np.linspace(0, len(dataset) - 1, args.examples, dtype=np.int64)
    items = [dataset[int(index)] for index in indices]
    dataset.close()
    targets = torch.stack([item["keypoints_xy"] for item in items]).div_(255.0)
    modality_inputs = {
        modality: torch.stack([item[modality] for item in items]) for modality in ("rgb", "event")
    }
    results = [
        _train_modality(
            modality,
            modality_inputs[modality],
            targets,
            steps=args.steps,
            learning_rate=args.learning_rate,
            seed=args.seed,
            device=device,
        )
        for modality in ("rgb", "event")
    ]
    report = {
        "schema_version": 1,
        "status": "pass" if all(result["passed"] for result in results) else "fail",
        "scope": "plumbing competence only; not a reportable baseline",
        "config": {
            "dataset": "fresh-synthetic-grouped-v1 train",
            "object": args.object,
            "examples": args.examples,
            "sample_ids": [item["sample_id"] for item in items],
            "sample_selection": "evenly spaced over grouped-v1 training membership",
            "seed": args.seed,
            "target_size_wh": [256, 256],
            "augmentation": None,
            "optimizer": "AdamW(weight_decay=0)",
        },
        "provenance": {
            "git_commit": _git_commit(),
            "uv_lock_sha256": _sha256(Path("uv.lock")),
            "split_index_sha256": _sha256(args.split_index),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        },
        "results": results,
    }
    _write_json(args.output, report)
    print(f"Tiny overfit competence: {report['status'].upper()}")
    for result in results:
        print(
            f"  {result['modality']}: {result['initial_rmse_px']:.2f} -> "
            f"{result['final_rmse_px']:.2f} px, "
            f"{result['steps_per_second']:.1f} steps/s, pass={result['passed']}"
        )
    print(f"  report: {args.output}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
