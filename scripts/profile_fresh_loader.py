"""Measure lazy FRESH paired-crop DataLoader throughput without model compute."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from saga_pose.data.dataset import FreshArchiveDataset, FreshItem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("data/fresh/raw"))
    parser.add_argument(
        "--split-index",
        type=Path,
        default=Path("data/fresh/index/synthetic-split-v1.json"),
    )
    parser.add_argument("--object", default="cassini", choices=("cassini", "satty", "soho"))
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/feasibility/data-loader-profile.json"),
    )
    return parser.parse_args()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def profile_loader(
    dataset: FreshArchiveDataset,
    *,
    batch_size: int,
    workers: int,
    sample_limit: int,
) -> dict[str, Any]:
    loader: DataLoader[FreshItem] = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )
    samples = 0
    batches = 0
    tensor_bytes = 0
    started = time.perf_counter()
    for batch in loader:
        current_batch = int(batch["rgb"].shape[0])
        if batches == 0:
            tensor_bytes = sum(
                int(batch[key].numel() * batch[key].element_size())
                for key in (
                    "rgb",
                    "event",
                    "clipping_ratio",
                    "keypoints_xy",
                    "intrinsics",
                    "pose_object_to_camera_mm",
                    "landmarks_object_mm",
                )
            )
        samples += current_batch
        batches += 1
        if samples >= sample_limit:
            break
    elapsed = time.perf_counter() - started
    return {
        "batch_size": batch_size,
        "workers": workers,
        "samples": samples,
        "batches": batches,
        "elapsed_seconds": elapsed,
        "samples_per_second": samples / elapsed,
        "first_batch_tensor_bytes": tensor_bytes,
    }


def main() -> int:
    args = parse_args()
    dataset = FreshArchiveDataset(
        args.raw_root,
        args.split_index,
        role="train",
        objects=(args.object,),
    )
    profiles = [
        profile_loader(
            dataset,
            batch_size=batch_size,
            workers=workers,
            sample_limit=args.samples,
        )
        for workers in (0, 2)
        for batch_size in (1, 4, 8)
    ]
    dataset.close()
    report = {
        "schema_version": 1,
        "dataset": "fresh-synthetic-grouped-v1",
        "object": args.object,
        "target_size_wh": [256, 256],
        "storage_mode": "zero-extraction ZIP random access",
        "device": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "profiles": profiles,
    }
    _write_json(args.output, report)
    print("FRESH lazy DataLoader profile")
    for profile in profiles:
        print(
            f"  workers={profile['workers']} batch={profile['batch_size']}: "
            f"{profile['samples_per_second']:.1f} pairs/s"
        )
    print(f"  report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
