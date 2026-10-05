"""Historical crop-v1 cache benchmark (produced outputs/feasibility/crop-cache-benchmark.json).

Superseded by scripts/build_crop_cache.py (schema 2, crop-v2). The schema-1 reader
it depended on was removed, so this script is retained only as provenance for
the frozen benchmark report and refuses to run.
"""

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
from typing import Any, Literal, cast

import numpy as np
import torch
from numpy.lib.format import open_memmap
from torch.utils.data import DataLoader, Dataset, Subset

from saga_pose.data.cache import FreshCropCacheDataset
from saga_pose.data.dataset import FreshArchiveDataset, FreshItem
from saga_pose.models.heatmaps import HRNetHeatmapRegressor, gaussian_keypoint_heatmaps

Modality = Literal["rgb", "event"]
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])
TENSOR_FIELDS = (
    "clipping_ratio",
    "keypoints_xy",
    "intrinsics",
    "pose_object_to_camera_mm",
    "landmarks_object_mm",
    "source_bbox_xyxy",
    "crop_xyxy",
)


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
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--timed-steps", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=249)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("data/fresh/cache/crop-v1/cassini-train-256"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/feasibility/crop-cache-benchmark.json"),
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


def _git_dirty() -> bool:
    completed = subprocess.run(
        ["git", "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    )
    return bool(completed.stdout.strip())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _evenly_spaced_indices(length: int, count: int) -> list[int]:
    if count <= 0 or count > length:
        raise ValueError(f"sample count must be in [1, {length}]")
    return np.linspace(0, length - 1, count, dtype=np.int64).tolist()


def _uint8_image(tensor: torch.Tensor) -> np.ndarray:
    return torch.round(tensor * 255.0).clamp(0, 255).to(torch.uint8).cpu().numpy()


def _build_cache(
    dataset: FreshArchiveDataset,
    indices: Sequence[int],
    cache_root: Path,
    split_sha256: str,
) -> dict[str, Any]:
    expected_ids = [dataset[index]["sample_id"] for index in indices]
    manifest_path = cache_root / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("sample_ids") != expected_ids:
            raise ValueError("Existing crop-cache shard uses different sample IDs")
        if manifest.get("split_index_sha256") != split_sha256:
            raise ValueError("Existing crop-cache shard uses a different split index")
        return {
            "reused": True,
            "build_seconds": 0.0,
            "manifest": manifest,
        }
    if cache_root.exists():
        raise RuntimeError(
            f"Refusing to overwrite incomplete cache directory: {cache_root}. "
            "Inspect or remove that exact derived-cache directory manually."
        )

    started = time.perf_counter()
    cache_root.mkdir(parents=True)
    first = dataset[indices[0]]
    prototypes: dict[str, np.ndarray] = {
        "rgb_uint8": _uint8_image(first["rgb"]),
        "event_uint8": _uint8_image(first["event"]),
    }
    for name in TENSOR_FIELDS:
        prototypes[name] = first[cast(Any, name)].cpu().numpy()

    maps: dict[str, np.memmap] = {}
    array_specs: dict[str, dict[str, Any]] = {}
    for name, prototype in prototypes.items():
        filename = f"{name}.npy"
        shape = (len(indices), *prototype.shape)
        maps[name] = open_memmap(
            cache_root / filename,
            mode="w+",
            dtype=prototype.dtype,
            shape=shape,
        )
        array_specs[name] = {
            "filename": filename,
            "shape": list(shape),
            "dtype": str(prototype.dtype),
        }

    sample_ids: list[str] = []
    object_names: list[str] = []
    for cache_index, source_index in enumerate(indices):
        item = dataset[source_index]
        sample_ids.append(item["sample_id"])
        object_names.append(item["object_name"])
        maps["rgb_uint8"][cache_index] = _uint8_image(item["rgb"])
        maps["event_uint8"][cache_index] = _uint8_image(item["event"])
        for name in TENSOR_FIELDS:
            maps[name][cache_index] = item[cast(Any, name)].cpu().numpy()

    for array in maps.values():
        array.flush()
    del maps
    for spec in array_specs.values():
        spec["sha256"] = _sha256(cache_root / cast(str, spec["filename"]))

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "complete",
        "cache_name": "fresh-crop-v1",
        "role": dataset.role,
        "sample_selection": "evenly spaced within frozen object/role ordering",
        "sample_ids": sample_ids,
        "object_names": object_names,
        "split_index_sha256": split_sha256,
        "preprocessing": {
            "target_size_wh": list(dataset.target_size_wh),
            "crop_padding_factor": dataset.crop_padding_factor,
            "crop_source": "annotation_xyxy_shared_oracle",
            "off_frame_policy": "constant_zero_padding",
            "resize": "Pillow bilinear",
            "clipping_threshold": dataset.clipping_threshold,
            "clipping_grid_size": dataset.clipping_grid_size,
        },
        "arrays": array_specs,
    }
    _write_json(manifest_path, manifest)
    return {
        "reused": False,
        "build_seconds": time.perf_counter() - started,
        "manifest": manifest,
    }


def _cache_size(cache_root: Path) -> int:
    return sum(path.stat().st_size for path in cache_root.iterdir() if path.is_file())


def _equivalence(
    raw_dataset: FreshArchiveDataset,
    source_indices: Sequence[int],
    cache_dataset: FreshCropCacheDataset,
) -> dict[str, Any]:
    positions = sorted({0, len(source_indices) // 2, len(source_indices) - 1})
    max_absolute_difference: dict[str, float] = {}
    exact_images = True
    metadata_match = True
    for position in positions:
        raw = raw_dataset[source_indices[position]]
        cached = cache_dataset[position]
        metadata_match &= (
            raw["sample_id"] == cached["sample_id"]
            and raw["object_name"] == cached["object_name"]
            and raw["role"] == cached["role"]
        )
        for name in ("rgb", "event", *TENSOR_FIELDS):
            raw_tensor = raw[cast(Any, name)]
            cached_tensor = cached[cast(Any, name)]
            difference = float(torch.max(torch.abs(raw_tensor - cached_tensor)))
            max_absolute_difference[name] = max(max_absolute_difference.get(name, 0.0), difference)
            if name in ("rgb", "event"):
                exact_images &= torch.equal(raw_tensor, cached_tensor)
    return {
        "positions": positions,
        "sample_metadata_match": metadata_match,
        "images_bit_exact_after_float_conversion": exact_images,
        "max_absolute_difference": max_absolute_difference,
        "pass": metadata_match and exact_images and max(max_absolute_difference.values()) == 0.0,
    }


def _loader(dataset: Dataset[FreshItem], batch_size: int, seed: int) -> DataLoader[FreshItem]:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(seed),
    )


def _profile(
    dataset: Dataset[FreshItem],
    *,
    storage: str,
    modality: Modality,
    warmup_steps: int,
    timed_steps: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    gc.collect()
    torch.cuda.empty_cache()
    first = dataset[0]
    model = HRNetHeatmapRegressor(first["keypoints_xy"].shape[0], pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = _loader(dataset, 16, seed)
    iterator = iter(loader)
    waits: list[float] = []
    computes: list[float] = []
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
        target_points = batch["keypoints_xy"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(inputs)
        targets, visibility = gaussian_keypoint_heatmaps(target_points)
        weights = visibility[:, :, None, None]
        loss = torch.sum((prediction - targets).square() * weights) / (
            weights.sum() * prediction.shape[-2] * prediction.shape[-1]
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite cache benchmark loss for {storage}/{modality}")
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize(device)
        compute_elapsed = time.perf_counter() - compute_started
        if step >= warmup_steps:
            waits.append(data_elapsed)
            computes.append(compute_elapsed)
            losses.append(float(loss.detach()))

    elapsed = sum(waits) + sum(computes)
    result = {
        "storage": storage,
        "modality": modality,
        "batch_size": 16,
        "warmup_steps": warmup_steps,
        "timed_steps": timed_steps,
        "data_wait_seconds_mean": float(np.mean(waits)),
        "data_wait_seconds_median": float(np.median(waits)),
        "compute_seconds_mean": float(np.mean(computes)),
        "complete_step_seconds_mean": elapsed / timed_steps,
        "end_to_end_samples_per_second": timed_steps * 16 / elapsed,
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "finite": bool(np.all(np.isfinite(losses))),
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    del iterator, loader, optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> int:
    raise SystemExit(
        "benchmark_crop_cache.py is a historical crop-v1 script; use scripts/build_crop_cache.py"
    )
    args = parse_args()
    if min(args.samples, args.warmup_steps, args.timed_steps, args.repeats) <= 0:
        raise ValueError("samples, steps, and repeats must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("the crop-cache compute benchmark requires CUDA")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    split_sha256 = _sha256(args.split_index)
    raw = FreshArchiveDataset(
        args.raw_root,
        args.split_index,
        role="train",
        objects=(args.object,),
    )
    source_indices = _evenly_spaced_indices(len(raw), args.samples)
    build = _build_cache(raw, source_indices, args.cache_root, split_sha256)
    cached = FreshCropCacheDataset(args.cache_root)
    equivalence = _equivalence(raw, source_indices, cached)
    if not equivalence["pass"]:
        raise RuntimeError("crop-cache equivalence check failed")

    raw_subset = cast(Dataset[FreshItem], Subset(raw, source_indices))
    device = torch.device("cuda")
    results: list[dict[str, Any]] = []
    for repeat in range(args.repeats):
        for modality in cast(tuple[Modality, ...], ("rgb", "event")):
            for storage, dataset in (("raw_zip", raw_subset), ("uint8_memmap", cached)):
                print(f"repeat={repeat + 1} {modality} {storage}", flush=True)
                result = _profile(
                    dataset,
                    storage=storage,
                    modality=modality,
                    warmup_steps=args.warmup_steps,
                    timed_steps=args.timed_steps,
                    seed=args.seed,
                    device=device,
                )
                result["repeat"] = repeat + 1
                results.append(result)
                print(
                    f"  {result['end_to_end_samples_per_second']:.2f} samples/s, "
                    f"wait={result['data_wait_seconds_mean']:.4f}s",
                    flush=True,
                )

    throughputs: dict[str, list[float]] = {"raw_zip": [], "uint8_memmap": []}
    for result in results:
        throughputs[cast(str, result["storage"])].append(
            cast(float, result["end_to_end_samples_per_second"])
        )
    raw_median = float(np.median(throughputs["raw_zip"]))
    cache_median = float(np.median(throughputs["uint8_memmap"]))
    speedup = cache_median / raw_median
    source_paths = (
        Path("scripts/benchmark_crop_cache.py"),
        Path("src/saga_pose/data/cache.py"),
        Path("src/saga_pose/data/dataset.py"),
        Path("src/saga_pose/models/heatmaps.py"),
    )
    report = {
        "schema_version": 1,
        "status": "pass" if speedup >= 2.0 else "stop",
        "scope": "bounded uint8 crop-cache feasibility; not an accuracy result",
        "decision_rule": "authorize full-cache expansion only at >=2x median end-to-end throughput",
        "config": {
            "dataset": "fresh-synthetic-grouped-v1",
            "object": args.object,
            "role": "train",
            "samples": args.samples,
            "selection": "evenly spaced within frozen train role",
            "batch_size": 16,
            "precision": "float32",
            "warmup_steps": args.warmup_steps,
            "timed_steps": args.timed_steps,
            "repeats": args.repeats,
            "seed": args.seed,
        },
        "provenance": {
            "git_commit": _git_commit(),
            "working_tree_dirty": _git_dirty(),
            "uv_lock_sha256": _sha256(Path("uv.lock")),
            "split_index_sha256": split_sha256,
            "fresh_dataset_manifest_sha256": _sha256(Path("docs/fresh-dataset-manifest.json")),
            "cache_manifest_sha256": _sha256(args.cache_root / "manifest.json"),
            "source_sha256": {str(path): _sha256(path) for path in source_paths},
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(device),
            "gpu_total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
        },
        "cache": {
            "root_policy": "data/fresh/cache/crop-v1; ignored derived data",
            "reused": build["reused"],
            "build_seconds": build["build_seconds"],
            "size_bytes": _cache_size(args.cache_root),
            "full_cache_estimate_bytes": 14_816_378_880,
            "equivalence": equivalence,
        },
        "results": results,
        "summary": {
            "raw_median_samples_per_second": raw_median,
            "cache_median_samples_per_second": cache_median,
            "speedup": speedup,
            "full_cache_expansion_authorized": speedup >= 2.0,
        },
    }
    _write_json(args.output, report)
    raw.close()
    print(f"median speedup: {speedup:.2f}x", flush=True)
    print(f"full-cache expansion authorized: {speedup >= 2.0}", flush=True)
    print(f"report: {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
