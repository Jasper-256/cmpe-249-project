"""Build the crop-v2 schema-2 caches for synthetic train/validation and real test."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from saga_pose.data.augment import TargetedSaturationConfig
from saga_pose.data.cache import FreshCropCacheDataset, build_crop_cache
from saga_pose.data.dataset import (
    PREPROCESSING_VERSION,
    FreshArchiveDataset,
    FreshRealArchiveDataset,
)
from saga_pose.data.real import FRESH_REPOSITORY_COMMIT

OBJECTS = ("cassini", "satty", "soho")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("data/fresh/raw"))
    parser.add_argument(
        "--split-index", type=Path, default=Path("data/fresh/index/synthetic-split-v1.json")
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("data/fresh/cache/crop-v2"))
    parser.add_argument("--targets", nargs="+", default=["synthetic", "real"])
    parser.add_argument("--objects", nargs="+", default=list(OBJECTS))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--tss-seed", type=int, default=249)
    parser.add_argument("--output", type=Path, default=Path("outputs/data/crop-cache-v2.json"))
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout.strip()


def _progress(label: str) -> Any:
    last = [0.0]

    def report(done: int, total: int) -> None:
        now = time.perf_counter()
        if now - last[0] > 30 or done == total:
            last[0] = now
            print(f"  {label}: {done:,}/{total:,}", flush=True)

    return report


def _summary(root: Path) -> dict[str, Any]:
    cache = FreshCropCacheDataset(root)
    clipping = np.asarray(cache.array("clipping_ratio_tss"))
    applied = np.asarray(cache.array("tss_applied"), dtype=bool)
    active = (clipping >= 0.01).reshape(len(cache), -1)
    natural_active = (np.asarray(cache.array("clipping_ratio")) >= 0.01).reshape(len(cache), -1)
    summary: dict[str, Any] = {
        "samples": len(cache),
        "bytes": sum(
            (root / spec["filename"]).stat().st_size for spec in cache.manifest["arrays"].values()
        ),
        "build_seconds": cache.manifest["build_seconds"],
        "natural_crops_with_active_patch": float(natural_active.any(axis=1).mean()),
        "tss_applied_fraction": float(applied.mean()),
    }
    if applied.any():
        summary["tss_applied_crops_with_active_patch"] = float(active[applied].any(axis=1).mean())
        summary["tss_applied_active_patch_fraction_median"] = float(
            np.median(active[applied].mean(axis=1))
        )
    return summary


def main() -> int:
    args = parse_args()
    tss = TargetedSaturationConfig()
    provenance = {
        "git_commit": _git("rev-parse", "HEAD"),
        "working_tree_dirty": bool(_git("status", "--porcelain")),
        "uv_lock_sha256": _sha256(Path("uv.lock")),
        "split_index_sha256": _sha256(args.split_index),
        "preprocessing_version": PREPROCESSING_VERSION,
        "tss_config": tss.as_dict(),
        "tss_seed": args.tss_seed,
        "fresh_addendum_commit": FRESH_REPOSITORY_COMMIT,
        "python": platform.python_version(),
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "cache_dir": str(args.cache_dir),
        "provenance": provenance,
        "caches": {},
    }
    if "synthetic" in args.targets:
        for object_name in args.objects:
            for role in ("train", "validation"):
                name = f"{object_name}-{role}"
                dataset = FreshArchiveDataset(
                    args.raw_root,
                    args.split_index,
                    role=role,
                    objects=(object_name,),
                    tss_config=tss,
                    tss_seed=args.tss_seed,
                )
                print(f"Building {name} ({len(dataset):,} samples)...", flush=True)
                build_crop_cache(
                    dataset,
                    dataset.sample_ids,
                    args.cache_dir / name,
                    role=role,
                    store_tss=True,
                    provenance=provenance,
                    num_workers=args.workers,
                    progress=_progress(name),
                )
                dataset.close()
                report["caches"][name] = _summary(args.cache_dir / name)
    if "real" in args.targets:
        # One cache per object: objects have different landmark counts (18 vs 26)
        # and, as in FRESH, each object has its own model.
        for object_name in args.objects:
            name = f"real-{object_name}"
            probe = FreshRealArchiveDataset(args.raw_root)
            sequences = sorted({sample.split("/", maxsplit=1)[0] for sample in probe.sample_ids})
            probe.close()
            selected = [sequence for sequence in sequences if sequence.startswith(object_name)]
            dataset_real = FreshRealArchiveDataset(args.raw_root, sequences=selected)
            print(f"Building {name} ({len(dataset_real):,} samples)...", flush=True)
            build_crop_cache(
                dataset_real,
                dataset_real.sample_ids,
                args.cache_dir / name,
                role="real_test",
                store_tss=False,
                provenance={**provenance, "annotation_sources": dataset_real.annotation_sources},
                num_workers=args.workers,
                progress=_progress(name),
            )
            dataset_real.close()
            # Only storage/format facts are summarized for real data; no error or
            # clipping statistics are inspected before the frozen evaluation.
            real_cache = FreshCropCacheDataset(args.cache_dir / name)
            report["caches"][name] = {
                "samples": len(real_cache),
                "sequences": sorted(set(real_cache.sequences)),
                "build_seconds": real_cache.manifest["build_seconds"],
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, Any] = {}
    if args.output.is_file():
        existing = json.loads(args.output.read_text(encoding="utf-8")).get("caches", {})
    report["caches"] = {**existing, **report["caches"]}
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"report: {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
