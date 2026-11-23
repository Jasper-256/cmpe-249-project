"""Build the event-supp-v1 supplement cache (activity64 and activity_support maps).

Reads the raw archives through the crop-v2 dataset classes and crop path, one
supplement split per crop-v2 split, in crop-v2's sample order. crop-v2 itself is
only read (order and alignment checks), never modified. Real splits are built
for the later frozen evaluation; no real statistic beyond storage facts and the
format-level alignment checks is computed or printed here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

from saga_pose.data.dataset import FreshArchiveDataset, FreshRealArchiveDataset
from saga_pose.data.real import FRESH_REPOSITORY_COMMIT
from saga_pose.data.representations import (
    ACTIVITY64_THRESHOLD,
    ACTIVITY_THRESHOLD,
    SUPPLEMENT_REPRESENTATIONS,
    SUPPLEMENT_VERSION,
)
from saga_pose.data.supplement import (
    EventSupplementSource,
    build_event_supplement,
    supplement_code_sha256,
)

OBJECTS = ("cassini", "satty", "soho")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("data/fresh/raw"))
    parser.add_argument(
        "--split-index", type=Path, default=Path("data/fresh/index/synthetic-split-v1.json")
    )
    parser.add_argument("--crop-cache-dir", type=Path, default=Path("data/fresh/cache/crop-v2"))
    parser.add_argument(
        "--out-dir", type=Path, default=Path("data/fresh/cache") / SUPPLEMENT_VERSION
    )
    parser.add_argument("--targets", nargs="+", default=["synthetic", "real"])
    parser.add_argument("--objects", nargs="+", default=list(OBJECTS))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("outputs/data/event-supp-v1.json"))
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


def main() -> int:
    args = parse_args()
    provenance = {
        "git_commit": _git("rev-parse", "HEAD"),
        "working_tree_dirty": bool(_git("status", "--porcelain")),
        "uv_lock_sha256": _sha256(Path("uv.lock")),
        "split_index_sha256": _sha256(args.split_index),
        "supplement_code_sha256": supplement_code_sha256(),
        "activity_threshold": ACTIVITY_THRESHOLD,
        "activity64_threshold": ACTIVITY64_THRESHOLD,
        "fresh_addendum_commit": FRESH_REPOSITORY_COMMIT,
        "python": platform.python_version(),
    }
    splits: list[tuple[str, FreshArchiveDataset | FreshRealArchiveDataset]] = []
    if "synthetic" in args.targets:
        for object_name in args.objects:
            for role in ("train", "validation"):
                splits.append(
                    (
                        f"{object_name}-{role}",
                        FreshArchiveDataset(
                            args.raw_root, args.split_index, role=role, objects=(object_name,)
                        ),
                    )
                )
    if "real" in args.targets:
        probe = FreshRealArchiveDataset(args.raw_root)
        sequences = sorted({sample.split("/", maxsplit=1)[0] for sample in probe.sample_ids})
        probe.close()
        for object_name in args.objects:
            selected = [sequence for sequence in sequences if sequence.startswith(object_name)]
            splits.append(
                (f"real-{object_name}", FreshRealArchiveDataset(args.raw_root, sequences=selected))
            )
    caches: dict[str, Any] = {}
    for name, dataset in splits:
        print(f"Building {name} ({len(dataset):,} samples)...", flush=True)
        manifest = build_event_supplement(
            EventSupplementSource(dataset),
            args.crop_cache_dir / name,
            args.out_dir / name,
            provenance=provenance,
            num_workers=args.workers,
            progress=_progress(name),
            statistics=not name.startswith("real-"),
        )
        dataset.close()
        root = args.out_dir / name
        entry: dict[str, Any] = {
            "samples": len(manifest["sample_ids"]),
            "bytes": sum(
                (root / spec["filename"]).stat().st_size for spec in manifest["arrays"].values()
            ),
            "build_seconds": manifest["build_seconds"],
            "manifest_sha256": _sha256(root / "manifest.json"),
            "source_crop_manifest_sha256": manifest["source_crop_manifest_sha256"],
        }
        if not name.startswith("real-"):
            entry["statistics"] = manifest["statistics"]
        caches[name] = entry
        print(f"  {name}: {entry['bytes'] / 2**30:.2f} GiB in {entry['build_seconds']:.0f}s")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, Any] = {}
    if args.output.is_file():
        existing = json.loads(args.output.read_text(encoding="utf-8")).get("caches", {})
    report = {
        "schema_version": 1,
        "version": SUPPLEMENT_VERSION,
        "representations": list(SUPPLEMENT_REPRESENTATIONS),
        "out_dir": args.out_dir.as_posix(),
        "provenance": provenance,
        "caches": {**existing, **caches},
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"report: {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
