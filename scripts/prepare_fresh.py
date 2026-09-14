"""Validate FRESH archives and build a tiny index without extracting data."""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

from saga_pose.data.archives import ARCHIVE_KINDS, open_fresh_archive

DEFAULT_SOFT_STORAGE_TARGET_BYTES = 20_000_000_000


def _validate_member(info: zipfile.ZipInfo) -> PurePosixPath:
    name = info.filename
    if "\\" in name:
        raise RuntimeError(f"Archive member uses a backslash path: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise RuntimeError(f"Unsafe archive path: {name!r}")
    if path.parts[0].endswith(":"):
        raise RuntimeError(f"Drive-qualified archive path: {name!r}")
    if stat.S_ISLNK(info.external_attr >> 16):
        raise RuntimeError(f"Symbolic link is not allowed: {name!r}")
    if info.flag_bits & 0x1:
        raise RuntimeError(f"Encrypted member is not supported: {name!r}")
    return path


def _archive_summary(kind: str, raw_root: Path) -> dict[str, Any]:
    extension_counts: Counter[str] = Counter()
    top_level_counts: Counter[str] = Counter()
    member_count = 0
    file_count = 0
    compressed_bytes = 0
    uncompressed_bytes = 0

    with open_fresh_archive(kind, raw_root) as archive:
        for info in archive.infolist():
            path = _validate_member(info)
            member_count += 1
            if info.is_dir():
                continue
            file_count += 1
            compressed_bytes += info.compress_size
            uncompressed_bytes += info.file_size
            extension_counts[path.suffix.lower() or "<none>"] += 1
            top_level_counts[path.parts[0]] += 1

    return {
        "kind": kind,
        "member_count": member_count,
        "file_count": file_count,
        "compressed_member_bytes": compressed_bytes,
        "uncompressed_member_bytes": uncompressed_bytes,
        "top_level_file_counts": dict(sorted(top_level_counts.items())),
        "extension_counts": dict(sorted(extension_counts.items())),
    }


def main() -> None:
    """Inspect central directories through lazy archive readers and save a summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("data/fresh/raw"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/fresh/index/archive-inventory.json"),
    )
    parser.add_argument(
        "--soft-storage-target-bytes",
        type=int,
        default=DEFAULT_SOFT_STORAGE_TARGET_BYTES,
        help="Preferred data/fresh working-set size; warnings only (default: 20 GB decimal).",
    )
    args = parser.parse_args()

    raw_files = [path for path in args.raw_root.iterdir() if path.is_file()]
    raw_bytes = sum(path.stat().st_size for path in raw_files)
    if raw_bytes > args.soft_storage_target_bytes:
        print(
            f"Warning: raw artifacts exceed the soft storage target: "
            f"{raw_bytes:,} > {args.soft_storage_target_bytes:,}",
            file=sys.stderr,
        )

    summaries = []
    for kind in ARCHIVE_KINDS:
        print(f"Inspecting {kind} archive without extraction...", flush=True)
        summaries.append(_archive_summary(kind, args.raw_root))

    report = {
        "schema_version": 1,
        "generated_at_unix": int(time.time()),
        "access_mode": "zero-copy-lazy-zip",
        "raw_root": str(args.raw_root.resolve()),
        "raw_file_count": len(raw_files),
        "raw_size_bytes": raw_bytes,
        "soft_storage_target_bytes": args.soft_storage_target_bytes,
        "bytes_over_soft_target_before_index": max(0, raw_bytes - args.soft_storage_target_bytes),
        "bytes_below_soft_target_before_index": max(0, args.soft_storage_target_bytes - raw_bytes),
        "archives": summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_name(f"{args.output.name}.part")
    with partial.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(partial, args.output)

    total_data_bytes = sum(
        path.stat().st_size for path in args.raw_root.parent.rglob("*") if path.is_file()
    )
    if total_data_bytes > args.soft_storage_target_bytes:
        print(
            f"Warning: total FRESH storage exceeds the soft target: "
            f"{total_data_bytes:,} > {args.soft_storage_target_bytes:,}",
            file=sys.stderr,
        )
    print(
        f"Wrote {args.output} ({args.output.stat().st_size:,} bytes); "
        f"total FRESH storage is {total_data_bytes:,} bytes.",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError, zipfile.BadZipFile) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
