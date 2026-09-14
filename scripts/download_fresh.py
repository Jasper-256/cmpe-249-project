"""Resumably download and verify the canonical FRESH Zenodo artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import TypedDict

import requests


class FileRecord(TypedDict):
    """A downloadable artifact from the committed FRESH manifest."""

    name: str
    size_bytes: int
    md5: str
    url: str


CHUNK_BYTES = 8 * 1024 * 1024
REPORT_INTERVAL_SECONDS = 5.0
MAX_ATTEMPTS = 8


def _human_bytes(size: float) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024
    raise AssertionError("unreachable")


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _verify(path: Path, record: FileRecord) -> bool:
    if not path.is_file() or path.stat().st_size != record["size_bytes"]:
        return False
    actual = _md5(path)
    if actual != record["md5"]:
        print(f"Checksum mismatch for {path.name}: expected {record['md5']}, got {actual}")
        return False
    return True


def _stream_response(response: requests.Response, part_path: Path, mode: str, total: int) -> None:
    downloaded = part_path.stat().st_size if part_path.exists() and mode == "ab" else 0
    started_at = time.monotonic()
    last_report = started_at
    initial_bytes = downloaded

    with part_path.open(mode) as output:
        for chunk in response.iter_content(chunk_size=CHUNK_BYTES):
            if not chunk:
                continue
            output.write(chunk)
            downloaded += len(chunk)
            now = time.monotonic()
            if now - last_report >= REPORT_INTERVAL_SECONDS:
                elapsed = max(now - started_at, 1e-9)
                rate = (downloaded - initial_bytes) / elapsed
                percent = 100 * downloaded / total
                print(
                    f"  {percent:6.2f}%  {_human_bytes(downloaded)} / "
                    f"{_human_bytes(total)}  {_human_bytes(rate)}/s",
                    flush=True,
                )
                last_report = now


def _download(record: FileRecord, destination: Path) -> None:
    final_path = destination / record["name"]
    part_path = destination / f"{record['name']}.part"
    expected_size = record["size_bytes"]

    if _verify(final_path, record):
        print(f"Verified existing {record['name']}", flush=True)
        return
    if final_path.exists():
        raise RuntimeError(f"Refusing to overwrite invalid completed file: {final_path}")
    if part_path.exists() and part_path.stat().st_size > expected_size:
        raise RuntimeError(f"Partial file is larger than its manifest size: {part_path}")

    for attempt in range(1, MAX_ATTEMPTS + 1):
        offset = part_path.stat().st_size if part_path.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        print(
            f"Downloading {record['name']} (attempt {attempt}/{MAX_ATTEMPTS}, "
            f"resume at {_human_bytes(offset)})",
            flush=True,
        )
        try:
            with requests.get(
                record["url"], headers=headers, stream=True, timeout=(30, 300)
            ) as response:
                if response.status_code == 416 and offset == expected_size:
                    break
                response.raise_for_status()
                resumed = offset > 0 and response.status_code == 206
                mode = "ab" if resumed else "wb"
                if offset and not resumed:
                    print("  Server did not honor Range; restarting this file.", flush=True)
                _stream_response(response, part_path, mode, expected_size)
            if part_path.stat().st_size == expected_size:
                break
            print(
                f"  Incomplete response: {_human_bytes(part_path.stat().st_size)} of "
                f"{_human_bytes(expected_size)}",
                flush=True,
            )
        except (OSError, requests.RequestException) as error:
            if attempt == MAX_ATTEMPTS:
                raise RuntimeError(f"Download failed for {record['name']}") from error
            delay = min(2**attempt, 30)
            print(f"  Transient failure: {error}; retrying in {delay}s", flush=True)
            time.sleep(delay)
    else:
        raise RuntimeError(f"Download attempts exhausted for {record['name']}")

    print(f"Verifying MD5 for {record['name']}...", flush=True)
    if not _verify(part_path, record):
        raise RuntimeError(f"Verification failed for partial file: {part_path}")
    os.replace(part_path, final_path)
    print(f"Verified {record['name']}", flush=True)


def _selected_files(records: Iterable[FileRecord], names: set[str]) -> list[FileRecord]:
    selected = [record for record in records if not names or record["name"] in names]
    known_names = {record["name"] for record in records}
    if unknown := names - known_names:
        raise ValueError(f"Unknown manifest file(s): {', '.join(sorted(unknown))}")
    return selected


def main() -> None:
    """Download selected artifacts and enforce size and MD5 verification."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("docs/fresh-dataset-manifest.json"),
    )
    parser.add_argument("--destination", type=Path, default=Path("data/fresh/raw"))
    parser.add_argument("--only", action="append", default=[], metavar="FILENAME")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    with args.manifest.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    records = _selected_files(manifest["files"], set(args.only))
    args.destination.mkdir(parents=True, exist_ok=True)

    missing_bytes = sum(
        record["size_bytes"]
        for record in records
        if not (args.destination / record["name"]).exists()
    )
    free_bytes = shutil.disk_usage(args.destination).free
    if not args.verify_only and free_bytes < missing_bytes + 2 * 1024**3:
        raise RuntimeError(
            f"Insufficient space: need at least {_human_bytes(missing_bytes + 2 * 1024**3)}, "
            f"have {_human_bytes(free_bytes)}"
        )

    if args.verify_only:
        failures = [
            record["name"]
            for record in records
            if not _verify(args.destination / record["name"], record)
        ]
        if failures:
            raise SystemExit(f"Missing or invalid: {', '.join(failures)}")
        print(f"Verified {len(records)} FRESH artifacts.")
        return

    for record in records:
        _download(record, args.destination)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
