"""Tests for zero-copy regular and multipart ZIP access."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

from saga_pose.data.archives import MultiPartFile, open_fresh_archive, read_fresh_member


def _zip_bytes() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("real/example/first.txt", "alpha")
        archive.writestr("real/example/second.bin", bytes(range(256)) * 8)
    return buffer.getvalue()


def _write_parts(root: Path, payload: bytes) -> list[Path]:
    boundaries = (31, 197)
    chunks = (
        payload[: boundaries[0]],
        payload[boundaries[0] : boundaries[1]],
        payload[boundaries[1] :],
    )
    paths = []
    for index, chunk in enumerate(chunks, start=1):
        path = root / f"real.zip.{index:03d}"
        path.write_bytes(chunk)
        paths.append(path)
    return paths


def test_multipart_stream_supports_cross_boundary_reads(tmp_path: Path) -> None:
    payload = _zip_bytes()
    parts = _write_parts(tmp_path, payload)

    with MultiPartFile(parts) as stream:
        assert stream.size == len(payload)
        stream.seek(25)
        assert stream.read(220) == payload[25:245]
        stream.seek(-40, io.SEEK_END)
        assert stream.read() == payload[-40:]


def test_zipfile_reads_virtual_multipart_archive(tmp_path: Path) -> None:
    parts = _write_parts(tmp_path, _zip_bytes())

    with MultiPartFile(parts) as stream, zipfile.ZipFile(stream) as archive:
        assert archive.read("real/example/first.txt") == b"alpha"
        assert archive.read("real/example/second.bin") == bytes(range(256)) * 8


def test_fresh_archive_helpers_do_not_create_combined_archive(tmp_path: Path) -> None:
    _write_parts(tmp_path, _zip_bytes())

    with open_fresh_archive("real", tmp_path) as archive:
        assert archive.namelist()[0] == "real/example/first.txt"
    assert not (tmp_path / "real.zip").exists()
    assert read_fresh_member("real", "real/example/first.txt", tmp_path) == b"alpha"
