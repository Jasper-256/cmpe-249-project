"""Zero-copy access to regular and split FRESH ZIP archives."""

from __future__ import annotations

import bisect
import io
import os
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO, Final, Literal

ArchiveKind = Literal["synthetic", "real", "models"]
ARCHIVE_KINDS: Final[tuple[ArchiveKind, ...]] = ("synthetic", "real", "models")
DEFAULT_MAX_MEMBER_BYTES: Final[int] = 512 * 1024**2


class MultiPartFile(io.RawIOBase):
    """Expose ordered file parts as one read-only seekable binary stream."""

    def __init__(self, parts: list[Path]) -> None:
        super().__init__()
        if not parts:
            raise ValueError("At least one file part is required")
        self._paths = tuple(parts)
        self._streams: list[BinaryIO] = [path.open("rb") for path in self._paths]
        self._sizes = tuple(path.stat().st_size for path in self._paths)
        offsets = [0]
        for size in self._sizes:
            offsets.append(offsets[-1] + size)
        self._offsets = tuple(offsets)
        self._size = self._offsets[-1]
        self._position = 0

    @property
    def size(self) -> int:
        """Return the total virtual stream length."""
        return self._size

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        self._check_closed()
        return self._position

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        self._check_closed()
        if whence == os.SEEK_SET:
            new_position = offset
        elif whence == os.SEEK_CUR:
            new_position = self._position + offset
        elif whence == os.SEEK_END:
            new_position = self._size + offset
        else:
            raise ValueError(f"Unsupported whence: {whence}")
        if new_position < 0:
            raise ValueError("Negative seek position")
        self._position = new_position
        return self._position

    def read(self, size: int = -1) -> bytes:
        self._check_closed()
        if self._position >= self._size or size == 0:
            return b""
        remaining = (
            self._size - self._position if size < 0 else min(size, self._size - self._position)
        )
        output = bytearray()
        while remaining:
            part_index = bisect.bisect_right(self._offsets, self._position) - 1
            offset_in_part = self._position - self._offsets[part_index]
            available = self._sizes[part_index] - offset_in_part
            requested = min(remaining, available)
            stream = self._streams[part_index]
            stream.seek(offset_in_part)
            chunk = stream.read(requested)
            if len(chunk) != requested:
                message = (
                    f"Short read from {self._paths[part_index]}: "
                    f"expected {requested}, got {len(chunk)}"
                )
                raise OSError(message)
            output.extend(chunk)
            self._position += len(chunk)
            remaining -= len(chunk)
        return bytes(output)

    def close(self) -> None:
        if not self.closed:
            for stream in self._streams:
                stream.close()
        super().close()

    def _check_closed(self) -> None:
        if self.closed:
            raise ValueError("I/O operation on closed multipart stream")


@contextmanager
def open_fresh_archive(kind: ArchiveKind, raw_root: Path) -> Iterator[zipfile.ZipFile]:
    """Open a FRESH archive without assembling or extracting it on disk."""
    if kind == "real":
        parts = [raw_root / f"real.zip.{index:03d}" for index in range(1, 4)]
        missing = [str(path) for path in parts if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing real archive part(s): {', '.join(missing)}")
        with MultiPartFile(parts) as stream, zipfile.ZipFile(stream) as archive:
            yield archive
        return

    archive_path = raw_root / f"{kind}.zip"
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        yield archive


def read_fresh_member(
    kind: ArchiveKind,
    member: str,
    raw_root: Path = Path("data/fresh/raw"),
    *,
    max_bytes: int | None = DEFAULT_MAX_MEMBER_BYTES,
) -> bytes:
    """Read one member into memory without creating an extracted copy."""
    with open_fresh_archive(kind, raw_root) as archive:
        info = archive.getinfo(member)
        if max_bytes is not None and info.file_size > max_bytes:
            raise ValueError(
                f"Refusing to load {info.file_size:,} bytes into memory; limit is {max_bytes:,}"
            )
        return archive.read(info)
