"""Frozen real-test contract: annotation corrections and author-defined slices.

The real sequences are evaluation-only. This module never inspects images; it
only resolves which ground-truth file is authoritative and which frames the
FRESH authors labelled as harsh-lighting or low-motion.

The Zenodo ``real/satty-4-far/test.json`` was superseded by the authors'
``addendum-data/corrected-annotations`` file in their repository. The two differ
by a median of about 59 degrees in rotation, so evaluating against the archive
copy would score that sequence against wrong labels. The repository has no
license file, so the correction is read from an ignored, commit-pinned checkout
and verified by SHA-256 instead of being copied into this project.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final, Literal, cast

import numpy as np
from numpy.typing import NDArray

from saga_pose.data.fresh import FreshMetadata, load_metadata, metadata_members

SliceName = Literal["harsh", "low-motion"]
SLICE_NAMES: Final[tuple[SliceName, ...]] = ("harsh", "low-motion")
SLICES_MEMBER: Final[str] = "real/harsh-and-slow-frames.json"

FRESH_REPOSITORY: Final[str] = "https://github.com/mohsij/space-event-rgb-fusion"
FRESH_REPOSITORY_COMMIT: Final[str] = "fb566826e950b5db1f5b7ba013e15c12c5a7d4af"
DEFAULT_ADDENDUM_ROOT: Final[Path] = Path("external/space-event-rgb-fusion/addendum-data")


@dataclass(frozen=True)
class AnnotationOverride:
    """An author-published replacement for one archive annotation file."""

    sequence: str
    relative_path: str
    sha256: str


ANNOTATION_OVERRIDES: Final[tuple[AnnotationOverride, ...]] = (
    AnnotationOverride(
        sequence="satty-4-far",
        relative_path="corrected-annotations/satty-4-far/test.json",
        sha256="59129ccf7c6c85a484ddcb008a253f3e129f6acc087cdb5f1d0f4a4448f6b634",
    ),
)


def sequence_name(member: str) -> str:
    """Return the real sequence directory name, e.g. ``satty-4-far``."""
    return PurePosixPath(member).parent.name


def frame_number(filename: str) -> int:
    """Parse the numeric frame prefix of ``00042_rgb.png``-style filenames."""
    return int(PurePosixPath(filename).name.split("_", maxsplit=1)[0])


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class _OverrideArchive:
    """Minimal ``ZipFile.read`` adapter so overrides reuse ``load_metadata``."""

    def __init__(self, member: str, payload: bytes) -> None:
        self._member = member
        self._payload = payload

    def read(self, member: str) -> bytes:
        if member != self._member:
            raise KeyError(member)
        return self._payload


def load_real_metadata(
    archive: zipfile.ZipFile,
    *,
    addendum_root: Path = DEFAULT_ADDENDUM_ROOT,
) -> tuple[dict[str, FreshMetadata], dict[str, str]]:
    """Load all 24 real sequences, applying verified author corrections.

    Returns ``(metadata_by_sequence, annotation_source_by_sequence)``. A missing or
    hash-mismatched override is an error: silently falling back to the archive
    copy would evaluate against known-wrong labels.
    """
    overrides = {override.sequence: override for override in ANNOTATION_OVERRIDES}
    metadata: dict[str, FreshMetadata] = {}
    sources: dict[str, str] = {}
    for member in metadata_members(archive, real=True):
        name = sequence_name(member)
        override = overrides.get(name)
        if override is None:
            metadata[name] = load_metadata(archive, member)
            sources[name] = f"archive:{member}"
            continue
        path = addendum_root / override.relative_path
        if not path.is_file():
            raise FileNotFoundError(
                f"{name} requires the corrected FRESH annotation at {path}; clone "
                f"{FRESH_REPOSITORY} at {FRESH_REPOSITORY_COMMIT} into external/"
            )
        payload = path.read_bytes()
        digest = _sha256_bytes(payload)
        if digest != override.sha256:
            raise ValueError(f"{path} SHA-256 {digest} does not match the pinned correction")
        metadata[name] = load_metadata(
            cast(zipfile.ZipFile, _OverrideArchive(member, payload)), member
        )
        sources[name] = f"override:{override.relative_path}@{FRESH_REPOSITORY_COMMIT[:12]}"
    return metadata, sources


def load_real_slices(archive: zipfile.ZipFile) -> dict[str, dict[SliceName, tuple[int, int]]]:
    """Return author-defined inclusive frame-number ranges per sequence and slice."""
    raw = cast(dict[str, dict[str, list[int]]], json.loads(archive.read(SLICES_MEMBER)))
    slices: dict[str, dict[SliceName, tuple[int, int]]] = {}
    for sequence, ranges in raw.items():
        parsed: dict[SliceName, tuple[int, int]] = {}
        for name in SLICE_NAMES:
            start, end = (int(value) for value in ranges[name])
            if end < start:
                raise ValueError(f"{sequence}/{name}: empty range {start}-{end}")
            parsed[name] = (start, end)
        slices[sequence] = parsed
    return slices


def slice_mask(
    frame_numbers: NDArray[np.int64] | list[int],
    frame_range: tuple[int, int],
) -> NDArray[np.bool_]:
    """Select frames inside an inclusive ``[start, end]`` frame-number range.

    The FRESH release does not document range inclusivity; the inclusive reading
    is recorded here once and used for every method so comparisons stay paired.
    """
    numbers = np.asarray(frame_numbers, dtype=np.int64)
    start, end = frame_range
    return (numbers >= start) & (numbers <= end)
