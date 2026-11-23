"""Versioned event-representation supplement cache (event-supp-v1).

The events-v1 study arms need event maps that crop-v2 does not store. Rather
than rebuilding crop-v2, a supplement cache holds only the new maps, one
directory per crop-v2 split, in exactly crop-v2's sample order. Maps are built
from the raw archives through the same dataset classes and crop path as
crop-v2; every build also recomputes the crop-v2 ``activity`` map and requires
it to match the cached one byte for byte, which proves the crops align.

A supplement is attached to a :class:`~saga_pose.data.cache.FreshCropCacheDataset`
only after a strict order check (sample IDs and frame numbers).
"""

from __future__ import annotations

import gc
import hashlib
import json
import shutil
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import torch
from numpy.lib.format import open_memmap
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from saga_pose.data.dataset import FreshArchiveDataset, FreshRealArchiveDataset
from saga_pose.data.representations import (
    SUPPLEMENT_REPRESENTATIONS,
    SUPPLEMENT_VERSION,
    native_event_representations,
    resize_unit_map,
    supplement_representation_maps,
)

SUPPLEMENT_SCHEMA_VERSION: Final[int] = 1
SUPPLEMENT_ARRAYS: Final[tuple[str, ...]] = tuple(
    f"event_{name}_uint8" for name in SUPPLEMENT_REPRESENTATIONS
)
CODE_FILES: Final[tuple[str, ...]] = (
    "representations.py",
    "supplement.py",
    "dataset.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def supplement_code_sha256() -> str:
    """Hash of the source files that define the supplement maps."""
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parent
    for name in CODE_FILES:
        digest.update(name.encode() + (root / name).read_bytes())
    return digest.hexdigest()


def check_supplement_order(
    crop_sample_ids: Sequence[str],
    crop_frame_numbers: NDArray[Any],
    supplement_sample_ids: Sequence[str],
    supplement_frame_numbers: NDArray[Any],
) -> None:
    """Raise ``ValueError`` unless the supplement rows are crop-v2's rows, in order."""
    if len(crop_sample_ids) != len(supplement_sample_ids):
        raise ValueError(
            f"supplement has {len(supplement_sample_ids)} samples, crop cache has "
            f"{len(crop_sample_ids)}"
        )
    if list(crop_sample_ids) != list(supplement_sample_ids):
        raise ValueError("supplement sample IDs differ from the crop cache order")
    if not np.array_equal(np.asarray(crop_frame_numbers), np.asarray(supplement_frame_numbers)):
        raise ValueError("supplement frame numbers differ from the crop cache order")


ArchiveDataset = FreshArchiveDataset | FreshRealArchiveDataset


class EventSupplementSource(Dataset[dict[str, Any]]):
    """Event-only items for the supplement builder (no RGB decoding, no TSS)."""

    def __init__(self, dataset: ArchiveDataset) -> None:
        super().__init__()
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample_id, frame, crop, event_crop = self.dataset.event_crop(index)
        neutral = self.dataset.event_neutral
        size = self.dataset.target_size_wh
        maps = supplement_representation_maps(event_crop, neutral, size)
        # Recomputed crop-v2 activity, compared byte for byte with the cache.
        check = resize_unit_map(native_event_representations(event_crop, neutral)["activity"], size)
        item: dict[str, Any] = {
            "sample_id": sample_id,
            "frame_number": frame,
            "crop_xyxy": torch.tensor(crop, dtype=torch.int64),
            "activity_check": torch.from_numpy(check[None].copy()),
        }
        for name in SUPPLEMENT_REPRESENTATIONS:
            item[f"event_{name}_uint8"] = torch.from_numpy(maps[name][None].copy())
        return item


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_event_supplement(
    source: EventSupplementSource,
    crop_root: Path,
    out_root: Path,
    *,
    provenance: dict[str, Any],
    num_workers: int = 8,
    batch_size: int = 32,
    progress: Callable[[int, int], None] | None = None,
    statistics: bool = True,
) -> dict[str, Any]:
    """Build one supplement split aligned with the crop-v2 split at ``crop_root``.

    Fails before writing anything unless the source order equals crop-v2's, and
    fails during the build on any frame-number, crop-box, or activity mismatch.
    An existing complete supplement is reused only if it matches exactly,
    including its version and supplement code hash; a changed definition must be
    built under a new version directory.
    ``statistics=False`` (real splits) records no content statistics.
    """
    crop_manifest_path = crop_root / "manifest.json"
    crop_manifest = cast(dict[str, Any], json.loads(crop_manifest_path.read_text("utf-8")))
    crop_manifest_sha = sha256_file(crop_manifest_path)
    sample_ids = cast(list[str], crop_manifest["sample_ids"])
    manifest_path = out_root / "manifest.json"
    if manifest_path.is_file():
        existing = cast(dict[str, Any], json.loads(manifest_path.read_text("utf-8")))
        if (
            existing.get("status") != "complete"
            or existing.get("schema_version") != SUPPLEMENT_SCHEMA_VERSION
            or existing.get("version") != SUPPLEMENT_VERSION
            or existing.get("code_sha256") != supplement_code_sha256()
            or existing.get("source_crop_manifest_sha256") != crop_manifest_sha
            or existing.get("sample_ids") != sample_ids
        ):
            raise ValueError(f"{out_root} exists with different or incomplete contents")
        return existing
    if out_root.exists():
        raise RuntimeError(f"Refusing to overwrite unmanifested supplement: {out_root}")
    source_ids = source.dataset.sample_ids
    if source_ids != sample_ids:
        raise ValueError(f"{crop_root.name}: source sample order differs from crop-v2")

    def crop_array(name: str) -> NDArray[Any]:
        spec = crop_manifest["arrays"][name]
        return cast(NDArray[Any], np.load(crop_root / spec["filename"], mmap_mode="r"))

    crop_frames = crop_array("frame_number")
    crop_boxes = crop_array("crop_xyxy")
    crop_activity = crop_array("event_activity_uint8")
    count = len(sample_ids)
    final_root = out_root
    work_root = final_root.with_name(final_root.name + ".partial")
    if work_root.exists():
        shutil.rmtree(work_root)
    work_root.mkdir(parents=True)
    started = time.perf_counter()
    shape = (count, *crop_activity.shape[1:])
    maps: dict[str, Any] = {
        name: open_memmap(work_root / f"{name}.npy", mode="w+", dtype=np.uint8, shape=shape)
        for name in SUPPLEMENT_ARRAYS
    }
    frames_out = open_memmap(
        work_root / "frame_number.npy", mode="w+", dtype=crop_frames.dtype, shape=(count,)
    )
    loader: DataLoader[dict[str, Any]] = DataLoader(
        source, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    written: list[str] = []
    differs_from_activity = 0
    active_fraction = {name: 0.0 for name in SUPPLEMENT_ARRAYS}
    offset = 0
    for batch in loader:
        size = len(batch["sample_id"])
        rows = slice(offset, offset + size)
        frames = cast(Tensor, batch["frame_number"]).numpy()
        if not np.array_equal(frames, crop_frames[rows]):
            raise RuntimeError(f"{crop_root.name}: frame numbers differ at rows {rows}")
        if not np.array_equal(cast(Tensor, batch["crop_xyxy"]).numpy(), crop_boxes[rows]):
            raise RuntimeError(f"{crop_root.name}: crop boxes differ at rows {rows}")
        check = cast(Tensor, batch["activity_check"]).numpy()
        if not np.array_equal(check, crop_activity[rows]):
            raise RuntimeError(f"{crop_root.name}: recomputed activity differs at rows {rows}")
        for name in SUPPLEMENT_ARRAYS:
            values = cast(Tensor, batch[name]).numpy()
            maps[name][rows] = values
            if statistics:
                active_fraction[name] += float((values > 0).mean()) * size
        if statistics:
            activity64 = cast(Tensor, batch[SUPPLEMENT_ARRAYS[0]]).numpy()
            differs_from_activity += int(np.any(activity64 != check, axis=(1, 2, 3)).sum())
        frames_out[rows] = frames
        written.extend(batch["sample_id"])
        offset += size
        if progress is not None:
            progress(offset, count)
    if written != sample_ids:
        raise RuntimeError("supplement build order does not match crop-v2")
    for array in (*maps.values(), frames_out):
        array.flush()
    # Windows cannot rename a directory whose files are still mapped.
    del array, frames_out
    maps.clear()
    gc.collect()
    arrays: dict[str, dict[str, Any]] = {}
    for name in (*SUPPLEMENT_ARRAYS, "frame_number"):
        filename = f"{name}.npy"
        array_shape = list(shape) if name != "frame_number" else [count]
        arrays[name] = {
            "filename": filename,
            "shape": array_shape,
            "dtype": "uint8" if name != "frame_number" else str(crop_frames.dtype),
            "sha256": sha256_file(work_root / filename),
        }
    manifest = {
        "schema_version": SUPPLEMENT_SCHEMA_VERSION,
        "version": SUPPLEMENT_VERSION,
        "status": "complete",
        "split": crop_root.name,
        "representations": list(SUPPLEMENT_REPRESENTATIONS),
        "code_sha256": supplement_code_sha256(),
        "source_crop_manifest_sha256": crop_manifest_sha,
        "source_preprocessing_version": crop_manifest["preprocessing_version"],
        "sample_ids": sample_ids,
        "order_check": "sample IDs, frame numbers, crop boxes, and recomputed crop-v2 "
        "activity maps all equal crop-v2 for every row",
        "statistics": {
            "samples_where_activity64_differs_from_activity": differs_from_activity,
            "mean_nonzero_fraction": {
                name: total / count for name, total in active_fraction.items()
            },
        }
        if statistics
        else None,
        "build_seconds": time.perf_counter() - started,
        "provenance": provenance,
        "arrays": arrays,
    }
    _write_manifest(work_root / "manifest.json", manifest)
    work_root.rename(final_root)
    return manifest


def open_event_supplement(
    supplement_root: Path,
    crop_manifest_path: Path,
    crop_sample_ids: Sequence[str],
    crop_frame_numbers: NDArray[Any],
) -> dict[str, NDArray[Any]]:
    """Open a supplement split's arrays after strict provenance and order checks."""
    manifest_path = supplement_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing completed event-supplement manifest: {manifest_path}")
    manifest = cast(dict[str, Any], json.loads(manifest_path.read_text("utf-8")))
    if (
        manifest.get("schema_version") != SUPPLEMENT_SCHEMA_VERSION
        or manifest.get("version") != SUPPLEMENT_VERSION
        or manifest.get("status") != "complete"
    ):
        raise ValueError(f"{supplement_root}: unsupported or incomplete event supplement")
    if manifest.get("split") != crop_manifest_path.parent.name:
        raise ValueError(f"{supplement_root}: built for split {manifest.get('split')!r}")
    if manifest["source_crop_manifest_sha256"] != sha256_file(crop_manifest_path):
        raise ValueError(f"{supplement_root}: built from a different crop-cache manifest")
    arrays: dict[str, NDArray[Any]] = {}
    for name, spec in cast(dict[str, dict[str, Any]], manifest["arrays"]).items():
        array = np.load(supplement_root / spec["filename"], mmap_mode="r", allow_pickle=False)
        if list(array.shape) != spec["shape"] or str(array.dtype) != spec["dtype"]:
            raise ValueError(f"{supplement_root}: array contract mismatch for {name}")
        arrays[name] = array
    check_supplement_order(
        crop_sample_ids,
        crop_frame_numbers,
        cast(list[str], manifest["sample_ids"]),
        arrays.pop("frame_number"),
    )
    return arrays
