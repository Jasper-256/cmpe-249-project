"""Versioned memory-mapped crop cache (schema 2, crop-v2 preprocessing contract)."""

from __future__ import annotations

import gc
import hashlib
import json
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import torch
from numpy.lib.format import open_memmap
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from saga_pose.data.dataset import PREPROCESSING_VERSION, FreshItem, ItemRole
from saga_pose.data.supplement import SUPPLEMENT_ARRAYS, open_event_supplement

SCHEMA_VERSION: Final[int] = 2
IMAGE_FIELDS: Final[tuple[str, ...]] = ("rgb", "rgb_tss")
MAP_FIELDS: Final[tuple[str, ...]] = ("event_raw", "event_activity", "event_magnitude")
TENSOR_FIELDS: Final[tuple[str, ...]] = (
    "clipping_ratio",
    "clipping_ratio_tss",
    "tss_applied",
    "frame_number",
    "keypoints_xy",
    "intrinsics",
    "pose_object_to_camera_mm",
    "landmarks_object_mm",
    "source_bbox_xyxy",
    "crop_xyxy",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _to_uint8(values: Tensor) -> NDArray[np.uint8]:
    return cast(
        NDArray[np.uint8],
        torch.round(values * 255.0).clamp(0, 255).to(torch.uint8).numpy(),
    )


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_crop_cache(
    dataset: Dataset[FreshItem],
    sample_ids: list[str],
    cache_root: Path,
    *,
    role: ItemRole,
    store_tss: bool,
    provenance: dict[str, Any],
    num_workers: int = 8,
    batch_size: int = 32,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Materialize every item of ``dataset`` into an immutable schema-2 cache.

    An existing complete cache is reused only if its sample IDs match exactly; a
    partial directory is never overwritten.
    """
    manifest_path = cache_root / "manifest.json"
    if manifest_path.is_file():
        manifest = cast(dict[str, Any], json.loads(manifest_path.read_text(encoding="utf-8")))
        if manifest.get("sample_ids") != sample_ids or manifest.get("status") != "complete":
            raise ValueError(f"{cache_root} exists with different or incomplete contents")
        return manifest
    if cache_root.exists():
        raise RuntimeError(f"Refusing to overwrite unmanifested cache directory: {cache_root}")
    count = len(sample_ids)
    if count == 0:
        raise ValueError("cannot build an empty crop cache")

    # Build into a sibling ".partial" directory and rename atomically on
    # success, so an interrupted build never leaves a directory that blocks
    # reruns. A stale partial directory is always this builder's own output.
    final_root = cache_root
    cache_root = final_root.with_name(final_root.name + ".partial")
    if cache_root.exists():
        shutil.rmtree(cache_root)
    started = time.perf_counter()
    cache_root.mkdir(parents=True)
    loader: DataLoader[FreshItem] = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=False,
    )
    maps: dict[str, Any] = {}
    specs: dict[str, dict[str, Any]] = {}
    sequences: list[str] = []
    object_names: list[str] = []
    written_ids: list[str] = []
    image_fields = IMAGE_FIELDS if store_tss else IMAGE_FIELDS[:1]
    offset = 0
    for batch in loader:
        batch_dict = cast(dict[str, Any], batch)
        arrays: dict[str, NDArray[Any]] = {}
        for name in (*image_fields, *MAP_FIELDS):
            arrays[f"{name}_uint8"] = _to_uint8(batch_dict[name])
        for name in TENSOR_FIELDS:
            arrays[name] = cast(Tensor, batch_dict[name]).numpy()
        if not maps:
            for name, array in arrays.items():
                filename = f"{name}.npy"
                shape = (count, *array.shape[1:])
                maps[name] = open_memmap(
                    cache_root / filename, mode="w+", dtype=array.dtype, shape=shape
                )
                specs[name] = {
                    "filename": filename,
                    "shape": list(shape),
                    "dtype": str(array.dtype),
                }
        size = len(batch_dict["sample_id"])
        for name, array in arrays.items():
            maps[name][offset : offset + size] = array
        written_ids.extend(batch_dict["sample_id"])
        sequences.extend(batch_dict["sequence"])
        object_names.extend(batch_dict["object_name"])
        offset += size
        if progress is not None:
            progress(offset, count)
    if written_ids != sample_ids:
        raise RuntimeError("cache build order does not match the requested sample IDs")
    for memmap in maps.values():
        memmap.flush()
    # Drop every memmap reference: Windows cannot rename a directory whose
    # files are still mapped.
    del memmap
    maps.clear()
    gc.collect()
    for spec in specs.values():
        spec["sha256"] = _sha256(cache_root / cast(str, spec["filename"]))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "preprocessing_version": PREPROCESSING_VERSION,
        "role": role,
        "sample_ids": sample_ids,
        "sequences": sequences,
        "object_names": object_names,
        "build_seconds": time.perf_counter() - started,
        "provenance": provenance,
        "arrays": specs,
    }
    _write_manifest(cache_root / "manifest.json", manifest)
    cache_root.rename(final_root)
    return manifest


class FreshCropCacheDataset(Dataset[FreshItem]):
    """Read one immutable schema-2 crop cache without touching raw archives.

    ``supplement_root`` attaches an event-supp-v1 split (same split name) after a
    strict order check; its maps then appear as extra ``event_*`` item keys.
    """

    def __init__(self, cache_root: Path, supplement_root: Path | None = None) -> None:
        super().__init__()
        manifest_path = cache_root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing completed crop-cache manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                "Unsupported crop-cache schema; rebuild with scripts/build_crop_cache.py"
            )
        if manifest.get("status") != "complete":
            raise ValueError("Crop-cache manifest is not complete")
        self.cache_root = cache_root
        self.manifest = cast(dict[str, Any], manifest)
        self.sample_ids = cast(list[str], manifest["sample_ids"])
        self.sequences = cast(list[str], manifest["sequences"])
        self.object_names = cast(list[str], manifest["object_names"])
        self.role = cast(ItemRole, manifest["role"])
        if not len(self.sample_ids) == len(self.sequences) == len(self.object_names):
            raise ValueError("Crop-cache sample metadata lengths differ")
        self.supplement_root = supplement_root
        self._arrays: dict[str, NDArray[Any]] = {}
        self._open_arrays()

    def _open_arrays(self) -> None:
        manifest = self.manifest
        cache_root = self.cache_root
        self._arrays = {}
        for name, spec in cast(dict[str, dict[str, Any]], manifest["arrays"]).items():
            path = cache_root / cast(str, spec["filename"])
            if not path.is_file():
                raise FileNotFoundError(f"Missing crop-cache array: {path}")
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            expected_shape = tuple(cast(list[int], spec["shape"]))
            if array.shape != expected_shape or str(array.dtype) != spec["dtype"]:
                raise ValueError(f"Crop-cache array contract mismatch for {name}")
            if array.shape[0] != len(self.sample_ids):
                raise ValueError(f"Crop-cache sample count mismatch for {name}")
            self._arrays[name] = array
        if self.supplement_root is not None:
            self._arrays.update(
                open_event_supplement(
                    self.supplement_root,
                    cache_root / "manifest.json",
                    self.sample_ids,
                    self._arrays["frame_number"],
                )
            )

    def __getstate__(self) -> dict[str, Any]:
        # Pickling a memmap copies the whole array; Windows-spawned DataLoader
        # workers instead reopen the memory maps from the manifest.
        state = dict(self.__dict__)
        state["_arrays"] = {}
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._open_arrays()

    def __len__(self) -> int:
        return len(self.sample_ids)

    def array(self, name: str) -> NDArray[Any]:
        """Return a read-only memory map (for vectorized analysis without items)."""
        return self._arrays[name]

    def _tensor(self, name: str, index: int) -> Tensor:
        # A writable copy avoids undefined behavior when PyTorch receives a
        # read-only memory-map view and allows DataLoader pinning.
        return torch.from_numpy(np.array(self._arrays[name][index], copy=True))

    def _unit(self, name: str, index: int) -> Tensor:
        return self._tensor(f"{name}_uint8", index).float().div_(255.0)

    def __getitem__(self, index: int) -> FreshItem:
        rgb = self._unit("rgb", index)
        rgb_tss = self._unit("rgb_tss", index) if "rgb_tss_uint8" in self._arrays else rgb
        event_raw = self._unit("event_raw", index)
        item = FreshItem(
            sample_id=self.sample_ids[index],
            object_name=self.object_names[index],
            sequence=self.sequences[index],
            frame_number=int(self._arrays["frame_number"][index]),
            role=self.role,
            rgb=rgb,
            rgb_tss=rgb_tss,
            event=event_raw.expand(3, -1, -1).clone(),
            event_raw=event_raw,
            event_activity=self._unit("event_activity", index),
            event_magnitude=self._unit("event_magnitude", index),
            clipping_ratio=self._tensor("clipping_ratio", index),
            clipping_ratio_tss=self._tensor("clipping_ratio_tss", index),
            tss_applied=self._tensor("tss_applied", index),
            keypoints_xy=self._tensor("keypoints_xy", index),
            intrinsics=self._tensor("intrinsics", index),
            pose_object_to_camera_mm=self._tensor("pose_object_to_camera_mm", index),
            landmarks_object_mm=self._tensor("landmarks_object_mm", index),
            source_bbox_xyxy=self._tensor("source_bbox_xyxy", index),
            crop_xyxy=self._tensor("crop_xyxy", index),
        )
        for name in SUPPLEMENT_ARRAYS:
            if name in self._arrays:
                key = name.removesuffix("_uint8")
                item[key] = self._unit(key, index)  # type: ignore[literal-required]
        return item
