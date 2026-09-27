"""Training-ready, zero-extraction FRESH paired-image dataset."""

from __future__ import annotations

import io
import json
import zipfile
from collections.abc import Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from saga_pose.analysis.saturation import clipped_pixel_mask, patch_fraction_map
from saga_pose.data.archives import open_fresh_archive
from saga_pose.data.fresh import FreshAnnotation, frame_member, load_metadata
from saga_pose.geometry.projection import (
    crop_resize_intrinsics,
    crop_resize_points,
    padded_square_bbox_xyxy,
)

SyntheticRole = Literal["train", "validation"]


class FreshItem(TypedDict):
    """One aligned and geometrically transformed training item."""

    sample_id: str
    object_name: str
    role: SyntheticRole
    rgb: Tensor
    event: Tensor
    clipping_ratio: Tensor
    keypoints_xy: Tensor
    intrinsics: Tensor
    pose_object_to_camera_mm: Tensor
    landmarks_object_mm: Tensor
    source_bbox_xyxy: Tensor
    crop_xyxy: Tensor


@dataclass(frozen=True)
class _SampleRecord:
    sample_id: str
    object_name: str
    metadata_member: str
    annotation: FreshAnnotation
    intrinsics: np.ndarray
    landmarks: np.ndarray


def _load_rgb(archive: zipfile.ZipFile, member: str) -> np.ndarray:
    with Image.open(io.BytesIO(archive.read(member))) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


class FreshArchiveDataset(Dataset[FreshItem]):
    """Read paired FRESH frames lazily from ``synthetic.zip``.

    Archive handles are opened per process on first access and discarded on
    pickle, making the dataset compatible with Windows spawned workers.
    """

    def __init__(
        self,
        raw_root: Path,
        split_index: Path,
        *,
        role: SyntheticRole,
        objects: Sequence[str] = ("cassini", "satty", "soho"),
        target_size_wh: tuple[int, int] = (256, 256),
        clipping_threshold: int = 254,
        clipping_grid_size: int = 16,
        crop_padding_factor: float = 1.2,
    ) -> None:
        super().__init__()
        self.raw_root = raw_root
        self.split_index = split_index
        self.role = role
        self.objects = tuple(objects)
        self.target_size_wh = target_size_wh
        self.clipping_threshold = clipping_threshold
        self.clipping_grid_size = clipping_grid_size
        self.crop_padding_factor = crop_padding_factor
        self._archive_context: AbstractContextManager[zipfile.ZipFile] | None = None
        self._archive: zipfile.ZipFile | None = None
        self._records = self._build_records()

    def _build_records(self) -> tuple[_SampleRecord, ...]:
        if not self.split_index.is_file():
            raise FileNotFoundError(
                f"Missing {self.split_index}; run scripts/create_synthetic_split.py"
            )
        split = json.loads(self.split_index.read_text(encoding="utf-8"))
        requested_ids: list[str] = []
        for object_name in self.objects:
            if object_name not in split["objects"]:
                raise ValueError(f"Unknown object in split: {object_name}")
            requested_ids.extend(split["objects"][object_name][f"{self.role}_ids"])
        requested_set = set(requested_ids)

        available: dict[str, _SampleRecord] = {}
        with open_fresh_archive("synthetic", self.raw_root) as archive:
            for object_name in self.objects:
                for filename in ("train.json", "test.json"):
                    member = f"synthetic/{object_name}/{filename}"
                    metadata = load_metadata(archive, member)
                    for annotation in metadata.annotations:
                        sample_id = f"{object_name}/{annotation.filename_rgb}"
                        if sample_id in requested_set:
                            available[sample_id] = _SampleRecord(
                                sample_id=sample_id,
                                object_name=object_name,
                                metadata_member=member,
                                annotation=annotation,
                                intrinsics=metadata.intrinsics,
                                landmarks=metadata.landmarks_3d,
                            )
        missing = requested_set - set(available)
        if missing:
            preview = ", ".join(sorted(missing)[:5])
            raise ValueError(f"Split references {len(missing)} missing samples: {preview}")
        return tuple(available[sample_id] for sample_id in requested_ids)

    def __len__(self) -> int:
        return len(self._records)

    def _get_archive(self) -> zipfile.ZipFile:
        if self._archive is None:
            context = open_fresh_archive("synthetic", self.raw_root)
            self._archive_context = context
            self._archive = context.__enter__()
        return self._archive

    def close(self) -> None:
        """Close the process-local archive handle if it was opened."""
        if self._archive_context is not None:
            self._archive_context.__exit__(None, None, None)
        self._archive_context = None
        self._archive = None

    def __getstate__(self) -> dict[str, object]:
        """Drop unpicklable archive handles before spawning a worker."""
        state = dict(self.__dict__)
        state["_archive_context"] = None
        state["_archive"] = None
        return state

    def __del__(self) -> None:
        self.close()

    def __getitem__(self, index: int) -> FreshItem:
        record = self._records[index]
        archive = self._get_archive()
        annotation = record.annotation
        rgb = _load_rgb(
            archive,
            frame_member(record.metadata_member, annotation.filename_rgb),
        )
        event = _load_rgb(
            archive,
            frame_member(record.metadata_member, annotation.filename_event),
        )
        crop = padded_square_bbox_xyxy(
            annotation.bbox_xyxy,
            padding_factor=self.crop_padding_factor,
        )
        rgb_crop = np.asarray(Image.fromarray(rgb).crop(crop), dtype=np.uint8)
        event_crop = np.asarray(Image.fromarray(event).crop(crop), dtype=np.uint8)

        target_width, target_height = self.target_size_wh
        rgb_resized = np.asarray(
            Image.fromarray(rgb_crop).resize(
                (target_width, target_height),
                resample=Image.Resampling.BILINEAR,
            ),
            dtype=np.uint8,
        )
        event_resized = np.asarray(
            Image.fromarray(event_crop).resize(
                (target_width, target_height),
                resample=Image.Resampling.BILINEAR,
            ),
            dtype=np.uint8,
        )
        clipping_ratio = patch_fraction_map(
            clipped_pixel_mask(rgb_crop, self.clipping_threshold),
            self.clipping_grid_size,
        )
        keypoints = crop_resize_points(annotation.keypoints, crop, self.target_size_wh)
        intrinsics = crop_resize_intrinsics(record.intrinsics, crop, self.target_size_wh)

        return FreshItem(
            sample_id=record.sample_id,
            object_name=record.object_name,
            role=self.role,
            rgb=torch.from_numpy(np.moveaxis(rgb_resized, -1, 0).copy()).float().div_(255.0),
            event=torch.from_numpy(np.moveaxis(event_resized, -1, 0).copy()).float().div_(255.0),
            clipping_ratio=torch.from_numpy(clipping_ratio.astype(np.float32)),
            keypoints_xy=torch.from_numpy(keypoints.astype(np.float32)),
            intrinsics=torch.from_numpy(intrinsics.astype(np.float32)),
            pose_object_to_camera_mm=torch.from_numpy(annotation.pose.astype(np.float32)),
            landmarks_object_mm=torch.from_numpy(record.landmarks.astype(np.float32)),
            source_bbox_xyxy=torch.tensor(annotation.bbox_xyxy, dtype=torch.float32),
            crop_xyxy=torch.tensor(crop, dtype=torch.int64),
        )
