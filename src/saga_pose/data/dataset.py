"""Training-ready, zero-extraction FRESH paired-image datasets (crop-v2 contract).

One item builder serves synthetic and real data so both domains pass through the
identical crop, resize, clipping, event-representation, and geometry path. Only
the format-defined event neutral (127 synthetic, 128 real) differs.
"""

from __future__ import annotations

import io
import json
import zipfile
from collections.abc import Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NotRequired, TypedDict

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from saga_pose.analysis.saturation import clipped_pixel_mask, patch_fraction_map
from saga_pose.data.archives import ArchiveKind, open_fresh_archive
from saga_pose.data.augment import TargetedSaturationConfig, targeted_specular_saturation
from saga_pose.data.fresh import FreshAnnotation, frame_member, load_metadata
from saga_pose.data.real import DEFAULT_ADDENDUM_ROOT, frame_number, load_real_metadata
from saga_pose.data.representations import (
    EVENT_REPRESENTATIONS,
    REAL_EVENT_NEUTRAL,
    SYNTHETIC_EVENT_NEUTRAL,
    EventRepresentation,
    event_representation_maps,
)
from saga_pose.geometry.projection import (
    crop_resize_intrinsics,
    crop_resize_points,
    padded_square_bbox_xyxy,
)

SyntheticRole = Literal["train", "validation"]
ItemRole = Literal["train", "validation", "real_test"]
PREPROCESSING_VERSION = "crop-v2"
RGB_PADDING_VALUE = 0
CLIPPING_THRESHOLD = 254
CLIPPING_GRID_SIZE = 16


class FreshItem(TypedDict):
    """One aligned and geometrically transformed item.

    ``rgb``/``event`` keep their crop-v1 meaning (natural RGB, raw event frame as
    three identical channels). ``rgb_tss`` and ``clipping_ratio_tss`` are the
    protocol-2 targeted-saturation versions and equal the natural values when
    TSS was not applied (always for real data). Event maps are single-channel.
    The event-supp-v1 maps are present only when a supplement cache is attached.
    """

    sample_id: str
    object_name: str
    sequence: str
    frame_number: int
    role: ItemRole
    rgb: Tensor
    rgb_tss: Tensor
    event: Tensor
    event_raw: Tensor
    event_activity: Tensor
    event_magnitude: Tensor
    event_activity64: NotRequired[Tensor]
    event_activity_support: NotRequired[Tensor]
    clipping_ratio: Tensor
    clipping_ratio_tss: Tensor
    tss_applied: Tensor
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
    sequence: str
    metadata_member: str
    annotation: FreshAnnotation
    intrinsics: np.ndarray
    landmarks: np.ndarray


def _load_rgb(archive: zipfile.ZipFile, member: str) -> np.ndarray:
    with Image.open(io.BytesIO(archive.read(member))) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def crop_with_constant_fill(
    image: np.ndarray,
    crop_xyxy: tuple[int, int, int, int],
    fill_value: int,
) -> np.ndarray:
    """Crop ``(H, W, C)`` uint8 pixels, filling out-of-image regions with a constant."""
    left, top, right, bottom = crop_xyxy
    height, width = image.shape[:2]
    if right <= left or bottom <= top:
        raise ValueError("crop must have positive area")
    output = np.full((bottom - top, right - left, *image.shape[2:]), fill_value, dtype=image.dtype)
    source_left, source_top = max(left, 0), max(top, 0)
    source_right, source_bottom = min(right, width), min(bottom, height)
    if source_right > source_left and source_bottom > source_top:
        output[
            source_top - top : source_bottom - top,
            source_left - left : source_right - left,
        ] = image[source_top:source_bottom, source_left:source_right]
    return output


def _resize_rgb(crop: NDArray[np.uint8], size_wh: tuple[int, int]) -> NDArray[np.uint8]:
    return np.asarray(
        Image.fromarray(crop).resize(size_wh, resample=Image.Resampling.BILINEAR),
        dtype=np.uint8,
    )


def _clipping_grid(crop: NDArray[np.uint8]) -> NDArray[np.float32]:
    return patch_fraction_map(
        clipped_pixel_mask(crop, CLIPPING_THRESHOLD), CLIPPING_GRID_SIZE
    ).astype(np.float32)


def _image_tensor(image: NDArray[np.uint8]) -> Tensor:
    return torch.from_numpy(np.moveaxis(image, -1, 0).copy()).float().div_(255.0)


def _map_tensor(values: NDArray[np.uint8]) -> Tensor:
    return torch.from_numpy(values[None].copy()).float().div_(255.0)


def build_item(
    *,
    sample_id: str,
    object_name: str,
    sequence: str,
    role: ItemRole,
    rgb: NDArray[np.uint8],
    event: NDArray[np.uint8],
    annotation: FreshAnnotation,
    intrinsics: NDArray[np.float64],
    landmarks: NDArray[np.float64],
    event_neutral: int,
    target_size_wh: tuple[int, int] = (256, 256),
    crop_padding_factor: float = 1.2,
    tss_config: TargetedSaturationConfig | None = None,
    tss_seed: int = 249,
) -> FreshItem:
    """Apply the frozen crop-v2 contract to one RGB/event pair.

    ``tss_config=None`` disables protocol-2 corruption (always the case for real
    and validation data unless explicitly requested).
    """
    crop = padded_square_bbox_xyxy(annotation.bbox_xyxy, padding_factor=crop_padding_factor)
    rgb_crop = crop_with_constant_fill(rgb, crop, RGB_PADDING_VALUE)
    event_crop = crop_with_constant_fill(event, crop, event_neutral)
    if tss_config is None:
        rgb_tss_crop, applied = rgb_crop, False
    else:
        rgb_tss_crop, metadata = targeted_specular_saturation(
            rgb_crop, sample_id, seed=tss_seed, config=tss_config
        )
        applied = bool(metadata["applied"])

    rgb_resized = _resize_rgb(rgb_crop, target_size_wh)
    rgb_tss_resized = _resize_rgb(rgb_tss_crop, target_size_wh) if applied else rgb_resized
    clipping = _clipping_grid(rgb_crop)
    clipping_tss = _clipping_grid(rgb_tss_crop) if applied else clipping
    maps = event_representation_maps(event_crop, event_neutral, target_size_wh)
    event_raw = _map_tensor(maps["raw"])
    keypoints = crop_resize_points(annotation.keypoints, crop, target_size_wh)
    camera = crop_resize_intrinsics(intrinsics, crop, target_size_wh)
    return FreshItem(
        sample_id=sample_id,
        object_name=object_name,
        sequence=sequence,
        frame_number=frame_number(annotation.filename_rgb),
        role=role,
        rgb=_image_tensor(rgb_resized),
        rgb_tss=_image_tensor(rgb_tss_resized),
        event=event_raw.expand(3, -1, -1).clone(),
        event_raw=event_raw,
        event_activity=_map_tensor(maps["activity"]),
        event_magnitude=_map_tensor(maps["magnitude"]),
        clipping_ratio=torch.from_numpy(clipping),
        clipping_ratio_tss=torch.from_numpy(clipping_tss),
        tss_applied=torch.tensor(applied),
        keypoints_xy=torch.from_numpy(keypoints.astype(np.float32)),
        intrinsics=torch.from_numpy(camera.astype(np.float32)),
        pose_object_to_camera_mm=torch.from_numpy(annotation.pose.astype(np.float32)),
        landmarks_object_mm=torch.from_numpy(landmarks.astype(np.float32)),
        source_bbox_xyxy=torch.tensor(annotation.bbox_xyxy, dtype=torch.float32),
        crop_xyxy=torch.tensor(crop, dtype=torch.int64),
    )


def event_input(item_or_batch: dict[str, Tensor], representation: EventRepresentation) -> Tensor:
    """Return the chosen event representation as 3 channels for pretrained backbones."""
    if representation not in EVENT_REPRESENTATIONS:
        raise ValueError(f"unknown event representation: {representation}")
    key = f"event_{representation}"
    if key not in item_or_batch:
        raise KeyError(
            f"{key} is missing; representation {representation!r} needs the event "
            "supplement cache (attach it with --supplement-dir)"
        )
    values = item_or_batch[key]
    channel_dim = values.ndim - 3
    return values.repeat_interleave(3, dim=channel_dim)


class _ArchiveBackedDataset(Dataset[FreshItem]):
    """Process-local lazy ZIP handle shared by the synthetic and real datasets."""

    archive_kind: ArchiveKind
    event_neutral: int
    target_size_wh: tuple[int, int]
    crop_padding_factor: float
    _records: tuple[_SampleRecord, ...]

    def __init__(self, raw_root: Path) -> None:
        super().__init__()
        self.raw_root = raw_root
        self._archive_context: AbstractContextManager[zipfile.ZipFile] | None = None
        self._archive: zipfile.ZipFile | None = None

    def _get_archive(self) -> zipfile.ZipFile:
        if self._archive is None:
            context = open_fresh_archive(self.archive_kind, self.raw_root)
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

    def event_crop(
        self, index: int
    ) -> tuple[str, int, tuple[int, int, int, int], NDArray[np.uint8]]:
        """Return ``(sample_id, frame_number, crop_xyxy, event_crop)`` for one sample.

        Only the event frame is decoded; the crop box and fill are those of
        :func:`build_item`, so supplement maps align with the crop-v2 cache.
        """
        record = self._records[index]
        annotation = record.annotation
        event = _load_rgb(
            self._get_archive(), frame_member(record.metadata_member, annotation.filename_event)
        )
        crop = padded_square_bbox_xyxy(
            annotation.bbox_xyxy, padding_factor=self.crop_padding_factor
        )
        return (
            record.sample_id,
            frame_number(annotation.filename_rgb),
            crop,
            crop_with_constant_fill(event, crop, self.event_neutral),
        )


class FreshArchiveDataset(_ArchiveBackedDataset):
    """Read paired synthetic FRESH frames lazily from ``synthetic.zip``.

    Archive handles are opened per process on first access and discarded on
    pickle, making the dataset compatible with Windows spawned workers.
    """

    archive_kind: ArchiveKind = "synthetic"
    event_neutral: int = SYNTHETIC_EVENT_NEUTRAL

    def __init__(
        self,
        raw_root: Path,
        split_index: Path,
        *,
        role: SyntheticRole,
        objects: Sequence[str] = ("cassini", "satty", "soho"),
        target_size_wh: tuple[int, int] = (256, 256),
        crop_padding_factor: float = 1.2,
        tss_config: TargetedSaturationConfig | None = None,
        tss_seed: int = 249,
    ) -> None:
        super().__init__(raw_root)
        self.split_index = split_index
        self.role = role
        self.objects = tuple(objects)
        self.target_size_wh = target_size_wh
        self.clipping_threshold = CLIPPING_THRESHOLD
        self.clipping_grid_size = CLIPPING_GRID_SIZE
        self.crop_padding_factor = crop_padding_factor
        self.tss_config = tss_config
        self.tss_seed = tss_seed
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
                                sequence=f"synthetic-{object_name}",
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

    @property
    def sample_ids(self) -> list[str]:
        return [record.sample_id for record in self._records]

    def __getitem__(self, index: int) -> FreshItem:
        record = self._records[index]
        archive = self._get_archive()
        annotation = record.annotation
        return build_item(
            sample_id=record.sample_id,
            object_name=record.object_name,
            sequence=record.sequence,
            role=self.role,
            rgb=_load_rgb(archive, frame_member(record.metadata_member, annotation.filename_rgb)),
            event=_load_rgb(
                archive, frame_member(record.metadata_member, annotation.filename_event)
            ),
            annotation=annotation,
            intrinsics=record.intrinsics,
            landmarks=record.landmarks,
            event_neutral=SYNTHETIC_EVENT_NEUTRAL,
            target_size_wh=self.target_size_wh,
            crop_padding_factor=self.crop_padding_factor,
            tss_config=self.tss_config,
            tss_seed=self.tss_seed,
        )


class FreshRealArchiveDataset(_ArchiveBackedDataset):
    """Frozen real-test frames with verified author annotation corrections.

    Uses the same annotation-box (shared/oracle) crop as the synthetic contract,
    so results are conditional pose estimation, not an end-to-end pipeline.
    """

    archive_kind: ArchiveKind = "real"
    event_neutral: int = REAL_EVENT_NEUTRAL

    def __init__(
        self,
        raw_root: Path,
        *,
        addendum_root: Path = DEFAULT_ADDENDUM_ROOT,
        sequences: Sequence[str] | None = None,
        target_size_wh: tuple[int, int] = (256, 256),
        crop_padding_factor: float = 1.2,
    ) -> None:
        super().__init__(raw_root)
        self.target_size_wh = target_size_wh
        self.crop_padding_factor = crop_padding_factor
        with open_fresh_archive("real", raw_root) as archive:
            metadata, self.annotation_sources = load_real_metadata(
                archive, addendum_root=addendum_root
            )
        names = sorted(metadata) if sequences is None else list(sequences)
        records: list[_SampleRecord] = []
        for name in names:
            sequence_metadata = metadata[name]
            for annotation in sequence_metadata.annotations:
                records.append(
                    _SampleRecord(
                        sample_id=f"{name}/{annotation.filename_rgb}",
                        object_name=name.split("-", maxsplit=1)[0],
                        sequence=name,
                        metadata_member=sequence_metadata.member,
                        annotation=annotation,
                        intrinsics=sequence_metadata.intrinsics,
                        landmarks=sequence_metadata.landmarks_3d,
                    )
                )
        self._records = tuple(records)

    def __len__(self) -> int:
        return len(self._records)

    @property
    def sample_ids(self) -> list[str]:
        return [record.sample_id for record in self._records]

    def __getitem__(self, index: int) -> FreshItem:
        record = self._records[index]
        archive = self._get_archive()
        annotation = record.annotation
        return build_item(
            sample_id=record.sample_id,
            object_name=record.object_name,
            sequence=record.sequence,
            role="real_test",
            rgb=_load_rgb(archive, frame_member(record.metadata_member, annotation.filename_rgb)),
            event=_load_rgb(
                archive, frame_member(record.metadata_member, annotation.filename_event)
            ),
            annotation=annotation,
            intrinsics=record.intrinsics,
            landmarks=record.landmarks,
            event_neutral=REAL_EVENT_NEUTRAL,
            target_size_wh=self.target_size_wh,
            crop_padding_factor=self.crop_padding_factor,
        )
