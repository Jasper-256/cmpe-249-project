"""Typed access to FRESH metadata stored inside the dataset archives."""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

DatasetRole = Literal["train", "validation", "real_test"]


@dataclass(frozen=True)
class FreshAnnotation:
    """One aligned RGB/event pair and its object-to-camera pose labels."""

    filename_rgb: str
    filename_event: str
    pose: NDArray[np.float64]
    keypoints: NDArray[np.float64]
    bbox_xyxy: tuple[float, float, float, float]
    index: int | None


@dataclass(frozen=True)
class FreshMetadata:
    """Contents of one object or real-sequence annotation file."""

    member: str
    landmarks_3d: NDArray[np.float64]
    wireframe_points: NDArray[np.float64]
    wireframe_faces: tuple[tuple[int, ...], ...]
    intrinsics: NDArray[np.float64]
    annotations: tuple[FreshAnnotation, ...]

    @property
    def object_name(self) -> str:
        return PurePosixPath(self.member).parent.name.split("-")[0]

    @property
    def role(self) -> DatasetRole:
        path = PurePosixPath(self.member)
        if path.parts[0] == "real":
            return "real_test"
        return "train" if path.stem == "train" else "validation"


def metadata_members(archive: zipfile.ZipFile, *, real: bool) -> list[str]:
    """Return canonical FRESH annotation members in deterministic order."""
    prefix = "real/" if real else "synthetic/"
    expected_names = {"test.json"} if real else {"train.json", "test.json"}
    return sorted(
        name
        for name in archive.namelist()
        if name.startswith(prefix) and PurePosixPath(name).name in expected_names
    )


def frame_member(metadata_member: str, filename: str) -> str:
    """Resolve a frame filename relative to its annotation member."""
    return str(PurePosixPath(metadata_member).parent / "frames" / filename)


def load_metadata(archive: zipfile.ZipFile, member: str) -> FreshMetadata:
    """Load and validate the basic types and array shapes of one JSON member."""
    raw = cast(dict[str, Any], json.loads(archive.read(member)))
    landmarks = np.asarray(raw["landmarks_3d"], dtype=np.float64)
    wireframe_points = np.asarray(raw["wireframe_points"], dtype=np.float64)
    wireframe_faces = tuple(
        tuple(int(index) for index in face)
        for face in cast(list[list[int]], raw["wireframe_faces"])
    )
    intrinsics = np.asarray(raw["intrinsics"], dtype=np.float64)

    if landmarks.ndim != 2 or landmarks.shape[1] != 3:
        raise ValueError(f"{member}: landmarks_3d must have shape (N, 3)")
    if wireframe_points.ndim != 2 or wireframe_points.shape[1] != 3:
        raise ValueError(f"{member}: wireframe_points must have shape (N, 3)")
    if any(len(face) < 3 for face in wireframe_faces):
        raise ValueError(f"{member}: each wireframe face must have at least three vertices")
    if intrinsics.shape != (3, 3):
        raise ValueError(f"{member}: intrinsics must have shape (3, 3)")

    annotations: list[FreshAnnotation] = []
    for row_index, annotation_raw in enumerate(cast(list[dict[str, Any]], raw["annotations"])):
        pose = np.asarray(annotation_raw["pose"], dtype=np.float64)
        keypoints = np.asarray(annotation_raw["keypoints"], dtype=np.float64).reshape(-1, 2)
        bbox_values = tuple(float(value) for value in annotation_raw["bbox"])
        if pose.shape != (4, 4):
            raise ValueError(f"{member} row {row_index}: pose must have shape (4, 4)")
        if keypoints.shape != (landmarks.shape[0], 2):
            raise ValueError(
                f"{member} row {row_index}: {keypoints.shape[0]} keypoints for "
                f"{landmarks.shape[0]} landmarks"
            )
        if len(bbox_values) != 4:
            raise ValueError(f"{member} row {row_index}: bbox must contain four values")
        index_raw = annotation_raw.get("index")
        annotations.append(
            FreshAnnotation(
                filename_rgb=str(annotation_raw["filename_rgb"]),
                filename_event=str(annotation_raw["filename_event"]),
                pose=pose,
                keypoints=keypoints,
                bbox_xyxy=bbox_values,
                index=int(index_raw) if index_raw is not None else None,
            )
        )

    return FreshMetadata(
        member=member,
        landmarks_3d=landmarks,
        wireframe_points=wireframe_points,
        wireframe_faces=wireframe_faces,
        intrinsics=intrinsics,
        annotations=tuple(annotations),
    )
