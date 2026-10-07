"""Real-test contract: pinned annotation overrides and author slices (fixture only)."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from saga_pose.data import real
from saga_pose.data.real import (
    AnnotationOverride,
    frame_number,
    load_real_metadata,
    load_real_slices,
    slice_mask,
)


def _metadata_json(translation_z: float) -> bytes:
    pose = np.eye(4)
    pose[2, 3] = translation_z
    payload = {
        "landmarks_3d": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "wireframe_points": [[0.0, 0.0, 0.0]],
        "wireframe_faces": [[0, 0, 0]],
        "intrinsics": [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]],
        "annotations": [
            {
                "filename_rgb": "00007_rgb.png",
                "filename_event": "00007_event.png",
                "pose": pose.tolist(),
                "keypoints": [0.0] * 8,
                "bbox": [1.0, 1.0, 5.0, 5.0],
            }
        ],
    }
    return json.dumps(payload).encode()


def _archive() -> zipfile.ZipFile:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("real/a-1-far/test.json", _metadata_json(1000.0))
        archive.writestr("real/b-1-far/test.json", _metadata_json(2000.0))
        archive.writestr(
            "real/harsh-and-slow-frames.json",
            json.dumps({"a-1-far": {"harsh": [0, 10], "low-motion": [5, 7]}}),
        )
    buffer.seek(0)
    return zipfile.ZipFile(buffer)


def test_override_replaces_archive_labels_and_is_hash_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corrected = _metadata_json(1234.0)
    path = tmp_path / "corrected" / "b.json"
    path.parent.mkdir()
    path.write_bytes(corrected)
    override = AnnotationOverride(
        sequence="b-1-far",
        relative_path="corrected/b.json",
        sha256=hashlib.sha256(corrected).hexdigest(),
    )
    monkeypatch.setattr(real, "ANNOTATION_OVERRIDES", (override,))

    metadata, sources = load_real_metadata(_archive(), addendum_root=tmp_path)

    assert sorted(metadata) == ["a-1-far", "b-1-far"]
    assert metadata["a-1-far"].annotations[0].pose[2, 3] == 1000.0
    assert metadata["b-1-far"].annotations[0].pose[2, 3] == 1234.0
    assert sources["b-1-far"].startswith("override:")

    path.write_bytes(_metadata_json(9999.0))
    with pytest.raises(ValueError, match="SHA-256"):
        load_real_metadata(_archive(), addendum_root=tmp_path)


def test_missing_override_is_an_error(tmp_path: Path) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("real/satty-4-far/test.json", _metadata_json(1.0))
    buffer.seek(0)
    with pytest.raises(FileNotFoundError, match="corrected FRESH annotation"):
        load_real_metadata(zipfile.ZipFile(buffer), addendum_root=tmp_path)


def test_slices_are_inclusive_frame_ranges() -> None:
    slices = load_real_slices(_archive())
    assert slices == {"a-1-far": {"harsh": (0, 10), "low-motion": (5, 7)}}
    numbers = [frame_number(f"{index:05d}_rgb.png") for index in (4, 5, 7, 8)]
    np.testing.assert_array_equal(
        slice_mask(numbers, slices["a-1-far"]["low-motion"]), [False, True, True, False]
    )
