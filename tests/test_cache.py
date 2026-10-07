"""Schema-2 crop cache: build/read round trip and immutability guards."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from saga_pose.data.augment import TargetedSaturationConfig
from saga_pose.data.cache import FreshCropCacheDataset, build_crop_cache
from saga_pose.data.dataset import FreshItem, build_item, event_input
from saga_pose.data.fresh import FreshAnnotation


def _item(index: int) -> FreshItem:
    rng = np.random.default_rng(index)
    rgb = rng.integers(0, 200, size=(60, 80, 3), dtype=np.uint8)
    event = np.full((60, 80, 3), 127, dtype=np.uint8)
    event[20:30, 30:40] = 255
    pose = np.eye(4)
    pose[2, 3] = 1500.0
    annotation = FreshAnnotation(
        filename_rgb=f"{index:05d}_rgb.png",
        filename_event=f"{index:05d}_event.png",
        pose=pose,
        keypoints=np.array([[40.0, 30.0], [50.0, 35.0]]),
        bbox_xyxy=(20.0, 10.0, 60.0, 50.0),
        index=None,
    )
    return build_item(
        sample_id=f"cassini/{index:05d}_rgb.png",
        object_name="cassini",
        sequence="synthetic-cassini",
        role="train",
        rgb=rgb,
        event=event,
        annotation=annotation,
        intrinsics=np.array([[500.0, 0.0, 40.0], [0.0, 500.0, 30.0], [0.0, 0.0, 1.0]]),
        landmarks=np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
        event_neutral=127,
        target_size_wh=(32, 32),
        tss_config=TargetedSaturationConfig(probability=1.0 if index % 2 else 0.0),
    )


class _Items(Dataset[FreshItem]):
    def __init__(self, count: int) -> None:
        self.items = [_item(index) for index in range(count)]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> FreshItem:
        return self.items[index]


def test_crop_cache_round_trip_is_exact_for_all_fields(tmp_path: Path) -> None:
    source = _Items(5)
    ids = [item["sample_id"] for item in source.items]
    root = tmp_path / "cache"
    build_crop_cache(
        source, ids, root, role="train", store_tss=True, provenance={}, num_workers=0, batch_size=2
    )
    cached = FreshCropCacheDataset(root)
    assert len(cached) == 5
    for index, expected in enumerate(source.items):
        actual = cached[index]
        assert actual["sample_id"] == expected["sample_id"]
        assert actual["frame_number"] == expected["frame_number"]
        for name in (
            "rgb",
            "rgb_tss",
            "event",
            "event_raw",
            "event_activity",
            "event_magnitude",
        ):
            # Images are already uint8-quantized values divided by 255.
            torch.testing.assert_close(actual[name], expected[name], atol=1e-6, rtol=0)  # type: ignore[literal-required]
        for name in ("clipping_ratio_tss", "tss_applied", "keypoints_xy", "intrinsics"):
            assert torch.equal(actual[name], expected[name])  # type: ignore[literal-required]
    assert bool(cached[1]["tss_applied"]) and not bool(cached[0]["tss_applied"])
    assert float(cached[1]["clipping_ratio_tss"].max()) > 0.0
    assert event_input(dict(cached[0]), "activity").shape == (3, 32, 32)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="different or incomplete"):
        build_crop_cache(source, ids[:2], root, role="train", store_tss=True, provenance={})


def test_crop_cache_rejects_incomplete_manifest(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text(
        json.dumps({"schema_version": 2, "status": "building"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not complete"):
        FreshCropCacheDataset(tmp_path)


def test_event_representations_are_polarity_agnostic() -> None:
    from saga_pose.data.representations import native_event_representations

    synthetic = np.array([[127, 191, 255]], dtype=np.uint8)
    real = np.array([[128, 64, 255, 130, 104]], dtype=np.uint8)
    synthetic_maps = native_event_representations(synthetic, 127)
    real_maps = native_event_representations(real, 128)
    # Both synthetic polarities are indistinguishable in both polarity-agnostic maps.
    np.testing.assert_array_equal(synthetic_maps["activity"], [[0.0, 1.0, 1.0]])
    np.testing.assert_array_equal(synthetic_maps["magnitude"], [[0.0, 1.0, 1.0]])
    np.testing.assert_array_equal(real_maps["activity"], [[0.0, 1.0, 1.0, 0.0, 0.0]])
    np.testing.assert_allclose(real_maps["magnitude"], [[0.0, 1.0, 1.0, 2 / 64, 24 / 64]])
    with pytest.raises(ValueError, match="channels differ"):
        native_event_representations(np.array([[[127, 0, 127]]], dtype=np.uint8), 127)
