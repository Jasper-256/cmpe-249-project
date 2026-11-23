"""events-v1 study arms: new representations, native noise, and supplement checks."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from saga_pose.data.dataset import event_input
from saga_pose.data.representations import (
    native_event_representations,
    native_supplement_representations,
    support_filter,
)
from saga_pose.data.sim2real import (
    NATIVE_NOISE_DENSITY_RANGE,
    Sim2RealConfig,
    augment_batch,
    augment_config,
    native_noise,
)
from saga_pose.data.supplement import check_supplement_order


def test_activity64_equals_activity_on_synthetic_levels() -> None:
    rng = np.random.default_rng(0)
    grey = rng.choice(np.array([127, 191, 255], dtype=np.uint8), size=(40, 50))
    frame = np.repeat(grey[..., None], 3, axis=2)
    crop_v2 = native_event_representations(frame, 127)
    supplement = native_supplement_representations(frame, 127)
    assert np.array_equal(supplement["activity64"], crop_v2["activity"])
    # Real residue between 32 and 63 is active for ``activity`` only.
    residue = native_supplement_representations(np.array([[128 + 40, 128 + 64]], np.uint8), 128)
    assert residue["activity64"].tolist() == [[0.0, 1.0]]


def test_support_filter_removes_isolated_and_keeps_connected() -> None:
    active = np.zeros((12, 12), dtype=bool)
    active[1, 1] = True  # isolated
    active[5, 2:9] = True  # edge segment
    active[9, 9] = active[10, 10] = True  # diagonal pair (8-connected)
    active[0, 11] = True  # isolated at the border
    kept = support_filter(active)
    assert not kept[1, 1] and not kept[0, 11]
    assert kept[5, 2:9].all()
    assert kept[9, 9] and kept[10, 10]
    frame = np.full((12, 12), 128, dtype=np.uint8)
    frame[active] = 255
    maps = native_supplement_representations(frame, 128)
    assert np.array_equal(maps["activity_support"].astype(bool), kept)


def test_native_noise_is_faint_and_respects_density() -> None:
    generator = torch.Generator().manual_seed(5)
    noise = native_noise(64, (256, 256), generator)
    assert noise.shape == (64, 1, 256, 256)
    assert float(noise.min()) >= 0.0 and float(noise.max()) <= 1.0
    nonzero = noise[noise > 0]
    # Area downsampling turns isolated native events into sub-unit haze.
    assert float((nonzero < 1.0).float().mean()) > 0.9
    # The mean of an area-downsampled binary grid is its density.
    means = noise.mean(dim=(1, 2, 3))
    low, high = NATIVE_NOISE_DENSITY_RANGE
    assert float(means.min()) >= low * 0.5 and float(means.max()) <= high * 1.05
    assert float(means.max()) - float(means.min()) > 0.05
    repeat = native_noise(64, (256, 256), torch.Generator().manual_seed(5))
    assert torch.equal(noise, repeat)


def _inputs() -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    base = torch.Generator().manual_seed(99)
    inputs = {
        "rgb": torch.rand(3, 3, 64, 64, generator=base),
        "event": (torch.rand(3, 1, 64, 64, generator=base) < 0.1).float().repeat(1, 3, 1, 1),
        "clipping": torch.rand(3, 16, 16, generator=base),
    }
    keypoints = torch.rand(3, 5, 2, generator=base) * 63
    return inputs, keypoints


def test_sim2real_v1_default_behaviour_unchanged() -> None:
    # Expected values were recorded from the pre-events-v1 code path with the
    # same inputs and generator seed.
    inputs, keypoints = _inputs()
    generator = torch.Generator().manual_seed(1234)
    out, moved, _ = augment_batch(inputs, keypoints, augment_config("sim2real-v1"), generator)
    assert augment_config("sim2real-v1") == Sim2RealConfig()
    expected = {
        "rgb": (16503.414811871015, 9634.41145969948),
        "event": (3709.127556261985, 1820.1051698214815),
        "clipping": (6.0, 0.625),
    }
    for key, (total, squares) in expected.items():
        values = out[key].double()
        assert float(values.sum()) == pytest.approx(total, rel=1e-6)
        assert float((values**2).sum()) == pytest.approx(squares, rel=1e-6)
    assert float(moved.double().sum()) == pytest.approx(993.6653709411621, rel=1e-6)
    # The generator advanced by exactly the same number of draws.
    assert float(torch.rand(1, generator=generator)) == pytest.approx(0.11291897296905518)


def test_named_event_variants_change_only_the_event_term() -> None:
    reference = Sim2RealConfig().as_dict()
    for name, field, value in (
        ("sim2real-v1-native-noise", "event_noise_mode", "native"),
        ("sim2real-v1-drop90", "event_dropout_max", 0.9),
    ):
        config = augment_config(name).as_dict()
        changed = {key for key in reference if reference[key] != config[key]}
        assert changed == {"version", field}
        assert config[field] == value
    native_config = augment_config("sim2real-v1-native-noise").as_dict()
    assert native_config["event_noise_native_density_range"] == (0.001, 0.20)
    assert native_config["event_noise_native_upsample"] == 2.5
    inputs, keypoints = _inputs()
    plain, _, _ = augment_batch(
        inputs, keypoints, Sim2RealConfig(), torch.Generator().manual_seed(3)
    )
    native, _, _ = augment_batch(
        inputs,
        keypoints,
        augment_config("sim2real-v1-native-noise"),
        torch.Generator().manual_seed(3),
    )
    # Geometry and photometry draws precede the event term, so RGB is identical.
    assert torch.equal(plain["rgb"], native["rgb"])
    assert float(native["event"].min()) >= 0.0 and float(native["event"].max()) <= 1.0


def test_supplement_order_check_raises_on_mismatch() -> None:
    ids = ["a/1", "a/2", "a/3"]
    frames = np.array([1, 2, 3])
    check_supplement_order(ids, frames, list(ids), frames.copy())
    with pytest.raises(ValueError, match="frame numbers"):
        check_supplement_order(ids, frames, list(ids), np.array([1, 3, 2]))
    with pytest.raises(ValueError, match="sample IDs"):
        check_supplement_order(ids, frames, ["a/1", "a/3", "a/2"], frames)
    with pytest.raises(ValueError, match="samples"):
        check_supplement_order(ids, frames, ids[:2], frames[:2])


def test_event_input_requires_attached_supplement() -> None:
    batch = {"event_activity": torch.zeros(2, 1, 4, 4)}
    assert event_input(batch, "activity").shape == (2, 3, 4, 4)
    with pytest.raises(KeyError, match="supplement"):
        event_input(batch, "activity64")
