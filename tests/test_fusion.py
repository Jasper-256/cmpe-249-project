"""Protocol properties of the fusion family (G4 checklist items)."""

from __future__ import annotations

import itertools

import pytest
import torch

from saga_pose.models.fusion import (
    FUSION_RULES,
    FusionHeatmapRegressor,
    LevelFusion,
    clipping_at_level,
    intervene_on_clipping,
)


def _features() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    return torch.randn(2, 8, 4, 4, generator=generator), torch.randn(
        2, 8, 4, 4, generator=generator
    )


@pytest.mark.parametrize("rule", FUSION_RULES)
def test_every_rule_produces_a_valid_fused_level(rule: str) -> None:
    fusion = LevelFusion(rule, 8)  # type: ignore[arg-type]
    rgb, event = _features()
    with torch.no_grad():
        output, gate = fusion(rgb, event, torch.zeros(2, 1, 4, 4), None)
    assert output.shape == rgb.shape
    if gate is not None:
        assert float(gate.min()) >= 0.0 and float(gate.max()) <= 1.0


def test_learned_clip_gate_is_monotone_non_increasing_in_clipping() -> None:
    fusion = LevelFusion("clip_gate", 8)
    with torch.no_grad():
        fusion.clip_slope.uniform_(-3.0, 3.0)  # type: ignore[union-attr]
        fusion.clip_bias.uniform_(-2.0, 2.0)  # type: ignore[union-attr]
    rgb, event = _features()
    levels = torch.linspace(0.0, 1.0, 11)
    gates = [fusion.gate(rgb, event, torch.full((2, 1, 4, 4), float(s))) for s in levels]
    for lower, higher in itertools.pairwise(gates):
        assert torch.all(higher <= lower + 1e-7)
    assert torch.all(gates[-1] < gates[0])


def test_constant_control_matches_parameters_and_ignores_clipping() -> None:
    learned = LevelFusion("clip_gate", 8)
    control = LevelFusion("clip_gate_constant", 8)
    assert sum(p.numel() for p in learned.parameters()) == sum(
        p.numel() for p in control.parameters()
    )
    rgb, event = _features()
    zero = control.gate(rgb, event, torch.zeros(2, 1, 4, 4))
    full = control.gate(rgb, event, torch.ones(2, 1, 4, 4))
    assert torch.equal(zero, full)


def test_fixed_gate_and_saturation_extremes() -> None:
    rgb, event = _features()
    fixed = LevelFusion("clip_gate_fixed", 8)
    assert sum(p.numel() for p in fixed.parameters()) == 0
    clipping = torch.zeros(2, 1, 4, 4)
    clipping[:, :, 0, 0] = 1.0
    gate = fixed.gate(rgb, event, clipping)
    assert float(gate[0, 0, 0, 0]) == 0.0 and float(gate[0, 0, 1, 1]) == 0.5
    learned = LevelFusion("clip_gate", 8)
    unclipped = learned.gate(rgb, event, torch.zeros(2, 1, 4, 4))
    saturated = learned.gate(rgb, event, torch.ones(2, 1, 4, 4))
    torch.testing.assert_close(unclipped, torch.full_like(unclipped, 0.5))
    assert float(saturated.max()) < 0.01


def test_interventions_preserve_marginal_or_bypass() -> None:
    clipping = torch.rand(3, 16, 16)
    generator = torch.Generator().manual_seed(1)
    shuffled = intervene_on_clipping(clipping, "shuffle", generator)
    torch.testing.assert_close(
        torch.sort(shuffled.flatten(1)).values, torch.sort(clipping.flatten(1)).values
    )
    assert not torch.equal(shuffled, clipping)
    assert float(intervene_on_clipping(clipping, "zero").abs().sum()) == 0.0


def test_clipping_grid_maps_to_every_level_scale() -> None:
    grid = torch.zeros(1, 16, 16)
    grid[0, 0, 0] = 1.0
    assert clipping_at_level(grid, (64, 64)).shape == (1, 1, 64, 64)
    assert float(clipping_at_level(grid, (64, 64))[0, 0, :4, :4].min()) == 1.0
    assert float(clipping_at_level(grid, (8, 8))[0, 0, 0, 0]) == 0.25


def test_fusion_regressor_forward_backward_and_gate_bypass() -> None:
    model = FusionHeatmapRegressor(
        4, backbone_name="efficientnet_b0", fusion="content_clip_gate", pretrained=False
    )
    rgb, event = torch.rand(1, 3, 256, 256), torch.rand(1, 3, 256, 256)
    clipping = torch.rand(1, 16, 16)
    heatmaps, gates = model(rgb, event, clipping, return_gates=True)
    assert heatmaps.shape == (1, 4, 64, 64)
    assert len(gates) == len(model.feature_channels)
    heatmaps.sum().backward()
    assert model.fusions[0].clip_slope.grad is not None  # type: ignore[union-attr]
    bypass = model(rgb, event, clipping, gate_override=0.5)
    mean = FusionHeatmapRegressor(4, backbone_name="efficientnet_b0", fusion="mean")
    mean.load_state_dict(
        {k: v for k, v in model.state_dict().items() if not k.startswith("fusions")}
    )
    torch.testing.assert_close(bypass, mean(rgb, event, clipping))


def test_signed_gate_starts_like_clip_gate_but_can_flip() -> None:
    rgb, event = _features()
    clipping = torch.full((2, 1, 4, 4), 0.3)
    monotone = LevelFusion("clip_gate", 8)
    signed = LevelFusion("clip_gate_signed", 8)
    torch.testing.assert_close(
        signed.gate(rgb, event, clipping), monotone.gate(rgb, event, clipping)
    )
    with torch.no_grad():
        signed.clip_weight.fill_(4.0)  # type: ignore[union-attr]
    assert torch.all(signed.gate(rgb, event, clipping) > 0.5)


def test_decoder_initialization_is_identical_across_fusion_rules() -> None:
    states = []
    for rule in ("mean", "concat", "content_gate"):
        torch.manual_seed(249)
        model = FusionHeatmapRegressor(4, backbone_name="efficientnet_b0", fusion=rule)  # type: ignore[arg-type]
        states.append(
            {k: v for k, v in model.state_dict().items() if k.startswith(("lateral", "head"))}
        )
        after = torch.rand(1)
        states[-1]["_rng_probe"] = after
    for other in states[1:]:
        for key, value in states[0].items():
            torch.testing.assert_close(other[key], value)
