"""Two-stream event-RGB heatmap models with a family of matched fusion rules.

Every variant shares the same two pretrained encoders (separate weights, same
architecture), the same per-level fusion insertion point (every pyramid level),
and the same heatmap decoder. Only the fusion rule changes, so differences can
be attributed to it. Gated rules compute ``fused = g * rgb + (1 - g) * event``.

Fusion rules
------------
``mean``                 g = 0.5, no parameters.
``concat``               1x1 conv over [rgb, event]; no explicit gate.
``content_gate``         g = sigmoid(conv1x1([rgb, event])): unrestricted, learned
                         reliability from content alone.
``clip_gate``            g = sigmoid(b - softplus(a) * s): the proposed monotone
                         gate driven only by the measured patch clipping ratio s.
``clip_gate_constant``   identical module and parameter count, s replaced by 0:
                         the parameter-matched no-physics control.
``clip_gate_fixed``      g = 0.5 * (1 - [s >= tau]), no parameters: mean fusion
                         with a fixed physical override toward events.
``content_clip_gate``    g = sigmoid(conv1x1([rgb, event]) - softplus(a) * s): does
                         explicit clipping add information beyond learned
                         content reliability?
``clip_gate_signed``     g = sigmoid(b + w * s) with unconstrained sign, initialized
                         at the same starting gate as ``clip_gate``: separates the
                         clipping *signal* from the *monotone constraint*.

Under the natural protocol s == 0 throughout training, so clipping slopes
receive no gradient: ``clip_gate`` and ``clip_gate_constant`` train identically
and the slope stays at its initialization. Natural-protocol clipping gates are
therefore hand-set priors at test time, not learned physics.

Fusion modules are built inside a forked RNG after the decoder, so every rule
shares the same decoder initialization for a given seed.

Evaluation-time interventions on the clipping input (``clip_mode``): ``normal``,
``shuffle`` (permute patch values within each sample, preserving the marginal),
and ``zero`` (bypass the physical signal). ``gate_override`` replaces every gate
by a constant (gate bypass, e.g. 0.5).
"""

from __future__ import annotations

from typing import Any, Final, Literal, cast

import timm
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from saga_pose.models.heatmaps import IMAGENET_MEAN, IMAGENET_STD, fuse_pyramid, normalize_inputs

FusionRule = Literal[
    "mean",
    "concat",
    "content_gate",
    "clip_gate",
    "clip_gate_constant",
    "clip_gate_fixed",
    "content_clip_gate",
    "clip_gate_signed",
]
FUSION_RULES: Final[tuple[FusionRule, ...]] = (
    "mean",
    "concat",
    "content_gate",
    "clip_gate",
    "clip_gate_constant",
    "clip_gate_fixed",
    "content_clip_gate",
    "clip_gate_signed",
)
ClipMode = Literal["normal", "shuffle", "zero"]
FIXED_GATE_THRESHOLD: Final[float] = 0.01
CLIP_SLOPE_INIT: Final[float] = 5.0


def clipping_at_level(clipping: Tensor, size: tuple[int, int]) -> Tensor:
    """Map a ``(B, G, G)`` patch clipping grid onto a feature level as ``(B, 1, H, W)``."""
    grid = clipping[:, None].float()
    height, width = size
    grid_height, grid_width = grid.shape[-2:]
    if (height, width) == (grid_height, grid_width):
        return grid
    if height >= grid_height:
        return F.interpolate(grid, size=size, mode="nearest")
    factor = grid_height // height
    if grid_height % height or grid_width % width or grid_width // width != factor:
        raise ValueError(f"cannot pool a {grid_height}x{grid_width} grid onto {size}")
    return F.avg_pool2d(grid, kernel_size=factor, stride=factor)


def intervene_on_clipping(
    clipping: Tensor, mode: ClipMode, generator: torch.Generator | None = None
) -> Tensor:
    """Apply an evaluation-time intervention to the clipping grid."""
    if mode == "normal":
        return clipping
    if mode == "zero":
        return torch.zeros_like(clipping)
    batch, height, width = clipping.shape
    flat = clipping.reshape(batch, -1)
    order = torch.argsort(
        torch.rand(flat.shape, generator=generator, device="cpu").to(clipping.device), dim=1
    )
    return torch.gather(flat, 1, order).reshape(batch, height, width)


def _inverse_softplus(value: float) -> float:
    return float(torch.log(torch.expm1(torch.tensor(value))))


class LevelFusion(nn.Module):
    """One fusion rule at one pyramid level with ``channels`` features."""

    def __init__(
        self, rule: FusionRule, channels: int, *, clip_slope_init: float = CLIP_SLOPE_INIT
    ) -> None:
        super().__init__()
        self.rule = rule
        self.content: nn.Conv2d | None = None
        self.mixer: nn.Conv2d | None = None
        self.clip_slope: nn.Parameter | None = None
        self.clip_bias: nn.Parameter | None = None
        self.clip_weight: nn.Parameter | None = None
        if rule == "concat":
            self.mixer = nn.Conv2d(2 * channels, channels, kernel_size=1)
        if rule in ("content_gate", "content_clip_gate"):
            content = nn.Conv2d(2 * channels, channels, kernel_size=1)
            # Zero init: every content gate starts as mean fusion (g = 0.5).
            nn.init.zeros_(content.weight)
            assert content.bias is not None
            nn.init.zeros_(content.bias)
            self.content = content
        if rule in ("clip_gate", "clip_gate_constant", "content_clip_gate"):
            # softplus(slope) starts at 5: a fully clipped patch begins at
            # g = sigmoid(-5) ~ 0.007, an unclipped one at g = 0.5.
            self.clip_slope = nn.Parameter(
                torch.full((channels,), _inverse_softplus(clip_slope_init))
            )
        if rule == "clip_gate_signed":
            # Same starting gate as clip_gate, but the sign is free to flip.
            self.clip_weight = nn.Parameter(torch.full((channels,), -clip_slope_init))
        if rule in ("clip_gate", "clip_gate_constant", "clip_gate_signed"):
            self.clip_bias = nn.Parameter(torch.zeros(channels))

    def forward(
        self, rgb: Tensor, event: Tensor, clipping: Tensor, gate_override: float | None
    ) -> tuple[Tensor, Tensor | None]:
        if self.rule == "concat":
            assert self.mixer is not None
            return self.mixer(torch.cat((rgb, event), dim=1)), None
        gate = self.gate(rgb, event, clipping)
        if gate_override is not None:
            gate = torch.full_like(gate, gate_override)
        return gate * rgb + (1.0 - gate) * event, gate

    def gate(self, rgb: Tensor, event: Tensor, clipping: Tensor) -> Tensor:
        if self.rule == "mean":
            return torch.full_like(rgb[:, :1], 0.5)
        if self.rule == "clip_gate_fixed":
            return 0.5 * (1.0 - (clipping >= FIXED_GATE_THRESHOLD).to(rgb.dtype))
        logits = torch.zeros_like(rgb[:, :1])
        if self.content is not None:
            logits = self.content(torch.cat((rgb, event), dim=1))
        if self.clip_bias is not None:
            logits = logits + self.clip_bias[None, :, None, None]
        if self.clip_slope is not None:
            signal = torch.zeros_like(clipping) if self.rule == "clip_gate_constant" else clipping
            logits = logits - F.softplus(self.clip_slope)[None, :, None, None] * signal
        if self.clip_weight is not None:
            logits = logits + self.clip_weight[None, :, None, None] * clipping
        return torch.sigmoid(logits)


class FusionHeatmapRegressor(nn.Module):
    """Two pretrained timm pyramids, per-level fusion, and the shared heatmap decoder."""

    def __init__(
        self,
        num_keypoints: int,
        *,
        backbone_name: str,
        fusion: FusionRule,
        pretrained: bool = False,
        decoder_channels: int = 64,
        backbone_kwargs: dict[str, Any] | None = None,
        clip_slope_init: float = CLIP_SLOPE_INIT,
    ) -> None:
        super().__init__()
        if fusion not in FUSION_RULES:
            raise ValueError(f"unknown fusion rule: {fusion}")
        self.fusion_rule = fusion
        self.backbone_name = backbone_name
        kwargs = backbone_kwargs or {}
        self.rgb_encoder = timm.create_model(
            backbone_name, pretrained=pretrained, features_only=True, **kwargs
        )
        self.event_encoder = timm.create_model(
            backbone_name, pretrained=pretrained, features_only=True, **kwargs
        )
        channels = tuple(cast(Any, self.rgb_encoder).feature_info.channels())
        self.feature_channels = channels
        pretrained_cfg = cast(dict[str, Any], getattr(self.rgb_encoder, "pretrained_cfg", {}))
        self.input_mean: tuple[float, ...] = tuple(pretrained_cfg.get("mean", IMAGENET_MEAN))
        self.input_std: tuple[float, ...] = tuple(pretrained_cfg.get("std", IMAGENET_STD))
        self.lateral = nn.ModuleList(
            nn.Conv2d(channel, decoder_channels, kernel_size=1) for channel in channels
        )
        self.head = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, decoder_channels),
            nn.GELU(),
            nn.Conv2d(decoder_channels, num_keypoints, kernel_size=1),
        )
        # Fusion modules draw random numbers (concat mixer) or none (gates); a
        # forked RNG keeps the global stream, and thus later initialization and
        # data order, identical across rules.
        with torch.random.fork_rng(devices=[]):
            # Seeded from the run seed so concat's mixer varies across seeds.
            torch.manual_seed(torch.initial_seed() + 1)
            self.fusions = nn.ModuleList(
                LevelFusion(fusion, channel, clip_slope_init=clip_slope_init)
                for channel in channels
            )

    def normalize(self, inputs: Tensor) -> Tensor:
        return normalize_inputs(inputs, self.input_mean, self.input_std)

    def fusion_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.fusions.parameters())

    def effective_fusion_parameter_count(self) -> int:
        """Fusion parameters that can receive gradient (the constant control's slope cannot)."""
        # Under the natural protocol every clipping slope is also dead because
        # s == 0 in training; report that case from the training protocol.
        dead = 0
        if self.fusion_rule == "clip_gate_constant":
            dead = sum(cast(nn.Parameter, fusion.clip_slope).numel() for fusion in self.fusions)
        return self.fusion_parameter_count() - dead

    @property
    def has_gate(self) -> bool:
        return self.fusion_rule not in ("concat", "mean")

    def forward(
        self,
        rgb: Tensor,
        event: Tensor,
        clipping: Tensor,
        *,
        clip_mode: ClipMode = "normal",
        gate_override: float | None = None,
        generator: torch.Generator | None = None,
        return_gates: bool = False,
    ) -> Tensor | tuple[Tensor, list[Tensor | None]]:
        """Return heatmaps for normalized ``rgb``/``event`` and a ``(B, G, G)`` clipping grid."""
        clipping = intervene_on_clipping(clipping, clip_mode, generator)
        rgb_features = cast(list[Tensor], self.rgb_encoder(rgb))
        event_features = cast(list[Tensor], self.event_encoder(event))
        fused: list[Tensor] = []
        gates: list[Tensor | None] = []
        for rgb_level, event_level, fusion in zip(
            rgb_features, event_features, self.fusions, strict=True
        ):
            level_clipping = clipping_at_level(
                clipping, cast(tuple[int, int], rgb_level.shape[-2:])
            )
            output, gate = fusion(rgb_level, event_level, level_clipping, gate_override)
            fused.append(output)
            gates.append(gate)
        heatmaps = cast(Tensor, self.head(fuse_pyramid(fused, self.lateral)))
        if return_gates:
            return heatmaps, gates
        return heatmaps
