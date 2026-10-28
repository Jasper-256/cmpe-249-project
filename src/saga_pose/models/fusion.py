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

``attention``            coarse-level cross-modal attention (the unrestricted-attention
                         comparator) at the coarse levels
                         (stride >= 16): RGB and event tokens share one
                         transformer layer whose output is added as a residual on
                         top of mean fusion (zero-initialized, so training starts
                         at mean fusion); finer levels use mean fusion.
``attention_clip``       the original S1 design: the same attention, with a per-head
                         monotone clipping gate g = sigmoid(b - softplus(a) * s)
                         entering as key biases log(g) for RGB tokens and
                         log(1 - g) for event tokens. The gate reweights attention
                         only: the mean-fusion term still carries saturated RGB, so
                         S1 cannot fully exclude a modality (unlike LevelFusion).

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

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from saga_pose.models.heatmaps import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    create_timm_backbone,
    fuse_pyramid,
    normalize_inputs,
)

FusionRule = Literal[
    "mean",
    "concat",
    "content_gate",
    "clip_gate",
    "clip_gate_constant",
    "clip_gate_fixed",
    "content_clip_gate",
    "clip_gate_signed",
    "attention",
    "attention_clip",
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
    "attention",
    "attention_clip",
)
ATTENTION_RULES: Final[tuple[FusionRule, ...]] = ("attention", "attention_clip")
ATTENTION_MIN_STRIDE: Final[int] = 16
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


class TransformerBlock(nn.Module):
    """Pre-norm self-attention block with an additive per-key attention bias.

    Written explicitly on ``scaled_dot_product_attention``: PyTorch's
    ``TransformerEncoderLayer`` switches to a fused inference fast path in eval
    mode that mishandled the per-head float mask here (constant 129 px validation
    RMSE while training loss fell), so train and eval now share one code path.
    """

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.heads = heads
        self.norm_attention = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.projection = nn.Linear(width, width)
        self.norm_mlp = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width)
        )

    def forward(self, tokens: Tensor, key_bias: Tensor | None = None) -> Tensor:
        """``tokens`` is (B, L, C); ``key_bias`` is (B, heads, L) added to every query row."""
        batch, length, width = tokens.shape
        query, key, value = (
            self.qkv(self.norm_attention(tokens))
            .reshape(batch, length, 3, self.heads, width // self.heads)
            .permute(2, 0, 3, 1, 4)
        )
        mask = None if key_bias is None else key_bias[:, :, None, :].to(query.dtype)
        attended = F.scaled_dot_product_attention(query, key, value, attn_mask=mask)
        tokens = tokens + self.projection(attended.transpose(1, 2).reshape(batch, length, width))
        return tokens + cast(Tensor, self.mlp(self.norm_mlp(tokens)))


class AttentionLevelFusion(nn.Module):
    """Cross-modal transformer fusion at one coarse level (residual on mean fusion)."""

    def __init__(
        self,
        channels: int,
        tokens: int,
        *,
        use_clip: bool,
        width: int = 128,
        heads: int = 4,
        clip_slope_init: float = CLIP_SLOPE_INIT,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.use_clip = use_clip
        self.project_rgb = nn.Linear(channels, width)
        self.project_event = nn.Linear(channels, width)
        self.modality = nn.Parameter(torch.zeros(2, width))
        self.position = nn.Parameter(torch.zeros(tokens, width))
        nn.init.trunc_normal_(self.position, std=0.02)
        self.layer = TransformerBlock(width, heads)
        self.output = nn.Linear(width, channels)
        # Zero init: the level starts exactly as mean fusion.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.clip_slope: nn.Parameter | None = None
        self.clip_bias: nn.Parameter | None = None
        if use_clip:
            self.clip_slope = nn.Parameter(torch.full((heads,), _inverse_softplus(clip_slope_init)))
            self.clip_bias = nn.Parameter(torch.zeros(heads))

    def forward(
        self, rgb: Tensor, event: Tensor, clipping: Tensor, gate_override: float | None
    ) -> tuple[Tensor, Tensor | None]:
        batch, channels, height, width = rgb.shape
        tokens = height * width
        if tokens != self.position.shape[0]:
            raise ValueError(
                f"attention level expects {self.position.shape[0]} tokens, got {tokens}; "
                "the position embedding is sized for 256 x 256 inputs"
            )
        rgb_tokens = self.project_rgb(rgb.flatten(2).transpose(1, 2))
        event_tokens = self.project_event(event.flatten(2).transpose(1, 2))
        sequence = torch.cat(
            (
                rgb_tokens + self.modality[0] + self.position[:tokens],
                event_tokens + self.modality[1] + self.position[:tokens],
            ),
            dim=1,
        )
        key_bias: Tensor | None = None
        gate_map: Tensor | None = None
        if self.use_clip:
            assert self.clip_slope is not None and self.clip_bias is not None
            signal = clipping.flatten(1)  # (B, N)
            gate = torch.sigmoid(
                self.clip_bias[None, :, None]
                - F.softplus(self.clip_slope)[None, :, None] * signal[:, None, :]
            )  # (B, heads, N)
            if gate_override is not None:
                gate = torch.full_like(gate, gate_override)
            epsilon = 1e-4
            key_bias = torch.cat(
                (torch.log(gate + epsilon), torch.log(1.0 - gate + epsilon)), dim=2
            )  # (B, heads, 2N)
            gate_map = gate.mean(dim=1).reshape(batch, 1, height, width)
        mixed = self.layer(sequence, key_bias)
        update = self.output((mixed[:, :tokens] + mixed[:, tokens:]) / 2.0)
        update = update.transpose(1, 2).reshape(batch, channels, height, width)
        return 0.5 * (rgb + event) + update, gate_map


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
        self.rgb_encoder = create_timm_backbone(backbone_name, pretrained=pretrained, **kwargs)
        self.event_encoder = create_timm_backbone(backbone_name, pretrained=pretrained, **kwargs)
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
            reductions = tuple(cast(Any, self.rgb_encoder).feature_info.reduction())
            modules: list[nn.Module] = []
            for channel, reduction in zip(channels, reductions, strict=True):
                if fusion in ATTENTION_RULES and reduction >= ATTENTION_MIN_STRIDE:
                    side = 256 // reduction
                    modules.append(
                        AttentionLevelFusion(
                            channel,
                            side * side,
                            use_clip=fusion == "attention_clip",
                            clip_slope_init=clip_slope_init,
                        )
                    )
                elif fusion in ATTENTION_RULES:
                    modules.append(LevelFusion("mean", channel))
                else:
                    modules.append(LevelFusion(fusion, channel, clip_slope_init=clip_slope_init))
            self.fusions = nn.ModuleList(modules)

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
        return self.fusion_rule not in ("concat", "mean", "attention")

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
