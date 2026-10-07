"""One input, loss, and evaluation path for unimodal and fusion heatmap models.

Every variant in the experiment matrix goes through these functions, so a
difference between runs cannot come from plumbing (normalization, protocol
input selection, decoding, PnP failure policy, or metric definitions).
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any, Literal, cast

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from saga_pose.data.dataset import FreshItem, event_input
from saga_pose.data.representations import EventRepresentation
from saga_pose.evaluation.pose import pose_error_arrays, recover_poses, summarize_pose_errors
from saga_pose.models.fusion import ClipMode, FusionHeatmapRegressor, FusionRule
from saga_pose.models.heatmaps import (
    SegFormerHeatmapRegressor,
    TimmPyramidHeatmapRegressor,
    decode_heatmap_argmax,
    gaussian_keypoint_heatmaps,
)

Protocol = Literal["natural", "tss"]
Modality = Literal["rgb", "event", "fusion"]


@dataclass(frozen=True)
class ModelSpec:
    """Everything that defines a model variant (serialized into every report)."""

    modality: Modality
    backbone: str = "hrnet_w18_small_v2.ms_in1k"
    fusion: FusionRule | None = None
    event_representation: EventRepresentation = "raw"
    decoder_channels: int = 64
    clip_slope_init: float = 5.0

    def __post_init__(self) -> None:
        if (self.modality == "fusion") != (self.fusion is not None):
            raise ValueError("fusion rule is required exactly when modality == 'fusion'")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


HeatmapModel = TimmPyramidHeatmapRegressor | SegFormerHeatmapRegressor | FusionHeatmapRegressor
HF_PREFIX = "hf:"


def build_model(spec: ModelSpec, num_keypoints: int, *, pretrained: bool) -> HeatmapModel:
    if spec.modality == "fusion":
        assert spec.fusion is not None
        if spec.backbone.startswith(HF_PREFIX):
            raise ValueError("fusion currently supports timm backbones only")
        return FusionHeatmapRegressor(
            num_keypoints,
            backbone_name=spec.backbone,
            fusion=spec.fusion,
            pretrained=pretrained,
            decoder_channels=spec.decoder_channels,
            clip_slope_init=spec.clip_slope_init,
        )
    if spec.backbone.startswith(HF_PREFIX):
        return SegFormerHeatmapRegressor(
            num_keypoints,
            pretrained=pretrained,
            decoder_channels=spec.decoder_channels,
            model_id=spec.backbone.removeprefix(HF_PREFIX),
        )
    return TimmPyramidHeatmapRegressor(
        num_keypoints,
        backbone_name=spec.backbone,
        pretrained=pretrained,
        decoder_channels=spec.decoder_channels,
    )


def tss_keep_mask(sample_ids: list[str], fraction: float, seed: int = 249) -> Tensor:
    """Deterministically keep TSS corruption for ``fraction`` of samples (by id hash)."""
    values = [
        int.from_bytes(
            hashlib.sha256(f"tss-fraction:{seed}:{sample}".encode()).digest()[:8], "little"
        )
        / 2**64
        for sample in sample_ids
    ]
    return torch.tensor(values) < fraction


def protocol_inputs(
    batch: FreshItem | dict[str, Any],
    spec: ModelSpec,
    protocol: Protocol,
    *,
    tss_fraction: float = 1.0,
) -> dict[str, Tensor]:
    """Select unnormalized protocol inputs: RGB (natural or TSS), event map, clipping grid.

    ``tss_fraction < 1`` (training-only ablation) reverts a deterministic subset of
    TSS-corrupted samples to their natural RGB and clipping grid, lowering the
    effective corruption rate to ``0.5 * tss_fraction`` under TSS-v1.
    """
    data = cast(dict[str, Any], batch)
    rgb_key, clip_key = (
        ("rgb_tss", "clipping_ratio_tss") if protocol == "tss" else ("rgb", "clipping_ratio")
    )
    rgb = data[rgb_key]
    clipping = data[clip_key]
    if protocol == "tss" and tss_fraction < 1.0:
        keep = tss_keep_mask(list(data["sample_id"]), tss_fraction)
        rgb = torch.where(keep[:, None, None, None], rgb, data["rgb"])
        clipping = torch.where(keep[:, None, None], clipping, data["clipping_ratio"])
    return {
        "rgb": rgb,
        "event": event_input(data, spec.event_representation),
        "clipping": clipping,
    }


def forward(
    model: HeatmapModel,
    spec: ModelSpec,
    inputs: dict[str, Tensor],
    device: torch.device,
    *,
    clip_mode: ClipMode = "normal",
    gate_override: float | None = None,
    generator: torch.Generator | None = None,
    return_gates: bool = False,
) -> Tensor | tuple[Tensor, list[Tensor | None]]:
    """Normalize with the backbone's own statistics and run the model."""
    if spec.modality == "fusion":
        fusion_model = cast(FusionHeatmapRegressor, model)
        output: Tensor | tuple[Tensor, list[Tensor | None]] = fusion_model(
            fusion_model.normalize(inputs["rgb"].to(device, non_blocking=True)),
            fusion_model.normalize(inputs["event"].to(device, non_blocking=True)),
            inputs["clipping"].to(device, non_blocking=True),
            clip_mode=clip_mode,
            gate_override=gate_override,
            generator=generator,
            return_gates=return_gates,
        )
        return output
    single = cast(TimmPyramidHeatmapRegressor | SegFormerHeatmapRegressor, model)
    heatmaps = single(single.normalize(inputs[spec.modality].to(device, non_blocking=True)))
    return (heatmaps, []) if return_gates else heatmaps


def heatmap_loss(prediction: Tensor, keypoints_xy: Tensor) -> Tensor:
    """Visibility-weighted mean squared heatmap error (the frozen HRNet objective)."""
    targets, visibility = gaussian_keypoint_heatmaps(keypoints_xy)
    weights = visibility[:, :, None, None]
    return torch.sum((prediction - targets).square() * weights) / (
        weights.sum().clamp_min(1.0) * prediction.shape[-2] * prediction.shape[-1]
    )


@torch.inference_mode()
def evaluate(
    model: HeatmapModel,
    loader: DataLoader[FreshItem],
    spec: ModelSpec,
    protocol: Protocol,
    device: torch.device,
    *,
    recover_pose: bool = True,
    clip_mode: ClipMode = "normal",
    gate_override: float | None = None,
    seed: int = 249,
    keep_frames: bool = False,
) -> dict[str, Any]:
    """Keypoint RMSE, pose metrics (PnP failures included), and gate diagnostics."""
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    squared_error = 0.0
    coordinate_count = 0
    frame_count = 0
    keypoints: list[np.ndarray] = []
    landmarks: list[np.ndarray] = []
    intrinsics: list[np.ndarray] = []
    ground_truth: list[np.ndarray] = []
    sample_ids: list[str] = []
    sequences: list[str] = []
    frame_numbers: list[np.ndarray] = []
    gate_clipped_sum = gate_clipped_count = gate_clean_sum = gate_clean_count = 0.0
    for batch in loader:
        inputs = protocol_inputs(batch, spec, protocol)
        heatmaps, gates = cast(
            tuple[Tensor, list[Tensor | None]],
            forward(
                model,
                spec,
                inputs,
                device,
                clip_mode=clip_mode,
                gate_override=gate_override,
                generator=generator,
                return_gates=True,
            ),
        )
        predicted, _ = decode_heatmap_argmax(heatmaps.float())
        predicted = predicted.cpu()
        targets = batch["keypoints_xy"]
        squared_error += float(torch.sum((predicted - targets) ** 2))
        coordinate_count += targets.numel()
        frame_count += targets.shape[0]
        # Gate response at the 16x16 (stride-16) level, where one feature cell is one patch.
        for gate in gates:
            if gate is None or gate.shape[-1] != inputs["clipping"].shape[-1]:
                continue
            mean_gate = gate.float().mean(dim=1).cpu()
            clipped = inputs["clipping"] >= 0.01
            gate_clipped_sum += float(mean_gate[clipped].sum())
            gate_clipped_count += float(clipped.sum())
            gate_clean_sum += float(mean_gate[~clipped].sum())
            gate_clean_count += float((~clipped).sum())
        if recover_pose or keep_frames:
            keypoints.append(predicted.double().numpy())
            landmarks.append(batch["landmarks_object_mm"].double().numpy())
            intrinsics.append(batch["intrinsics"].double().numpy())
            ground_truth.append(batch["pose_object_to_camera_mm"].double().numpy())
            sample_ids.extend(batch["sample_id"])
            sequences.extend(batch["sequence"])
            frame_numbers.append(np.asarray(batch["frame_number"]))
    result: dict[str, Any] = {
        "frame_count": frame_count,
        "coordinate_rmse_px": float(np.sqrt(squared_error / coordinate_count)),
        "protocol": protocol,
        "clip_mode": clip_mode,
        "gate_override": gate_override,
        "gate_mean_clipped_patches": gate_clipped_sum / gate_clipped_count
        if gate_clipped_count
        else None,
        "gate_mean_unclipped_patches": gate_clean_sum / gate_clean_count
        if gate_clean_count
        else None,
    }
    if not (recover_pose or keep_frames):
        return result
    stacked_keypoints = np.concatenate(keypoints)
    poses, valid, reprojection = recover_poses(
        stacked_keypoints, np.concatenate(landmarks), np.concatenate(intrinsics)
    )
    stacked_truth = np.concatenate(ground_truth)
    result["pose"] = summarize_pose_errors(stacked_truth, poses, valid, reprojection)
    if keep_frames:
        errors = pose_error_arrays(stacked_truth, poses)
        result["frames"] = {
            "sample_id": sample_ids,
            "sequence": sequences,
            "frame_number": np.concatenate(frame_numbers),
            "pnp_valid": valid,
            "rotation_error_deg": errors["rotation_error_deg"],
            "translation_error_m": errors["translation_error_m"],
            "predicted_keypoints_xy": stacked_keypoints.astype(np.float32),
            "predicted_pose_mm": poses.astype(np.float32),
        }
    return result
