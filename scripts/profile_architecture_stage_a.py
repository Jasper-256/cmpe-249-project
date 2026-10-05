"""Stage A static feasibility screen for locally supported architecture candidates."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import timm
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from saga_pose.data.cache import FreshCropCacheDataset
from saga_pose.data.dataset import FreshItem
from saga_pose.models.heatmaps import (
    SegFormerHeatmapRegressor,
    TimmPyramidHeatmapRegressor,
    gaussian_keypoint_heatmaps,
)

Modality = Literal["rgb", "event"]
ArchitectureModel = TimmPyramidHeatmapRegressor | SegFormerHeatmapRegressor
MAXIMUM_FINEST_STRIDE = 8
PEAK_VRAM_LIMIT_BYTES = int(7.2 * 2**30)
LicenseStatus = Literal["cleared", "cleared_noncommercial_research", "blocked"]


@dataclass(frozen=True)
class Candidate:
    id: str
    architecture: str
    family: str
    provider: Literal["timm", "transformers"]
    model_name: str
    pretraining: str
    license: str
    license_status: LicenseStatus
    scientific_question: str
    source_flops: str
    backbone_kwargs: dict[str, Any]


CANDIDATES = (
    Candidate(
        id="A0",
        architecture="HRNet-W18-Small-v2",
        family="high-resolution CNN",
        provider="timm",
        model_name="hrnet_w18_small_v2.ms_in1k",
        pretraining="supervised ImageNet-1k",
        license="MIT (timm model card)",
        license_status="cleared",
        scientific_question="verified high-resolution CNN reference",
        source_flops="2.42G at 224 for the official classifier",
        backbone_kwargs={},
    ),
    Candidate(
        id="A1",
        architecture="EfficientNet-B0",
        family="MBConv CNN",
        provider="timm",
        model_name="efficientnet_b0.ra_in1k",
        pretraining="supervised ImageNet-1k with timm RandAugment recipe",
        license="Apache-2.0 (timm)",
        license_status="cleared",
        scientific_question="can an efficient conventional CNN make attention unnecessary?",
        source_flops="0.39G at 224",
        backbone_kwargs={},
    ),
    Candidate(
        id="A2",
        architecture="TinyViT-5M",
        family="compact hierarchical transformer",
        provider="timm",
        model_name="tiny_vit_5m_224.dist_in22k_ft_in1k",
        pretraining="IN-22k CLIP distillation then IN-1k fine-tuning",
        license="MIT (official Microsoft repository)",
        license_status="cleared",
        scientific_question="parameter-envelope test of hierarchical token mixing",
        source_flops="1.3G MACs at 224",
        backbone_kwargs={},
    ),
    Candidate(
        id="A3",
        architecture="SegFormer MiT-B0",
        family="hierarchical dense-prediction transformer",
        provider="transformers",
        model_name="nvidia/mit-b0",
        pretraining="supervised ImageNet-1k",
        license="NVIDIA non-commercial research license",
        license_status="cleared_noncommercial_research",
        scientific_question="does a native dense-transformer hierarchy beat generic encoders?",
        source_flops="full SegFormer-B0 reports 8.4G at 512; encoder measured by latency",
        backbone_kwargs={},
    ),
    Candidate(
        id="A4",
        architecture="SwinV2-T window8-256",
        family="shifted-window hierarchical transformer",
        provider="timm",
        model_name="swinv2_tiny_window8_256.ms_in1k",
        pretraining="supervised ImageNet-1k at 256",
        license="MIT (official Microsoft repository)",
        license_status="cleared",
        scientific_question="upper-capacity control for mature window attention",
        source_flops="5.9G at 256",
        backbone_kwargs={},
    ),
    Candidate(
        id="A5",
        architecture="MobileViTv2-1.0",
        family="CNN-transformer hybrid",
        provider="timm",
        model_name="mobilevitv2_100.cvnets_in1k",
        pretraining="supervised ImageNet-1k",
        license="Apple sample-code license; subcomponent terms apply",
        license_status="cleared",
        scientific_question="does convolutional locality plus light global context win?",
        source_flops="about 1.8G at 256",
        backbone_kwargs={},
    ),
    Candidate(
        id="A6",
        architecture="EfficientViT-M2",
        family="cascaded-group-attention hybrid",
        provider="timm",
        model_name="efficientvit_m2.r224_in1k",
        pretraining="supervised ImageNet-1k",
        license="MIT (official Microsoft repository)",
        license_status="cleared",
        scientific_question="does very low nominal compute retain localization detail?",
        source_flops="about 0.20G at 224",
        backbone_kwargs={"img_size": 256},
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("data/fresh/cache/crop-v2/cassini-train"),
    )
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--timed-steps", type=int, default=10)
    parser.add_argument("--inference-warmup", type=int, default=10)
    parser.add_argument("--inference-steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=249)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/architecture/stage-a-profile.json"),
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _git_dirty() -> bool:
    completed = subprocess.run(
        ["git", "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    )
    return bool(completed.stdout.strip())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _synchronize(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def _distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def _model(candidate: Candidate, num_keypoints: int) -> ArchitectureModel:
    if candidate.provider == "transformers":
        return SegFormerHeatmapRegressor(
            num_keypoints,
            pretrained=True,
            decoder_channels=64,
            model_id=candidate.model_name,
        )
    return TimmPyramidHeatmapRegressor(
        num_keypoints,
        backbone_name=candidate.model_name,
        pretrained=True,
        decoder_channels=64,
        backbone_kwargs=candidate.backbone_kwargs,
    )


def _feature_shapes(
    model: ArchitectureModel,
    one_input: Tensor,
) -> list[list[int]]:
    model.eval()
    with torch.inference_mode():
        features = model.pyramid_features(one_input)
    return [list(feature.shape) for feature in features]


def _feature_strides(
    model: ArchitectureModel,
    shapes: list[list[int]],
) -> list[int]:
    strides: list[int] = []
    for shape, channels in zip(shapes, model.feature_channels, strict=True):
        if shape[1] == channels:
            spatial_width = shape[-1]
        elif shape[-1] == channels:
            spatial_width = shape[2]
        else:
            raise ValueError("feature shape does not match the declared channels")
        strides.append(256 // spatial_width)
    return strides


def _inference_profile(
    model: ArchitectureModel,
    one_input: Tensor,
    *,
    warmup: int,
    steps: int,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            model(one_input)
        _synchronize(device)
        timings: list[float] = []
        for _ in range(steps):
            started = time.perf_counter()
            output = model(one_input)
            _synchronize(device)
            timings.append(time.perf_counter() - started)
    return {
        "output_shape": list(output.shape),
        "batch_size": 1,
        "latency_seconds": _distribution(timings),
        "samples_per_second": 1.0 / float(np.mean(timings)),
    }


def _train_profile(
    candidate: Candidate,
    batch: FreshItem,
    modality: Modality,
    *,
    warmup: int,
    steps: int,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, Any], ArchitectureModel]:
    num_keypoints = batch["keypoints_xy"].shape[1]
    last_error: str | None = None
    for microbatch_size in (16, 8, 4, 2, 1):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        gc.collect()
        torch.cuda.empty_cache()
        try:
            model = _model(candidate, num_keypoints).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            inputs = model.normalize(batch[modality].to(device, non_blocking=True))
            target_points = batch["keypoints_xy"].to(device, non_blocking=True)
            targets, visibility = gaussian_keypoint_heatmaps(target_points)
            accumulation_steps = 16 // microbatch_size
            timings: list[float] = []
            losses: list[float] = []
            torch.cuda.reset_peak_memory_stats(device)
            model.train()
            for step in range(warmup + steps):
                _synchronize(device)
                started = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                total_loss = torch.zeros((), device=device)
                for offset in range(0, 16, microbatch_size):
                    end = offset + microbatch_size
                    prediction = model(inputs[offset:end])
                    local_targets = targets[offset:end]
                    local_weights = visibility[offset:end, :, None, None]
                    loss = torch.sum((prediction - local_targets).square() * local_weights) / (
                        local_weights.sum()
                        * prediction.shape[-2]
                        * prediction.shape[-1]
                        * accumulation_steps
                    )
                    if not torch.isfinite(loss):
                        raise RuntimeError("non-finite loss")
                    loss.backward()
                    total_loss += loss.detach()
                optimizer.step()
                _synchronize(device)
                elapsed = time.perf_counter() - started
                if step >= warmup:
                    timings.append(elapsed)
                    losses.append(float(total_loss))
            gradients_finite = all(
                parameter.grad is None or bool(torch.all(torch.isfinite(parameter.grad)))
                for parameter in model.parameters()
            )
            result = {
                "modality": modality,
                "effective_batch_size": 16,
                "microbatch_size": microbatch_size,
                "gradient_accumulation_steps": accumulation_steps,
                "warmup_steps": warmup,
                "timed_steps": steps,
                "optimizer": "AdamW(lr=1e-3, weight_decay=1e-4)",
                "finite_losses": bool(np.all(np.isfinite(losses))),
                "finite_gradients": gradients_finite,
                "first_loss": losses[0],
                "last_loss": losses[-1],
                "effective_step_seconds": _distribution(timings),
                "training_samples_per_second": 16 * steps / sum(timings),
                "peak_vram_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "peak_vram_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            }
            del optimizer, inputs, target_points, targets, visibility
            return result, model
        except torch.OutOfMemoryError as error:
            last_error = f"{type(error).__name__}: {error}"
            if "model" in locals():
                del model
            gc.collect()
            torch.cuda.empty_cache()
    raise RuntimeError(f"candidate cannot train with microbatch 1: {last_error}")


def _profile_candidate(
    candidate: Candidate,
    batch: FreshItem,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    result: dict[str, Any] = {"candidate": asdict(candidate), "status": "failed"}
    try:
        modality_results: list[dict[str, Any]] = []
        retained_model: ArchitectureModel | None = None
        for modality in cast(tuple[Modality, ...], ("rgb", "event")):
            profile, model = _train_profile(
                candidate,
                batch,
                modality,
                warmup=args.warmup_steps,
                steps=args.timed_steps,
                seed=args.seed,
                device=device,
            )
            modality_results.append(profile)
            if retained_model is not None:
                del retained_model
            retained_model = model

        assert retained_model is not None
        one_input = retained_model.normalize(batch["rgb"][:1].to(device))
        feature_shapes = _feature_shapes(retained_model, one_input)
        inference = _inference_profile(
            retained_model,
            one_input,
            warmup=args.inference_warmup,
            steps=args.inference_steps,
            device=device,
        )
        encoder_parameters = sum(p.numel() for p in retained_model.backbone.parameters())
        decoder_parameters = sum(p.numel() for p in retained_model.lateral.parameters()) + sum(
            p.numel() for p in retained_model.head.parameters()
        )
        total_parameters = sum(p.numel() for p in retained_model.parameters())
        if candidate.provider == "timm":
            pretrained_cfg = cast(dict[str, Any], retained_model.backbone.pretrained_cfg)
            pretrained_provenance = {
                "hf_hub_id": pretrained_cfg.get("hf_hub_id"),
                "url": pretrained_cfg.get("url"),
                "input_size": pretrained_cfg.get("input_size"),
                "mean": pretrained_cfg.get("mean"),
                "std": pretrained_cfg.get("std"),
            }
        else:
            pretrained_provenance = {
                "hf_hub_id": candidate.model_name,
                "url": f"https://huggingface.co/{candidate.model_name}",
                "input_size": [3, 256, 256],
                "mean": list(retained_model.input_mean),
                "std": list(retained_model.input_std),
            }
        pretrained_provenance["applied_normalization"] = {
            "mean": list(retained_model.input_mean),
            "std": list(retained_model.input_std),
        }
        result.update(
            {
                "status": "pass",
                "feature_shapes": feature_shapes,
                "feature_strides": _feature_strides(retained_model, feature_shapes),
                "parameters": {
                    "encoder": encoder_parameters,
                    "decoder": decoder_parameters,
                    "total": total_parameters,
                    "trainable": sum(
                        p.numel() for p in retained_model.parameters() if p.requires_grad
                    ),
                },
                "pretrained_provenance": pretrained_provenance,
                "inference": inference,
                "training": modality_results,
            }
        )
        strides = cast(list[int], result["feature_strides"])
        criteria = {
            "output_is_jx64x64": inference["output_shape"][1:]
            == [batch["keypoints_xy"].shape[1], 64, 64],
            "finite_losses": all(profile["finite_losses"] for profile in modality_results),
            "finite_gradients": all(profile["finite_gradients"] for profile in modality_results),
            "peak_vram_below_7_2_gib": all(
                profile["peak_vram_allocated_bytes"] < PEAK_VRAM_LIMIT_BYTES
                for profile in modality_results
            ),
            # The audit rejects packages whose finest feature cannot localize at
            # stride 4/8 without an undeclared upsampling neck.
            f"finest_feature_stride_at_most_{MAXIMUM_FINEST_STRIDE}": min(strides)
            <= MAXIMUM_FINEST_STRIDE,
            "license_and_weights_resolved": candidate.license_status != "blocked"
            and bool(pretrained_provenance.get("hf_hub_id")),
        }
        result["stage_a_criteria"] = criteria
        result["passes_stage_a"] = all(criteria.values())
        del retained_model, one_input
    except Exception as error:  # Preserve candidate-specific failures in the report.
        result["error"] = f"{type(error).__name__}: {error}"
        result["passes_stage_a"] = False
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> int:
    args = parse_args()
    if (
        min(
            args.warmup_steps,
            args.timed_steps,
            args.inference_warmup,
            args.inference_steps,
        )
        <= 0
    ):
        raise ValueError("all warm-up and timing counts must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("Stage A profiling requires CUDA")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda")
    dataset = FreshCropCacheDataset(args.cache_root)
    loader: DataLoader[FreshItem] = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    batch = next(iter(loader))
    results: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        print(f"Profiling {candidate.id} {candidate.architecture}...", flush=True)
        result = _profile_candidate(candidate, batch, args, device)
        results.append(result)
        if result["status"] == "pass":
            training = cast(list[dict[str, Any]], result["training"])
            print(
                f"  pass={result['passes_stage_a']} params={result['parameters']['total']:,} "
                f"rgb={training[0]['training_samples_per_second']:.1f}/s "
                f"event={training[1]['training_samples_per_second']:.1f}/s",
                flush=True,
            )
        else:
            print(f"  failed: {result['error']}", flush=True)

    source_paths = (
        Path("scripts/profile_architecture_stage_a.py"),
        Path("src/saga_pose/models/heatmaps.py"),
        Path("src/saga_pose/data/cache.py"),
    )
    report = {
        "schema_version": 1,
        "status": "complete",
        "scope": "Stage A static/compute feasibility; not a convergence or accuracy result",
        "decision_rule": {
            "required": [
                "finite forward/loss/backward/update for RGB and event",
                "Jx64x64 output through the shared decoder",
                "effective batch 16 with peak allocated VRAM below 7.2 GiB",
                f"finest feature stride at most {MAXIMUM_FINEST_STRIDE}",
                "resolved weights and license provenance (non-commercial research allowed)",
                "inputs normalized with each package's own pretraining statistics",
            ],
            "stage_b_is_separate": True,
        },
        "config": {
            "dataset": "fresh-crop-v1 Cassini train shard",
            "samples_consumed_per_step": 16,
            "precision": "float32",
            "warmup_steps": args.warmup_steps,
            "timed_steps": args.timed_steps,
            "inference_warmup": args.inference_warmup,
            "inference_steps": args.inference_steps,
            "seed": args.seed,
            "target_output": "18x64x64 Gaussian heatmaps",
        },
        "provenance": {
            "git_commit": _git_commit(),
            "working_tree_dirty": _git_dirty(),
            "uv_lock_sha256": _sha256(Path("uv.lock")),
            "cache_manifest_sha256": _sha256(args.cache_root / "manifest.json"),
            "source_sha256": {str(path): _sha256(path) for path in source_paths},
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "timm": timm.__version__,
            "gpu": torch.cuda.get_device_name(device),
            "gpu_total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
        },
        "results": results,
    }
    _write_json(args.output, report)
    print(f"report: {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
