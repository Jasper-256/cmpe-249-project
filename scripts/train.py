"""Train one unimodal or fusion heatmap variant on the crop-v2 synthetic caches.

Selection uses synthetic validation only (minimum keypoint RMSE under the run's
protocol). Real sequences are never loaded here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, cast

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import timm
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from saga_pose.data.cache import FreshCropCacheDataset
from saga_pose.data.dataset import FreshItem
from saga_pose.models.fusion import FUSION_RULES, FusionHeatmapRegressor
from saga_pose.training.engine import (
    ModelSpec,
    Protocol,
    build_model,
    evaluate,
    forward,
    heatmap_loss,
    protocol_inputs,
)

OBJECTS = ("cassini", "satty", "soho")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/fresh/cache/crop-v2"))
    parser.add_argument("--objects", nargs="+", default=list(OBJECTS))
    parser.add_argument("--modality", choices=("rgb", "event", "fusion"), required=True)
    parser.add_argument("--fusion", choices=FUSION_RULES, default=None)
    parser.add_argument("--backbone", default="hrnet_w18_small_v2.ms_in1k")
    parser.add_argument(
        "--event-representation", choices=("raw", "activity", "magnitude"), default="raw"
    )
    parser.add_argument("--protocol", choices=("natural", "tss"), default="natural")
    parser.add_argument("--clip-slope-init", type=float, default=5.0)
    parser.add_argument(
        "--tss-fraction", type=float, default=1.0, help="training-only TSS strength ablation"
    )
    parser.add_argument("--train-stride", type=int, default=1, help="keep every k-th train frame")
    parser.add_argument("--train-limit", type=int, default=None, help="evenly spaced subset size")
    parser.add_argument("--validation-limit", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--schedule", choices=("cosine", "constant"), default="cosine")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-improvement-px", type=float, default=0.05)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument(
        "--init-from",
        type=Path,
        default=None,
        help="fine-tuning: initialize all weights from a selected checkpoint (spec must match)",
    )
    parser.add_argument("--seed", type=int, default=249)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--overfit",
        type=int,
        default=0,
        help="Stage-B mode: N evenly spaced train samples, full-batch updates, no validation",
    )
    parser.add_argument("--overfit-steps", type=int, default=200)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/runs"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout.strip()


def _source_state_sha256(run_dir: Path) -> str:
    """Hash tracked diffs plus untracked source so dirty-tree runs stay reproducible.

    The exact patch is saved next to the report as ``source_state.patch``.
    """
    diff = _git("diff", "HEAD", "--", "src", "scripts", "configs", "tests", "pyproject.toml")
    untracked = _git(
        "ls-files", "--others", "--exclude-standard", "src", "scripts", "configs", "tests"
    )
    digest = hashlib.sha256(diff.encode())
    contents = [diff, "\n# untracked files\n"]
    for name in sorted(filter(None, untracked.splitlines())):
        payload = Path(name).read_bytes()
        digest.update(name.encode() + payload)
        contents.append(f"\n# --- {name}\n" + payload.decode("utf-8", errors="replace"))
    (run_dir / "source_state.patch").write_text("".join(contents), encoding="utf-8")
    return digest.hexdigest()


def _evenly(length: int, count: int | None) -> list[int]:
    if count is None or count >= length:
        return list(range(length))
    return np.linspace(0, length - 1, count, dtype=np.int64).tolist()


def _datasets(args: argparse.Namespace) -> tuple[Dataset[FreshItem], Dataset[FreshItem]]:
    train_parts: list[Dataset[FreshItem]] = []
    validation_parts: list[Dataset[FreshItem]] = []
    for object_name in args.objects:
        train = FreshCropCacheDataset(args.cache_dir / f"{object_name}-train")
        indices = list(range(0, len(train), args.train_stride))
        train_parts.append(Subset(train, indices))
        validation_parts.append(FreshCropCacheDataset(args.cache_dir / f"{object_name}-validation"))
    train_all: Dataset[FreshItem] = ConcatDataset(train_parts)
    validation_all: Dataset[FreshItem] = ConcatDataset(validation_parts)
    train_all = Subset(train_all, _evenly(len(train_all), args.train_limit))  # type: ignore[arg-type]
    validation_all = Subset(validation_all, _evenly(len(validation_all), args.validation_limit))  # type: ignore[arg-type]
    return train_all, validation_all


def _loader(
    dataset: Dataset[FreshItem], *, batch_size: int, shuffle: bool, seed: int, workers: int
) -> DataLoader[FreshItem]:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        generator=torch.Generator().manual_seed(seed),
    )


def _lr_lambda(schedule: str, warmup: int, total: int) -> Any:
    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        if schedule == "constant":
            return 1.0
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return factor


def _parameter_report(model: torch.nn.Module) -> dict[str, int]:
    report = {
        "total": sum(p.numel() for p in model.parameters()),
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    if isinstance(model, FusionHeatmapRegressor):
        report["fusion_specific"] = model.fusion_parameter_count()
        report["fusion_specific_effective"] = model.effective_fusion_parameter_count()
    return report


def main() -> int:
    args = parse_args()
    if not 0.0 < args.tss_fraction <= 1.0:
        raise ValueError("--tss-fraction must lie in (0, 1]")
    if args.tss_fraction < 1.0 and args.protocol != "tss":
        raise ValueError("--tss-fraction applies only to the tss protocol")
    spec = ModelSpec(
        modality=args.modality,
        backbone=args.backbone,
        fusion=args.fusion,
        event_representation=args.event_representation,
        clip_slope_init=args.clip_slope_init,
    )
    protocol = cast(Protocol, args.protocol)
    if not torch.cuda.is_available():
        raise RuntimeError("training requires CUDA")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

    if args.overfit:
        # Frozen Stage-B semantics: N samples evenly spaced over the full grouped
        # training membership, independent of --train-limit.
        args.train_limit = None
    train_set, validation_set = _datasets(args)
    if args.overfit:
        train_set = Subset(train_set, _evenly(len(train_set), args.overfit))  # type: ignore[arg-type]
    first = train_set[0]
    num_keypoints = int(first["keypoints_xy"].shape[0])
    model = build_model(
        spec, num_keypoints, pretrained=not args.no_pretrained and args.init_from is None
    ).to(device)
    if args.init_from is not None:
        initial_state = torch.load(args.init_from, map_location=device, weights_only=True)
        if initial_state["spec"] != spec.as_dict():
            raise ValueError(f"{args.init_from} was trained with a different model spec")
        model.load_state_dict(initial_state["model"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    batch_size = args.overfit or args.batch_size
    train_loader = _loader(
        train_set,
        batch_size=batch_size,
        shuffle=not args.overfit,
        seed=args.seed,
        workers=0 if args.overfit else args.workers,
    )
    validation_loader = _loader(
        validation_set, batch_size=32, shuffle=False, seed=args.seed, workers=args.workers
    )
    steps_per_epoch = len(train_loader)
    total_steps = args.overfit_steps if args.overfit else steps_per_epoch * args.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        _lr_lambda(
            "constant" if args.overfit else args.schedule,
            0 if args.overfit else args.warmup_steps,
            total_steps,
        ),
    )
    autocast_dtype = torch.bfloat16 if args.precision == "bf16" else None
    run_dir = args.output_dir / args.name
    checkpoint_path = args.checkpoint_dir / args.name / "best.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    # Provenance is captured before training so later edits cannot change it.
    source_provenance = {
        "git_commit": _git("rev-parse", "HEAD"),
        "working_tree_dirty": bool(_git("status", "--porcelain")),
        "source_state_sha256": _source_state_sha256(run_dir),
    }
    torch.cuda.reset_peak_memory_stats(device)

    def train_step(batch: FreshItem) -> float:
        model.train()
        inputs = protocol_inputs(batch, spec, protocol, tss_fraction=args.tss_fraction)
        targets = batch["keypoints_xy"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            "cuda", dtype=autocast_dtype or torch.float32, enabled=autocast_dtype is not None
        ):
            prediction = cast(torch.Tensor, forward(model, spec, inputs, device))
        loss = heatmap_loss(prediction.float(), targets)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite loss")
        loss.backward()
        optimizer.step()
        scheduler.step()
        return float(loss.detach())

    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    best_rmse, best_epoch, stale = float("inf"), 0, 0
    # Training state is persisted every epoch so a failure after training (for
    # example a transient cuDNN error during evaluation) can finish evaluation
    # from the saved best checkpoint without retraining.
    state_path = checkpoint_path.with_name("training_state.json")
    resumed_evaluation = False
    if not args.overfit and state_path.is_file() and checkpoint_path.is_file():
        saved = json.loads(state_path.read_text(encoding="utf-8"))
        if saved.get("training_complete") and saved.get("spec") == spec.as_dict():
            history = saved["history"]
            best_rmse, best_epoch = saved["best_rmse"], saved["best_epoch"]
            resumed_evaluation = True
            print(f"[{args.name}] training already complete; resuming evaluation", flush=True)

    # Mid-training resume: last.pt holds model, optimizer, scheduler, data-order
    # and RNG state after every epoch, so a crash costs one epoch, not a run.
    last_path = checkpoint_path.with_name("last.pt")
    resume_key = json.dumps(
        {
            key: str(value)
            for key, value in sorted(vars(args).items())
            if key not in ("workers", "output_dir", "checkpoint_dir")
        }
    )
    start_epoch = 1
    if not args.overfit and not resumed_evaluation and last_path.is_file():
        last = torch.load(last_path, map_location="cpu", weights_only=True)
        if last.get("resume_key") == resume_key:
            model.load_state_dict(last["model"])
            optimizer.load_state_dict(last["optimizer"])
            scheduler.load_state_dict(last["scheduler"])
            cast(torch.Generator, train_loader.generator).set_state(last["data_generator"])
            torch.set_rng_state(last["torch_rng"])
            torch.cuda.set_rng_state_all(last["cuda_rng"])
            history = json.loads(last["history_json"])
            best_rmse, best_epoch, stale = last["best_rmse"], last["best_epoch"], last["stale"]
            start_epoch = int(last["epoch"]) + 1
            print(f"[{args.name}] resuming training at epoch {start_epoch}", flush=True)

    def evaluate_with_retry(*positional: Any, **keyword: Any) -> dict[str, Any]:
        try:
            return evaluate(*positional, **keyword)
        except RuntimeError as error:
            # Transient driver/cuDNN failures under memory pressure: retry once.
            print(f"[{args.name}] evaluation error, retrying once: {error}", flush=True)
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
            return evaluate(*positional, **keyword)

    if args.overfit:
        batch = next(iter(train_loader))
        initial = evaluate(model, train_loader, spec, protocol, device, recover_pose=False)
        losses = [train_step(batch) for _ in range(args.overfit_steps)]
        final = evaluate(model, train_loader, spec, protocol, device, recover_pose=True)
        reduction = 1.0 - final["coordinate_rmse_px"] / initial["coordinate_rmse_px"]
        outcome: dict[str, Any] = {
            "stage": "B_tiny_overfit",
            "initial_rmse_px": initial["coordinate_rmse_px"],
            "final": final,
            "loss_first_last": [losses[0], losses[-1]],
            "rmse_reduction": reduction,
            "passes": final["coordinate_rmse_px"] <= 5.0 and reduction >= 0.9,
        }
    else:
        if args.init_from is not None and not resumed_evaluation and start_epoch == 1:
            # Fine-tuning: the starting checkpoint is epoch 0, so a fine-tune that
            # only makes the model worse can never be selected over it.
            initial = evaluate(model, validation_loader, spec, protocol, device, recover_pose=False)
            best_rmse, best_epoch = initial["coordinate_rmse_px"], 0
            torch.save(
                {"model": model.state_dict(), "spec": spec.as_dict(), "epoch": 0},
                checkpoint_path,
            )
            print(f"[{args.name}] epoch 0 (initial) val_rmse={best_rmse:.2f}px", flush=True)
        for epoch in range(start_epoch, 0 if resumed_evaluation else args.epochs + 1):
            epoch_started = time.perf_counter()
            losses = [train_step(batch) for batch in train_loader]
            validation = evaluate(
                model, validation_loader, spec, protocol, device, recover_pose=False
            )
            rmse = validation["coordinate_rmse_px"]
            improved = rmse <= best_rmse - args.min_improvement_px
            if improved:
                best_rmse, best_epoch, stale = rmse, epoch, 0
                temporary = checkpoint_path.with_suffix(".tmp")
                torch.save(
                    {"model": model.state_dict(), "spec": spec.as_dict(), "epoch": epoch},
                    temporary,
                )
                temporary.replace(checkpoint_path)
            else:
                stale += 1
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": float(np.mean(losses)),
                    "validation_rmse_px": rmse,
                    "gate_mean_clipped": validation["gate_mean_clipped_patches"],
                    "gate_mean_unclipped": validation["gate_mean_unclipped_patches"],
                    "learning_rate": scheduler.get_last_lr()[0],
                    "epoch_seconds": time.perf_counter() - epoch_started,
                    "checkpoint": improved,
                }
            )
            print(
                f"[{args.name}] epoch {epoch}/{args.epochs} loss={np.mean(losses):.6f} "
                f"val_rmse={rmse:.2f}px best={best_rmse:.2f}@{best_epoch} "
                f"({time.perf_counter() - epoch_started:.0f}s)",
                flush=True,
            )
            state_path.write_text(
                json.dumps(
                    {
                        "spec": spec.as_dict(),
                        "history": history,
                        "best_rmse": best_rmse,
                        "best_epoch": best_epoch,
                        "training_complete": False,
                    }
                ),
                encoding="utf-8",
            )
            temporary_last = last_path.with_suffix(".tmp")
            torch.save(
                {
                    "resume_key": resume_key,
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "data_generator": cast(torch.Generator, train_loader.generator).get_state(),
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all(),
                    "history_json": json.dumps(history),
                    "best_rmse": best_rmse,
                    "best_epoch": best_epoch,
                    "stale": stale,
                },
                temporary_last,
            )
            temporary_last.replace(last_path)
            if stale >= args.patience:
                break
        if not resumed_evaluation:
            state_path.write_text(
                json.dumps(
                    {
                        "spec": spec.as_dict(),
                        "history": history,
                        "best_rmse": best_rmse,
                        "best_epoch": best_epoch,
                        "training_complete": True,
                    }
                ),
                encoding="utf-8",
            )
        # Training is complete; the resume snapshot is no longer needed.
        last_path.unlink(missing_ok=True)
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
        evaluations: dict[str, Any] = {}
        for evaluation_protocol in ("natural", "tss"):
            key = f"validation_{evaluation_protocol}"
            evaluations[key] = evaluate_with_retry(
                model,
                validation_loader,
                spec,
                cast(Protocol, evaluation_protocol),
                device,
                keep_frames=evaluation_protocol == protocol,
            )
        frames = evaluations[f"validation_{protocol}"].pop("frames")
        np.savez_compressed(run_dir / "validation_frames.npz", **frames)
        if spec.modality == "fusion":
            # Interventions always use TSS validation: natural validation never
            # clips, so shuffling or zeroing its clipping input would test nothing.
            for mode in ("shuffle", "zero"):
                evaluations[f"validation_tss_clip_{mode}"] = evaluate_with_retry(
                    model, validation_loader, spec, "tss", device, clip_mode=mode
                )
            if cast(FusionHeatmapRegressor, model).has_gate:
                evaluations["validation_tss_gate_bypass_0.5"] = evaluate_with_retry(
                    model, validation_loader, spec, "tss", device, gate_override=0.5
                )
        outcome = {
            "stage": "train",
            "selected_epoch": best_epoch,
            "selected_validation_rmse_px": best_rmse,
            "epochs_completed": len(history),
            "resumed_evaluation": resumed_evaluation,
            "evaluations": evaluations,
        }

    elapsed = time.perf_counter() - started
    report = {
        "schema_version": 1,
        "name": args.name,
        "spec": spec.as_dict(),
        "protocol": protocol,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "data": {
            "train_samples": len(train_set),  # type: ignore[arg-type]
            "validation_samples": len(validation_set),  # type: ignore[arg-type]
            "train_steps_per_epoch": steps_per_epoch,
            "cache_manifests_sha256": {
                f"{name}-{role}": _sha256(args.cache_dir / f"{name}-{role}" / "manifest.json")
                for name in args.objects
                for role in ("train", "validation")
            },
        },
        "parameters": _parameter_report(model),
        "history": history,
        "outcome": outcome,
        "compute": {
            "elapsed_seconds": elapsed,
            "peak_vram_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "precision": args.precision,
            "gpu": torch.cuda.get_device_name(device),
        },
        "provenance": {
            **source_provenance,
            "uv_lock_sha256": _sha256(Path("uv.lock")),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "timm": timm.__version__,
            "checkpoint": str(checkpoint_path) if not args.overfit else None,
        },
    }
    (run_dir / "report.json").write_text(
        json.dumps(report, indent=2, default=float) + "\n", encoding="utf-8"
    )
    print(f"[{args.name}] report: {run_dir / 'report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
