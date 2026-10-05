"""Generate the main experiment matrix YAML (priority-ordered tiers)."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

RULES = (
    "mean",
    "concat",
    "content_gate",
    "clip_gate",
    "clip_gate_constant",
    "clip_gate_fixed",
    "content_clip_gate",
)
OBJECTS = ("cassini", "satty", "soho")
CENTRAL = ("content_gate", "clip_gate", "clip_gate_constant", "content_clip_gate")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", default="hrnet_w18_small_v2.ms_in1k")
    parser.add_argument("--tag", default="hrnet")
    parser.add_argument("--train-stride", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--output", type=Path, default=Path("configs/experiments/main_v1.yaml"))
    args = parser.parse_args()
    runs: list[dict[str, object]] = []
    # Tier 1: unimodal references, then every fusion rule under both protocols.
    # Under the natural protocol clip_gate and clip_gate_constant train identically
    # (s == 0); their difference is reported as the nondeterminism noise floor.
    for protocol in ("natural", "tss"):
        runs.append({"name": f"u-rgb-{protocol}", "modality": "rgb", "protocol": protocol})
    for representation in ("raw", "activity"):
        runs.append(
            {
                "name": f"u-event-{representation}",
                "modality": "event",
                "event_representation": representation,
            }
        )
    # Central TSS rules first (they answer the pre-registered primary question),
    # then the remaining TSS rules, then the natural protocol.
    ordered_rules = [
        ("tss", rule)
        for rule in ("content_gate", "content_clip_gate", "clip_gate", "clip_gate_constant", "mean")
    ]
    ordered_rules += [("tss", rule) for rule in RULES if ("tss", rule) not in ordered_rules]
    ordered_rules += [("natural", rule) for rule in RULES]
    for protocol, rule in ordered_rules:
        if rule == "clip_gate_signed":
            continue
        runs.append(
            {
                "name": f"f-{rule}-{protocol}-activity",
                "modality": "fusion",
                "fusion": rule,
                "protocol": protocol,
                "event_representation": "activity",
            }
        )
    # Tier 2a: gate-design ablations under TSS (natural would give these no gradient).
    runs.append(
        {
            "name": "f-clip_gate_signed-tss-activity",
            "modality": "fusion",
            "fusion": "clip_gate_signed",
            "protocol": "tss",
        }
    )
    for rule in ("clip_gate", "content_clip_gate"):
        runs.append(
            {
                "name": f"f-{rule}-tss-activity-slope0.1",
                "modality": "fusion",
                "fusion": rule,
                "protocol": "tss",
                "clip_slope_init": 0.1,
            }
        )
    # Tier 2b: event-representation ablation inside fusion.
    for protocol in ("natural", "tss"):
        for rule in ("mean", "content_gate", "content_clip_gate"):
            runs.append(
                {
                    "name": f"f-{rule}-{protocol}-raw",
                    "modality": "fusion",
                    "fusion": rule,
                    "protocol": protocol,
                    "event_representation": "raw",
                }
            )
    # Tier 2c: TSS strength sweep (training-only; effective corruption 12.5%).
    for rule in ("content_gate", "content_clip_gate"):
        runs.append(
            {
                "name": f"f-{rule}-tss0.25-activity",
                "modality": "fusion",
                "fusion": rule,
                "protocol": "tss",
                "tss_fraction": 0.25,
            }
        )
    # Tier 3: additional seeds for the central TSS comparison.
    for seed in (250, 251):
        for rule in CENTRAL:
            runs.append(
                {
                    "name": f"f-{rule}-tss-activity-s{seed}",
                    "modality": "fusion",
                    "fusion": rule,
                    "protocol": "tss",
                    "event_representation": "activity",
                    "seed": seed,
                }
            )
    # Tier 4 (audit addition): seeds for the remaining pre-registered TSS baselines.
    for seed in (250, 251):
        for rule in ("mean", "clip_gate_fixed"):
            runs.append(
                {
                    "name": f"f-{rule}-tss-activity-s{seed}",
                    "modality": "fusion",
                    "fusion": rule,
                    "protocol": "tss",
                    "seed": seed,
                }
            )
        runs.append(
            {"name": f"u-rgb-tss-s{seed}", "modality": "rgb", "protocol": "tss", "seed": seed}
        )
    # FRESH trains one model per satellite (objects differ in landmark count), so
    # every method is three per-object runs; analyses pool them into 24 sequences.
    runs = [
        {**run, "name": f"{run['name']}-{object_name}", "objects": [object_name]}
        for run in runs
        for object_name in OBJECTS
    ]
    document = {
        "description": (
            "Main fusion matrix v1: per-object models for all three synthetic objects, "
            "grouped-v1 split, crop-v2 "
            "cache, synthetic-validation checkpoint selection under each run's protocol. "
            "Real data are never used here."
        ),
        "defaults": {
            "backbone": args.backbone,
            "train_stride": args.train_stride,
            "epochs": args.epochs,
            "batch_size": 16,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "schedule": "cosine",
            "warmup_steps": 300,
            "patience": 4,
            "min_improvement_px": 0.05,
            "protocol": "natural",
            "event_representation": "activity",
            "seed": 249,
            "workers": 0,
            "output_dir": f"outputs/runs/main-{args.tag}",
        },
        "runs": runs,
    }
    args.output.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    print(f"{len(runs)} runs -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
