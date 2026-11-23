"""Generate the trust-v1 matrix (configs/experiments/trust_v1.yaml).

Track B of the post-round-2 plan, trained on synthetic data only with the
frozen sim2real-v1 protocol (same backbone, data, augmentation, optimizer and
budget as round 2):

* seeds: two more seeds (250, 251) for the eight core round-2 configurations,
  so round-2 contrasts can be reported with seed variation;
* dose-response: mean fusion with a fixed event weight w in
  {0, 0.25, 0.5, 0.75, 1}, three seeds, to test whether real-data degradation
  grows with reliance on the simulated event stream;
* decoupled clipping: clipping attenuates RGB only (no event contribution),
  attenuates RGB with a fixed event weight of 0.5, and a shuffled-clipping
  control, three seeds.

Runs are ordered seed-major (seed 249 of the new families first), so every
finished block is a complete replicate if compute runs out.
"""

from __future__ import annotations

from pathlib import Path

import yaml

OBJECTS = ("cassini", "satty", "soho")
CORE: list[tuple[str, dict[str, str]]] = [
    ("s2r-u-rgb-tss", {"modality": "rgb"}),
    ("s2r-f-mean-tss-activity", {"modality": "fusion", "fusion": "mean"}),
    ("s2r-f-content_gate-tss-activity", {"modality": "fusion", "fusion": "content_gate"}),
    (
        "s2r-f-content_clip_gate-tss-activity",
        {"modality": "fusion", "fusion": "content_clip_gate"},
    ),
    ("s2r-f-clip_gate-tss-activity", {"modality": "fusion", "fusion": "clip_gate"}),
    (
        "s2r-f-clip_gate_constant-tss-activity",
        {"modality": "fusion", "fusion": "clip_gate_constant"},
    ),
    ("s2r-f-attention-tss-activity", {"modality": "fusion", "fusion": "attention"}),
    ("s2r-f-attention_clip-tss-activity", {"modality": "fusion", "fusion": "attention_clip"}),
]
DOSE = [
    (f"tr-f-fixed_w{w}-tss-activity", {"modality": "fusion", "fusion": f"fixed_w{w}"})
    for w in ("000", "025", "050", "075", "100")
]
DECOUPLED = [
    (f"tr-f-{rule}-tss-activity", {"modality": "fusion", "fusion": rule})
    for rule in ("clip_rgb_only", "clip_rgb_event", "clip_rgb_only_shuffled")
]
SEEDS = (249, 250, 251)


def main() -> int:
    runs = []
    for seed in SEEDS:
        families = DOSE + DECOUPLED if seed == 249 else CORE + DOSE + DECOUPLED
        for object_name in OBJECTS:
            for name, entry in families:
                runs.append(
                    {
                        "name": f"{name}-s{seed}-{object_name}",
                        **entry,
                        "seed": seed,
                        "objects": [object_name],
                    }
                )
    matrix = {
        "description": "trust-v1 (track B): extra seeds for the round-2 core, a fixed "
        "event-weight dose-response, and decoupled clipping arms. sim2real-v1 protocol; "
        "checkpoints selected on augmented synthetic validation. Real data are never "
        "used here.",
        "defaults": {
            "backbone": "hrnet_w18_small_v2.ms_in1k",
            "train_stride": 1,
            "epochs": 8,
            "batch_size": 16,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "schedule": "cosine",
            "warmup_steps": 300,
            "patience": 3,
            "min_improvement_px": 0.05,
            "protocol": "tss",
            "event_representation": "activity",
            "augment": "sim2real-v1",
            "seed": 249,
            "workers": 0,
            "output_dir": "outputs/runs/trust-v1",
            "checkpoint_dir": "checkpoints/trust-v1",
        },
        "runs": runs,
    }
    out = Path("configs/experiments/trust_v1.yaml")
    out.write_text(yaml.safe_dump(matrix, sort_keys=False, width=88), encoding="utf-8")
    print(f"{len(runs)} runs -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
