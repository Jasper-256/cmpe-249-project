"""Generate the sim2real-v1 core matrix (configs/experiments/sim2real_v1.yaml).

Nine configurations x three per-object models, all glare-trained (TSS) with
the sim2real-v1 augmentation. Same backbone, data contract, optimizer, and
learning rate as main_v1; the epoch budget is doubled because augmented
training converges more slowly (budget chosen on synthetic validation only).
"""

from __future__ import annotations

from pathlib import Path

import yaml

OBJECTS = ("cassini", "satty", "soho")
CONFIGURATIONS: list[tuple[str, dict[str, str]]] = [
    ("s2r-u-rgb-tss", {"modality": "rgb"}),
    ("s2r-u-event-activity", {"modality": "event"}),
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
    # FRESH argues learned attention fusion overfits synthetic data; test it directly.
    ("s2r-f-attention-tss-activity", {"modality": "fusion", "fusion": "attention"}),
    ("s2r-f-attention_clip-tss-activity", {"modality": "fusion", "fusion": "attention_clip"}),
]


def main() -> int:
    runs = []
    # Object-major order: one object's full comparison finishes before the next starts.
    for object_name in OBJECTS:
        for name, entry in CONFIGURATIONS:
            runs.append({"name": f"{name}-{object_name}", **entry, "objects": [object_name]})
    matrix = {
        "description": "sim2real-v1 core matrix: glare-trained (TSS) per-object models with "
        "in-plane roll, scale/shift, photometric, blur, and event-noise augmentation. "
        "Checkpoints are selected on the deterministically augmented synthetic validation "
        "set. Real data are never used here.",
        "defaults": {
            "backbone": "hrnet_w18_small_v2.ms_in1k",
            "train_stride": 4,
            "epochs": 30,
            "batch_size": 16,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "schedule": "cosine",
            "warmup_steps": 300,
            "patience": 6,
            "min_improvement_px": 0.05,
            "protocol": "tss",
            "event_representation": "activity",
            "augment": "sim2real-v1",
            "seed": 249,
            "workers": 0,
            "output_dir": "outputs/runs/sim2real-hrnet",
            "checkpoint_dir": "checkpoints/sim2real-hrnet",
        },
        "runs": runs,
    }
    path = Path("configs/experiments/sim2real_v1.yaml")
    path.write_text(yaml.safe_dump(matrix, sort_keys=False), encoding="utf-8")
    print(f"{len(runs)} runs -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
