"""Generate the events-v1 matrix (configs/experiments/events_v1.yaml).

One-factor event-only arms against the round-2 reference ``s2r-u-event-activity``
(binary activity, sim2real-v1). Each arm changes exactly one thing; every other
setting is copied from sim2real_v1.yaml:

* ``support``: ``activity_support`` (8-neighbour background-activity filter);
* ``noise``: native-scale noise events (``sim2real-v1-native-noise``);
* ``drop``: event dropout up to 0.9 (``sim2real-v1-drop90``).

``magnitude`` and ``activity64`` are not trained: on synthetic data both maps
equal binary activity, so training is identical to the reference (verified on
2026-11-23: ev-mag and ev-thr64 seed 249 cassini reproduced the reference's
3.70 px validation RMSE exactly). They differ only on real frames, so they are
test-time representation swaps of the reference checkpoints.

The reference ``a0-activity`` is added for seeds 250 and 251 only; seed 249 is
``s2r-u-event-activity-<object>`` in outputs/runs/sim2real-hrnet. Runs are
seed-major, so every finished seed block is a complete replicate.
"""

from __future__ import annotations

from pathlib import Path

import yaml

OBJECTS = ("cassini", "satty", "soho")
SEEDS = (249, 250, 251)
REFERENCE: tuple[str, dict[str, str]] = ("a0-activity", {})
ARMS: list[tuple[str, dict[str, str]]] = [
    ("support", {"event_representation": "activity_support"}),
    ("noise", {"augment": "sim2real-v1-native-noise"}),
    ("drop", {"augment": "sim2real-v1-drop90"}),
]


def main() -> int:
    source = yaml.safe_load(Path("configs/experiments/sim2real_v1.yaml").read_text("utf-8"))
    defaults = dict(source["defaults"])
    defaults.update(
        {
            "supplement_dir": "data/fresh/cache/event-supp-v1",
            "output_dir": "outputs/runs/events-v1",
            "checkpoint_dir": "checkpoints/events-v1",
        }
    )
    runs = []
    for seed in SEEDS:
        families = ARMS if seed == 249 else [REFERENCE, *ARMS]
        for object_name in OBJECTS:
            for arm, entry in families:
                runs.append(
                    {
                        "name": f"ev-{arm}-s{seed}-{object_name}",
                        "modality": "event",
                        **entry,
                        "seed": seed,
                        "objects": [object_name],
                    }
                )
    matrix = {
        "description": "events-v1: one-factor event-only arms (representation, threshold, "
        "support filter, native-scale noise, strong dropout) against the sim2real-v1 "
        "activity reference. Every other setting is copied from sim2real_v1.yaml; "
        "checkpoints are selected on synthetic validation augmented with the run's own "
        "augmentation. Real data are never used here.",
        "defaults": defaults,
        "runs": runs,
    }
    out = Path("configs/experiments/events_v1.yaml")
    out.write_text(yaml.safe_dump(matrix, sort_keys=False, width=88), encoding="utf-8")
    print(f"{len(runs)} runs -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
