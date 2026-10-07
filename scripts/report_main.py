"""Write a Markdown results table for the main matrix (synthetic validation only).

Each method's three per-object runs are pooled frame-weighted. Columns report the
run's own-protocol and cross-protocol validation, interventions on TSS
validation, gate response, parameters, and a flag when any object selected its
last epoch (possible truncation). Incomplete methods are listed separately.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

OBJECTS = ("cassini", "satty", "soho")


def _pooled(runs: list[dict[str, Any]], key: str, field: str) -> float | None:
    total = weight = 0.0
    for report in runs:
        evaluation = report["outcome"]["evaluations"].get(key)
        if evaluation is None:
            return None
        frames = evaluation["frame_count"]
        if field == "rmse":
            value = evaluation["coordinate_rmse_px"]
        elif field in ("gate_clipped", "gate_unclipped"):
            value = evaluation[f"gate_mean_{field.split('_')[1]}_patches"]
            if value is None:
                return None
        else:
            value = evaluation["pose"]["all_frames"][field]
        total += value * frames
        weight += frames
    return total / weight


def _fmt(value: float | None, digits: int = 2) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for path in sorted(args.run_root.glob("*/report.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        name = report["name"]
        for object_name in OBJECTS:
            if name.endswith(f"-{object_name}"):
                grouped[name[: -len(object_name) - 1]][object_name] = report
    complete = {m: r for m, r in grouped.items() if len(r) == len(OBJECTS)}
    partial = sorted(set(grouped) - set(complete))
    lines = [
        f"# Main matrix — synthetic validation ({len(complete)} complete methods)",
        "",
        "Pooled over the three per-object models (frame-weighted means; PnP failures",
        "counted with the FRESH default pose). `own` = validation under the training",
        "protocol; `nat`/`tss` = natural / TSS-corrupted validation. Interventions are",
        "keypoint RMSE (px) on TSS validation. Gate = mean RGB trust at the stride-16",
        "level on TSS validation (clipped / unclipped patches). Real data not used.",
        "",
        "| method | protocol | nat RMSE | nat ARE° | nat ATE m | nat 10°/10mm | tss RMSE "
        "| tss ARE° "
        "| tss ATE m | tss 10°/10mm | shuffle | zero | bypass | gate clip/clean | fusion params "
        "| last-epoch? |",
        "|" + "---|" * 17,
    ]
    for method in sorted(complete, key=lambda m: (complete[m]["cassini"]["protocol"], m)):
        runs = [complete[method][o] for o in OBJECTS]
        first = runs[0]
        gate_clipped = _pooled(runs, "validation_tss", "gate_clipped")
        gate_clean = _pooled(runs, "validation_tss", "gate_unclipped")
        last = any(r["outcome"]["selected_epoch"] == r["config"]["epochs"] for r in runs)
        # Spread of the last five per-epoch validation RMSEs, averaged over objects:
        # a large value means the selected checkpoint is a noisy draw.
        noise = sum(
            float(np.std([h["validation_rmse_px"] for h in r["history"][-5:]])) for r in runs
        ) / len(runs)
        lines.append(
            "| "
            + " | ".join(
                [
                    method,
                    first["protocol"],
                    _fmt(_pooled(runs, "validation_natural", "rmse")),
                    _fmt(_pooled(runs, "validation_natural", "are_degrees_mean")),
                    _fmt(
                        _pooled(runs, "validation_natural", "absolute_translation_error_m_mean"), 4
                    ),
                    _fmt(_pooled(runs, "validation_natural", "success_joint_10_deg_10_mm"), 3),
                    _fmt(_pooled(runs, "validation_tss", "rmse")),
                    _fmt(_pooled(runs, "validation_tss", "are_degrees_mean")),
                    _fmt(_pooled(runs, "validation_tss", "absolute_translation_error_m_mean"), 4),
                    _fmt(_pooled(runs, "validation_tss", "success_joint_10_deg_10_mm"), 3),
                    _fmt(_pooled(runs, "validation_tss_clip_shuffle", "rmse")),
                    _fmt(_pooled(runs, "validation_tss_clip_zero", "rmse")),
                    _fmt(_pooled(runs, "validation_tss_gate_bypass_0.5", "rmse")),
                    f"{_fmt(gate_clipped, 3)} / {_fmt(gate_clean, 3)}",
                    str(first["parameters"].get("fusion_specific", "—")),
                    "yes" if last else "no",
                    f"{noise:.2f}",
                ]
            )
            + " |"
        )
    if partial:
        lines += ["", "In progress: " + ", ".join(partial)]
    output = args.output or args.run_root / "results.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
