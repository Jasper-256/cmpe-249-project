"""Tabulate synthetic-validation results of completed runs (CSV + Markdown)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output", type=Path, default=None, help="defaults to <run_root>/summary")
    return parser.parse_args()


def _row(report: dict[str, Any]) -> dict[str, Any]:
    spec = report["spec"]
    outcome = report["outcome"]
    row: dict[str, Any] = {
        "name": report["name"],
        "modality": spec["modality"],
        "fusion": spec.get("fusion") or "",
        "backbone": spec["backbone"],
        "event_repr": spec["event_representation"],
        "protocol": report["protocol"],
        "seed": report["config"]["seed"],
        "params_total": report["parameters"]["total"],
        "params_fusion": report["parameters"].get("fusion_specific", ""),
        "peak_vram_gib": round(report["compute"]["peak_vram_allocated_bytes"] / 2**30, 2),
        "minutes": round(report["compute"]["elapsed_seconds"] / 60, 1),
    }
    if outcome["stage"] == "B_tiny_overfit":
        row.update(
            {
                "stage": "B",
                "rmse_px": round(outcome["final"]["coordinate_rmse_px"], 2),
                "initial_rmse_px": round(outcome["initial_rmse_px"], 2),
                "passes": outcome["passes"],
            }
        )
        return row
    row["stage"] = "train"
    row["selected_epoch"] = outcome["selected_epoch"]
    row["epochs"] = outcome["epochs_completed"]
    # Selection at the final epoch means the budget may have truncated training.
    row["selected_is_last"] = outcome["selected_epoch"] == report["config"]["epochs"]
    for key, evaluation in outcome["evaluations"].items():
        short = key.replace("validation_", "")
        row[f"{short}:rmse_px"] = round(evaluation["coordinate_rmse_px"], 2)
        pose = evaluation.get("pose")
        if pose:
            frames = pose["all_frames"]
            row[f"{short}:are_med_deg"] = round(frames["are_degrees_median"], 3)
            row[f"{short}:ate_med_m"] = round(frames["absolute_translation_error_m_median"], 4)
            row[f"{short}:succ_10deg_10mm"] = round(frames["success_joint_10_deg_10_mm"], 3)
            row[f"{short}:pnp_valid"] = round(pose["pnp_valid_fraction"], 3)
        if evaluation.get("gate_mean_clipped_patches") is not None:
            row[f"{short}:gate_clipped"] = round(evaluation["gate_mean_clipped_patches"], 3)
            row[f"{short}:gate_unclipped"] = round(evaluation["gate_mean_unclipped_patches"], 3)
    return row


def main() -> int:
    args = parse_args()
    rows = [
        _row(json.loads(path.read_text(encoding="utf-8")))
        for path in sorted(args.run_root.glob("*/report.json"))
    ]
    if not rows:
        raise SystemExit(f"no reports under {args.run_root}")
    output = args.output or args.run_root / "summary"
    output.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        columns.extend(key for key in row if key not in columns)
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += [
        "| " + " | ".join(str(row.get(column, "")) for column in columns) + " |" for row in rows
    ]
    output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output.with_suffix(".md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
