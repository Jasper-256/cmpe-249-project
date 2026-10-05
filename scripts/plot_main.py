"""Synthetic-validation figures for the main matrix (per-object runs pooled).

Figures (written to ``<run_root>/figures``):
1. ``validation_pose.png`` — frame-weighted median-of-objects is avoided; we pool
   per-object frame means of ARE/ATE on the run's own protocol validation, one
   horizontal bar per method, grouped by training protocol.
2. ``interventions.png`` — TSS-validation keypoint RMSE under normal input,
   clipping shuffle, clipping zero, and gate bypass for TSS-trained fusion runs.
3. ``gate_response.png`` — mean stride-16 gate in clipped vs unclipped patches.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OBJECTS = ("cassini", "satty", "soho")
# Reference categorical slots (fixed order, never cycled).
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
TEXT = "#1f1f1e"
MUTED = "#6b6a65"
GRID = "#e4e3dc"


def _style(axis: Any) -> None:
    axis.grid(axis="x", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_color(MUTED)
    axis.tick_params(colors=MUTED, labelcolor=TEXT, length=0)


def _methods(run_root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for path in sorted(run_root.glob("*/report.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        if report["outcome"]["stage"] != "train":
            continue
        name = report["name"]
        for object_name in OBJECTS:
            if name.endswith(f"-{object_name}"):
                grouped[name[: -len(object_name) - 1]][object_name] = report
    return {method: runs for method, runs in grouped.items() if len(runs) == len(OBJECTS)}


def _pooled(runs: dict[str, dict[str, Any]], key: str, metric: str) -> float:
    total = weight = 0.0
    for report in runs.values():
        evaluation = report["outcome"]["evaluations"][key]
        frames = evaluation["frame_count"]
        if metric == "rmse":
            value = evaluation["coordinate_rmse_px"]
        else:
            value = evaluation["pose"]["all_frames"][metric]
        total += value * frames
        weight += frames
    return total / weight


def _bar_rows(axis: Any, labels: list[str], groups: list[tuple[str, list[float | None]]]) -> None:
    height = 0.8 / len(groups)
    for index, (group_label, values) in enumerate(groups):
        positions = [row + (index - (len(groups) - 1) / 2) * height for row in range(len(labels))]
        bars = [(p, v) for p, v in zip(positions, values, strict=True) if v is not None]
        axis.barh(
            [p for p, _ in bars],
            [v for _, v in bars],
            height=height * 0.9,
            color=SERIES[index],
            label=group_label,
            edgecolor="white",
            linewidth=1.0,
        )
        for position, value in bars:
            axis.text(value, position, f" {value:.2f}", va="center", fontsize=7, color=MUTED)
    axis.set_yticks(range(len(labels)), labels)
    axis.invert_yaxis()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    methods = _methods(args.run_root)
    if not methods:
        raise SystemExit("no complete per-object method triplets yet")
    figures = args.run_root / "figures"
    figures.mkdir(exist_ok=True)

    base_names = sorted({name.replace("-natural", "").replace("-tss", "") for name in methods})
    labels = base_names
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.35 * len(labels) + 1.5), sharey=True)
    for axis, metric, title in (
        (axes[0], "are_degrees_mean", "Mean rotation error (deg)"),
        (axes[1], "absolute_translation_error_m_mean", "Mean translation error (m)"),
    ):
        groups = []
        for protocol in ("natural", "tss"):
            values: list[float | None] = []
            for base in labels:
                candidates = [
                    name
                    for name in methods
                    if f"-{protocol}" in name and name.replace(f"-{protocol}", "") == base
                ]
                values.append(
                    _pooled(methods[candidates[0]], f"validation_{protocol}", metric)
                    if candidates
                    else None
                )
            groups.append((f"trained & validated: {protocol}", values))
        _bar_rows(axis, labels, groups)
        axis.set_title(title, color=TEXT, fontsize=10, loc="left")
        _style(axis)
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle(
        "Synthetic validation (own protocol), three per-object models pooled; PnP failures counted",
        color=TEXT,
        fontsize=10,
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(figures / "validation_pose.png", dpi=160)
    plt.close(fig)

    tss_fusion = sorted(
        name
        for name, runs in methods.items()
        if runs["cassini"]["spec"]["modality"] == "fusion" and runs["cassini"]["protocol"] == "tss"
    )
    if tss_fusion:
        conditions = (
            ("normal", "validation_tss"),
            ("clipping shuffled", "validation_tss_clip_shuffle"),
            ("clipping zeroed", "validation_tss_clip_zero"),
            ("gate bypass 0.5", "validation_tss_gate_bypass_0.5"),
        )
        groups = []
        for label, key in conditions:
            groups.append(
                (
                    label,
                    [
                        _pooled(methods[name], key, "rmse")
                        if key in methods[name]["cassini"]["outcome"]["evaluations"]
                        else None
                        for name in tss_fusion
                    ],
                )
            )
        fig, axis = plt.subplots(figsize=(8, 0.5 * len(tss_fusion) + 1.5))
        _bar_rows(axis, tss_fusion, groups)
        axis.set_title(
            "TSS-trained fusion: TSS-validation keypoint RMSE (px) under interventions",
            color=TEXT,
            fontsize=10,
            loc="left",
        )
        axis.legend(frameon=False, fontsize=8, loc="lower right")
        _style(axis)
        fig.tight_layout()
        fig.savefig(figures / "interventions.png", dpi=160)
        plt.close(fig)

        gated = [
            name
            for name in tss_fusion
            if methods[name]["cassini"]["outcome"]["evaluations"]["validation_tss"].get(
                "gate_mean_clipped_patches"
            )
            is not None
        ]
        if gated:
            fig, axis = plt.subplots(figsize=(8, 0.5 * len(gated) + 1.5))
            groups = []
            for label, field in (
                ("clipped patches (s >= 0.01)", "gate_mean_clipped_patches"),
                ("unclipped patches", "gate_mean_unclipped_patches"),
            ):
                values = []
                for name in gated:
                    pooled = [
                        report["outcome"]["evaluations"]["validation_tss"][field]
                        for report in methods[name].values()
                    ]
                    values.append(sum(pooled) / len(pooled))
                groups.append((label, values))
            _bar_rows(axis, gated, groups)
            axis.set_title(
                "Mean RGB trust g at the stride-16 level (TSS validation; 0 = events, 1 = RGB)",
                color=TEXT,
                fontsize=10,
                loc="left",
            )
            axis.legend(frameon=False, fontsize=8, loc="lower right")
            _style(axis)
            fig.tight_layout()
            fig.savefig(figures / "gate_response.png", dpi=160)
            plt.close(fig)
    print(f"figures: {figures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
