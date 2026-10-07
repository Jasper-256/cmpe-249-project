"""Generate the standard figure set for the main matrix (synthetic validation only).

Writes PNGs to ``outputs/figures/`` from saved run reports, per-frame validation
predictions, and the crop-v2 caches. No model inference and no real-test data.

Figures
-------
training_curves.png      Validation keypoint RMSE per epoch (mean over objects).
gate_dynamics.png        Mean RGB trust in clipped patches per epoch (gated rules).
method_comparison.png    Pose error on natural vs TSS validation per method.
corrupted_vs_clean.png   TSS-trained methods on corrupted vs clean frames.
per_object.png           Corrupted-frame rotation error per object.
error_cdf.png            CDF of rotation error on TSS validation.
example_inputs.png       RGB, TSS RGB, clipping grid, and event encodings.
prediction_overlays.png  Ground-truth vs predicted keypoints on corrupted frames.
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
import numpy as np
from matplotlib.patches import Patch

from saga_pose.data.cache import FreshCropCacheDataset

OBJECTS = ("cassini", "satty", "soho")
# One fixed hue per fusion rule (identity follows the entity, never its rank).
RULE_COLORS = {
    "mean": "#2a78d6",
    "concat": "#eb6834",
    "content_gate": "#1baf7a",
    "clip_gate": "#eda100",
    "clip_gate_constant": "#e87ba4",
    "clip_gate_fixed": "#008300",
    "content_clip_gate": "#4a3aa7",
    "clip_gate_signed": "#e34948",
}
UNIMODAL_COLORS = {"rgb": "#52514e", "event": "#a3a29b"}
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3dc"


def _style(axis: Any, grid_axis: str = "y") -> None:
    axis.set_facecolor(SURFACE)
    axis.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(MUTED)
    axis.tick_params(colors=MUTED, labelcolor=TEXT, labelsize=8)


def _figure(*args: Any, **kwargs: Any) -> tuple[Any, Any]:
    fig, axes = plt.subplots(*args, **kwargs)
    fig.patch.set_facecolor(SURFACE)
    return fig, axes


def _load_methods(run_root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for path in sorted(run_root.glob("*/report.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        if report["outcome"]["stage"] != "train":
            continue
        name = report["name"]
        for object_name in OBJECTS:
            if name.endswith(f"-{object_name}"):
                report["_dir"] = str(path.parent)
                grouped[name[: -len(object_name) - 1]][object_name] = report
    return {method: runs for method, runs in grouped.items() if len(runs) == len(OBJECTS)}


def _color(report: dict[str, Any]) -> tuple[str, str]:
    spec = report["spec"]
    if spec["modality"] == "fusion":
        return RULE_COLORS[spec["fusion"]], "-"
    if spec["modality"] == "event":
        return UNIMODAL_COLORS["event"], "--" if spec["event_representation"] == "raw" else ":"
    return UNIMODAL_COLORS["rgb"], "-" if report["protocol"] == "natural" else "-."


def _label(method: str) -> str:
    label = method.replace("f-", "").replace("u-", "")
    return label if method.startswith("u-") else label.replace("-activity", "")


def _pooled(runs: dict[str, dict[str, Any]], key: str, field: str) -> float | None:
    total = weight = 0.0
    for report in runs.values():
        evaluation = report["outcome"]["evaluations"].get(key)
        if evaluation is None:
            return None
        frames = evaluation["frame_count"]
        value = (
            evaluation["coordinate_rmse_px"]
            if field == "rmse"
            else evaluation["pose"]["all_frames"][field]
        )
        total += value * frames
        weight += frames
    return total / weight


def _is_primary(method: str) -> bool:
    """Main rules only (no seeds, slope, raw, or strength ablations)."""
    if any(tag in method for tag in ("-s25", "slope", "tss0.")):
        return False
    return not (method.startswith("f-") and "-raw" in method)


def _neutral_legend(axis: Any, labels: tuple[str, str], loc: str) -> None:
    """Legend for faded/solid encodings without implying a particular rule colour."""
    handles = [Patch(color=MUTED, alpha=0.45), Patch(color=MUTED, alpha=1.0)]
    axis.legend(handles, labels, frameon=False, fontsize=8, loc=loc)


def training_curves(methods: dict[str, Any], out: Path) -> None:
    fig, axes = _figure(1, 3, figsize=(15, 4.6), sharey=True)
    panels = (
        ("Unimodal baselines", lambda m, r: r["spec"]["modality"] != "fusion"),
        (
            "Fusion, TSS-trained",
            lambda m, r: r["spec"]["modality"] == "fusion" and r["protocol"] == "tss",
        ),
        (
            "Fusion, natural-trained",
            lambda m, r: r["spec"]["modality"] == "fusion" and r["protocol"] == "natural",
        ),
    )
    for axis, (title, keep) in zip(axes, panels, strict=True):
        for method, runs in sorted(methods.items()):
            first = runs["cassini"]
            if not _is_primary(method) or not keep(method, first):
                continue
            curves = [[h["validation_rmse_px"] for h in r["history"]] for r in runs.values()]
            # Early stopping differs per object; average only epochs all objects reached.
            length = min(len(curve) for curve in curves)
            mean_curve = [np.mean([curve[i] for curve in curves]) for i in range(length)]
            color, style = _color(first)
            axis.plot(
                range(1, length + 1),
                mean_curve,
                color=color,
                linestyle=style,
                linewidth=2,
                label=_label(method),
            )
        axis.set_title(title, color=TEXT, fontsize=10, loc="left")
        axis.set_xlabel("epoch", color=MUTED, fontsize=8)
        axis.set_yscale("log")
        _style(axis)
        axis.legend(frameon=False, fontsize=7)
    axes[0].set_ylabel(
        "validation keypoint RMSE, px (own protocol, mean of objects)", color=MUTED, fontsize=8
    )
    fig.suptitle("Training curves", x=0.01, ha="left", color=TEXT, fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "training_curves.png", dpi=160)
    plt.close(fig)


def gate_dynamics(methods: dict[str, Any], out: Path) -> None:
    fig, axis = _figure(figsize=(8, 4.2))
    for method, runs in sorted(methods.items()):
        first = runs["cassini"]
        if first["protocol"] != "tss" or first["spec"]["modality"] != "fusion":
            continue
        if first["history"][0]["gate_mean_clipped"] is None:
            continue
        curves = [[h["gate_mean_clipped"] for h in r["history"]] for r in runs.values()]
        length = min(len(curve) for curve in curves)
        mean_curve = [np.mean([c[i] for c in curves]) for i in range(length)]
        color, _ = _color(first)
        style = ":" if "slope" in method else "-"
        axis.plot(
            range(1, length + 1),
            mean_curve,
            color=color,
            linestyle=style,
            linewidth=2,
            label=_label(method),
        )
    axis.axhline(0.5, color=MUTED, linewidth=0.8, linestyle="--")
    axis.text(1, 0.51, "mean fusion (g = 0.5)", color=MUTED, fontsize=7)
    axis.set_ylim(0, 1)
    axis.set_xlabel("epoch", color=MUTED, fontsize=8)
    axis.set_ylabel("mean RGB trust g in clipped patches", color=MUTED, fontsize=8)
    axis.set_title(
        "Gate dynamics on TSS validation (0 = events, 1 = RGB)", color=TEXT, fontsize=10, loc="left"
    )
    _style(axis)
    axis.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "gate_dynamics.png", dpi=160)
    plt.close(fig)


def method_comparison(methods: dict[str, Any], out: Path) -> None:
    names = [
        m
        for m in sorted(methods, key=lambda m: (methods[m]["cassini"]["protocol"], m))
        if _is_primary(m)
    ]
    fig, axes = _figure(1, 2, figsize=(13, 0.38 * len(names) + 1.6), sharey=True)
    for axis, field, title in (
        (axes[0], "are_degrees_mean", "Mean rotation error (deg)"),
        (axes[1], "absolute_translation_error_m_mean", "Mean translation error (m)"),
    ):
        for offset, (key, label, alpha) in enumerate(
            (
                ("validation_natural", "natural validation", 0.45),
                ("validation_tss", "TSS validation", 1.0),
            )
        ):
            values = [_pooled(methods[n], key, field) for n in names]
            colors = [_color(methods[n]["cassini"])[0] for n in names]
            positions = np.arange(len(names)) + (offset - 0.5) * 0.38
            axis.barh(
                positions,
                values,
                height=0.36,
                color=colors,
                alpha=alpha,
                edgecolor=SURFACE,
                linewidth=1,
                label=label,
            )
            for position, value in zip(positions, values, strict=True):
                axis.text(
                    value,
                    position,
                    f" {value:.2f}" if field.startswith("are") else f" {value:.3f}",
                    va="center",
                    fontsize=6.5,
                    color=MUTED,
                )
        axis.set_yticks(
            range(len(names)), [f"{_label(n)} [{methods[n]['cassini']['protocol']}]" for n in names]
        )
        axis.invert_yaxis()
        axis.set_title(title, color=TEXT, fontsize=10, loc="left")
        _style(axis, "x")
    _neutral_legend(axes[1], ("natural validation", "TSS validation"), "lower right")
    fig.suptitle(
        "Synthetic validation, pooled over three per-object models (faded = natural, solid = TSS)",
        x=0.01,
        ha="left",
        color=TEXT,
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out / "method_comparison.png", dpi=160)
    plt.close(fig)


def _split_errors(methods: dict[str, Any], applied: dict[str, bool]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for method, runs in methods.items():
        if runs["cassini"]["protocol"] != "tss":
            continue
        per_object: dict[str, dict[str, np.ndarray]] = {}
        for object_name in OBJECTS:
            frames = np.load(Path(runs[object_name]["_dir"]) / "validation_frames.npz")
            mask = np.asarray([applied[s] for s in frames["sample_id"]])
            per_object[object_name] = {
                "corrupted": frames["rotation_error_deg"][mask],
                "clean": frames["rotation_error_deg"][~mask],
            }
        result[method] = per_object
    return result


def corrupted_vs_clean(split: dict[str, Any], methods: dict[str, Any], out: Path) -> None:
    names = sorted(
        (m for m in split if _is_primary(m)),
        key=lambda m: np.concatenate([split[m][o]["corrupted"] for o in OBJECTS]).mean(),
    )
    fig, axis = _figure(figsize=(9, 0.42 * len(names) + 1.4))
    for offset, (part, alpha) in enumerate((("clean", 0.45), ("corrupted", 1.0))):
        values = [np.concatenate([split[m][o][part] for o in OBJECTS]).mean() for m in names]
        positions = np.arange(len(names)) + (offset - 0.5) * 0.38
        axis.barh(
            positions,
            values,
            height=0.36,
            color=[_color(methods[m]["cassini"])[0] for m in names],
            alpha=alpha,
            edgecolor=SURFACE,
            label=f"{part} frames",
        )
        for position, value in zip(positions, values, strict=True):
            axis.text(value, position, f" {value:.2f}", va="center", fontsize=7, color=MUTED)
    axis.set_yticks(range(len(names)), [_label(n) for n in names])
    axis.invert_yaxis()
    axis.set_xlabel("mean rotation error on TSS validation (deg)", color=MUTED, fontsize=8)
    axis.set_title(
        "TSS-trained models: corrupted vs clean validation frames",
        color=TEXT,
        fontsize=10,
        loc="left",
    )
    _style(axis, "x")
    _neutral_legend(axis, ("clean frames", "corrupted frames"), "center right")
    fig.tight_layout()
    fig.savefig(out / "corrupted_vs_clean.png", dpi=160)
    plt.close(fig)


def per_object(split: dict[str, Any], methods: dict[str, Any], out: Path) -> None:
    names = sorted(m for m in split if _is_primary(m))
    fig, axes = _figure(1, 3, figsize=(14, 0.38 * len(names) + 1.6), sharey=True)
    for axis, object_name in zip(axes, OBJECTS, strict=True):
        means = [split[m][object_name]["corrupted"].mean() for m in names]
        medians = [np.median(split[m][object_name]["corrupted"]) for m in names]
        colors = [_color(methods[m]["cassini"])[0] for m in names]
        axis.barh(range(len(names)), means, color=colors, edgecolor=SURFACE, height=0.7)
        axis.scatter(medians, range(len(names)), color=TEXT, s=12, zorder=3, label="median")
        for row, value in enumerate(means):
            axis.text(value, row, f" {value:.1f}", va="center", fontsize=7, color=MUTED)
        axis.set_title(object_name, color=TEXT, fontsize=10, loc="left")
        axis.set_xlabel("corrupted-frame rotation error (deg); bar = mean", color=MUTED, fontsize=8)
        _style(axis, "x")
    axes[0].set_yticks(range(len(names)), [_label(n) for n in names])
    axes[0].invert_yaxis()
    axes[2].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle(
        "Per-object breakdown (TSS-trained, corrupted TSS-validation frames)",
        x=0.01,
        ha="left",
        color=TEXT,
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out / "per_object.png", dpi=160)
    plt.close(fig)


def error_cdf(methods: dict[str, Any], out: Path) -> None:
    fig, axis = _figure(figsize=(8, 4.6))
    chosen = [
        m for m in sorted(methods) if _is_primary(m) and methods[m]["cassini"]["protocol"] == "tss"
    ]
    for method in chosen:
        errors = np.sort(
            np.concatenate(
                [
                    np.load(Path(methods[method][o]["_dir"]) / "validation_frames.npz")[
                        "rotation_error_deg"
                    ]
                    for o in OBJECTS
                ]
            )
        )
        color, style = _color(methods[method]["cassini"])
        axis.plot(
            errors,
            np.arange(1, errors.size + 1) / errors.size,
            color=color,
            linestyle=style,
            linewidth=2,
            label=_label(method),
        )
    axis.axvline(10, color=MUTED, linestyle="--", linewidth=0.8)
    axis.text(10.3, 0.05, "10° success threshold", color=MUTED, fontsize=7)
    axis.set_xscale("log")
    axis.set_xlabel("rotation error on TSS validation (deg, log scale)", color=MUTED, fontsize=8)
    axis.set_ylabel("fraction of frames", color=MUTED, fontsize=8)
    axis.set_title(
        "Cumulative rotation-error distribution (TSS-trained models)",
        color=TEXT,
        fontsize=10,
        loc="left",
    )
    _style(axis, "both")
    axis.legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "error_cdf.png", dpi=160)
    plt.close(fig)


def _keypoints(axis: Any, points: np.ndarray, color: str, marker: str, label: str) -> None:
    if marker == "o":
        axis.scatter(
            points[:, 0],
            points[:, 1],
            s=16,
            facecolors="none",
            edgecolors=color,
            marker=marker,
            linewidths=1.2,
            label=label,
        )
    else:
        axis.scatter(
            points[:, 0],
            points[:, 1],
            s=16,
            color=color,
            marker=marker,
            linewidths=1.2,
            label=label,
        )


def example_inputs(cache_dir: Path, out: Path) -> None:
    cache = FreshCropCacheDataset(cache_dir / "soho-validation")
    applied = np.flatnonzero(np.asarray(cache.array("tss_applied"), dtype=bool))
    picks = applied[np.linspace(0, applied.size - 1, 3, dtype=int)]
    columns = (
        "RGB",
        "RGB + TSS",
        "clipping grid s",
        "event raw",
        "event activity",
        "event magnitude",
    )
    fig, axes = _figure(len(picks), len(columns), figsize=(15, 2.7 * len(picks)))
    for row, index in enumerate(picks):
        item = cache[int(index)]
        images = [
            item["rgb"].permute(1, 2, 0).numpy(),
            item["rgb_tss"].permute(1, 2, 0).numpy(),
            item["clipping_ratio_tss"].numpy(),
            item["event_raw"][0].numpy(),
            item["event_activity"][0].numpy(),
            item["event_magnitude"][0].numpy(),
        ]
        for column, (image, title) in enumerate(zip(images, columns, strict=True)):
            axis = axes[row, column]
            if image.ndim == 2:
                axis.imshow(
                    image,
                    cmap="magma" if column == 2 else "gray",
                    vmin=0,
                    vmax=1,
                    interpolation="nearest",
                )
            else:
                axis.imshow(image)
            if column in (0, 1):
                _keypoints(axis, item["keypoints_xy"].numpy(), "#1baf7a", "o", "ground truth")
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(title, color=TEXT, fontsize=9)
    fig.suptitle(
        "crop-v2 inputs (SOHO synthetic validation; TSS-corrupted samples; green = GT keypoints)",
        x=0.01,
        ha="left",
        color=TEXT,
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out / "example_inputs.png", dpi=140)
    plt.close(fig)


def prediction_overlays(methods: dict[str, Any], cache_dir: Path, out: Path) -> None:
    chosen = [
        m
        for m in (
            "u-rgb-tss",
            "f-mean-tss-activity",
            "f-clip_gate-tss-activity",
            "f-content_clip_gate-tss-activity",
        )
        if m in methods
    ]
    if not chosen:
        return
    cache = FreshCropCacheDataset(cache_dir / "soho-validation")
    index_of = {sample: i for i, sample in enumerate(cache.sample_ids)}
    frames = {
        m: np.load(Path(methods[m]["soho"]["_dir"]) / "validation_frames.npz") for m in chosen
    }
    reference = frames[chosen[0]]
    applied = np.asarray(cache.array("tss_applied"), dtype=bool)
    candidates = [i for i, s in enumerate(reference["sample_id"]) if applied[index_of[s]]]
    picks = [candidates[i] for i in np.linspace(0, len(candidates) - 1, 3, dtype=int)]
    fig, axes = _figure(len(picks), len(chosen), figsize=(3.2 * len(chosen), 3.2 * len(picks)))
    for row, frame_index in enumerate(picks):
        sample = reference["sample_id"][frame_index]
        item = cache[index_of[sample]]
        for column, method in enumerate(chosen):
            axis = axes[row, column]
            axis.imshow(item["rgb_tss"].permute(1, 2, 0).numpy())
            _keypoints(axis, item["keypoints_xy"].numpy(), "#1baf7a", "o", "ground truth")
            position = list(frames[method]["sample_id"]).index(sample)
            predicted = frames[method]["predicted_keypoints_xy"][position]
            color, _ = _color(methods[method]["cassini"])
            _keypoints(axis, predicted, color, "x", "prediction")
            error = frames[method]["rotation_error_deg"][position]
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_title(f"{_label(method)}\nrotation error {error:.1f}°", color=TEXT, fontsize=8)
    fig.suptitle(
        "Keypoint predictions on TSS-corrupted SOHO validation frames "
        "(circles = GT, crosses = prediction)",
        x=0.01,
        ha="left",
        color=TEXT,
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out / "prediction_overlays.png", dpi=140)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("outputs/runs/main-hrnet"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/fresh/cache/crop-v2"))
    parser.add_argument("--output", type=Path, default=Path("outputs/figures"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    methods = _load_methods(args.run_root)
    applied: dict[str, bool] = {}
    for object_name in OBJECTS:
        cache = FreshCropCacheDataset(args.cache_dir / f"{object_name}-validation")
        applied.update(
            zip(
                cache.sample_ids,
                np.asarray(cache.array("tss_applied"), dtype=bool).tolist(),
                strict=True,
            )
        )
    split = _split_errors(methods, applied)
    training_curves(methods, args.output)
    gate_dynamics(methods, args.output)
    method_comparison(methods, args.output)
    corrupted_vs_clean(split, methods, args.output)
    per_object(split, methods, args.output)
    error_cdf(methods, args.output)
    example_inputs(args.cache_dir, args.output)
    prediction_overlays(methods, args.cache_dir, args.output)
    print(f"figures: {sorted(p.name for p in args.output.glob('*.png'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
