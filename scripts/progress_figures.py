"""Figures and summary data for the progress check-in report.

Writes four figures to ``outputs/figures`` (architecture, seed spread,
sim-to-real gap, real predictions) and ``outputs/report/progress_data.json``.
Real-test numbers are read from the frozen G6 reports that exist so far; they
are descriptive only (no checkpoint, threshold, or design choice uses them).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from saga_pose.data.cache import FreshCropCacheDataset

RUNS = Path("outputs/runs/main-hrnet")
REAL = Path("outputs/real")
FIG = Path("outputs/figures")
OBJECTS = ("cassini", "satty", "soho")
SURFACE, TEXT, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3dc"
GLARE, ACCENT = "#c97a00", "#2a5bd7"
COLORS = {
    "mean": "#2a78d6",
    "concat": "#eb6834",
    "content_gate": "#1baf7a",
    "clip_gate": "#eda100",
    "clip_gate_constant": "#e87ba4",
    "clip_gate_fixed": "#008300",
    "content_clip_gate": "#4a3aa7",
    "attention": "#7a5c3e",
    "rgb": "#52514e",
    "event": "#a3a29b",
}
RANDOM_ROTATION_DEG = float(
    np.degrees(np.pi / 2 + 2 / np.pi)
)  # mean angle of a uniform random rotation

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10})


def _style(ax: Any, grid_axis: str = "x") -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=TEXT, labelsize=8.5)


def _synthetic(method: str, key: str) -> dict[str, float] | None:
    total = np.zeros(3)
    frames = 0
    for obj in OBJECTS:
        path = RUNS / f"{method}-{obj}" / "report.json"
        if not path.is_file():
            return None
        evaluation = json.loads(path.read_text(encoding="utf-8"))["outcome"]["evaluations"][key]
        n = evaluation["frame_count"]
        pose = evaluation["pose"]["all_frames"]
        total += n * np.array(
            [
                pose["are_degrees_mean"],
                1000 * pose["absolute_translation_error_m_mean"],
                100 * pose["success_joint_10_deg_10_mm"],
            ]
        )
        frames += n
    are, ate, success = total / frames
    return {"are_deg": are, "ate_mm": ate, "success_pct": success}


def _real(method: str) -> dict[str, float] | None:
    weights, rows = [], []
    for obj in OBJECTS:
        path = REAL / f"{method}-{obj}" / "real_report.json"
        if not path.is_file():
            return None
        frames = json.loads(path.read_text(encoding="utf-8"))["results"]["primary"]["analysis"]
        overall = frames["overall_frames"]
        weights.append(overall["frames"])
        rows.append(
            [
                overall["are_degrees_mean"],
                overall["are_degrees_median"],
                1000 * overall["ate_m_median"],
                100 * overall["success_rotation_below_10_deg"],
                100 * overall["success_translation_below_10_mm"],
            ]
        )
    values = np.average(np.array(rows), axis=0, weights=weights)
    keys = ("are_deg", "are_median_deg", "ate_median_mm", "rot10_pct", "trans10mm_pct")
    return {k: float(v) for k, v in zip(keys, values, strict=True)} | {"frames": int(sum(weights))}


# --- 1. architecture -------------------------------------------------------------------------


def architecture() -> None:
    fig, ax = plt.subplots(figsize=(10, 3.9))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)
    ax.axis("off")

    def box(x, y, w, h, title, sub="", edge=MUTED, face="white", color=TEXT):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.25,rounding_size=1.2",
                linewidth=1.1,
                edgecolor=edge,
                facecolor=face,
            )
        )
        ax.text(
            x + w / 2,
            y + h / 2 + (1.3 if sub else 0),
            title,
            ha="center",
            va="center",
            fontsize=9.5,
            weight="bold",
            color=color,
        )
        if sub:
            ax.text(
                x + w / 2, y + h / 2 - 1.8, sub, ha="center", va="center", fontsize=7.6, color=MUTED
            )

    def arrow(x0, y0, x1, y1, color=MUTED, style="-|>"):
        ax.add_patch(
            FancyArrowPatch(
                (x0, y0),
                (x1, y1),
                arrowstyle=style,
                mutation_scale=11,
                linewidth=1.1,
                color=color,
                shrinkA=2,
                shrinkB=2,
            )
        )

    box(0.5, 29, 18, 8, "RGB crop", "256 x 256")
    box(
        0.5,
        16,
        18,
        8,
        "Clipping map s",
        "16 x 16, from RGB pixels",
        edge=GLARE,
        face="#fbf0dc",
        color=GLARE,
    )
    box(0.5, 3, 18, 8, "Event frame", "raw / activity")
    box(23, 29, 19, 8, "RGB encoder", "HRNet-W18, ImageNet init")
    box(23, 3, 19, 8, "Event encoder", "HRNet-W18, own weights")
    box(
        47,
        12,
        18,
        16,
        "Fusion rule",
        "at all 5 feature scales\n(the only part that varies)",
        edge=ACCENT,
        face="#e8eefc",
        color=ACCENT,
    )
    box(71, 14, 12, 12, "Heatmap\ndecoder", "64 x 64")
    box(87, 14, 12, 12, "PnP solver", "keypoints -> R, t")
    arrow(19, 33, 22.5, 33)
    arrow(19, 7, 22.5, 7)
    arrow(42.5, 33, 50, 28.5)
    arrow(42.5, 7, 50, 11.5)
    arrow(19, 20, 46.5, 20, color=GLARE)
    arrow(65.5, 20, 70.5, 20)
    arrow(83.5, 20, 86.5, 20)
    ax.text(32.5, 21.2, "used by the gates only", fontsize=7.5, color=GLARE, ha="center")
    rules = (
        "mean  |  concat  |  content gate  |  clipping gate  g = sigmoid(b - softplus(a) * s)  |  "
        "content + clipping gate  |  coarse attention (+ clipping bias)"
    )
    ax.text(
        50,
        0.2,
        "Fusion rules compared:  " + rules,
        ha="center",
        va="bottom",
        fontsize=7.6,
        color=MUTED,
    )
    fig.tight_layout(pad=0.4)
    fig.savefig(FIG / "architecture.png", dpi=200, facecolor="white")
    plt.close(fig)


# --- 2. seed spread --------------------------------------------------------------------------

SEED_ROWS = [
    ("RGB only", "u-rgb-tss", "rgb"),
    ("Mean fusion", "f-mean-tss-activity", "mean"),
    ("Content gate", "f-content_gate-tss-activity", "content_gate"),
    ("Content + clipping gate", "f-content_clip_gate-tss-activity", "content_clip_gate"),
    ("Clipping gate", "f-clip_gate-tss-activity", "clip_gate"),
    ("Constant control (no s)", "f-clip_gate_constant-tss-activity", "clip_gate_constant"),
    ("Fixed clipping gate", "f-clip_gate_fixed-tss-activity", "clip_gate_fixed"),
]


def seed_spread() -> list[dict[str, Any]]:
    out = []
    for label, method, key in SEED_ROWS:
        per_seed = [
            _synthetic(f"{method}{suffix}", "validation_tss") for suffix in ("", "-s250", "-s251")
        ]
        vals = [v for v in per_seed if v is not None]
        are = np.array([v["are_deg"] for v in vals])
        succ = np.array([v["success_pct"] for v in vals])
        ate = np.array([v["ate_mm"] for v in vals])
        out.append(
            {
                "label": label,
                "method": method,
                "color": COLORS[key],
                "seeds": len(vals),
                "are_each": are.round(2).tolist(),
                "are_mean": float(are.mean()),
                "are_sd": float(are.std(ddof=1)),
                "ate_mean": float(ate.mean()),
                "ate_sd": float(ate.std(ddof=1)),
                "success_mean": float(succ.mean()),
                "success_sd": float(succ.std(ddof=1)),
            }
        )
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    fig.patch.set_facecolor(SURFACE)
    for i, row in enumerate(out):
        y = len(out) - 1 - i
        ax.barh(y, row["are_mean"], color=row["color"], alpha=0.28, height=0.62)
        ax.errorbar(row["are_mean"], y, xerr=row["are_sd"], color=row["color"], capsize=3, lw=1.2)
        ax.scatter(
            row["are_each"],
            [y] * len(row["are_each"]),
            color=row["color"],
            s=18,
            zorder=3,
            edgecolor="white",
            linewidth=0.6,
        )
        ax.text(
            1.02,
            y,
            f"{row['are_mean']:.1f} ± {row['are_sd']:.1f}",
            transform=ax.get_yaxis_transform(),
            va="center",
            ha="left",
            fontsize=8,
            color=TEXT,
        )
    ax.set_yticks(range(len(out)), [r["label"] for r in reversed(out)])
    ax.set_xlabel("mean rotation error on glare (TSS) validation, degrees", color=MUTED)
    ax.set_xlim(0, 18)
    _style(ax)
    ax.set_title(
        "Three training seeds per model (dots), mean and seed SD (bar, whisker)",
        loc="left",
        color=TEXT,
    )
    fig.tight_layout(rect=(0, 0, 0.9, 1))
    fig.savefig(FIG / "seed_spread.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return out


# --- 3. sim-to-real ----------------------------------------------------------------------------

REAL_ROWS = [
    ("RGB only", "u-rgb-natural", "natural", "rgb"),
    ("RGB only", "u-rgb-tss", "tss", "rgb"),
    ("Event only", "u-event-activity", "natural", "event"),
    ("Mean fusion", "f-mean-natural-activity", "natural", "mean"),
    ("Mean fusion", "f-mean-tss-activity", "tss", "mean"),
    ("Concat fusion", "f-concat-natural-activity", "natural", "concat"),
    ("Content gate", "f-content_gate-tss-activity", "tss", "content_gate"),
    ("Content + clipping gate", "f-content_clip_gate-tss-activity", "tss", "content_clip_gate"),
    ("Clipping gate", "f-clip_gate-tss-activity", "tss", "clip_gate"),
    ("Coarse attention", "f-attention-tss-activity", "tss", "attention"),
]


def sim_to_real() -> list[dict[str, Any]]:
    out = []
    for label, method, protocol, key in REAL_ROWS:
        synthetic = _synthetic(method, "validation_natural")
        real = _real(method)
        if synthetic is None or real is None:
            continue
        out.append(
            {
                "label": label,
                "method": method,
                "trained": protocol,
                "color": COLORS[key],
                "synthetic": synthetic,
                "real": real,
            }
        )
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    fig.patch.set_facecolor(SURFACE)
    for i, row in enumerate(out):
        y = len(out) - 1 - i
        s, r = row["synthetic"]["are_deg"], row["real"]["are_deg"]
        ax.plot([s, r], [y, y], color=GRID, lw=2.2, zorder=1)
        ax.scatter([s], [y], color=row["color"], s=30, zorder=3, marker="o")
        ax.scatter(
            [r], [y], color=row["color"], s=34, zorder=3, marker="D", edgecolor=TEXT, linewidth=0.5
        )
        ax.text(r + 2, y, f"{r:.0f}", va="center", fontsize=8)
    ax.axvline(RANDOM_ROTATION_DEG, color=MUTED, ls="--", lw=1)
    ax.text(
        RANDOM_ROTATION_DEG + 1.5,
        -0.65,
        "random\nguess",
        ha="left",
        va="center",
        fontsize=7.5,
        color=MUTED,
    )
    ax.set_ylim(-1.2, len(out) - 0.5)
    labels = [
        f"{r['label']} ({'glare' if r['trained'] == 'tss' else 'clean'}-trained)"
        for r in reversed(out)
    ]
    ax.set_yticks(range(len(out)), labels)
    ax.set_xlim(0, 142)
    ax.set_xlabel(
        "mean rotation error, degrees   (o synthetic validation,  ◆ real test)", color=MUTED
    )
    _style(ax)
    ax.set_title(
        "Synthetic-to-real gap: every model fails on the real sequences", loc="left", color=TEXT
    )
    fig.tight_layout()
    fig.savefig(FIG / "sim_to_real.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return out


# --- 4. real predictions -----------------------------------------------------------------------


def real_predictions(method: str = "f-mean-tss-activity") -> None:
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 5.2))
    fig.patch.set_facecolor("white")
    for col, obj in enumerate(OBJECTS):
        cache = FreshCropCacheDataset(Path("data/fresh/cache/crop-v2") / f"real-{obj}")
        frames = np.load(REAL / f"{method}-{obj}" / "frames_primary.npz", allow_pickle=True)
        errors = np.asarray(frames["rotation_error_deg"])
        j = int(
            np.argsort(errors)[len(errors) // 2]
        )  # median-error frame: representative, not chosen
        item = cache[{s: i for i, s in enumerate(cache.sample_ids)}[str(frames["sample_id"][j])]]
        gt = item["keypoints_xy"].numpy()
        pred = np.asarray(frames["predicted_keypoints_xy"][j])
        event = item["event_raw"].numpy()
        event = event[0] if event.ndim == 3 else event
        for row, image in enumerate((item["rgb"].permute(1, 2, 0).numpy(), event)):
            ax = axes[row, col]
            ax.imshow(image, cmap="gray" if row else None)
            ax.scatter(gt[:, 0], gt[:, 1], s=13, c="#2bd96b", label="ground truth")
            ax.scatter(pred[:, 0], pred[:, 1], s=15, c="#ff4d4d", marker="x", label="predicted")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(
                    f"{frames['sequence'][j]}  (rotation error {errors[j]:.0f}°)", fontsize=8.5
                )
        axes[0, 0].set_ylabel("RGB", fontsize=9)
        axes[1, 0].set_ylabel("events", fontsize=9)
    axes[0, 0].legend(loc="lower left", fontsize=7, framealpha=0.85)
    fig.tight_layout(pad=0.5)
    fig.savefig(FIG / "real_predictions.png", dpi=200, facecolor="white")
    plt.close(fig)


# --- 5. tables -----------------------------------------------------------------------------------

SYNTHETIC_ROWS = [
    ("RGB only", "u-rgb-natural", "clean"),
    ("RGB only", "u-rgb-tss", "glare"),
    ("Event only", "u-event-activity", "clean"),
    ("Mean fusion", "f-mean-natural-activity", "clean"),
    ("Mean fusion", "f-mean-tss-activity", "glare"),
    ("Concat fusion", "f-concat-tss-activity", "glare"),
    ("Content gate", "f-content_gate-tss-activity", "glare"),
    ("Content + clipping gate", "f-content_clip_gate-tss-activity", "glare"),
    ("Clipping gate", "f-clip_gate-tss-activity", "glare"),
    ("Constant control (no s)", "f-clip_gate_constant-tss-activity", "glare"),
    ("Fixed clipping gate", "f-clip_gate_fixed-tss-activity", "glare"),
    ("Coarse attention", "f-attention-tss-activity", "glare"),
    ("Attention + clipping bias", "f-attention_clip-tss-activity", "glare"),
]


def synthetic_table() -> list[dict[str, Any]]:
    rows = []
    for label, method, trained in SYNTHETIC_ROWS:
        clean = _synthetic(method, "validation_natural")
        glare = _synthetic(method, "validation_tss")
        report = json.loads(
            (RUNS / f"{method}-cassini" / "report.json").read_text(encoding="utf-8")
        )
        params = report.get("parameters") or report["outcome"]["parameters"]
        rows.append(
            {
                "label": label,
                "method": method,
                "trained": trained,
                "clean": clean,
                "glare": glare,
                "total_params": params["total"],
                "fusion_params": params.get("fusion_specific"),
            }
        )
    return rows


# --- 6. current study (sim2real-v1) ----------------------------------------------------------

S2R = Path("outputs/runs/sim2real-hrnet")
S2R_ROWS = [
    ("RGB only", "s2r-u-rgb-tss", "u-rgb-tss", "rgb"),
    ("Event only", "s2r-u-event-activity", "u-event-activity", "event"),
    ("Mean fusion", "s2r-f-mean-tss-activity", "f-mean-tss-activity", "mean"),
    ("Content gate", "s2r-f-content_gate-tss-activity", None, "content_gate"),
    ("Content + clipping gate", "s2r-f-content_clip_gate-tss-activity", None, "content_clip_gate"),
    ("Clipping gate", "s2r-f-clip_gate-tss-activity", "f-clip_gate-tss-activity", "clip_gate"),
    (
        "Constant control (no s)",
        "s2r-f-clip_gate_constant-tss-activity",
        None,
        "clip_gate_constant",
    ),
    ("Coarse attention", "s2r-f-attention-tss-activity", None, "attention"),
    ("Attention + clipping bias", "s2r-f-attention_clip-tss-activity", None, "attention"),
    ("RGB only, clean-trained", "s2r-u-rgb-natural", "u-rgb-natural", "rgb"),
    (
        "Mean fusion, clean-trained",
        "s2r-f-mean-natural-activity",
        "f-mean-natural-activity",
        "mean",
    ),
]


def current_study() -> dict[str, Any]:
    """Synthetic results of the augmented study on the objects finished so far."""
    gap_path = Path("outputs/sim2real/baseline_gap.json")
    before = json.loads(gap_path.read_text(encoding="utf-8"))["runs"] if gap_path.is_file() else {}
    keys = {
        "clean": "validation_natural_sim2real",
        "glare": "validation_tss_sim2real",
        "glare_plain": "validation_tss",
    }
    rows = []
    for label, method, earlier, colour in S2R_ROWS:
        objects, frames = [], 0
        total = {k: np.zeros(3) for k in keys}
        old = np.zeros(2)
        old_frames = 0
        for obj in OBJECTS:
            path = S2R / f"{method}-{obj}" / "report.json"
            if not path.is_file():
                continue
            report = json.loads(path.read_text(encoding="utf-8"))
            objects.append(obj)
            for name, key in keys.items():
                evaluation = report["outcome"]["evaluations"][key]
                n = evaluation["frame_count"]
                pose = evaluation["pose"]["all_frames"]
                total[name] += n * np.array(
                    [
                        pose["are_degrees_mean"],
                        pose["are_degrees_median"],
                        100 * pose["success_joint_10_deg_10_mm"],
                    ]
                )
            frames += n
            if earlier and f"{earlier}-{obj}" in before:
                entry = before[f"{earlier}-{obj}"]
                old += entry["natural_sim2real"]["frames"] * np.array(
                    [entry["natural_sim2real"]["are_deg"], entry["tss_sim2real"]["are_deg"]]
                )
                old_frames += entry["natural_sim2real"]["frames"]
        if not objects:
            continue
        row: dict[str, Any] = {
            "label": label,
            "method": method,
            "objects": objects,
            "color": COLORS[colour],
        }
        for name in keys:
            are, median, success = total[name] / frames
            row[name] = {"are_deg": are, "are_median_deg": median, "success_pct": success}
        if old_frames:
            row["earlier"] = {
                "clean_are_deg": old[0] / old_frames,
                "glare_are_deg": old[1] / old_frames,
            }
        rows.append(row)
    planned = len(S2R_ROWS) * len(OBJECTS)
    done = sum(len(r["objects"]) for r in rows)

    with_earlier = [r for r in rows if "earlier" in r]
    if with_earlier:
        fig, ax = plt.subplots(figsize=(7.2, 0.52 * len(with_earlier) + 1.3))
        fig.patch.set_facecolor(SURFACE)
        for i, row in enumerate(with_earlier):
            y = len(with_earlier) - 1 - i
            a, b = row["earlier"]["glare_are_deg"], row["glare"]["are_deg"]
            ax.plot([a, b], [y, y], color=GRID, lw=2.2, zorder=1)
            ax.scatter([a], [y], color=row["color"], s=34, marker="X", zorder=3)
            ax.scatter(
                [b],
                [y],
                color=row["color"],
                s=38,
                marker="o",
                zorder=3,
                edgecolor=TEXT,
                linewidth=0.5,
            )
            ax.text(a + 2.5, y, f"{a:.0f}", va="center", fontsize=8)
            ax.text(b + 2.5, y, f"{b:.1f}", va="center", fontsize=8)
        ax.set_yticks(
            range(len(with_earlier)),
            [
                f"{r['label']} ({', '.join(o.capitalize() for o in r['objects'])})"
                for r in reversed(with_earlier)
            ],
        )
        ax.set_xlim(0, 125)
        ax.set_ylim(-0.6, len(with_earlier) - 0.4)
        ax.set_xlabel(
            "rotation error on augmented glare validation, degrees  (x before, o now)",
            color=MUTED,
        )
        _style(ax)
        fig.tight_layout()
        fig.savefig(FIG / "augmentation_effect.png", dpi=200, facecolor=SURFACE)
        plt.close(fig)
    return {"rows": rows, "models_done": done, "models_planned": planned}


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    architecture()
    seeds = seed_spread()
    gap = sim_to_real()
    real_predictions()
    reports = sorted(
        p.parent.name for p in REAL.glob("*/real_report.json") if not p.parent.name.startswith("g-")
    )
    manifest = json.loads((REAL / "freeze_manifest.json").read_text(encoding="utf-8"))
    analyses = {}
    for condition in ("primary", "primary_ransac"):
        path = REAL / f"analysis_{condition}.json"
        if path.is_file():
            analyses[condition] = json.loads(path.read_text(encoding="utf-8"))
    g6_log = Path("checkpoints/g6.log")
    data = {
        "seeds": seeds,
        "synthetic_table": synthetic_table(),
        "sim_to_real": gap,
        "real_reports": len(reports),
        "frozen_runs": len(manifest["runs"]),
        "freeze_commit": manifest["git_commit"],
        "g6_complete": bool(analyses)
        and g6_log.is_file()
        and g6_log.read_text(encoding="utf-8", errors="replace")
        .rstrip()
        .splitlines()[-1]
        .startswith("G6 done"),
        "analyses": analyses,
        "current_study": current_study(),
    }
    out = Path("outputs/report/progress_data.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    print(f"real reports {len(reports)}/{len(manifest['runs'])}; wrote figures and {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
