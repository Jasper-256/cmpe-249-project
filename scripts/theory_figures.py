"""Explanatory figures for the theory guide (outputs/figures/theory_*.png).

Diagrams of the fusion rules, the gate function, the configuration matrix, the
real-versus-synthetic pose coverage (labels only), and sim2real-v1 samples.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from saga_pose.data.cache import FreshCropCacheDataset
from saga_pose.data.dataset import event_input
from saga_pose.data.sim2real import Sim2RealConfig, augment_batch

FIG = Path("outputs/figures")
CACHE = Path("data/fresh/cache/crop-v2")
INK, MUTED, GRID = "#14171f", "#5a6273", "#e4e3dc"
RGB_C, EV_C, GLARE, ACCENT, GOOD = "#b4352c", "#2a78d6", "#c97a00", "#2448c8", "#1b7f4b"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})


def _box(
    ax: Any,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    color: str = MUTED,
    face: str = "white",
    size: float = 8.5,
    bold: bool = False,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=1.1,
            edgecolor=color,
            facecolor=face,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=size,
        color=color if color != MUTED else INK,
        weight="bold" if bold else "normal",
    )


def _arrow(ax: Any, a: tuple[float, float], b: tuple[float, float], color: str = MUTED) -> None:
    ax.add_patch(
        FancyArrowPatch(
            a,
            b,
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=1.0,
            color=color,
            shrinkA=1,
            shrinkB=1,
        )
    )


def fusion_rules() -> None:
    panels = [
        ("Mean", "0 params", None, "F = (R + E) / 2", False),
        ("Concat", "2.80 M", "1x1 conv", "F = conv([R, E])", False),
        (
            "Content gate",
            "2.80 M",
            "gate from R, E",
            "g = sigmoid(conv([R, E]))\nF = g R + (1 - g) E",
            False,
        ),
        (
            "Clipping gate (proposed)",
            "3,968",
            "gate from s",
            "g = sigmoid(b - softplus(a) s)\nF = g R + (1 - g) E",
            True,
        ),
        (
            "Content + clipping gate",
            "2.80 M",
            "gate from R, E, s",
            "g = sigmoid(conv([R, E]) - softplus(a) s)\nF = g R + (1 - g) E",
            True,
        ),
        (
            "Coarse attention",
            "0.90 M",
            "transformer block",
            "tokens of R and E attend to each other\nF = mean + residual (strides 16, 32)",
            False,
        ),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.2))
    for ax, (title, params, middle, formula, uses_s) in zip(axes.flat, panels, strict=True):
        ax.set_xlim(0, 4)
        ax.set_ylim(0, 3)
        ax.axis("off")
        ax.text(0.05, 2.86, title, fontsize=10, weight="bold", color=INK, va="top")
        ax.text(3.95, 2.86, params, fontsize=8, color=MUTED, va="top", ha="right")
        _box(ax, 0.05, 1.85, 1.0, 0.42, "RGB  R", RGB_C, size=8.5)
        _box(ax, 0.05, 0.75, 1.0, 0.42, "events  E", EV_C, size=8.5)
        if uses_s:
            _box(ax, 0.05, 1.3, 1.0, 0.42, "clipping  s", GLARE, face="#fbf0dc", size=8.5)
        if middle:
            _box(ax, 1.5, 1.25, 1.4, 0.52, middle, ACCENT, face="#e8eefc", size=8)
            _arrow(ax, (1.05, 2.06), (1.6, 1.77))
            _arrow(ax, (1.05, 0.96), (1.6, 1.25))
            if uses_s:
                _arrow(ax, (1.05, 1.51), (1.5, 1.51), GLARE)
            _arrow(ax, (2.9, 1.51), (3.3, 1.51))
        else:
            _arrow(ax, (1.05, 2.06), (3.3, 1.65))
            _arrow(ax, (1.05, 0.96), (3.3, 1.37))
        _box(ax, 3.3, 1.25, 0.6, 0.52, "F", GOOD, face="#e2f3ea", bold=True)
        ax.text(
            2.0,
            0.3,
            formula,
            fontsize=8,
            ha="center",
            va="center",
            color=INK,
            family="DejaVu Sans Mono",
            linespacing=1.5,
        )
    fig.tight_layout(pad=0.6)
    fig.savefig(FIG / "theory_fusion_rules.png", dpi=200, facecolor="white")
    plt.close(fig)


def gate_curve() -> None:
    s = np.linspace(0, 1, 200)
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    for slope, label, color in (
        (5.0, "strong start (slope 5)", GLARE),
        (0.1, "weak start (slope 0.1)", ACCENT),
    ):
        ax.plot(s, 1 / (1 + np.exp(-(0.0 - slope * s))), color=color, lw=2, label=label)
    ax.axhline(0.5, color=MUTED, lw=0.8, ls="--")
    ax.text(0.99, 0.515, "mean fusion (g = 0.5)", ha="right", fontsize=7.5, color=MUTED)
    ax.set_xlabel("share of clipped pixels in the patch, s")
    ax.set_ylabel("trust in RGB, g")
    ax.set_ylim(0, 0.62)
    ax.grid(color=GRID, lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "theory_gate_curve.png", dpi=200, facecolor="white")
    plt.close(fig)


def config_matrix() -> None:
    rows = [
        "RGB only",
        "Event only",
        "Mean",
        "Concat",
        "Content gate",
        "Clipping gate",
        "Constant control",
        "Fixed clipping gate",
        "Content + clipping gate",
        "Attention",
        "Attention + clipping bias",
    ]
    cols = [
        "clean\ntraining",
        "glare\ntraining",
        "raw events\n(clean)",
        "raw events\n(glare)",
        "weak glare\n(25%)",
        "gate\nablations",
        "extra seeds\n(x2)",
        "fine-tune",
    ]
    # counts of configurations per cell
    grid = np.zeros((len(rows), len(cols)), dtype=int)
    cells = {
        "RGB only": {0: 1, 1: 1, 6: 2, 7: 1},
        "Event only": {0: 1, 2: 1},
        "Mean": {0: 1, 1: 1, 2: 1, 3: 1, 6: 2, 7: 1},
        "Concat": {0: 1, 1: 1},
        "Content gate": {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 6: 2, 7: 1},
        "Clipping gate": {0: 1, 1: 1, 5: 2, 6: 2},
        "Constant control": {0: 1, 1: 1, 6: 2},
        "Fixed clipping gate": {0: 1, 1: 1, 6: 2},
        "Content + clipping gate": {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 2, 7: 1},
        "Attention": {0: 1, 1: 1},
        "Attention + clipping bias": {1: 1},
    }
    for r, name in enumerate(rows):
        for c, n in cells[name].items():
            grid[r, c] = n
    assert grid.sum() == 50, grid.sum()
    fig, ax = plt.subplots(figsize=(8.6, 4.3))
    for r in range(len(rows)):
        for c in range(len(cols)):
            n = grid[r, c]
            face = "#f4f5f8" if n == 0 else ("#cfdaf7" if n == 1 else "#8fa8ee")
            ax.add_patch(
                plt.Rectangle((c, len(rows) - 1 - r), 0.94, 0.9, facecolor=face, edgecolor="none")
            )
            if n:
                ax.text(
                    c + 0.47,
                    len(rows) - 1 - r + 0.45,
                    str(n),
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=INK,
                )
    ax.set_xlim(0, len(cols))
    ax.set_ylim(0, len(rows))
    ax.set_xticks(np.arange(len(cols)) + 0.47, cols, fontsize=8)
    ax.set_yticks(np.arange(len(rows)) + 0.45, list(reversed(rows)), fontsize=8.5)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    totals = grid.sum(0)
    for c, t in enumerate(totals):
        ax.text(c + 0.47, -0.45, f"{t}", ha="center", fontsize=8.5, color=MUTED)
    ax.text(len(cols) + 0.1, -0.45, "= 50", fontsize=9, weight="bold", color=INK, va="center")
    ax.text(-0.1, -0.45, "configurations", ha="right", fontsize=8.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(FIG / "theory_config_matrix.png", dpi=200, facecolor="white")
    plt.close(fig)


def _geodesic(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    trace = np.einsum("nij,mij->nm", a, b)
    return np.degrees(np.arccos(np.clip((trace - 1) / 2, -1, 1)))


def _roll_free(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    m = np.einsum("mij,nkj->nmik", b, a)
    trace = np.hypot(m[..., 0, 0] + m[..., 1, 1], m[..., 1, 0] - m[..., 0, 1]) + m[..., 2, 2]
    return np.degrees(np.arccos(np.clip((trace - 1) / 2, -1, 1)))


def pose_coverage() -> None:
    plain, free, validation = [], [], []
    for obj in ("cassini", "satty", "soho"):

        def rotations(split: str) -> np.ndarray:
            cache = FreshCropCacheDataset(CACHE / split)
            return np.asarray(cache.array("pose_object_to_camera_mm"))[:, :3, :3].astype(float)

        train = rotations(f"{obj}-train")[::3]
        real = rotations(f"real-{obj}")
        real = real[np.linspace(0, len(real) - 1, 500).astype(int)]
        plain.append(_geodesic(real, train).min(1))
        free.append(_roll_free(real, train).min(1))
        validation.append(_geodesic(rotations(f"{obj}-validation")[::2], train).min(1))
    bins = np.linspace(0, 90, 46)
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    ax.hist(
        np.concatenate(validation),
        bins=bins,
        density=True,
        color=MUTED,
        alpha=0.35,
        label="synthetic validation frames",
    )
    ax.hist(
        np.concatenate(plain), bins=bins, density=True, color=RGB_C, alpha=0.6, label="real frames"
    )
    ax.hist(
        np.concatenate(free),
        bins=bins,
        density=True,
        histtype="step",
        color=GOOD,
        lw=1.8,
        label="real frames, in-plane roll allowed",
    )
    ax.set_xlabel("rotation distance to the nearest training pose, degrees")
    ax.set_ylabel("density")
    ax.set_yticks([])
    ax.grid(axis="x", color=GRID, lw=0.8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "theory_pose_coverage.png", dpi=200, facecolor="white")
    plt.close(fig)


def augmentation_samples() -> None:
    cache = FreshCropCacheDataset(CACHE / "cassini-validation")
    items = [cache[i] for i in (40, 330)]
    keys = (
        "rgb_tss",
        "event_activity",
        "event_raw",
        "event_magnitude",
        "clipping_ratio_tss",
        "keypoints_xy",
    )
    batch = {k: torch.stack([it[k] for it in items]) for k in keys}  # type: ignore[literal-required]
    inputs = {
        "rgb": batch["rgb_tss"],
        "event": event_input(batch, "activity"),
        "clipping": batch["clipping_ratio_tss"],
    }
    fig, axes = plt.subplots(2, 6, figsize=(10.5, 3.7))
    for row in range(2):
        single = {k: v[row : row + 1] for k, v in inputs.items()}
        kp = batch["keypoints_xy"][row : row + 1]
        axes[row, 0].imshow(single["rgb"][0].permute(1, 2, 0))
        axes[row, 0].scatter(kp[0, :, 0], kp[0, :, 1], s=6, c="#2bd96b")
        axes[row, 1].imshow(single["event"][0, 0], cmap="gray")
        for col in range(2):
            out, moved, _ = augment_batch(
                single, kp, Sim2RealConfig(), torch.Generator().manual_seed(11 + 5 * row + col)
            )
            axes[row, 2 + 2 * col].imshow(out["rgb"][0].permute(1, 2, 0))
            axes[row, 2 + 2 * col].scatter(moved[0, :, 0], moved[0, :, 1], s=6, c="#2bd96b")
            axes[row, 3 + 2 * col].imshow(out["event"][0, 0], cmap="gray")
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, 255)
        ax.set_ylim(255, 0)
    for col, title in enumerate(
        (
            "original RGB",
            "original events",
            "augmented RGB",
            "augmented events",
            "augmented RGB",
            "augmented events",
        )
    ):
        axes[0, col].set_title(title, fontsize=8.5)
    fig.tight_layout(pad=0.4)
    fig.savefig(FIG / "theory_augmentation.png", dpi=170, facecolor="white")
    plt.close(fig)


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    fusion_rules()
    gate_curve()
    config_matrix()
    pose_coverage()
    augmentation_samples()
    print("theory figures written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
