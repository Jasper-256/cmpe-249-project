# ruff: noqa: RUF001, E501  (report prose: typographic dashes and long sentences are intended)
"""Build the progress check-in PDF, centred on the current (augmented) study.

The earlier unaugmented exploration is summarised as a negative transfer
result. Run ``scripts/progress_figures.py`` and ``scripts/theory_figures.py``
first, then ``uv run --with reportlab python scripts/build_checkin.py``.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from build_progress_report import (
    DATA,
    FRESH,
    H1,
    H2,
    META,
    WIDTH,
    bullets,
    figure,
    fmt_int,
    footer,
    highlight,
    link,
    p,
    summary_box,
    table,
)
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import CondPageBreak, KeepTogether, SimpleDocTemplate, Spacer

OUT = Path("outputs/report/progress-checkin.pdf")
REPO = "https://github.com/Jasper-256/cmpe-249-project"
DEMO = "https://advaitshinde.com/#ongoing"


def h2(text: str) -> list[Any]:
    return [CondPageBreak(1.6 * inch), p(text, H2)]


def build() -> Path:
    d = json.loads(DATA.read_text(encoding="utf-8"))
    study = d["current_study"]
    rows = study["rows"]
    by = {r["label"]: r for r in rows}
    done, planned = study["models_done"], study["models_planned"]
    objects_all = sorted({o for r in rows for o in r["objects"]})
    seeds = {r["label"]: r for r in d["seeds"]}
    syn = {(r["label"], r["trained"]): r for r in d["synthetic_table"]}
    real = d["sim_to_real"]
    lo, hi = min(r["real"]["are_deg"] for r in real), max(r["real"]["are_deg"] for r in real)
    fusion = [
        r
        for r in rows
        if r["label"] not in ("RGB only", "Event only")
        and "clean-trained" not in r["label"]
        and len(r["objects"]) == len(objects_all)
    ]
    f_lo, f_hi = (
        min(r["glare"]["are_deg"] for r in fusion),
        max(r["glare"]["are_deg"] for r in fusion),
    )
    rgb, mean = by["RGB only"], by["Mean fusion"]
    names = " and ".join(o.capitalize() if o != "soho" else "SOHO" for o in objects_all)
    analysis = d["analyses"]["primary"]

    s: list[Any] = []
    s.append(p("Glare-aware event–RGB fusion for spacecraft pose", H1))
    s.append(
        p(
            "Advait Shinde, Jasper Morgal  ·  CMPE 249  ·  Progress check-in, "
            f"{date.today():%B} {date.today().day}, {date.today().year}<br/>"
            f"Code: {link(REPO, 'github.com/Jasper-256/cmpe-249-project')} (branch "
            "<font name='Sans-Bold'>fusion-matrix-v2</font>)<br/>"
            f"Interactive demo of the problem: {link(DEMO, 'advaitshinde.com/#ongoing')}",
            META,
        )
    )
    s.append(Spacer(1, 10))
    s.append(
        summary_box(
            [
                "<b>Question.</b> Does telling a fusion network where the RGB image is blown out "
                "by glare help it combine RGB and event cameras for spacecraft pose, beyond what a "
                "network of the same size learns by itself?",
                f"<b>Current study.</b> 33 models (11 configurations × 3 spacecraft) trained on "
                f"synthetic data with a sim-to-real augmentation recipe; {done} are done. On "
                f"synthetic validation that mimics real conditions they reach about "
                f"{min(r['glare']['are_deg'] for r in rows):.0f}–{mean['glare']['are_deg']:.0f}° rotation error "
                f"with fusion, against {rgb['glare']['are_deg']:.1f}° for RGB alone under glare.",
                "<b>So far</b> the fusion rules are tied on synthetic data: the explicit clipping "
                "signal neither helps nor hurts. The real-sequence evaluation, which decides the "
                "question, runs when training finishes (about a day).",
                "<b>Earlier exploration.</b> A first pass without augmentation (150 models) gave "
                "clear synthetic findings but did not transfer to real sequences. That negative "
                "result shaped the current recipe (Section 6).",
            ]
        )
    )

    # 1 ---------------------------------------------------------------------------------------
    s.extend(h2("1. The question"))
    s.append(
        p(
            "Docking with a spacecraft requires its 6-DoF pose (rotation and translation) from "
            "camera images. Sunlight off reflective surfaces saturates RGB cameras exactly when "
            "precision matters. Event cameras report per-pixel brightness changes instead of "
            "absolute intensity, so they keep seeing edges where RGB is white. The FRESH paper "
            f"({link(FRESH, 'Jawaid et al., 2025')}) fuses the two with a learning-free RANSAC "
            "step and argues that learned attention fusion would overfit synthetic training data."
        )
    )
    s.append(
        p(
            "<b>Our question:</b> does an explicit, local signal of <i>where</i> RGB is clipped "
            "help a learned fusion network trust the right sensor, beyond what a network of the "
            "same size learns without it? We also test FRESH's concern about attention directly. "
            "A clean negative answer is an acceptable outcome; the experiment is built to "
            "separate the physical signal from model capacity."
        )
    )

    # 2 ---------------------------------------------------------------------------------------
    s.extend(h2("2. Data and task"))
    s.append(
        p(
            "<b>Data.</b> FRESH has three spacecraft mock-ups (Cassini, Satty, SOHO). Training "
            "uses synthetic renders with simulated events; testing uses 24 real lab sequences "
            "(7,894 frames) filmed under harsh lighting. <b>Task.</b> From an RGB crop and an "
            "event frame, predict 18 keypoints (26 for SOHO) as heatmaps, then solve PnP for the "
            "pose. <b>Metrics.</b> Rotation error (°), translation error (mm), and FRESH's "
            "success rates (share of frames under 10° and under 10 mm). Failed pose solves are "
            "scored, not dropped."
        )
    )
    s.append(
        p(
            "<b>Glare in training.</b> The synthetic images were rendered with soft light and "
            "never clip, so a clipping-driven model would never see its input change. We add "
            "<i>targeted specular saturation</i> (TSS): bright blooming highlights on the "
            "spacecraft in half the training images. The <b>clipping map <i>s</i></b> is the "
            "share of saturated pixels in each of 16×16 patches of the RGB crop."
        )
    )
    s.append(
        figure(
            "example_inputs.png",
            "<b>Figure 1.</b> Model inputs for three synthetic samples: RGB, RGB with TSS glare, "
            "the clipping map <i>s</i>, and three event encodings. Green circles are ground-truth "
            "keypoints.",
        )
    )

    # 3 ---------------------------------------------------------------------------------------
    s.extend(h2("3. Model and what varies"))
    s.append(
        figure(
            "architecture.png",
            "<b>Figure 2.</b> Two pretrained HRNet-W18 encoders (one per sensor) are fused at all "
            "five feature scales, followed by a shared heatmap decoder and PnP. Only the fusion "
            "rule changes between models.",
            width=WIDTH * 0.92,
        )
    )
    rules = [
        ["Fusion rule", "Fusion params", "What it tests"],
        ["Mean", "0", "Simplest possible fusion"],
        [
            "Content gate",
            fmt_int(syn[("Content gate", "glare")]["fusion_params"]),
            "Learned per-location reliability, no clipping signal",
        ],
        [
            "<b>Clipping gate</b> (proposed)",
            fmt_int(syn[("Clipping gate", "glare")]["fusion_params"]),
            "g = σ(b − softplus(a)·s): trust in RGB falls as clipping rises",
        ],
        [
            "Constant control",
            fmt_int(syn[("Constant control (no s)", "glare")]["fusion_params"]),
            "The same module with s = 0: separates the signal from capacity",
        ],
        [
            "Content + clipping gate",
            fmt_int(syn[("Content + clipping gate", "glare")]["fusion_params"]),
            "Does adding s to a learned gate help?",
        ],
        [
            "Coarse attention",
            fmt_int(syn[("Coarse attention", "glare")]["fusion_params"]),
            "FRESH's concern: attention over both sensors' tokens",
        ],
        [
            "Attention + clipping bias",
            fmt_int(syn[("Attention + clipping bias", "glare")]["fusion_params"]),
            "Attention steered away from clipped RGB patches",
        ],
    ]
    from build_progress_report import CELL, HEAD

    rules_t = [
        [p(r[0], CELL), r[1], p(r[2], CELL)] if j else [r[0], r[1], p(r[2], HEAD)]
        for j, r in enumerate(rules)
    ]
    s.append(
        table(
            rules_t,
            [0.28, 0.13, 0.59],
            right_from=1,
            note="<b>Table 1.</b> Fusion rules in the current study, plus RGB-only and event-only "
            "baselines. Encoders, decoder, data, optimizer and budget are identical across models. "
            "The companion theory guide explains each rule.",
        )
    )

    # 4 ---------------------------------------------------------------------------------------
    s.extend(h2("4. Training recipe"))
    s.append(
        p(
            "Synthetic and real images differ in ways unrelated to the question: the synthetic "
            "set has no in-plane camera roll, a black background, no blur or sensor noise, and "
            "clean events. Every model is therefore trained with <b>sim2real-v1</b> augmentation: "
            "random in-plane rotation (±180°), scale and shift, applied identically to RGB, "
            "events and the clipping map; brightness, contrast, colour, background, blur and "
            "noise on RGB; dropped and added events. The recipe follows the augmentation "
            "families FRESH reports for its own networks. Magnitudes are fixed in code, and "
            "checkpoints are selected on an augmented copy of synthetic validation, never on "
            "real data."
        )
    )
    s.append(
        figure(
            "theory_augmentation.png",
            "<b>Figure 3.</b> Left: an original synthetic RGB crop and its events (top clean, "
            "bottom with TSS glare). Right: two random augmentations of each. Keypoints (green) "
            "move with the image.",
        )
    )
    s.append(
        p(
            "<b>Design.</b> Nine glare-trained configurations (Table 1 plus the two baselines) "
            "and two clean-trained controls (RGB-only, mean fusion) that test whether glare "
            "exposure still matters once general augmentation is on. Each is trained once per "
            "spacecraft, on all training frames, with one seed to start: 33 models, about 35 "
            "GPU-hours on a laptop RTX 5060. Every run records its config, seed, code and data "
            "hashes."
        )
    )

    # 5 ---------------------------------------------------------------------------------------
    s.extend(h2(f"5. Results so far (synthetic, {done} of {planned} models)"))
    tr = [
        [
            "Method",
            "Spacecraft done",
            "Clean rot. °",
            "Glare rot. ° (mean / median)",
            "Glare ≤10°/10 mm",
        ]
    ]
    hl = []
    for i, r in enumerate(rows, start=1):
        tr.append(
            [
                r["label"],
                ", ".join(o.capitalize() if o != "soho" else "SOHO" for o in r["objects"]),
                f"{r['clean']['are_deg']:.2f}",
                f"{r['glare']['are_deg']:.2f} / {r['glare']['are_median_deg']:.2f}",
                f"{r['glare']['success_pct']:.0f}%",
            ]
        )
        if r["label"] == "Clipping gate":
            hl.append(i)
    t = table(
        tr,
        [0.3, 0.2, 0.13, 0.22, 0.15],
        right_from=2,
        note="<b>Table 2.</b> Augmented synthetic validation (rotated, rescaled, photometrically "
        "altered, noisy events), pooled over the spacecraft finished so far; rows with fewer "
        "spacecraft are not directly comparable. Single seed. Checkpoints are selected on the "
        "same set, so these are development numbers. Highlighted: the proposed rule.",
    )
    highlight(t, hl)
    s.append(t)
    s.append(
        bullets(
            [
                f"<b>Fusion beats RGB under glare</b> ({mean['glare']['are_deg']:.1f}° against "
                f"{rgb['glare']['are_deg']:.1f}° mean rotation error on {names}); medians are close, "
                "so the gain comes from the hard glare frames.",
                f"<b>Fusion rules are tied.</b> On {names} they span {f_lo:.1f}–{f_hi:.1f}°. "
                "The clipping gate does not beat its constant control, and adding the clipping "
                "signal to a content gate changes nothing measurable. With one seed, differences "
                "of this size are noise.",
                "<b>Event-only is strong</b> once trained with augmentation on all frames, so "
                "fusion's margin over the best single sensor is small on synthetic data. Real "
                "event frames are noisier than simulated ones; the real test will show whether "
                "this holds.",
                "<b>Attention trains as well as simpler fusion</b> on synthetic data (Cassini so "
                "far). Whether it transfers is the direct test of FRESH's concern.",
            ]
        )
    )
    if (Path("outputs/figures") / "augmentation_effect.png").is_file():
        s.append(
            figure(
                "augmentation_effect.png",
                "<b>Figure 4.</b> Effect of the training recipe on the augmented glare validation "
                "set. Crosses: models from the earlier exploration, trained without augmentation. "
                "Circles: the current models.",
                width=WIDTH * 0.86,
            )
        )

    # 6 ---------------------------------------------------------------------------------------
    s.extend(h2("6. Earlier exploration: a negative transfer result"))
    cg, ccg = (
        analysis["seeds"]["f-content_gate-tss-activity"],
        analysis["seeds"]["f-content_clip_gate-tss-activity"],
    )
    s.append(
        p(
            "Before the current study we trained 150 models (50 configurations × 3 spacecraft) "
            "with the same architecture but no augmentation other than TSS, froze them, and "
            "evaluated each once on the real sequences under a pre-registered analysis. "
            "The theory guide lists the configurations."
        )
    )
    s.append(
        bullets(
            [
                "<b>Synthetic findings.</b> Glare exposure in training decides robustness: "
                "clean-trained models collapse under glare (25–35°). Fusion halves the error "
                f"under glare ({seeds['Mean fusion']['are_mean']:.1f} ± {seeds['Mean fusion']['are_sd']:.1f}° "
                f"against {seeds['RGB only']['are_mean']:.1f} ± {seeds['RGB only']['are_sd']:.1f}° for "
                "RGB-only, three seeds). Fusion rules are within seed noise of each other. A "
                "robust RANSAC pose solver helps every model about as much as the fusion rule.",
                f"<b>No transfer to real data.</b> Every model, including the single-sensor "
                f"baselines, had {lo:.0f}–{hi:.0f}° mean rotation error on the real sequences "
                "(a random guess averages 126.5°) and under 2% of frames within 10 mm, where "
                "FRESH reports 29–45% for its own models. Labels and geometry check out: PnP on "
                "the ground-truth keypoints recovers the true pose to within 0.01°.",
                "<b>Cause.</b> All synthetic poses have zero in-plane roll, while a typical "
                "real frame needs 42–83° of roll to match any training pose (Figure 5). "
                "Rotating synthetic validation images reproduces the failure without real data "
                "(4° → 44–90°), and the full augmentation reproduces its size (Figure 4, "
                "crosses). Appearance differences add to it.",
                "<b>Pre-registered comparison</b>, reported for completeness: content + clipping "
                "gate minus content gate, "
                f"{analysis['primary']['are_deg']['mean_difference']:+.1f}° rotation (95% CI "
                f"{analysis['primary']['are_deg']['ci95'][0]:+.1f} to {analysis['primary']['are_deg']['ci95'][1]:+.1f}; "
                f"three seeds: {ccg['are_mean_deg']:.0f} ± {ccg['are_sd_deg']:.0f}° against "
                f"{cg['are_mean_deg']:.0f} ± {cg['are_sd_deg']:.0f}°). With both models near chance, "
                "no translation gain, and no loss when the clipping map is zeroed at test time, "
                "we do not interpret it.",
            ]
        )
    )
    s.append(
        figure(
            "theory_pose_coverage.png",
            "<b>Figure 5.</b> Rotation distance from each frame to its nearest training pose "
            "(pose labels only). Real frames are far from the training poses unless in-plane "
            "roll is allowed; then all are within 10°.",
            width=WIDTH * 0.74,
        )
    )
    s.append(
        p(
            "<b>What this means for the current study.</b> The real sequences have now been "
            "evaluated once, and the roll range in our recipe was motivated by their pose "
            "labels. The coming real evaluation is therefore not a blind first look; we will "
            "report it next to the earlier one and say so. No checkpoint, threshold or "
            "augmentation magnitude was or will be tuned on real results."
        )
    )

    # 7 ---------------------------------------------------------------------------------------
    s.extend(h2("7. Questions for you"))
    s.append(
        p(
            "We will finish the study as planned: the remaining training, then one evaluation on "
            "all 24 real sequences, reported with the disclosure in Section 6. Our goal beyond "
            "the course is a publishable result, and these are the points where your judgement "
            "would help most."
        )
    )
    s.append(
        bullets(
            [
                "<b>Contribution.</b> If the models transfer but the clipping gate still ties "
                "with plain averaging, the paper would be a controlled negative result: explicit "
                "saturation gating does not help event–RGB fusion once models are trained with "
                "glare and sim-to-real augmentation, together with what transfer on FRESH "
                "requires (the roll gap, augmentation, the pose solver). Is that enough for a "
                "workshop or conference paper, and what would you add to strengthen it?",
                "<b>Attention.</b> So far attention is applied at the two coarsest scales, on top "
                "of mean fusion. We have extra GPU capacity (Colab) and would like to explore "
                "more: cross-attention at every scale, attention replacing rather than "
                "supplementing mean fusion, and token-level gating by the clipping map. Which of "
                "these would make the most convincing test of FRESH's claim that learned "
                "attention overfits synthetic data?",
                "<b>Resolution.</b> We chose 256 px input early on so that the full comparison "
                "(150 models, then 33) would fit on one laptop GPU; FRESH uses 512 px, which costs "
                "about four times as much. We plan to treat 512 px as future scope, or as a final "
                "run for the best few models. Would reviewers expect the 512 px comparison in the "
                "main results?",
            ],
            numbered=True,
        )
    )

    # 8 ---------------------------------------------------------------------------------------
    s.extend(h2("8. What is left"))
    s.append(
        KeepTogether(
            bullets(
                [
                    f"<b>Finish training</b> the remaining {planned - done} models (about a day).",
                    "<b>Real evaluation:</b> freeze and hash the checkpoints, evaluate each once, "
                    "paired sequence-level bootstrap for the comparisons in Table 1, with both "
                    "pose solvers and the clipping interventions (shuffle, zero, bypass).",
                    "<b>Two more seeds</b> for the central comparison if the single-seed results "
                    "are close.",
                    "<b>Efficiency:</b> latency, throughput and peak memory per fusion rule.",
                    "<b>Failure analysis</b> by lighting severity and event quality, then the "
                    "final report.",
                ]
            )
        )
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=letter,
        leftMargin=0.85 * inch,
        rightMargin=0.85 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.8 * inch,
        title="Glare-aware event-RGB fusion for spacecraft pose: progress check-in",
        author="Advait Shinde, Jasper Morgal",
    )
    doc.build(s, onFirstPage=footer, onLaterPages=footer)
    return OUT


if __name__ == "__main__":
    print(build())
