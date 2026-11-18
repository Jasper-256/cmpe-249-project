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
    CELL,
    DATA,
    FRESH,
    H1,
    H2,
    HEAD,
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
    objects_all = sorted({o for r in rows for o in r["objects"]})
    seeds = {r["label"]: r for r in d["seeds"]}
    syn = {(r["label"], r["trained"]): r for r in d["synthetic_table"]}
    real = d["sim_to_real"]
    lo, hi = min(r["real"]["are_deg"] for r in real), max(r["real"]["are_deg"] for r in real)
    rgb, mean = by["RGB only"], by["Mean fusion"]
    " and ".join(o.capitalize() if o != "soho" else "SOHO" for o in objects_all)
    analysis = d["analyses"]["primary"]
    rv = d["real_v2"]
    rr = {r["label"]: r for r in rv["rows"]}
    ra, rr_ransac = rv["analyses"]["primary"], rv["analyses"]["primary_ransac"]
    rgb_r, ev_r = rr["RGB only"], rr["Event only"]

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
                f"synthetic data with a sim-to-real augmentation recipe, frozen, and evaluated "
                f"once on the 24 real sequences. The models now transfer: RGB-only reaches "
                f"{rgb_r['are_deg']:.0f}° mean rotation error on real data "
                f"({rgb_r['objects']['cassini']['are_deg']:.0f}° on Cassini), where every model "
                "of the earlier pass was at chance.",
                "<b>Answer so far: no.</b> On real data the clipping signal does not help, and in "
                "its plain form it hurts: removing it at test time improves the clipping gate. "
                "The models that relied most on events under glare (attention, clipping gate) "
                "were the best on synthetic data and the worst fusion rules on real data, and "
                "RGB-only beats every fusion rule.",
                "<b>Why.</b> Event-only models fail on real data "
                f"({ev_r['are_deg']:.0f}°) although they are among the best on synthetic data: the "
                "simulated events do not resemble real ones, and fusion inherits that gap. "
                "Single seed; SOHO fails for every model.",
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
    s.extend(h2("5. Synthetic results (all 33 models)"))
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
        "altered, noisy events), pooled over the three spacecraft. Single seed. Checkpoints are selected on the "
        "same set, so these are development numbers. Highlighted: the proposed rule.",
    )
    highlight(t, hl)
    s.append(t)
    s.append(
        bullets(
            [
                f"<b>Fusion beats RGB under glare</b> ({mean['glare']['are_deg']:.1f}° against "
                f"{rgb['glare']['are_deg']:.1f}° mean rotation error); medians are close, so the "
                "gain comes from the hard glare frames, mostly on SOHO.",
                "<b>Rules that can move trust away from clipped RGB do best:</b> attention with "
                f"the clipping bias ({by['Attention + clipping bias']['glare']['are_deg']:.1f}°), "
                f"attention ({by['Coarse attention']['glare']['are_deg']:.1f}°), event-only "
                f"({by['Event only']['glare']['are_deg']:.1f}°) and the clipping gate "
                f"({by['Clipping gate']['glare']['are_deg']:.1f}°, against "
                f"{by['Constant control (no s)']['glare']['are_deg']:.1f}° for its constant control).",
                "<b>Glare exposure still matters</b> with full augmentation: clean-trained models "
                f"reach about {by['RGB only, clean-trained']['glare']['are_deg']:.0f}° under glare.",
                "These are development numbers: one seed, and checkpoints were selected on the "
                "same validation set.",
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
    s.extend(h2("6. Real-sequence results (evaluated once)"))
    s.append(
        p(
            "All 33 checkpoints were hashed into a freeze manifest, then each was evaluated once "
            "on all 24 real sequences (7,894 frames). The comparisons and their Holm family were "
            "committed before any real result existed. Differences are paired over sequences, "
            "with 95% bootstrap intervals; negative means the first model is better."
        )
    )
    s.append(
        figure(
            "real_v2.png",
            "<b>Figure 6.</b> Left: mean rotation error on augmented synthetic glare validation. "
            "Right: the same models on the real sequences, per spacecraft (markers) and pooled "
            "(bar). Rows are sorted by real error. A random rotation averages 126.5°.",
            width=WIDTH * 0.96,
        )
    )
    rt = [["Method", "Rotation ° (mean / median)", "Transl. mm", "≤10°", "≤10° and 10 mm"]]
    rhl = []
    for i, r in enumerate(sorted(rv["rows"], key=lambda r: r["are_deg"]), start=1):
        rt.append(
            [
                r["label"],
                f"{r['are_deg']:.1f} / {r['are_median_deg']:.1f}",
                f"{r['ate_mm']:.0f}",
                f"{r['rot10_pct']:.0f}%",
                f"{r['joint_pct']:.1f}%",
            ]
        )
        if r["label"] == "Clipping gate":
            rhl.append(i)
    t = table(
        rt,
        [0.34, 0.24, 0.13, 0.11, 0.18],
        right_from=1,
        note="<b>Table 3.</b> Real sequences, pooled over the three spacecraft, EPnP pose solver, "
        "single seed. FRESH reports 29–45% joint success for its own models at 512 px; the "
        "protocols differ, so the numbers are not directly comparable. Highlighted: the "
        "proposed rule.",
    )
    highlight(t, rhl)
    s.append(t)

    short = {
        "s2r-u-rgb-tss": "RGB only",
        "s2r-u-event-activity": "Event only",
        "s2r-f-mean-tss-activity": "Mean",
        "s2r-f-content_gate-tss-activity": "Content gate",
        "s2r-f-content_clip_gate-tss-activity": "Content + clipping",
        "s2r-f-clip_gate-tss-activity": "Clipping gate",
        "s2r-f-clip_gate_constant-tss-activity": "Constant control",
        "s2r-f-attention-tss-activity": "Attention",
        "s2r-f-attention_clip-tss-activity": "Attention + clipping",
        "s2r-u-rgb-natural": "RGB only, clean",
        "s2r-f-mean-natural-activity": "Mean, clean",
    }
    ransac = {(c["candidate"], c["baseline"]): c for c in rr_ransac["comparisons"]}
    ct = [["Comparison", "EPnP Δ° [95% CI]", "Holm p", "RANSAC Δ°", "Harsh frames Δ°"]]
    chl = []
    for i, c in enumerate(ra["comparisons"], start=1):
        o = c["overall"]["are_deg"]
        holm = o.get("p_holm_tss_family")
        if c["family"] == "primary":
            holm_text = f"{o['p_two_sided']:.2f} (primary)"
            chl.append(i)
        elif holm is None:
            holm_text = "exploratory"
        else:
            holm_text = "<0.001" if holm < 0.001 else f"{holm:.3f}"
        rc = ransac[(c["candidate"], c["baseline"])]["overall"]["are_deg"]
        ct.append(
            [
                p(f"{short[c['candidate']]} vs {short[c['baseline']]}", CELL),
                f"{o['mean_difference']:+.1f} [{o['ci95'][0]:+.1f}, {o['ci95'][1]:+.1f}]",
                holm_text,
                f"{rc['mean_difference']:+.1f}",
                f"{c['harsh']['are_deg']['mean_difference']:+.1f}",
            ]
        )
    t = table(
        ct,
        [0.34, 0.24, 0.15, 0.12, 0.15],
        right_from=1,
        note="<b>Table 4.</b> Pre-registered paired comparisons on the real sequences "
        "(rotation error, first model minus second; negative favours the first). Harsh frames: "
        "frames the FRESH authors labelled as harsh lighting.",
    )
    highlight(t, chl)
    s.append(t)
    iv = ra["interventions"]["s2r-f-clip_gate-tss-activity"]
    att = next(
        c
        for c in ra["comparisons"]
        if c["candidate"] == "s2r-f-attention-tss-activity"
        and c["baseline"] == "s2r-f-mean-tss-activity"
    )
    s.append(
        bullets(
            [
                "<b>The recipe fixed transfer.</b> Every glare-trained model is far from chance "
                f"on Cassini and Satty; RGB-only gets {rgb_r['rot10_pct']:.0f}% of frames within "
                f"10° and {rgb_r['joint_pct']:.0f}% within 10° and 10 mm. Translation limits the "
                "joint rate. Glare training is worth about 25–35° on real data (Table 4, last "
                "rows).",
                "<b>The clipping signal does not help.</b> The primary comparison is a tie "
                f"({ra['primary']['are_deg']['mean_difference']:+.1f}°). The clipping gate is "
                "worse than its constant control and than plain averaging, and zeroing its "
                f"clipping map at test time <i>improves</i> it by {-iv['clip_zero']['delta_are_deg']:.1f}°: "
                "on real data the signal moves trust in the wrong direction. The one place it "
                "helps is as a bias inside attention.",
                "<b>Synthetic rankings reverse.</b> The rules that relied most on events under "
                "glare were the best on synthetic data and the worst fusion rules on real data. "
                f"Attention is {att['overall']['are_deg']['mean_difference']:.0f}° worse than plain "
                "averaging, which supports FRESH's concern about learned attention, at least in "
                "this form.",
                f"<b>The event branch is the bottleneck.</b> Event-only fails ({ev_r['are_deg']:.0f}°), "
                "and every fusion rule is worse than RGB-only. Our reading: simulated events differ "
                "from real ones far more than rendered RGB differs from real RGB, so moving trust "
                "to events under glare moves it to the less reliable sensor. This is an "
                "interpretation that we have not yet measured directly.",
                "<b>SOHO fails for every model</b> (62–90°), including RGB-only. It has the most "
                "keypoints and the largest synthetic glare errors; we have not diagnosed it yet.",
            ]
        )
    )

    # 7 ---------------------------------------------------------------------------------------
    s.extend(h2("7. Earlier exploration: a negative transfer result"))
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
            "<b>Disclosure.</b> The real sequences had been evaluated once before the current "
            "study, and the roll range in our recipe was motivated by their pose labels, so the "
            "current real evaluation was not a blind first look. No checkpoint, threshold or "
            "augmentation magnitude was tuned on real results."
        )
    )

    # 8 ---------------------------------------------------------------------------------------
    s.extend(h2("8. Questions for you"))
    s.append(
        p(
            "Our goal beyond the course is a publishable result, and these are the points where "
            "your judgement would help most."
        )
    )
    s.append(
        bullets(
            [
                "<b>Contribution.</b> The result is now a controlled negative one with a "
                "mechanism: an explicit clipping signal, and learned fusion more generally, help "
                "on synthetic data but hurt on real data because they shift trust to a simulated "
                "event stream that does not transfer. Together with what transfer on FRESH "
                "requires (roll coverage, augmentation, glare exposure), is this a publishable "
                "contribution, and what would make it convincing: more seeds, a second dataset, "
                "or a fix that closes the event gap?",
                "<b>Next experiment.</b> We see two directions: (a) make the event branch "
                "transfer (realistic noise and thresholds in the event simulation, event-side "
                "augmentation, or a little unlabelled real event data), then re-test the "
                "clipping gate; or (b) stronger attention variants (full multi-scale "
                "cross-attention, attention replacing mean fusion), which we can run on Colab. "
                "Our view is that (a) comes first, since any fusion rule is limited by the weaker "
                "sensor. Which would you prioritise?",
                "<b>Held-out data.</b> Further method changes will be informed by these real "
                "results, so the 24 real sequences can no longer serve as an untouched test set. "
                "Would you accept splitting them into development and test sequences (by "
                "spacecraft or lighting condition), or should we keep them closed and report the "
                "next round as exploratory?",
                "<b>Resolution.</b> We chose 256 px early so the full comparison would fit on one "
                "laptop GPU; FRESH uses 512 px. We plan 512 px as future scope or for the best few "
                "models. Would reviewers expect it in the main results?",
            ],
            numbered=True,
        )
    )

    # 9 ---------------------------------------------------------------------------------------
    s.extend(h2("9. What is left"))
    s.append(
        KeepTogether(
            bullets(
                [
                    "<b>Two more seeds</b> for the central comparisons (clipping gate against "
                    "constant control and mean; attention against mean; RGB-only against mean).",
                    "<b>Measure the event gap directly:</b> compare real and simulated event "
                    "statistics (rate, noise, polarity balance, edge sharpness) and relate "
                    "per-sequence event quality to fusion error.",
                    "<b>SOHO diagnosis:</b> keypoint error per landmark and the effect of the "
                    "RANSAC solver.",
                    "<b>Failure analysis</b> by lighting severity and low-motion frames, and "
                    "efficiency (latency, throughput, peak memory) per fusion rule.",
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
