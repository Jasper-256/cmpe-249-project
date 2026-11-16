# ruff: noqa: RUF001, E501  (report prose: typographic dashes and long sentences are intended)
"""Build the progress check-in PDF from outputs/report/progress_data.json.

Run ``scripts/progress_figures.py`` first. Needs reportlab, which is not a
project dependency: ``uv run --with reportlab python scripts/build_progress_report.py``.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import matplotlib
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

DATA = Path("outputs/report/progress_data.json")
FIG = Path("outputs/figures")
OUT = Path("outputs/report/round1-detail.pdf")
REPO = "https://github.com/Jasper-256/cmpe-249-project/tree/fusion-matrix-v1"
DEMO = "https://advaitshinde.com/#ongoing"
FRESH = "https://arxiv.org/abs/2507.05698"

INK = colors.HexColor("#14171f")
MUTED = colors.HexColor("#5a6273")
RULE = colors.HexColor("#d9dce3")
ACCENT = colors.HexColor("#2448c8")
SOFT = colors.HexColor("#f2f4f8")
GLARE = colors.HexColor("#9a5d00")

fonts = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
for name, file in {
    "Serif": "STIXGeneral.ttf",
    "Serif-Bold": "STIXGeneralBol.ttf",
    "Serif-Italic": "STIXGeneralItalic.ttf",
    "Sans": "DejaVuSans.ttf",
    "Sans-Bold": "DejaVuSans-Bold.ttf",
}.items():
    pdfmetrics.registerFont(TTFont(name, str(fonts / file)))
pdfmetrics.registerFontFamily(
    "Serif", normal="Serif", bold="Serif-Bold", italic="Serif-Italic", boldItalic="Serif-Bold"
)
pdfmetrics.registerFontFamily(
    "Sans", normal="Sans", bold="Sans-Bold", italic="Sans", boldItalic="Sans-Bold"
)

BODY = ParagraphStyle(
    "body", fontName="Serif", fontSize=10.6, leading=14.2, textColor=INK, spaceAfter=6
)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=9.4, leading=12.4, spaceAfter=3)
CAPTION = ParagraphStyle(
    "caption",
    parent=BODY,
    fontSize=8.9,
    leading=11.6,
    textColor=MUTED,
    spaceBefore=3,
    spaceAfter=10,
)
H1 = ParagraphStyle(
    "h1", fontName="Sans-Bold", fontSize=15.5, leading=20, textColor=INK, spaceAfter=4
)
H2 = ParagraphStyle(
    "h2",
    fontName="Sans-Bold",
    fontSize=11.6,
    leading=15,
    textColor=INK,
    spaceBefore=12,
    spaceAfter=5,
    keepWithNext=1,
)
META = ParagraphStyle("meta", fontName="Sans", fontSize=8.6, leading=12, textColor=MUTED)
CELL = ParagraphStyle("cell", fontName="Sans", fontSize=7.9, leading=10, textColor=INK)
CELL_R = ParagraphStyle("cellr", parent=CELL, alignment=2)
HEAD = ParagraphStyle("head", parent=CELL, fontName="Sans-Bold", textColor=MUTED, fontSize=7.4)
HEAD_R = ParagraphStyle("headr", parent=HEAD, alignment=2)
BOX = ParagraphStyle("box", parent=BODY, fontSize=10.2, leading=13.6, alignment=TA_LEFT)

WIDTH = letter[0] - 2 * 0.85 * inch


def p(text: str, style: ParagraphStyle = BODY) -> Paragraph:
    return Paragraph(text, style)


def bullets(items: list[str], style: ParagraphStyle = BODY, numbered: bool = False) -> ListFlowable:
    return ListFlowable(
        [ListItem(p(t, style), leftIndent=14) for t in items],
        bulletType="1" if numbered else "bullet",
        bulletFontName="Sans",
        bulletFontSize=8 if not numbered else 9,
        leftIndent=14,
        spaceAfter=4,
    )


def figure(name: str, caption: str, width: float = WIDTH) -> KeepTogether:
    path = FIG / name
    with PILImage.open(path) as image:
        w, h = image.size
    return KeepTogether([Image(str(path), width=width, height=width * h / w), p(caption, CAPTION)])


def table(rows: list[list[Any]], widths: list[float], right_from: int = 1, note: str = "") -> Any:
    styled = []
    for r, row in enumerate(rows):
        cells = []
        for c, value in enumerate(row):
            if isinstance(value, Paragraph):
                cells.append(value)
                continue
            if r == 0:
                cells.append(p(str(value), HEAD_R if c >= right_from else HEAD))
            else:
                cells.append(p(str(value), CELL_R if c >= right_from else CELL))
        styled.append(cells)
    t = Table(styled, colWidths=[w * WIDTH for w in widths], repeatRows=1)
    t.setStyle(
        TableStyle(
            [
                ("LINEBELOW", (0, 0), (-1, 0), 0.8, INK),
                ("LINEBELOW", (0, 1), (-1, -1), 0.3, RULE),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 2.6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.6),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return KeepTogether([t, p(note, CAPTION)]) if note else t


def highlight(t: Any, rows: list[int]) -> None:
    table_ = t._content[0] if isinstance(t, KeepTogether) else t
    for r in rows:
        table_.setStyle(TableStyle([("BACKGROUND", (0, r), (-1, r), SOFT)]))


def summary_box(items: list[str]) -> Table:
    inner = [
        p(
            "<b>Where we are</b>",
            ParagraphStyle("bh", parent=BOX, fontName="Sans-Bold", fontSize=9.6, textColor=ACCENT),
        )
    ]
    inner.append(bullets(items, BOX))
    t = Table([[inner]], colWidths=[WIDTH])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), SOFT),
                ("LINEBEFORE", (0, 0), (0, -1), 2.2, ACCENT),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return t


def footer(canvas: Any, doc: Any) -> None:
    canvas.saveState()
    canvas.setFont("Sans", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(0.85 * inch, 0.5 * inch, "CMPE 249 progress check-in  ·  Shinde, Morgal")
    canvas.drawRightString(letter[0] - 0.85 * inch, 0.5 * inch, str(doc.page))
    canvas.restoreState()


def fmt_int(n: int | None) -> str:
    if n is None:
        return "—"
    if n >= 1_000_000:
        return f"{n / 1e6:.2f} M"
    return f"{n:,}"


def link(url: str, text: str) -> str:
    return f'<link href="{url}" color="#2448c8">{text}</link>'


def build() -> Path:
    d = json.loads(DATA.read_text(encoding="utf-8"))
    seeds = {r["label"]: r for r in d["seeds"]}
    real = {(r["label"], r["trained"]): r for r in d["sim_to_real"]}
    syn = {(r["label"], r["trained"]): r for r in d["synthetic_table"]}
    done, total = d["real_reports"], d["frozen_runs"]
    complete = d["g6_complete"]
    frames = next(iter(real.values()))["real"]["frames"]
    best_real = min(r["real"]["are_deg"] for r in d["sim_to_real"])
    worst_real = max(r["real"]["are_deg"] for r in d["sim_to_real"])
    max_trans = max(r["real"]["trans10mm_pct"] for r in d["sim_to_real"])
    mean_s, rgb_s = seeds["Mean fusion"], seeds["RGB only"]
    fusion_seeds = [r for r in d["seeds"] if r["label"] != "RGB only"]
    lo = min(r["are_mean"] for r in fusion_seeds)
    hi = max(r["are_mean"] for r in fusion_seeds)
    sd_lo = min(r["are_sd"] for r in fusion_seeds)
    sd_hi = max(r["are_sd"] for r in fusion_seeds)

    story: list[Any] = []
    story.append(p("Glare-aware event–RGB fusion for spacecraft pose", H1))
    story.append(
        p(
            "Advait Shinde, Jasper Morgal  ·  CMPE 249  ·  "
            f"Progress check-in, {date.today():%B} {date.today().day}, {date.today().year}<br/>"
            f"Code: {link(REPO, 'github.com/Jasper-256/cmpe-249-project')} (branch "
            f"<font name='Sans-Bold'>fusion-matrix-v1</font>)<br/>"
            f"Interactive demo of the problem: {link(DEMO, 'advaitshinde.com/#ongoing')}",
            META,
        )
    )
    story.append(Spacer(1, 10))
    story.append(
        summary_box(
            [
                f"We built the full pipeline and trained {total} models on synthetic data under "
                "one shared protocol: one model per spacecraft for each of 50 configurations.",
                "On synthetic data the picture is clear. Training with glare is what makes models "
                f"robust, and adding events halves the error under glare ({rgb_s['are_mean']:.1f}° "
                f"→ {mean_s['are_mean']:.1f}°). The explicit clipping signal does not beat plain "
                "averaging: across three seeds, all fusion rules are within noise of each other.",
                "On the real sequences every model fails, including RGB-only and event-only, with "
                f"mean rotation errors of {best_real:.0f}–{worst_real:.0f}°. We verified the "
                "labels and geometry. The main cause: the synthetic data has no in-plane camera "
                "roll and we trained without augmentation, so real frames show orientations and "
                "appearance the models never saw.",
                "We are retraining the core comparison with augmentation and will evaluate on "
                "the real set a second time, reported as post-hoc. We would like your view on "
                "that protocol and on framing (Section 7).",
            ]
        )
    )

    # 1 ---------------------------------------------------------------------------------------
    story.append(p("1. The question", H2))
    story.append(
        p(
            "Docking with a spacecraft requires knowing its 6-DoF pose (rotation and translation) "
            "from camera images. Sunlight off reflective surfaces saturates RGB cameras exactly "
            "when precision matters. Event cameras report per-pixel brightness changes instead of "
            "absolute intensity, so they do not saturate the same way. The FRESH paper "
            f"({link(FRESH, 'Jawaid et al., 2025')}) fuses the two with a learning-free RANSAC "
            "step and argues that learned attention fusion would overfit synthetic training data."
        )
    )
    story.append(
        p(
            "<b>Our question:</b> does an explicit, local signal of <i>where</i> the RGB image is "
            "clipped help a learned fusion network trust the right sensor, beyond what a network "
            "of the same size learns without that signal? A clean negative answer is an "
            "acceptable outcome; the experiment is designed to separate the physical signal "
            "from capacity."
        )
    )

    # 2 ---------------------------------------------------------------------------------------
    story.append(p("2. Data, task and a first surprise", H2))
    story.append(
        p(
            "<b>Data.</b> FRESH has three spacecraft mock-ups (Cassini, Satty, SOHO). Training "
            "uses synthetic renders with simulated events. Testing uses 24 real lab sequences "
            f"({frames:,} frames) filmed in a lab under harsh lighting. These are touched once, after every "
            "model is frozen. <b>Task.</b> From an RGB crop and an event frame, predict 18 "
            "keypoints (26 for SOHO) as heatmaps, then solve PnP for the pose. <b>Metrics.</b> "
            "Mean rotation error (°), translation error (mm), and FRESH's success rates: the share "
            "of frames under 10° and under 10 mm. Failed PnP solves are scored, not dropped."
        )
    )
    story.append(
        p(
            "<b>Synthetic images never clip.</b> Before training, we checked 384 sampled crops: "
            "no pixel exceeded 233 of 255, because FRESH rendered with diffuse lighting on "
            "purpose. A clipping-driven gate would never see its input change. We added "
            "<i>targeted specular saturation</i> (TSS): synthetic glare spots on the spacecraft "
            "in half the training images. Every configuration is trained twice, on clean and on "
            "glare data."
        )
    )
    story.append(
        figure(
            "example_inputs.png",
            "<b>Figure 1.</b> Model inputs for three synthetic SOHO samples: RGB, RGB with TSS "
            "glare, the 16×16 clipping map <i>s</i> derived from it, and three event encodings. "
            "Green circles are ground-truth keypoints.",
        )
    )

    # 3 ---------------------------------------------------------------------------------------
    story.append(p("3. Model and what varies", H2))
    story.append(
        figure(
            "architecture.png",
            "<b>Figure 2.</b> Pipeline. Two pretrained HRNet-W18 encoders (one per sensor) are "
            "fused at all five feature scales, followed by a shared heatmap decoder and PnP. Data, "
            "encoders, decoder, optimizer budget, epochs and checkpoint rule are identical across "
            "models; only the fusion rule changes.",
        )
    )
    rules = [
        ["Fusion rule", "Fusion params", "What it tests"],
        ["Mean", "0", "Simplest possible fusion"],
        [
            "Concat (1×1 conv)",
            fmt_int(syn[("Concat fusion", "glare")]["fusion_params"]),
            "Unconstrained learned mixing",
        ],
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
            "Same module with s = 0: separates the signal from capacity",
        ],
        ["Fixed clipping gate", "0", "Hard rule: switch to events wherever a patch clips"],
        [
            "Content + clipping gate",
            fmt_int(syn[("Content + clipping gate", "glare")]["fusion_params"]),
            "Pre-registered primary: does adding s to a learned gate help?",
        ],
        [
            "Coarse attention (± clipping bias)",
            fmt_int(syn[("Coarse attention", "glare")]["fusion_params"]),
            "FRESH's concern: attention over both sensors' tokens",
        ],
    ]
    rules = [
        [p(r[0], CELL), r[1], p(r[2], CELL)] if j else [r[0], r[1], p(r[2], HEAD)]
        for j, r in enumerate(rules)
    ]
    story.append(
        table(
            rules,
            [0.28, 0.13, 0.59],
            right_from=1,
            note="<b>Table 1.</b> Fusion rules. RGB-only and event-only baselines use one encoder "
            f"({fmt_int(syn[('RGB only', 'glare')]['total_params'])} parameters); fused models "
            f"have {fmt_int(syn[('Mean fusion', 'glare')]['total_params'])}–"
            f"{fmt_int(syn[('Content gate', 'glare')]['total_params'])}. Interventions at test "
            "time: shuffle or zero s, or force the gate to 0.5.",
        )
    )

    # 4 ---------------------------------------------------------------------------------------
    story.append(p("4. What we ran", H2))
    ran = [
        ["Stage", "What", "Models"],
        [
            "Data audit",
            "Fixed a wrong real label (one sequence off by a median 59°), mismatched "
            "synthetic/real event encodings, event-crop padding, and an evaluator that silently "
            "dropped failed pose solves",
            "—",
        ],
        [
            "Backbone screen",
            "Six efficient backbones (EfficientNet-B0, TinyViT, SegFormer, SwinV2, "
            "MobileViTv2, EfficientViT) against HRNet on a small overfit test; HRNet kept",
            "7",
        ],
        [
            "Main matrix",
            "8 fusion rules + 2 unimodal baselines × clean/glare training; raw vs "
            "activity events; gate ablations (unconstrained sign, weak init); weaker glare (25%); "
            "two extra seeds for the central models",
            "138",
        ],
        [
            "Fine-tuning",
            "Final candidate, the primary pair and RGB-only continue on all training "
            "frames at a lower learning rate",
            "12",
        ],
        [
            "Solver control",
            "Re-solve every model's keypoints with RANSAC PnP, to separate fusion "
            "from solver effects",
            "all",
        ],
        [
            "Frozen real test",
            "Hash all checkpoints, evaluate each once on the real sequences, "
            "then the pre-registered paired sequence bootstrap with Holm correction",
            f"{done}/{total}" if not complete else str(total),
        ],
    ]
    ran = [[p(r[0], CELL), p(r[1], CELL), r[2]] if j else r for j, r in enumerate(ran)]
    story.append(
        table(
            ran,
            [0.17, 0.71, 0.12],
            right_from=2,
            note="<b>Table 2.</b> About 50 GPU-hours on one laptop RTX 5060. Every run saves its "
            "resolved config, seed, git commit, lockfile and dataset hashes, and selection rule.",
        )
    )

    # 5 ---------------------------------------------------------------------------------------
    section5 = p("5. Synthetic results", H2)
    rows = [
        [
            "Method",
            "Trained on",
            "Clean val. rot. °",
            "Glare val. rot. °",
            "Glare val. transl. mm",
            "Glare ≤10°/10 mm",
        ]
    ]
    best_glare = min(r["glare"]["are_deg"] for r in d["synthetic_table"])
    hl = []
    for i, r in enumerate(d["synthetic_table"], start=1):
        rows.append(
            [
                r["label"],
                r["trained"],
                f"{r['clean']['are_deg']:.2f}",
                f"{r['glare']['are_deg']:.2f}",
                f"{r['glare']['ate_mm']:.1f}",
                f"{r['glare']['success_pct']:.1f}%",
            ]
        )
        if r["glare"]["are_deg"] == best_glare or r["label"] == "Clipping gate":
            hl.append(i)
    t = table(
        rows,
        [0.27, 0.11, 0.15, 0.15, 0.17, 0.15],
        right_from=2,
        note="<b>Table 3.</b> Synthetic validation, single seed, pooled over the three "
        "per-object models. Glare validation has TSS applied to about half the frames. "
        "Checkpoints are selected on the same validation set, so these are development "
        "numbers. Highlighted: best under glare, and the proposed rule.",
    )
    highlight(t, hl)
    story.append(KeepTogether([section5, t]))
    story.append(
        figure(
            "seed_spread.png",
            "<b>Figure 3.</b> Three training seeds for the central glare-trained models. Fusion "
            f"rules span {lo:.1f}–{hi:.1f}° with seed standard deviations of "
            f"{sd_lo:.1f}–{sd_hi:.1f}°: their differences are within seed noise. Fusion versus "
            "RGB-only is not.",
            width=WIDTH * 0.86,
        )
    )
    story.append(
        bullets(
            [
                "<b>Glare in training decides robustness.</b> Every clean-trained model that uses "
                "RGB collapses under glare (25–35°); every glare-trained fusion model stays "
                "under 10°.",
                f"<b>Fusion helps.</b> Glare-trained mean fusion reaches {mean_s['are_mean']:.1f} ± "
                f"{mean_s['are_sd']:.1f}° against {rgb_s['are_mean']:.1f} ± {rgb_s['are_sd']:.1f}° "
                "for RGB-only (three seeds); events alone reach 9.1°.",
                "<b>The clipping signal does not help on synthetic data.</b> The proposed gate is "
                "indistinguishable from its constant control. Strongly initialized gates send "
                "clipped patches almost entirely to events and do worst on exactly the glare "
                "frames (Figure 4). A weakly initialized gate barely moves from 50/50, which "
                "suggests the data favour mild routing at most.",
                "<b>The pose solver matters as much as the fusion rule.</b> A robust RANSAC "
                "re-solve improves every model (glare-trained mean fusion: 5.8° → 4.7°). With the "
                "same solver, learned fusion matches FRESH-style geometric fusion; the "
                "geometric baseline's earlier lead came from its solver.",
            ],
            numbered=True,
        )
    )
    story.append(
        figure(
            "corrupted_vs_clean.png",
            "<b>Figure 4.</b> Glare-trained models on the clean and the glare-corrupted halves of "
            "the glare validation set. Gates that act strongly on <i>s</i> lose the most on "
            "corrupted frames.",
            width=WIDTH * 0.8,
        )
    )
    story.append(
        figure(
            "training_curves.png",
            "<b>Figure 5.</b> Validation keypoint error during training (log scale). All fusion "
            "rules converge similarly; early stopping uses patience 4.",
        )
    )

    # 6 ---------------------------------------------------------------------------------------
    story.append(p("6. Real test" + ("" if complete else " (preliminary)"), H2))
    status = (
        f"All {total} frozen models have been evaluated and the pre-registered analysis has run."
        if complete
        else f"{done} of {total} frozen models have been evaluated, including every central "
        "single-seed comparison. A GPU-driver crash interrupted the evaluation on November 10; "
        "it was resumed on November 15, and finished reports are never re-run."
    )
    story.append(p(status))
    rrows = [
        [
            "Model",
            "Trained on",
            "Synthetic rot. °",
            "Real rot. ° (mean / median)",
            "Real ≤10°",
            "Real ≤10 mm",
        ]
    ]
    for r in d["sim_to_real"]:
        rr = r["real"]
        rrows.append(
            [
                r["label"],
                "glare" if r["trained"] == "tss" else "clean",
                f"{r['synthetic']['are_deg']:.1f}",
                f"{rr['are_deg']:.0f} / {rr['are_median_deg']:.0f}",
                f"{rr['rot10_pct']:.1f}%",
                f"{rr['trans10mm_pct']:.1f}%",
            ]
        )
    ref = len(rrows)
    for label, value in (
        ("FRESH RGB-only (reported)", "29%"),
        ("FRESH event-only (reported)", "32%"),
        ("FRESH RANSAC fusion (reported)", "45%"),
    ):
        rrows.append([p(f"<i>{label}</i>", CELL), "clean", "—", "—", "—", value])
    t = table(
        rrows,
        [0.30, 0.11, 0.14, 0.2, 0.11, 0.14],
        right_from=2,
        note="<b>Table 4.</b> Real sequences, frames pooled over the three objects. A random "
        "rotation guess averages 126.5°. FRESH rows are source-reported (mean over sequences, "
        "their detector crop, 512 px input), not reproduced; they are a rough reference only.",
    )
    highlight(t, list(range(ref, len(rrows))))
    story.append(t)
    story.append(
        figure(
            "sim_to_real.png",
            "<b>Figure 6.</b> The same models on clean synthetic validation (circles) and on the "
            "real test (diamonds).",
            width=WIDTH * 0.86,
        )
    )
    story.append(
        figure(
            "real_predictions.png",
            "<b>Figure 7.</b> Glare-trained mean fusion on real frames at its median error for "
            "each object (frames chosen by rank, not by eye). Predictions land on the spacecraft "
            "but on the wrong parts. Satty shows real glare far heavier than our synthetic TSS, "
            "and real event frames are much noisier than simulated ones.",
            width=WIDTH * 0.82,
        )
    )
    story.append(p("<b>What we checked.</b>"))
    story.append(
        bullets(
            [
                "<b>Labels and geometry are correct.</b> Solving PnP from the ground-truth "
                "keypoints recovers the ground-truth pose (median error below 0.01°) on all three "
                "objects, so intrinsics, units and pose conventions are right.",
                "<b>The models find the spacecraft but not its parts.</b> Predicted keypoints sit "
                "on the object, yet most are assigned to the wrong part, so PnP returns an "
                "unrelated orientation (Figure 7).",
                "<b>The gap is in our training, not the task.</b> FRESH's own HRNet models put "
                f"29–45% of real frames within 10 mm; ours stay under {max(2, round(max_trans + 0.5))}%. "
                "FRESH trains at 512 px with random rotation and translation, blur, colour "
                "jitter, noise and event noise. We trained at 256 px with no augmentation "
                "beyond TSS.",
                "<b>The synthetic data has no in-plane roll.</b> Every synthetic pose has exactly "
                "zero camera roll, while a typical real frame needs 42–83° of roll to match any "
                "training pose (Figure 8). A synthetic-only test agrees: rotating synthetic "
                "validation images in-plane raises a round-1 RGB model's error from 4° to 44°.",
            ]
        )
    )
    story.append(
        figure(
            "theory_pose_coverage.png",
            "<b>Figure 8.</b> Rotation distance from each frame to its nearest training pose "
            "(pose labels only). Real frames are far from the training poses unless in-plane "
            "roll is allowed; then all are within 10°.",
            width=WIDTH * 0.74,
        )
    )
    analysis = (d.get("analyses") or {}).get("primary", {})

    def seed_real(method: str) -> str:
        entry = analysis["seeds"][method]
        return f"{entry['are_mean_deg']:.0f} ± {entry['are_sd_deg']:.0f}°"

    prim = analysis.get("primary")
    if prim:
        a, t_ = prim["are_deg"], prim["ate_m"]
        story.append(
            p(
                "<b>Pre-registered primary comparison</b> (content + clipping gate minus content "
                f"gate, glare-trained, 24 sequences): rotation {a['mean_difference']:+.1f}° "
                f"(95% CI {a['ci95'][0]:+.1f} to {a['ci95'][1]:+.1f}), translation "
                f"{1000 * t_['mean_difference']:+.0f} mm (95% CI {1000 * t_['ci95'][0]:+.0f} to "
                f"{1000 * t_['ci95'][1]:+.0f}). The rotation difference repeats across three "
                f"seeds ({seed_real('f-content_clip_gate-tss-activity')} against "
                f"{seed_real('f-content_gate-tss-activity')}). We do not read it as support for "
                "the hypothesis: both models are near chance, translation does not improve, and "
                "zeroing the clipping map at test time does not hurt the clipping model. It is "
                "reported as pre-registered and re-tested in round 2."
            )
        )

    # 7 ---------------------------------------------------------------------------------------
    story.append(p("7. Questions for you", H2))
    story.append(
        bullets(
            [
                "<b>A second real evaluation.</b> Our protocol treated the real sequences as a "
                "one-time test, and that test was spent on models that could not transfer. We "
                "are retraining with augmentation (chosen on synthetic validation; the roll "
                "range is motivated by the real pose labels) and plan to evaluate on the real "
                "set once more, reporting both rounds and labelling round 2 as post-hoc. Is that "
                "acceptable, or should we hold out part of the real data (for example 8 of 24 "
                "sequences) for the final number?",
                "<b>Framing.</b> If real accuracy recovers but the clipping gate still does not "
                "beat plain averaging, is a controlled negative result on explicit saturation "
                "gating, plus an analysis of what transfer required, a sufficient contribution?",
                "<b>Attention.</b> We test attention only at the two coarsest scales, added on "
                "top of mean fusion. Is that a fair test of the FRESH authors' concern that "
                "learned attention overfits, or should we add full multi-scale cross-attention?",
                "<b>Resolution.</b> FRESH uses 512 px input; we use 256 px. If round 2 falls "
                "short, is 512 px the right next step, given it costs about four times the "
                "compute?",
            ],
            numbered=True,
        )
    )

    # 8 ---------------------------------------------------------------------------------------
    story.append(p("8. What is left", H2))
    story.append(
        bullets(
            [
                "<b>Running now: round 2.</b> 33 models (11 configurations × 3 spacecraft) with "
                "the sim2real-v1 augmentation (in-plane roll, scale and shift, brightness, "
                "contrast, background, blur and noise on RGB; dropped and added events), trained "
                "on all frames: RGB-only, event-only, mean, content gate, content + clipping "
                "gate, clipping gate, constant control, attention, attention + clipping bias, "
                "plus clean-trained RGB-only and mean fusion as controls.",
                "<b>Round-2 real evaluation</b>, frozen and run once, reported beside round 1.",
                "<b>Efficiency table:</b> parameters are done (Table 1); latency, throughput and "
                "peak memory per fusion rule still need to be measured under one timing protocol.",
                "<b>Failure analysis</b> by lighting severity and event quality, using the "
                "harsh-lighting and low-motion sequence labels FRESH provides.",
                "<b>Final report</b> and figures.",
            ]
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
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return OUT


if __name__ == "__main__":
    print(build())
