# ruff: noqa: RUF001, E501  (report prose: typographic dashes and long sentences are intended)
"""Build the theory guide PDF: methods, the 50 configurations, and the failure audit.

Run ``scripts/progress_figures.py`` and ``scripts/theory_figures.py`` first, then
``uv run --with reportlab python scripts/build_theory_guide.py``.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from build_progress_report import (
    BODY,
    CELL,
    DATA,
    FRESH,
    H1,
    H2,
    HEAD,
    META,
    MUTED,
    REPO,
    WIDTH,
    bullets,
    figure,
    link,
    p,
    table,
)
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import CondPageBreak, SimpleDocTemplate, Spacer

OUT = Path("outputs/report/theory-guide.pdf")
H3 = ParagraphStyle(
    "h3", parent=H2, fontSize=10, leading=13, spaceBefore=8, spaceAfter=3, keepWithNext=1
)
MONO = ParagraphStyle(
    "mono", parent=BODY, fontName="Sans", fontSize=8.8, leading=12.5, leftIndent=14, spaceAfter=6
)


def footer(canvas: Any, doc: Any) -> None:
    canvas.saveState()
    canvas.setFont("Sans", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(0.85 * inch, 0.5 * inch, "CMPE 249  ·  Theory guide  ·  Shinde, Morgal")
    canvas.drawRightString(letter[0] - 0.85 * inch, 0.5 * inch, str(doc.page))
    canvas.restoreState()


def rows(data: list[list[str]], bold_first: bool = True) -> list[list[Any]]:
    out: list[list[Any]] = [[p(c, HEAD) for c in data[0]]]
    for row in data[1:]:
        out.append(
            [p(f"<b>{c}</b>" if bold_first and i == 0 else c, CELL) for i, c in enumerate(row)]
        )
    return out


def build() -> Path:
    d = json.loads(DATA.read_text(encoding="utf-8"))
    seeds = {r["label"]: r for r in d["seeds"]}
    mean_s, rgb_s = seeds["Mean fusion"], seeds["RGB only"]
    real = d["sim_to_real"]
    lo, hi = min(r["real"]["are_deg"] for r in real), max(r["real"]["are_deg"] for r in real)
    s: list[Any] = []

    s.append(p("How the project works: methods, experiments, and what went wrong", H1))
    s.append(
        p(
            f"Advait Shinde, Jasper Morgal  ·  CMPE 249  ·  Theory guide, {date.today():%B} "
            f"{date.today().day}, {date.today().year}<br/>Companion to the progress check-in. "
            f"Code: {link(REPO, 'github.com/Jasper-256/cmpe-249-project')}",
            META,
        )
    )
    s.append(Spacer(1, 8))
    s.append(
        p(
            "This guide explains every technique in the project from the ground up: how pose is "
            "estimated, how the two cameras are combined, what each of the 50 trained "
            "configurations is for, how results are judged, and an audit of everything that "
            "limited the first round of results."
        )
    )

    # 1 -----------------------------------------------------------------------------------
    s.extend([CondPageBreak(1.6 * inch), p("1. The task: pose from keypoints", H2)])
    s.append(
        p(
            "<b>Pose</b> means where the spacecraft is and how it is turned relative to the "
            "camera: a rotation (3 numbers) and a translation (3 numbers), six degrees of freedom. "
            "We do not predict those six numbers directly. We use the standard two-step route, "
            "the same one the FRESH paper uses:"
        )
    )
    s.append(
        bullets(
            [
                "<b>Keypoints.</b> Each spacecraft has a fixed set of 3D landmarks on its CAD "
                "model (18 for Cassini and Satty, 26 for SOHO): corners, antenna tips, panel "
                "edges. The network looks at a 256×256 crop and outputs one 64×64 <b>heatmap</b> "
                "per landmark, a picture that is bright where it thinks that landmark is. The "
                "brightest pixel gives the landmark's 2D position.",
                "<b>PnP</b> (Perspective-n-Point). Given the 3D landmark positions, their "
                "predicted 2D positions, and the camera's focal length, geometry gives the one "
                "pose that projects the 3D points onto those 2D points. We use EPnP followed by "
                "Levenberg–Marquardt refinement. A <b>RANSAC</b> variant repeatedly solves from "
                "random subsets and keeps the pose most keypoints agree with, so a few wild "
                "keypoints do not ruin the answer.",
            ],
            numbered=True,
        )
    )
    s.append(
        p(
            "This is why keypoint <i>identity</i> matters. If the network puts dots on the "
            "spacecraft but labels a panel corner as an antenna tip, PnP receives wrong "
            "3D-to-2D matches and returns a wrong rotation and a wrong distance together. "
            "That is the failure we see on real data."
        )
    )
    s.append(
        p(
            "<b>Metrics.</b> Rotation error is the angle between the predicted and true "
            "orientation, in degrees (a random guess averages 126.5°). Translation error is the "
            "distance between predicted and true position. FRESH's success rates are the share of "
            "frames under 10° and under 10 mm. When PnP fails outright we score a default pose "
            "instead of dropping the frame, so a method cannot look better by failing."
        )
    )

    # 2 -----------------------------------------------------------------------------------
    s.extend([CondPageBreak(1.6 * inch), p("2. Two cameras, glare, and the clipping map", H2)])
    s.append(
        p(
            "An <b>RGB camera</b> measures absolute brightness. Each pixel tops out at 255; "
            "under direct sunlight large regions hit that ceiling (they <i>clip</i>) and all "
            "detail inside them is gone. An <b>event camera</b> has no frames: each pixel "
            "independently reports when its brightness <i>changes</i>. It has a far larger "
            "dynamic range, so it keeps seeing edges where RGB is white. Its weakness is the "
            "opposite one: with little motion there are few events, and strong light produces "
            "spurious ones. We accumulate events over a short window into an image so a normal "
            "network can read them, in three encodings: <i>raw</i> (the dataset's grey frame), "
            "<i>activity</i> (did anything change here, yes or no) and <i>magnitude</i> (how "
            "strongly)."
        )
    )
    s.append(
        p(
            "<b>The clipping map <i>s</i></b> is our physical signal. For each of 16×16 patches "
            "of the RGB crop it stores the share of pixels whose brightest channel is at least "
            "254. It is measured, not learned: it says exactly where RGB has lost information."
        )
    )
    s.append(
        p(
            "<b>Targeted specular saturation (TSS).</b> The synthetic training images were "
            "rendered with soft lighting and never clip (maximum 233 of 255 in our check), so a "
            "network trained on them never sees <i>s</i> change. TSS paints one to three bright, "
            "blooming highlights on the spacecraft in half of the training images. "
            "“Clean” training uses the original images; “glare” training uses TSS."
        )
    )
    s.append(
        figure(
            "example_inputs.png",
            "<b>Figure 1.</b> What the network sees for three synthetic samples: RGB, RGB with "
            "TSS glare, the clipping map <i>s</i>, and the three event encodings. Green circles "
            "are the true keypoints.",
        )
    )

    # 3 -----------------------------------------------------------------------------------
    s.extend([CondPageBreak(1.6 * inch), p("3. The shared network", H2)])
    s.append(
        figure(
            "architecture.png",
            "<b>Figure 2.</b> The pipeline every fused model shares.",
            width=WIDTH * 0.92,
        )
    )
    s.append(
        p(
            "Each camera gets its own <b>encoder</b>, an HRNet-W18 (a convolutional network that "
            "keeps a high-resolution stream alongside coarser ones, which suits keypoint "
            "localisation; it is also FRESH's backbone). Both start from ImageNet weights. An "
            "encoder turns the image into <b>feature maps</b> at five scales, from fine "
            "(128×128, edges) to coarse (8×8, whole-object shape). <b>Fusion</b> combines the RGB "
            "and event feature maps at every scale. A small <b>decoder</b> merges the scales "
            "into the heatmaps. Everything except the fusion rule is identical across models "
            "(data, encoders, decoder, loss, learning rate, epochs, checkpoint rule), so a "
            "difference between two models can only come from how they fuse."
        )
    )

    # 4 -----------------------------------------------------------------------------------
    s.extend([CondPageBreak(1.6 * inch), p("4. The fusion methods", H2)])
    s.append(
        p(
            "Write R for the RGB feature map and E for the event feature map at one scale, and F "
            "for the fused result. Most rules compute a <b>gate</b> g between 0 and 1 at every "
            "location: g near 1 trusts RGB, g near 0 trusts events."
        )
    )
    s.append(
        figure(
            "theory_fusion_rules.png",
            "<b>Figure 3.</b> Six of the fusion rules. The number is the count of parameters the "
            "fusion step adds on top of the two encoders.",
        )
    )
    methods = [
        ["Method", "How it works", "Why it is in the study"],
        [
            "RGB only / Event only",
            "One encoder, no fusion.",
            "Baselines: what each sensor achieves alone.",
        ],
        [
            "Mean",
            "F = (R + E) / 2. No parameters.",
            "The simplest fusion. Anything more complex has to beat it.",
        ],
        [
            "Concat",
            "Stack R and E, mix with a learned 1×1 convolution.",
            "Unconstrained learned mixing, with no notion of trust.",
        ],
        [
            "Content gate",
            "A small network looks at R and E and outputs g for every location.",
            "“Learned reliability”: the network works out for itself where RGB is bad. The main competitor.",
        ],
        [
            "Clipping gate (proposed)",
            "g = sigmoid(b − softplus(a)·s). Two numbers per channel; g can only fall as clipping s rises.",
            "The idea under test: trust is driven by the measured clipping, with ~700× fewer parameters than the content gate.",
        ],
        [
            "Constant control",
            "The same module with s replaced by zero.",
            "Same size, no physical signal. If the clipping gate beats this, the signal helped, not the parameter count.",
        ],
        [
            "Fixed clipping gate",
            "Mean fusion, but switch fully to events in any patch that clips. Nothing learned.",
            "A hand-written rule: is the physics alone enough?",
        ],
        [
            "Content + clipping gate",
            "The content gate's output minus a learned multiple of s.",
            "The pre-registered primary comparison against the content gate: does adding s to a learned gate help?",
        ],
        [
            "Signed clipping gate",
            "The clipping gate without the “can only fall” constraint.",
            "Ablation: is forcing the direction helpful or harmful?",
        ],
        [
            "Coarse attention",
            "At the two coarsest scales, RGB and event patches become tokens in a transformer block and attend to each other; the result is added to mean fusion.",
            "FRESH argued learned attention would overfit synthetic data. This tests a form of it.",
        ],
        [
            "Attention + clipping bias",
            "The same block, with attention toward RGB tokens reduced where s is high.",
            "The original proposal: physics-guided attention.",
        ],
    ]
    s.append(
        table(
            rows(methods),
            [0.19, 0.42, 0.39],
            right_from=9,
            note="<b>Table 1.</b> All fusion methods. sigmoid squashes any number into 0–1; "
            "softplus keeps a number positive, which is what makes the gate one-directional.",
        )
    )
    s.append(
        figure(
            "theory_gate_curve.png",
            "<b>Figure 4.</b> The clipping gate as a function of clipping. The slope a is "
            "learned. Started strong (the default), a fully clipped patch hands almost "
            "everything to events; started weak, the gate stays close to plain averaging. "
            "Both starts were trained as an ablation.",
            width=WIDTH * 0.55,
        )
    )
    s.append(p("Checks that are not models", H3))
    s.append(
        bullets(
            [
                "<b>Interventions</b> on a trained gated model at test time: <i>shuffle</i> the "
                "clipping map between images, <i>zero</i> it, or <i>bypass</i> the gate by forcing "
                "g = 0.5. If accuracy does not change, the model was not really using the signal.",
                "<b>Geometric (FRESH-style) fusion</b>: no learned fusion at all. Take the "
                "keypoints of the RGB-only and event-only models and let RANSAC PnP pick the pose "
                "most of them support.",
                "<b>Solver control</b>: re-solve every model's keypoints with the same RANSAC "
                "solver, so a better solver is not mistaken for better fusion.",
            ]
        )
    )

    # 5 -----------------------------------------------------------------------------------
    s.extend([CondPageBreak(1.6 * inch), p("5. The 50 configurations and 150 models", H2)])
    s.append(
        p(
            "A <b>configuration</b> is one complete recipe: a method, a training protocol (clean "
            "or glare), an event encoding, a random seed, and any ablation setting. Each "
            "configuration is trained <b>three times, once per spacecraft</b>, because the "
            "keypoints are specific to each spacecraft (FRESH does the same). 50 configurations "
            "× 3 spacecraft = 150 trained models. Results for a configuration pool its three "
            "models."
        )
    )
    s.append(
        figure(
            "theory_config_matrix.png",
            "<b>Figure 5.</b> The 50 configurations. Rows are methods, columns are what was "
            "varied; each number is how many configurations sit in that cell.",
            width=WIDTH * 0.95,
        )
    )
    groups = [
        ["Group", "Count", "Question it answers"],
        [
            "Clean and glare training",
            "20",
            "Every method under both protocols (attention + clipping bias needs glare, so it has no clean run; event-only is unaffected by TSS, so it has no glare run). Does glare exposure in training matter, and for which methods?",
        ],
        [
            "Raw events",
            "7",
            "Event-only and three fusion methods with the dataset's raw event frames instead of the activity encoding. Does the encoding matter?",
        ],
        [
            "Weak glare (25%)",
            "2",
            "The primary pair with glare in an eighth of the images instead of half. Is the result sensitive to how much glare is used?",
        ],
        [
            "Gate ablations",
            "3",
            "Signed gate, and weak-start versions of the clipping gate and the content + clipping gate. Why do clipping gates behave as they do?",
        ],
        [
            "Extra seeds",
            "14",
            "Seven central glare-trained configurations repeated with two more random seeds. How much of a difference is luck?",
        ],
        [
            "Fine-tuning",
            "4",
            "Mean fusion, the primary pair and RGB-only continued on all training frames at a tenth of the learning rate. Does more data help the final model?",
        ],
    ]
    s.append(
        table(
            rows(groups),
            [0.2, 0.08, 0.72],
            right_from=9,
            note="<b>Table 2.</b> The groups in Figure 5.",
        )
    )

    # 6 -----------------------------------------------------------------------------------
    s.extend([CondPageBreak(1.6 * inch), p("6. How results are judged", H2)])
    s.append(
        bullets(
            [
                "<b>Split.</b> Synthetic data is divided into training and validation by blocks "
                "of consecutive frames with a 5° guard, so validation views are not "
                "near-duplicates of training views. The real sequences are a separate test set.",
                "<b>Checkpoint selection.</b> A model is saved after every epoch; we keep the one "
                "with the lowest keypoint error on synthetic validation. Real data plays no part.",
                "<b>Seeds.</b> Training twice with different random seeds gives different "
                "numbers. Three seeds for the central models tell us how big that noise is. "
                "On glare validation it is 1.1–2.2° while the fusion methods span only "
                "6.8–8.6°, so their ranking is not reliable.",
                "<b>Paired bootstrap.</b> To compare two models on the 24 real sequences, we "
                "resample sequences with replacement thousands of times and look at the spread "
                "of the difference. That gives a 95% interval that respects the fact that frames "
                "within a sequence are not independent.",
                "<b>Pre-registration and freeze.</b> Before touching real data we wrote down the "
                "one primary comparison and a family of 13 secondary ones (with Holm correction, "
                "which raises the bar when many comparisons are made), then hashed all 150 "
                "checkpoints. This prevents choosing the comparison after seeing the answer.",
            ]
        )
    )

    # 7 -----------------------------------------------------------------------------------
    s.extend([CondPageBreak(1.6 * inch), p("7. What round 1 showed", H2)])
    s.append(
        bullets(
            [
                "<b>Synthetic.</b> Glare-trained fusion is robust where RGB-only is not "
                f"({mean_s['are_mean']:.1f}° versus {rgb_s['are_mean']:.1f}° mean rotation error "
                "on glare validation, three seeds). Clean-trained models collapse under glare. No "
                "fusion method reliably beats plain averaging, and the clipping gate does not "
                "beat its constant control.",
                f"<b>Real.</b> Every model fails: {lo:.0f}–{hi:.0f}° mean rotation error, under "
                "2% of frames within 10 mm, against 29–45% reported by FRESH for its own models. "
                "With everything near chance, round 1 cannot say anything about fusion on real "
                "data.",
            ]
        )
    )

    # 8 -----------------------------------------------------------------------------------
    s.extend([CondPageBreak(1.6 * inch), p("8. Audit: what limited round 1", H2)])
    s.append(
        p(
            "The first two rows explain most of the real-data failure. The rest are smaller or "
            "affect interpretation. “Fixed” means the round-2 retraining addresses it."
        )
    )
    s.append(
        figure(
            "theory_pose_coverage.png",
            "<b>Figure 6.</b> The roll gap, from pose labels only. Grey: synthetic validation "
            "frames are within about 20° of a training pose. Red: most real frames are much "
            "further away. Green: once the image may be rotated in-plane, every real frame is "
            "within 10° of a training pose. The synthetic data contains no in-plane roll at all.",
            width=WIDTH * 0.8,
        )
    )
    audit = [
        ["Issue", "What we found", "Effect", "Status"],
        [
            "No in-plane roll in training",
            "All synthetic poses have exactly zero roll; real frames need a median 42–83° of roll to match a training pose (Figure 6).",
            "Major. Models see unfamiliar image orientations and mislabel parts.",
            "Fixed: ±180° roll augmentation",
        ],
        [
            "No appearance augmentation",
            "Synthetic: black background, no blur or noise, clean events. Real: lit background, blur, noisy events. FRESH trains with all of these augmentations; we used none.",
            "Major. Even real frames near a training pose mostly fail.",
            "Fixed: sim2real-v1 (Figure 7)",
        ],
        [
            "A quarter of the training frames",
            "To fit 150 models into the budget, each used every fourth frame. Fine-tuning on all frames lowered synthetic error (5.8° → 5.2°).",
            "Moderate. Fewer viewpoints seen.",
            "Fixed: all frames",
        ],
        [
            "Half the resolution of FRESH",
            "256 px input and 64 px heatmaps, against 512 and 128. One heatmap pixel is 4 input pixels.",
            "Unknown, possibly large for small parts.",
            "Open; next if round 2 falls short",
        ],
        [
            "Glare only touches RGB",
            "TSS brightens the RGB image but leaves events unchanged. Real glare also creates spurious events and washes out event contrast.",
            "Fusion looks safer on synthetic glare than it really is.",
            "Partly: event noise and dropout added",
        ],
        [
            "Synthetic glare is mild",
            "TSS adds a few highlights; real glare can swallow most of the spacecraft.",
            "The clipping gate is trained on easier cases than it will meet.",
            "Open",
        ],
        [
            "Attention was tested in a limited form",
            "Only at the two coarsest scales, added on top of mean fusion and starting from zero, so it begins as mean fusion.",
            "A fair first test, but not full attention fusion. Any conclusion about FRESH's claim must say so.",
            "Open; disclosed",
        ],
        [
            "The learned clipping gate barely learns",
            "Its two parameters per channel stay near their starting values, so its behaviour is mostly its initialisation.",
            "“Clipping gate loses” partly means “a strong prior loses”.",
            "Ablated (weak start); open",
        ],
        [
            "Means hide outliers",
            "On synthetic glare validation the median error is 3.0° but the mean is 5.8°; the 2.3% of frames above 30° contribute 28% of the mean.",
            "Rankings by mean are sensitive to a few failures.",
            "Report medians alongside means",
        ],
        [
            "Selection and reporting on the same set",
            "Checkpoints are chosen on the validation set that is then reported.",
            "Synthetic numbers are slightly optimistic for every model.",
            "Disclosed",
        ],
        [
            "One seed for most configurations",
            "Seed noise is as large as most differences between fusion methods.",
            "Single-seed rankings are unreliable.",
            "Three seeds for central models; round 2 starts with one",
        ],
        [
            "Basic pose solver as primary",
            "EPnP without RANSAC. The robust solver improves every model about as much as the choice of fusion.",
            "Solver and fusion effects can be confused.",
            "Controlled: both solvers reported",
        ],
        [
            "Ground-truth crop",
            "The crop comes from the annotated bounding box, not a detector.",
            "Results are pose-given-a-good-crop, not a full pipeline. Not comparable to FRESH's detector-based numbers.",
            "Stated limitation",
        ],
        [
            "Real test used before transfer worked",
            "The one-shot real evaluation was spent on models that could not transfer.",
            "Round 2 on real data is post-hoc, not pre-registered.",
            "Disclosed; both rounds reported",
        ],
        [
            "No efficiency measurements",
            "Latency, throughput and memory per method are not yet measured.",
            "Cannot yet argue the 3,968-parameter gate is cheaper in practice.",
            "Open",
        ],
    ]
    s.append(
        table(
            rows(audit),
            [0.19, 0.37, 0.24, 0.2],
            right_from=9,
            note="<b>Table 3.</b> Failure points and limitations, most important first.",
        )
    )

    # 9 -----------------------------------------------------------------------------------
    s.extend([CondPageBreak(1.6 * inch), p("9. Round 2: what is being retrained", H2)])
    s.append(
        figure(
            "theory_augmentation.png",
            "<b>Figure 7.</b> sim2real-v1. Left: an original synthetic RGB crop and its events "
            "(top clean, bottom with TSS glare). Right: two random augmentations of each. The "
            "same rotation, scale and shift are applied to RGB, events and the clipping map, and "
            "the keypoints (green) move with them.",
        )
    )
    s.append(
        p(
            "<b>Augmentation</b> means randomly altering each training image so the network "
            "cannot rely on things that differ between synthetic and real data. sim2real-v1 "
            "applies: in-plane rotation up to ±180°, scale 0.8–1.15, shift up to 8%; on RGB, "
            "brightness, contrast and colour changes, a lit background, blur and sensor noise; "
            "on events, randomly dropped events, added noise events and blur. The magnitudes "
            "are fixed in code and follow the kinds of augmentation FRESH reports. The roll "
            "range was motivated by the real pose labels, which is a use of test-set "
            "information and is disclosed."
        )
    )
    s.append(
        p(
            "<b>Round 2 trains 11 configurations × 3 spacecraft = 33 models</b>: RGB-only, "
            "event-only, mean, content gate, content + clipping gate, clipping gate, constant "
            "control, attention, and attention + clipping bias, all glare-trained; plus "
            "clean-trained RGB-only and mean fusion, to check whether glare training still "
            "matters once general augmentation is on. Checkpoints are selected on an augmented "
            "copy of synthetic validation. The models are then frozen and evaluated once on the "
            "real sequences, and reported next to round 1."
        )
    )
    s.append(p("What each outcome would mean", H3))
    s.append(
        bullets(
            [
                "<b>Real accuracy recovers and a clipping-aware method wins:</b> the physical "
                "signal helps where it matters, under real glare.",
                "<b>Real accuracy recovers and methods tie:</b> how the cameras are fused matters "
                "less than what the network is trained on. A useful negative result.",
                "<b>Attention transfers as well as simpler fusion:</b> evidence against the "
                "concern that learned attention must overfit synthetic data (for this limited "
                "form of attention).",
                "<b>Real accuracy does not recover:</b> the remaining suspects are resolution "
                "and how glare affects events, in that order.",
            ]
        )
    )
    s.append(
        p(
            f"Reference: Jawaid, Märtens and Chin, FRESH dataset and RANSAC fusion, "
            f"{link(FRESH, 'arXiv:2507.05698')}. FRESH numbers quoted here are as reported by "
            "the authors and were not reproduced.",
            META,
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
        title="How the project works: methods, experiments, and what went wrong",
        author="Advait Shinde, Jasper Morgal",
    )
    doc.build(s, onFirstPage=footer, onLaterPages=footer)
    return OUT


if __name__ == "__main__":
    print(build())
