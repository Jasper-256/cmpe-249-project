# Research Log

Record web-verified findings that materially affect project decisions. Prefer
primary sources and include the exact version or commit when available.

## 2026-09-13 — Foundation and roadmap review

| Source | Verified fact | Decision affected |
| --- | --- | --- |
| [FRESH paper](https://doi.org/10.1016/j.ast.2025.111039) | FRESH provides optically and temporally aligned synthetic/real event-RGB data and a learning-free RANSAC fusion method. | Keep the geometric method as a same-dataset P0 reference; do not claim it proves attention overfits. |
| [FRESH author repository](https://github.com/mohsij/space-event-rgb-fusion) | The public landing repository points to the dataset record and citation; implementation availability must be audited separately. | Do not assume a ready-to-run baseline from the landing repository. |
| [FRESH dataset](https://zenodo.org/records/15861758) | Canonical public dataset record referenced by the authors. | Use this as the dataset provenance root and create a checksum manifest before preprocessing. |
| [UE-Pose publisher record](https://doi.org/10.1016/j.asr.2026.08.069) | UE-Pose uses per-keypoint evidential uncertainty for fusion and pose recovery on FRESH. | Treat UE-Pose as the closest P1 competitor and avoid an unsupported SOTA claim. |
| [UE-Pose repository](https://github.com/JEFFLzx/UE-Pose) | Author-linked public code location. | Pin a commit and inspect protocol equivalence before comparison. |
| [SNGAF](https://doi.org/10.3390/rs17173112) | Sparse mixture-of-experts gating for RGB-event fusion predates this project in another task. | Frame novelty around the explicit physical clipping constraint and causal controls, not generic gating. |
| [GatedFusion-Net](https://doi.org/10.1016/j.inffus.2025.103986) | Recent per-pixel multimodal gating reports efficiency benefits but also mostly static lighting-dependent behavior under sensor loss. | Include sensor-failure interventions and prediction-dependence tests. |
| [PyTorch Windows installation](https://docs.pytorch.org/get-started/locally/) | Windows requires a supported 64-bit Python and an accelerator-specific PyTorch build; CUDA availability must be tested after install. | Standardize on managed Python 3.12 and run a CUDA smoke test. |
| [uv PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/) | uv supports explicit CUDA indexes; its documented current stable example uses PyTorch 2.11/CUDA 13.0, while current release information lists newer compatible builds. | Pin the CUDA 13.0 index and lock the resolved environment rather than relying on ambient packages. |

## 2026-09-13 — FRESH canonical manifest

The [Zenodo API record](https://zenodo.org/api/records/15861758) identifies DOI
`10.5281/zenodo.15861758`, open access under CC BY 4.0, and five artifacts totaling
19,537,865,799 bytes. Exact sizes, URLs, and publisher-provided MD5 values are
preserved in `docs/fresh-dataset-manifest.json`. Downloads use `.part` files,
HTTP range resumption, atomic completion, size validation, and MD5 verification.

All five raw artifacts were downloaded to `data/fresh/raw` and independently
reverified against the committed manifest on 2026-09-13. Zenodo temporarily
returned HTTP 504 responses from both API and direct record routes between file
transfers; retries succeeded after the service recovered. No alternate mirror or
client was substituted, preserving canonical provenance.

The raw artifacts alone occupy 19,537,865,799 bytes, while all archives contain
about 100.2 GB of uncompressed members. The adopted access path opens the regular
ZIPs directly and presents the three real ZIP parts as one virtual, seekable
read-only stream. Samples are decoded into memory on demand; no combined archive
or extracted dataset copy is created. The user clarified that approximately
20 GB is a soft target rather than a hard cap, so bounded selective caches may be
added later when profiling demonstrates a material benefit.

## 2026-09-23 — Focused SOTA and implementation audit

Accessed 2026-09-23.

| Source | Verified fact | Decision affected |
| --- | --- | --- |
| [FRESH full paper](https://arxiv.org/html/2507.05698v3) | The method uses separate RGB/event landmark regressors with dropout uncertainty, cross-modal RANSAC, cross-modal keypoint distance, uncertainty-based fallback, and temporal event-box handling. It deliberately excludes simulated harsh-light augmentation. Translation error is reported in metres with a 10 mm success threshold. | Treat FRESH as a complete geometric/uncertainty pipeline, not “plain RANSAC”; screen natural synthetic clipping before proposing training-time reliance on it. |
| [UE-Pose publisher record](https://doi.org/10.1016/j.asr.2026.08.069) | UE-Pose uses EfficientNet-B0 plus a multi-scale decoder, per-keypoint evidential uncertainty, adaptive fusion/outlier gating, topology validation, and uncertainty-aware pose recovery on FRESH. | It is the strongest baseline capable of making explicit clipping unnecessary. |
| [UE-Pose repository](https://github.com/JEFFLzx/UE-Pose), commit `2568f1023d02e15b70ae47dce48e8ec931fa4961` | The official snapshot has no top-level license or dependency lock. `ransac-UEPose.py` normalizes each modality's uncertainty by a median computed over the complete test prediction set and contains hard-coded result/data paths. `EventRGBAlignedDataset.py` reads bounding boxes from annotations and constructs CUDA objects inside the dataset. The advertised deformable-convolution class applies an ordinary convolution scaled by offset magnitude rather than offset sampling. | Keep the checkout isolated under ignored `external/`; do not copy code. Distinguish author-faithful/transductive reproduction from our non-transductive protocol and budget reproduction risk explicitly. |
| [FRESH predecessor repository](https://github.com/mohsij/spacecraft-pose-estimation), commit `dcbd3339f0cfd0665aff62117505279b61074517` | The repository is MIT-licensed and documents an older general detector-landmark-PnP pipeline for ICRA 2023/SPEC work, not a turnkey implementation of every component in the 2025 FRESH paper. | Reuse only as an attributed reproduction reference; do not assume it is the exact published FRESH fusion implementation. |
| [CVPRW 2026 event-only cascaded pose paper](https://openaccess.thecvf.com/content/CVPR2026W/AI4Space/html/Zhang_A_Cascaded_Architecture_for_Illumination-Robust_6-DoF_Spacecraft_Pose_Estimation_Using_CVPRW_2026_paper.html) | The method ranks second on the SPADES/SPARK event-camera track using event-only keypoint detection, geometric refinement, and temporal smoothing. | Relevant adjacent SOTA, but not a directly comparable FRESH event-RGB fusion baseline; temporal/event-representation work remains out of core scope. |

Local annotation audit found exact object-to-camera reprojection agreement to
floating-point precision for sampled synthetic and real labels. Translation
entries are millimetre-scale (`1500` for the synthetic 1.5 m case and about
`847` for a real 0.8 m case), so evaluation must convert to metres once. The
synthetic `test.json` files contain 137 held-out filenames per object, while
`train.json` contains 13,505 and shares no RGB/event filename pair. Their integer
`index` fields are reused and are not globally unique identifiers.

## 2026-09-23 — Initial feasibility results and strategy stop

The predeclared G1A screen sampled 64 pairs from each object and synthetic role
(384 aligned pairs total; seed 249). No sampled target-crop pixel reached 250,
254, or 255 before resize. Per-crop maximum channel values ranged from 184 to
233 (median 206). Consequently, no 16-by-16-grid patch contained at least 1%
clipped pixels. The observed threshold-254 prevalence was 0%; the rule-of-three
95% upper bound is 3/384 = 0.78%, below the predeclared 10% go criterion. Event
frames were not empty: median target-crop activity was 3.88% of pixels. Lazy ZIP
decode plus analysis processed approximately 39 aligned pairs/s without creating
an extracted cache. The full local report is
`outputs/feasibility/saturation-signal.json`.

This result stops the natural-data learned-gate strategy before GPU training.
It does not falsify the possibility that clipping is useful on real harsh-light
data; it shows that unmodified synthetic FRESH cannot teach a response to a cue
that never activates. The bounded next strategy is a separately named shared
synthetic-corruption protocol applied identically to all learned baselines. The
[Albumentations 2.0.8 RandomSunFlare documentation](https://albumentations.ai/explore/transform/RandomSunFlare/docs/)
provides a `physics_based` method already available in the locked environment.
Its signal coverage and determinism must pass a synthetic-only check before use.

That bounded rescue also failed its frozen criterion. On 96 stratified crops,
the 24-pixel-radius `physics_based` profile produced no threshold-254 patch
activation. The 48-pixel-radius profile activated 15.6% of crops, but the median
fraction of active patches remained 0%. Both transforms replayed deterministically
and preserved shape, so the failure concerns signal coverage rather than software
nondeterminism. No third parameterization will be tuned. The result is stored in
`outputs/feasibility/flare-augmentation.json`; the project proceeds to verified
metrics, unimodal baselines, and only then a fixed keypoint-level clipping
intervention.

The full contract audit then validated all 48,820 annotations: every referenced
RGB/event member exists, all sampled images are 800×720 RGB, rotations are valid,
and object-to-camera landmark reprojection has a maximum residual of
`3.41e-13` pixels. There are no exact filename-pair or pose overlaps between the
synthetic train and validation JSONs. 13,229 annotations have boxes extending
beyond the image boundary while retaining a valid intersection, so cropping must
clamp rather than reject them. Report: `data/fresh/index/contract-audit.json`.

The subsequent synthetic split audit found that the official holdout is a
systematic subsample of a smooth render trajectory. Adjacent training frames are
one numeric index and approximately one rotation degree from validation. Across
all 411 official validation images, the nearest adjacent training crop had a
median 64-bit dHash distance of 0 and median normalized pixel difference of
0.0066; 172 met the frozen near-duplicate proxy. Report:
`data/fresh/index/synthetic-split-similarity.json`.

For project model selection, `fresh-synthetic-grouped-v1` now unions the two
synthetic annotation files, selects eight contiguous 80-frame validation blocks
per object, and removes every other pose within five SO(3) degrees of any
validation pose. Each object retains 11,920 training frames and 640 validation
frames; 1,082 guard frames are unused. Membership hashes and exact IDs are in
the local generated index `data/fresh/index/synthetic-split-v1.json`; the tracked
algorithm/configuration is `configs/data/fresh_grouped_v1.yaml`. This split is
used for all project variants, while author-faithful reproduction results keep
the original split and remain explicitly incomparable unless rerun on grouped-v1.

The training-ready lazy dataset was profiled on the RTX 5060 Laptop system using
256×256 paired crops from Cassini. With no extracted cache, zero workers reached
55.4, 66.6, and 66.3 pairs/s at batch sizes 1, 4, and 8. Two spawned workers were
slower (45.1–50.0 pairs/s), consistent with compressed-archive contention and
process startup overhead. No derived image cache is justified for the pilot.
Report: `outputs/feasibility/data-loader-profile.json`.

Windows Application Control later blocked uv's copied virtual-environment Python
launcher and console-script wrappers. A local ignored workaround replaces only
`.venv/Scripts/python.exe` with a hard link to the trusted uv-managed Python and
places the matching runtime DLLs beside it; the original launcher is retained as
`python.blocked-launcher.exe`. `uv run --locked python` again detects CUDA and the
RTX 5060. Ruff and mypy console commands run, while pytest must be invoked as
`uv run --locked python -m pytest` because the generated `pytest.exe` wrapper is
still blocked. No dependency or lock version changed.

## 2026-09-24 — Unimodal tiny-overfit gate

A direct-coordinate pilot network with 1,253,700 parameters was trained without
augmentation on the same 16 evenly spaced Cassini grouped-v1 training samples.
After 400 full-batch steps, RGB coordinate RMSE decreased from 152.95 to 0.02
pixels and event RMSE from 149.26 to 0.77 pixels. Both passed the predeclared
criterion of at most 5 pixels and at least 90% reduction. Each run used about
302 MB peak allocated VRAM and completed in 4–5 seconds on the RTX 5060 Laptop
GPU. The immutable local report records config, sample IDs, git commit, uv-lock
hash, split-index hash, device, and learning curves at
`outputs/feasibility/tiny-overfit.json`.

This establishes data/target-path competence only. It is not a reportable
architecture or evidence of validation generalization; the next bounded test is
a short grouped train/validation pilot with deterministic PnP pose recovery.

## 2026-09-24 — Short direct-coordinate pilot and crop confound

The first frozen 1,024-train/256-validation Cassini pilot failed for both
modalities: RGB/event keypoint RMSE was 57.52/57.82 pixels, median ARE was
146.61/131.96 degrees, and median absolute translation error was 1.008/0.971 m.
All PnP calls returned a numerical pose, so solver availability was not the
failure. The local report is
`outputs/feasibility/short-unimodal-pilot-tight-crop.json`.

Before rejecting the architecture, a preprocessing audit found that the tight
clamped crop placed at least one landmark outside the network input in 29% of
the sampled validation frames. This differed materially from the official
FRESH/UE-Pose loader, which constructs a square crop and applies a 1.2 scale
factor before the affine warp. The project protocol now uses a
center-preserving square annotation crop with 1.2 padding and constant-zero
outside-image pixels. This is deliberately shared across RGB/event and all
future learned variants; it is not claimed to reproduce the official loader's
asymmetric RGB/event border constants exactly.

The single crop-corrected rerun put every sampled landmark inside the input and
improved RGB/event RMSE to 34.46/33.02 pixels, but it still failed the frozen
20-pixel criterion. Median ARE worsened to 162.03/167.35 degrees and median ATE
remained 0.661/0.436 m, with 100% numerical PnP solutions. Report:
`outputs/feasibility/short-unimodal-pilot.json`. The direct-coordinate head is
therefore rejected without another hyperparameter attempt.

The next controlled change is spatial keypoint prediction. FRESH and UE-Pose
use HRNet heatmaps, and current timm exposes HRNet feature pyramids through its
documented `features_only=True` interface. The proposed pilot uses the smaller
HRNet-W18-Small-v2 ImageNet-1k backbone plus a shared-size heatmap decoder to fit
the 8 GB device. The pretrained artifact is 62.6 MB with SHA-256
`cd7436f6de68d589ac0d05b90e49199c9ad28669381a25250e9667aa9e5b670e` according
to its official model card. Sources accessed 2026-09-24:

- https://github.com/JEFFLzx/UE-Pose/blob/2568f1023d02e15b70ae47dce48e8ec931fa4961/lib/dataset/EventRGBAlignedDataset.py
- https://huggingface.co/docs/timm/feature_extraction
- https://huggingface.co/timm/hrnet_w18_small_v2.ms_in1k

## 2026-09-26 — Short heatmap pilot and transformer decision

The first 1,024-train/256-validation HRNet heatmap pilot was intentionally
limited to three epochs. It failed the frozen competence criteria, but its
learning curves had not converged: RGB validation keypoint RMSE fell from 61.76
pixels before training to 48.66, 39.92, and 32.56 pixels; event RMSE fell from
73.18 to 64.46, 45.57, and 44.62 pixels. Both modalities still produced valid
PnP solutions for at least 96.9% of validation examples. The result therefore
supports one bounded checkpoint-selected extension, not indefinite tuning:
maximum 12 epochs, minimum 0.25-pixel validation improvement, patience three,
and selection by synthetic-validation keypoint RMSE. All other experimental
factors remain fixed. Local report:
`outputs/feasibility/short-heatmap-pilot.json`.

A current architecture audit examined recent spacecraft-pose, event-camera,
and dense-foundation-model work. FastPose-ViT (WACV 2026) and SPACE-HOP
(CVPRW 2026) are scientifically relevant but change the pose formulation, so
neither is a controlled backbone replacement for the keypoint/PnP protocol.
DINOv3 ViT-S/16 is the smallest released DINOv3 ViT at 21M parameters; its
weights require an access request and its custom license requires separate
acceptance. Hiera-T and SwinV2-T each measure about 27M encoder parameters in
the locked timm environment. C-RADIOv4's currently released SO400M/H family is
outside the dual-stream 8 GB budget, and RADIO1D checkpoints are not yet
released.

The only predeclared P0 transformer challenger is therefore TinyViT-5M. In the
locked environment it measures 5.07M encoder parameters and returns a four-level
64/128/160/320-channel hierarchy, close to the current 5.25M HRNet encoder.
Microsoft reports the available TinyViT-5M weights were distilled during
ImageNet-22k pretraining and fine-tuned on ImageNet-1k. This makes it a bounded
test of transformer inductive bias without introducing a fivefold capacity
increase. It will run only if HRNet remains incompetent; details and stop rules
are in `docs/transformer-architecture-audit.md`.

Two newer efficient families were also checked rather than assuming recency
implies suitability. TinyNeXt-M (ICCV 2025) reports 2.3M classification
parameters and 75.3% ImageNet-1k top-1, but it is absent from the locked timm
catalog and its official repository currently has only 15 commits. LowFormer's
2026 edge-GPU extension offers pretrained E-series checkpoints through a
standalone external implementation, but its public model table does not report
E-series parameter counts and provides no spacecraft or keypoint evidence.
They remain deployment watchlist items, not a better first representation test
than the stable, strongly pretrained, parameter-matched TinyViT path.

Recent event literature shifts the event branch decision away from training an
independent large transformer from scratch. STP (ICCV 2025) uses a lightweight
spatiotemporal prompt to transfer image-pretrained knowledge to event tasks,
while EventDAM and Depth AnyEvent use RGB-to-event feature transfer or
distillation for depth. These results do not validate event-frame spacecraft
pose, but they motivate frozen/lightweight adaptation as a later study. The core
FRESH input is a rendered event frame, so a raw-event temporal prompt remains
out of scope until the central protocol passes.

Sources accessed 2026-09-26:

- https://github.com/facebookresearch/dinov3
- https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md
- https://github.com/facebookresearch/hiera
- https://github.com/microsoft/Cream/tree/main/TinyViT
- https://huggingface.co/timm/tiny_vit_5m_224.dist_in22k_ft_in1k
- https://github.com/NVlabs/RADIO
- https://github.com/yuffeenn/TinyNeXt
- https://github.com/altair199797/lowformer
- https://openaccess.thecvf.com/content/WACV2026/html/Ancey_FastPose-ViT_A_Vision_Transformer_for_Real-Time_Spacecraft_Pose_Estimation_WACV_2026_paper.html
- https://openaccess.thecvf.com/content/CVPR2026W/AI4Space/html/Dutta_SPACE-HOP_Spacecraft_6-DoF_Pose_Estimation_using_Embedding-Predictive_Pretraining_and_Hopf_CVPRW_2026_paper.html
- https://openaccess.thecvf.com/content/ICCV2025/html/Liang_Efficient_Event_Camera_Data_Pretraining_with_Adaptive_Prompt_Fusion_ICCV_2025_paper.html
- https://openaccess.thecvf.com/content/ICCV2025/html/Zhu_Depth_Any_Event_Stream_Enhancing_Event-based_Monocular_Depth_Estimation_via_ICCV_2025_paper.html
- https://openaccess.thecvf.com/content/ICCV2025/html/Bartolomei_Depth_AnyEvent_A_Cross-Modal_Distillation_Paradigm_for_Event-Based_Monocular_Depth_ICCV_2025_paper.html

## 2026-09-26 — HRNet convergence gate passed

The predeclared checkpoint-selected extension passed every frozen competence
criterion without early stopping. The RGB checkpoint selected at epoch 12
achieved 7.620 px keypoint RMSE, 100% valid PnP, 3.877 degrees median ARE, and
0.01693 m median absolute translation error. The event checkpoint selected at
epoch 11 achieved 9.161 px, 100%, 4.005 degrees, and 0.01933 m. Both used the
same 1,024 grouped-v1 Cassini training examples, 256 validation examples,
ImageNet-pretrained HRNet heatmap architecture, crop, optimizer, batch size,
seed, and PnP path. No real FRESH sequence was used.

RGB and event training took 759.0 and 789.3 seconds respectively, or 16.19 and
15.57 samples/s. Peak PyTorch allocations were 1.859 and 1.860 GB. Batch-16
forward-only profiling measured 38.61/39.28 ms per batch and 414.5/407.3
samples/s, showing that forward latency alone badly overstates end-to-end
training throughput. Report:
`outputs/feasibility/short-heatmap-convergence.json`.

This result freezes HRNet-W18-Small-v2 as the core encoder family and prevents a
backbone rabbit hole. The transformer audit remains useful, but TinyViT-5M is no
longer a baseline rescue; it is a post-core architecture-robustness experiment.
DINOv3 remains a gated-license frozen/adapter study only. The next compute gate
will test complete train steps at batch 16/32 and FP32/BF16, following current
PyTorch AMP guidance, before estimating the multi-seed fusion budget. Precision
will be shared across claimed method comparisons and chosen from synthetic
numerical stability, never real performance.

Source accessed 2026-09-26:

- https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html

## 2026-09-26 — Complete-step compute screen and cloud trigger

The frozen compute screen measured five warm-up and 20 timed optimizer steps for
RGB/event, FP32/BF16, and batch sizes 16/32. Every trial had finite losses and
gradients, and the RTX 5060 Laptop GPU reports native BF16 support. At batch 16,
FP32/BF16 throughput was 23.77/23.61 samples/s for RGB and 25.03/25.52 for
event. BF16 reduced peak PyTorch allocation from 1.73 to 0.97 GiB but did not
materially improve end-to-end throughput. Batch 32 was slower for every pairing
(19.46–21.72 samples/s) while using 3.39 GiB in FP32 or 1.85 GiB in BF16.

The bottleneck is measured rather than inferred: batch-16 data wait averaged
474–508 ms per step, while synchronized forward, heatmap loss, backward, and
AdamW update took 154–170 ms. Larger batches increased data wait to 1.26–1.38 s.
FP32 batch 16 therefore remains frozen; BF16 is only a memory contingency that
would require convergence parity before use in a claimed comparison. Report:
`outputs/feasibility/training-compute-profile.json`.

At the measured 23.77 samples/s, 12 passes over all 35,760 grouped training
examples would require about 5.0 hours per unimodal model before recurring
validation and checkpoint overhead. A dual-encoder fusion model will be slower.
This makes a one-time USD 10–20 rented-GPU budget potentially useful only after
the exact fusion matrix is implemented and smoke-tested. Current provider prices
have deliberately not been compared yet because the required GPU memory,
end-to-end cached throughput, and job hours are not frozen.

The profile justifies one bounded storage optimization. Caching both 3-channel
modalities as exact uint8 256-by-256 crops for all 35,760 train and 1,920
validation examples would occupy 14,816,378,880 bytes (13.80 GiB), plus a small
versioned manifest. This raises local project data above the 20 GB soft target
but remains far below a roughly 100 GB raw extraction and has a measured target:
remove the 74–76% step-time data wait. First build one immutable shard under
`data/fresh/cache/` and expand only if it at least doubles complete-step
throughput. Raw archives remain untouched.
- https://docs.opencv.org/4.13.0/d5/d1f/calib3d_solvePnP.html

## 2026-09-26 — Spatial heatmap tiny-overfit gate

The first frozen heatmap attempt used batch size 4 for 200 optimizer updates.
Although heatmap loss continued to decrease, each of the 16 examples appeared
in only 42–60 updates. RGB/event decoded RMSE reached 9.88/15.92 pixels and
failed the predeclared 5-pixel, 90%-reduction criterion. This was diagnosed as
an inconsistent competence-test exposure budget rather than answered by an
arbitrary learning-rate or architecture change: the earlier direct-coordinate
gate used the entire 16-sample subset on every update.

A one-update batch-16 diagnostic measured only 1.83 GB peak allocated VRAM on
the RTX 5060 Laptop GPU. Repeating the otherwise unchanged 200-update gate with
all 16 samples per update passed: RGB RMSE decreased from 65.85 to 1.77 pixels
(97.3%) and event RMSE from 74.21 to 4.99 pixels (93.3%). Each modality used
1.87 GB peak VRAM and approximately 30 seconds. The machine-readable report is
`outputs/feasibility/heatmap-tiny-overfit.json`.

This establishes memorization competence only. It authorizes one bounded
Cassini grouped train/validation pilot with the same pretrained HRNet heatmap
architecture and shared EPnP+LM path; it is not evidence of validation or real
performance. Current official timm documentation still identifies
`features_only=True` as the supported multi-scale feature-pyramid interface,
and the selected HRNet model card identifies the weights as ImageNet-1k and MIT
licensed. Sources accessed 2026-09-26:

- https://huggingface.co/docs/timm/feature_extraction
- https://huggingface.co/timm/hrnet_w18_small_v2.ms_in1k

## 2026-09-27 — Backbone decision reopened as a bounded architecture study

The HRNet convergence result remains valid and is now the verified reference,
not an automatic final-architecture decision. It establishes that the frozen
FRESH crop, heatmap, and PnP path can produce competent RGB and accumulated-event
poses. It does not establish that a high-resolution CNN is better than a compact
transformer/hybrid, that one family is optimal for both modalities, or that a
practical pretrained-model gain is caused by architecture rather than
pretraining.

A current audit therefore separates three questions: (1) the best practical,
legally accessible pretrained package; (2) a reduced-confound architecture
comparison using ImageNet-1k initialization where possible; and (3) event-native
temporal models, which require a separately named raw-event study. The bounded
candidate set contains HRNet-W18-Small-v2, EfficientNet-B0, TinyViT-5M,
SegFormer MiT-B0, SwinV2-T, MobileViTv2-1.0, EfficientViT-M2, ViTPose-S,
DINOv3 ViT-S/16, and an STP-style prompt. Ten is an upper bound, not a training
target.

Recent evidence challenges both easy stories. FastPose-ViT (WACV 2026),
SPACE-HOP (CVPRW 2026), PVSPE, SO-PEN, and A3DKS show that ViT, hierarchical
window-attention, and EfficientViT designs are viable in spacecraft pose tasks,
but all change the head, loss, data, or evaluation. STP and TESPEC (ICCV 2025)
show benefits from event-specific spatiotemporal pretraining, but both consume
event sequences/histograms rather than the frozen accumulated event image.
DINOv3 ViT-S/16 supplies strong dense self-supervised features but introduces a
21M-parameter, 1.689B-image-pretraining and custom-license confound; its weights
will not be downloaded without explicit scope/license clearance.

The staged decision is now cache gate -> static Stage A -> fixed tiny-overfit
Stage B -> at most four new Stage C convergence packages. The existing HRNet
result is never discarded or silently rerun. A model advances only for a
meaningful pose-metric gain, a material efficiency gain, or modality-rank
evidence; lower keypoint RMSE alone is insufficient. Full details and stop rules
are in `docs/transformer-architecture-audit.md` and `PLAN.md`.

Sources accessed 2026-09-27:

- https://doi.org/10.1016/j.ast.2025.111039
- https://openaccess.thecvf.com/content/WACV2026/html/Ancey_FastPose-ViT_A_Vision_Transformer_for_Real-Time_Spacecraft_Pose_Estimation_WACV_2026_paper.html
- https://openaccess.thecvf.com/content/CVPR2026W/AI4Space/html/Dutta_SPACE-HOP_Spacecraft_6-DoF_Pose_Estimation_using_Embedding-Predictive_Pretraining_and_Hopf_CVPRW_2026_paper.html
- https://openaccess.thecvf.com/content/ICCV2025/papers/Liang_Efficient_Event_Camera_Data_Pretraining_with_Adaptive_Prompt_Fusion_ICCV_2025_paper.pdf
- https://openaccess.thecvf.com/content/ICCV2025/html/Mohammadi_TESPEC_Temporally-Enhanced_Self-Supervised_Pretraining_for_Event_Cameras_ICCV_2025_paper.html
- https://github.com/facebookresearch/dinov3
- https://github.com/microsoft/Cream/tree/main/TinyViT
- https://github.com/microsoft/Swin-Transformer
- https://github.com/NVlabs/SegFormer
- https://github.com/apple/ml-cvnets
- https://github.com/microsoft/Cream/tree/main/EfficientViT
- https://github.com/ViTAE-Transformer/ViTPose
- https://proceedings.mlr.press/v97/tan19a.html
- https://www.sciencedirect.com/science/article/abs/pii/S0952197626013576
- https://arxiv.org/abs/2606.11619

## 2026-09-27 — Uint8 crop-cache gate passed

The versioned cache experiment materialized one 256-sample Cassini training
shard under ignored `data/fresh/cache/crop-v1/`. It stores both 3-channel
modalities as uint8 memory-mapped arrays and the existing transformed geometry
as typed arrays. The manifest records the exact sample ordering, preprocessing,
split hash, array shapes/dtypes, and per-array SHA-256 checksums. Re-reading raw
items at the first, middle, and final shard positions showed bit-exact RGB/event
tensors and zero maximum absolute difference for every cached geometry field.
Raw archives were not modified or extracted.

Across three repeats and both modalities, median complete HRNet training
throughput rose from 18.36 samples/s for lazy ZIP reads to 89.50 samples/s for
the memory-mapped cache, a 4.87x speedup. Mean batch data wait fell from about
0.70-0.72 seconds to 0.016-0.020 seconds. The shard occupies 101,071,330 bytes
(96.39 MiB) and took 12.23 seconds to build. This passes the predeclared 2x rule
and demonstrates that a faster GPU alone would not have fixed the earlier input
bottleneck.

The estimated 14,816,378,880-byte (13.80 GiB) full cache is now permitted but
not built immediately. Stage A/B use the small shard; only objects/roles needed
by Stage C survivors will be expanded. This keeps the architecture screen from
spending disk before cheap candidate elimination. Machine-readable evidence:
`outputs/feasibility/crop-cache-benchmark.json`.

## 2026-10-05 — Contract audit fixes before the fusion matrix

An end-to-end audit (code, outputs, FRESH paper, author repositories, and the
local archives) found several contract problems. Each fix below was reviewed by
an independent agent before the next began.

1. **PnP failures now count.** The FRESH paper states that when PnP cannot
   produce a result the pipeline "outputs a default 0 vector for both
   translation and rotation". The pilot evaluator silently dropped failed
   solves. `saga_pose.evaluation.pose` now substitutes identity rotation and
   zero translation, reports failures-included metrics as primary, and keeps
   valid-only values as a diagnostic. The frozen HRNet reference had 100% PnP
   validity, so its numbers are unchanged.
2. **Stage A applies its written criteria.** The pass rule previously checked
   only shape, finiteness, and VRAM. It now also requires a finest feature
   stride of at most 8 and resolved license/weights, and every backbone receives
   its own pretraining normalization (MobileViTv2/cvnets expects unnormalized
   `[0, 1]` inputs, not ImageNet statistics). Rerun: A0–A5 pass (SegFormer A3
   now profiled, non-commercial research license); EfficientViT-M2 (A6) is
   rejected because its finest feature is stride 16.
3. **Resize label corrected.** Training uses Pillow BILINEAR, which antialiases
   when downsampling; the config said `opencv_linear`. Geometry is unaffected
   (both use half-pixel centres; regression test added). The G1A conclusion is
   unaffected because any bilinear filter is a convex combination and cannot
   create clipping from a 233 maximum.
4. **Real annotation correction.** The FRESH repository
   (https://github.com/mohsij/space-event-rgb-fusion, commit
   `fb566826e950b5db1f5b7ba013e15c12c5a7d4af`) ships
   `addendum-data/corrected-annotations/satty-4-far/test.json`. It differs from
   the Zenodo copy by a median 58.9 degrees rotation, 40.3 mm translation, and
   52.9 px keypoints over 272 of 273 frames. `saga_pose.data.real` applies it
   from the ignored pinned checkout (SHA-256 verified; not copied because the
   repository has no license file) and fails loudly if it is missing.
5. **Author-defined real slices.** `real/harsh-and-slow-frames.json` gives one
   `[start, end]` frame range per sequence for `harsh` and `low-motion`. Frame
   numbers equal annotation order in all 24 sequences. Ranges are read as
   inclusive; slices overlap in the `*-2-*` sequences, so they are reported as
   overlapping subsets, never as a partition.
6. **Event encoding mismatch.** Synthetic event frames are grey with exactly
   three values (127 neutral; 191 and 255, both brighter). Real event frames are
   grey and continuous with neutral 128 and both darker and brighter polarities
   (format check on three frames; no error or clipping statistics were
   computed). FRESH and UE-Pose consume raw frames and pad event crops with 127;
   this project padded with 0, which never occurs in synthetic frames. Event
   crops now pad with the format neutral. Three representations are compared as
   predeclared arms: `raw` (author-faithful), `activity`
   (`|x - neutral| >= 32`), and `magnitude` (`min(1, |x - neutral| / 64)`). Both
   constants derive from the synthetic encoding (64 = 191 - 127); none is tuned
   on real data. On synthetic data `activity` and `magnitude` are identical, so
   one trained model serves both arms, which differ only in the real-test encoding.
7. **Decoder resampling retained.** A reviewer suggested average-pooling the
   stride-2 pyramid level instead of nearest subsampling. On the fixed
   16-sample RGB overfit (seed 249, constant LR 1e-3, 200 updates) average pooling
   reached 6.26 px versus 1.70 px for the frozen nearest decoder, which also
   reproduces the original 1.77 px gate. The frozen decoder is kept.

### Two-protocol study design (user decision 2026-10-05)

Natural synthetic data never clips (G1A) and Albumentations flare failed coverage
(S1A). The study now runs two separately named protocols with identical models,
heads, budgets, and selection rules:

- **Natural:** unmodified synthetic data. The learned clipping gate receives no
  training signal; fixed physical gates and learned-reliability controls remain
  testable on real data.
- **TSS (targeted specular saturation, `tss-v1`):** a deterministic per-sample
  corruption adding 1–3 anisotropic white Gaussian highlights with bloom,
  centred on object pixels, at native crop resolution with probability 0.5.
  Event frames are unchanged (real glare can also create spurious events, a
  stated limitation). Frozen from synthetic train crops only: 100% of corrupted
  crops contain a patch with at least 1% clipping at 254, the median corrupted
  crop has 16% active patches (p10–p90 7–24%), and uncorrupted crops stay at 0%,
  meeting the S1A coverage criteria the flare profiles failed. Validation is
  corrupted with the same rule for protocol-TSS checkpoint selection.

The fusion family (`saga_pose.models.fusion`) shares two pretrained encoders,
per-level fusion at every pyramid level, and the frozen decoder: `mean`,
`concat`, `content_gate` (unrestricted learned reliability), `clip_gate`
(monotone, proposed), `clip_gate_constant` (parameter-matched no-physics
control), `clip_gate_fixed` (no parameters), and `content_clip_gate` (does
clipping add beyond learned reliability?). Shuffle, zero, and gate-bypass
interventions are built into the forward pass.

### Fusion-stack review findings and design consequences (2026-10-05)

An independent review of the fusion family and training stack produced these
binding interpretation rules and changes before any matrix training:

- **Natural-protocol clipping gates are hand-set priors.** Natural synthetic
  RGB never clips, so the clipping slope receives no gradient. `clip_gate` and
  `clip_gate_constant` therefore train identically under the natural protocol;
  their difference is reported only as the GPU-nondeterminism noise floor. Any
  real-test effect of a natural-protocol clipping gate is the effect of its
  initialization (softplus slope 5), in the same class as `clip_gate_fixed`.
- **Initialization confound removed.** Fusion modules are now built after the
  decoder inside a forked RNG, so decoder initialization is identical across
  fusion rules for a given seed (tested).
- **Monotone-prior confound probed.** `clip_gate_signed` (unconstrained sign,
  same starting gate) separates the clipping signal from the monotone
  constraint, and a slope-initialization ablation (0.1 instead of 5) tests
  whether the strong initial prior drives any TSS result.
- **TSS circularity probed.** TSS saturates RGB only and the clipping map
  locates it exactly. Unimodal TSS baselines, a training-only TSS strength sweep
  (`--tss-fraction 0.25`, effective rate 12.5%), and the natural protocol bound
  this; real-test clipping is the only out-of-protocol evidence. A post-G6
  descriptive check will relate real clipping to real RGB-only error.
- **Interventions** (clipping shuffle/zero, gate bypass) are always evaluated on
  TSS validation, where clipping exists; gate bypass is omitted for rules
  without a gate.
- **No photometric/geometric training augmentation** is used for any variant.
  This is shared by every comparison but limits absolute sim-to-real accuracy.
- **Provenance:** every run stores a hash and copy of the exact source diff;
  checkpoints are written atomically; the real test set is evaluated once per
  checkpoint and re-evaluation is refused.

## 2026-10-05 — Bounded architecture screen resolved: retain HRNet

Stage B (16 evenly spaced Cassini grouped-v1 training samples, 200 full-batch
updates, constant LR, pass = RMSE <= 5 px and >= 90% reduction) under crop-v2:

| Candidate | RGB @1e-3 | Event @1e-3 | RGB @2e-4 | Event @2e-4 |
| --- | ---: | ---: | ---: | ---: |
| A0 HRNet-W18-Small-v2 | **1.70 (pass)** | 10.40 | — | 11.11 |
| A1 EfficientNet-B0 | 38.72 | 49.91 | 46.27 | 65.35 |
| A2 TinyViT-5M | 18.80 | 19.06 | 36.65 | 41.98 |
| A3 SegFormer MiT-B0 | 95.97 | 124.66 | 101.69 | 94.22 |
| A4 SwinV2-T | 84.18 | 93.63 | 55.56 | 81.04 |
| A5 MobileViTv2-1.0 | 15.35 | 22.05 | 35.13 | 42.69 |

Final keypoint RMSE in px. The single predeclared LR fallback made most
candidates worse, so the failure is not an excessive learning rate: under the
shared shallow decoder (1x1 laterals summed at 64x64), coarser-stride encoders
leave the trivial near-zero heatmap solution far more slowly, and SegFormer/SwinV2
collapse toward corner predictions. Per the bounded rule, every challenger is
rejected and HRNet is retained for both modalities. This is a bounded null for
*this decoder and budget*, not evidence that CNNs beat transformers.

HRNet Stage C under crop-v2 reproduces the frozen reference within noise
(RGB 7.84 px / 3.60 deg / 14.0 mm median; event-raw 10.15 px / 4.65 deg /
23.8 mm; 100% PnP valid; both still improving at epoch 12). The event Stage B
miss (10.4 px raw; 7.46 px with the activity representation) follows the switch
to neutral event padding; the event path remains competent at Stage C, and the
activity encoding overfits faster than raw. Evidence: `outputs/runs/screen/`.

Main matrix `configs/experiments/main_v1.yaml` (111 per-object runs; protocol
frozen in `docs/fusion-protocol.md`) launched with HRNet.

### Correction after audit (2026-10-05)

- The Stage B mechanism above is overstated. EfficientNet-B0 and MobileViTv2
  expose stride-2 features like HRNet and still failed, and SwinV2 and
  SegFormer-event improved at 2e-4. The supported statement is narrower: under
  one shallow decoder frozen on HRNet, constant LR without warmup, 200 updates,
  and a gate calibrated on HRNet, no challenger fit 16 samples. The cause is
  undiagnosed; EfficientNet's failure also shows this is not a CNN-vs-transformer
  result.
- **Protocol exception (HRNet event).** Under crop-v2 HRNet-event fails Stage B
  (10.40 px raw at 1e-3, 11.11 px at 2e-4, 7.46 px activity; its crop-v1 pass was
  marginal at 4.99 px). It proceeds under the incumbent exemption because it
  passes Stage C competence, but its Stage C is worse than the crop-v1 reference
  (median ARE 4.65 vs 4.00 deg, +16%; median ATE 23.8 vs 19.3 mm, +23%; single
  seed, no variance estimate). The link to neutral padding is temporal, not
  tested. The `activity` representation used as the fusion default has no Stage C
  run; its unimodal main-matrix run (`u-event-activity-*`) is its first.
- The natural `clip_gate` vs `clip_gate_constant` pair shares seed, init, and
  data order; it measures GPU nondeterminism only. Seed variance comes from the
  three-seed TSS rules.
- **Primary comparison (pre-registered before any fusion result):**
  `f-content_clip_gate-tss-activity` vs `f-content_gate-tss-activity`, overall
  sequence-mean ARE and ATE (co-primary). All other comparisons are secondary;
  the TSS-protocol secondary family is Holm-corrected, the rest are exploratory.
- Cross-protocol comparisons (TSS-trained vs natural-trained) select checkpoints
  on different validation sets; that confound is stated wherever they appear.
- Real-image clipping prevalence (label-free: share of real crops with any
  patch s >= 0.01, overall and on the harsh slice) is a predeclared G6
  descriptive analysis, computed only after every checkpoint is frozen, so that
  it cannot influence design.

## 2026-10-05 — Interim synthetic results (single seed; not the frozen real test)

Main-matrix runs completed so far, pooled over the three per-object models
(frame-weighted means on synthetic validation; PnP failures counted):

| Method | Train protocol | Natural ARE | TSS ARE | TSS ATE | TSS 10°/10 mm | Gate clipped / clean |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| RGB only | natural | 3.70° | 34.52° | 228 mm | 35.9% | — |
| RGB only | TSS | 5.36° | 13.72° | 62 mm | 38.2% | — |
| Event only (raw) | natural | 9.65° | 9.65° | 33 mm | 47.3% | — |
| Event only (activity) | natural | 9.06° | 9.06° | 38 mm | 42.4% | — |
| Fusion, content gate | TSS | 5.81° | 7.45° | 35 mm | 53.2% | 0.471 / 0.491 |
| Fusion, content + clipping gate | TSS | 4.27° | 7.45° | 37 mm | 46.2% | 0.133 / 0.506 |

Interim reading (descriptive, one seed, synthetic only): TSS breaks RGB-only
models; TSS training recovers part of the loss at a clean-data cost; learned
fusion beats both single sensors under TSS. The hybrid clipping gate routes
clipped patches strongly toward events and its synthetic predictions depend on
the clipping input (TSS-validation RMSE 11.07 px; shuffled 11.60; zeroed 11.66),
but its pose outcome versus the content gate is mixed (lower RMSE and clean ARE,
higher TSS ATE, lower 10°/10 mm success). The pre-registered real-test
bootstrap and seeds 250/251 decide the primary comparison.

Operational note: one run failed with a transient
`CUDNN_STATUS_EXECUTION_FAILED_CUDART` during post-training evaluation under host
memory pressure; it was retrained and reproduced its training curve exactly
(best 6.23 px at epoch 11). The trainer now retries a failed evaluation once and
persists per-epoch training state so evaluation can resume without retraining.

### Interim mechanism check: where the clipping gates lose (2026-10-05, synthetic, one seed)

Splitting TSS-validation errors by whether each frame was actually corrupted
(`scripts/analyze_tss_split.py`):

| TSS-trained method | Corrupted ARE / ATE | Clean ARE / ATE |
| --- | --- | --- |
| Mean fusion | **6.96° / 34.2 mm** | 4.78° / 21.5 mm |
| Content gate | 9.08° / 36.2 mm | 5.91° / 32.9 mm |
| Content + clipping gate | 10.88° / 52.6 mm | **4.22°** / 22.2 mm |
| Clipping gate | 11.82° / 54.9 mm | 4.73° / 23.4 mm |
| Constant control | 11.70° / 53.7 mm | 6.09° / 35.9 mm |
| RGB only (TSS) | 22.76° / 106.8 mm | 5.21° / 19.4 mm |

The clipping gates are worst exactly on the corrupted frames they target and
competitive on clean frames. Routing clipped patches almost entirely to events
(mean RGB trust ~0.12) appears to discard partially intact RGB context that
mean fusion retains; accumulated event frames are sparse edge responses inside a
16-pixel patch. Hypothesis to test post-matrix (synthetic only): the
`slope0.1` ablations (weaker initial prior) and the signed gate already in the
matrix address whether a softer or learned-sign response recovers this loss.

### Audit correction to the interim mechanism check (2026-10-05)

An independent recomputation confirmed the arithmetic but changes the reading:

- **Concentrated in one object.** Corrupted-frame mean ARE, mean fusion vs
  fixed gate: cassini 3.81 vs 4.86 deg, satty 10.47 vs 11.92, soho 6.68 vs 24.06.
  Pooled medians are 3.44 vs 5.23 deg (about 1.5x, not 1.9x); frames above
  45 deg number 19 vs 51 of 931.
- **Selection bias on the reported set.** Checkpoints are selected by RMSE on
  the same synthetic validation set that is reported, and soho validation curves
  are noisy (mean fusion soho reached 6.1 px at its selected epoch with
  neighbouring epochs at 15-18 px). Synthetic numbers are therefore optimistic
  and noisy; the frozen real test is the unbiased estimate.
- **Single-seed noise is as large as the effect.** `clip_gate_constant`
  (gate ~0.497, functionally near-mean) reaches 11.70 deg on corrupted frames
  versus 6.96 deg for mean fusion.
- **The learned clipping gate does not move from its initialization**
  (soho clipped-patch gate 0.119 at epoch 1, 0.120 at the end), so this tier does
  not test whether a gate can learn; the `slope0.1` and signed-gate ablations do.
- **Mechanism not supported as stated.** Error correlates weakly with clipped
  area (Spearman 0.17-0.28 for every method), and the fixed-gate-minus-mean gap
  does not grow with clipped area (median +1.26/+1.19/+1.49 deg by tercile).

The "routing to events discards RGB context" statement above is withdrawn as a
finding and kept only as a hypothesis. Conclusions wait for seeds 250/251, the
gate ablations, and the pre-registered real-test analysis. Only selected
checkpoints are stored, so epoch-to-epoch variance is reported from the saved
per-epoch validation curves rather than re-evaluated.

## 2026-10-06 — Tier 1 complete (54 runs; synthetic validation, one seed)

Natural-protocol fusion, pooled over objects (clean ARE / TSS ARE, mean deg):
mean 3.27 / 30.67; concat 3.18 / 25.35; content gate 3.15 / 28.35; clipping gate
3.44 / 33.11; constant control 3.44 / 33.46; fixed gate 3.27 / 32.30; hybrid
3.15 / 30.85. Every natural-trained fusion collapses under TSS like RGB-only
(34.52), so exposure to saturation in training, not the fusion rule, determines
robustness. Design sanity checks hold exactly: natural `clip_gate` equals
`clip_gate_constant`, the fixed gate equals mean fusion, and the hybrid equals the
content gate on clean validation (identical metrics), because clipping is absent
during natural training. Their TSS differences isolate the untrained clipping
prior, which never helps (fixed and hybrid priors hurt slightly; zeroing the
clipping input of the natural clipping gate is mildly better than using it).
Full tables: `outputs/runs/main-hrnet/results.md`, `tss_split.md`.
