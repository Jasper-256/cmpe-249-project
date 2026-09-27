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
- https://docs.opencv.org/4.13.0/d5/d1f/calib3d_solvePnP.html
