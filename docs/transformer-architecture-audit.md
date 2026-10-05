# Bounded architecture exploration audit

**Decision date:** 2026-09-27  
**Scope:** synthetic-only architecture selection for the frozen FRESH
landmark-heatmap plus PnP protocol  
**Status:** candidate set/protocol frozen; Stage 0 passed; Stage A in progress

## Decision in one sentence

HRNet-W18-Small-v2 remains the verified reference, but it is no longer frozen as
the final encoder. Run a bounded, synthetic-only screen of compact CNN,
transformer, hybrid, and event-specific alternatives; eliminate candidates with
static and tiny-overfit evidence before any additional 1,024/256 convergence
run.

## Research question and claims that are not yet justified

> Under the available data and compute, how do modern CNN, transformer, and
> hybrid encoders compare for RGB and accumulated-event spacecraft landmark
> localization, and do the two modalities prefer different inductive biases?

The existing HRNet result establishes a competent high-resolution CNN reference:

| Modality | Keypoint RMSE | Valid PnP | Median ARE | Median ATE |
| --- | ---: | ---: | ---: | ---: |
| RGB | 7.620 px | 100% | 3.877 deg | 0.01693 m |
| Event | 9.161 px | 100% | 4.005 deg | 0.01933 m |

It does **not** establish that CNNs are better than transformers, that a shared
encoder family is optimal for both modalities, or that the practical winner is
caused by architecture rather than pretraining. No real FRESH sequence may be
used to answer those questions.

## Closest recent work and what it does not answer

| Work | Relevant evidence | Why it is not a drop-in core candidate |
| --- | --- | --- |
| FRESH (Aerospace Science and Technology 2026) | Aligned event-RGB spacecraft data and learning-free RANSAC fusion; event cameras help harsh lighting but degrade under low motion. | It supplies the benchmark and geometric baseline, not a controlled learned-backbone comparison. |
| UE-Pose (Advances in Space Research 2026) | EfficientNet-B0/FPN landmark network with evidential, per-keypoint uncertainty and geometry-aware fusion. | It is the strongest same-dataset CNN threat, but the public path has crop, test-normalization, licensing, and environment caveats that must be reported. |
| FastPose-ViT (WACV 2026) | A ViT can support real-time spacecraft pose regression with crop/projective correction. | It changes the output representation and solver; use it as viability evidence, not as a heatmap-backbone result. |
| SPACE-HOP (CVPRW 2026) | JEPA-style ViT pretraining plus Hopf-grid SO(3) classification/offset regression. | It changes both pretraining and pose formulation. |
| PVSPE (Acta Astronautica 2024), SO-PEN (Advances in Space Research 2026), A3DKS (Engineering Applications of AI 2026) | Pyramid transformers, modified Swin attention, and EfficientViT have all been applied to spacecraft pose/keypoint tasks. | Their heads, losses, datasets, and evaluation protocols differ; source-reported rankings are not transferable to FRESH. |
| PAID-ViT (arXiv 2026) | Patch reliability and illumination-disentangled ViT features directly target difficult illumination. | It is a recent preprint and changes supervision/architecture together; it motivates the question but cannot validate our claim. |
| STP (ICCV 2025) | A 2.2M-trainable-parameter spatiotemporal prompt transfers frozen RGB backbones to event tasks. | It consumes temporal event streams and constructs a dense event stack; one accumulated FRESH event image cannot reproduce its mechanism. |
| TESPEC (ICCV 2025) | Event-specific temporal masked pretraining with recurrent Swin-T/7 improves detection, segmentation, and depth. | It requires recurrent raw-event histograms and a different pretraining corpus. It is a later representation study, not a core accumulated-frame backbone swap. |
| DINOv3 (2025) | Frozen self-supervised features are strong on dense tasks; ViT-S/16 is the smallest released ViT. | It is 21M parameters, stride 16, pretrained on 1.689B images, gated, and governed by a custom license with prohibited-use clauses relevant to dual-use review. |

The strongest result that could make a transformer unnecessary is still the
competent HRNet reference. The strongest result that could make the proposed
physical gate unnecessary is an unrestricted learned model that already infers
RGB corruption and modality reliability. Both are falsification threats and must
remain in the final comparison.

## Candidate matrix: factual and implementation audit

FLOPs/MACs are source-reported classification values at the stated resolution
unless marked for measurement. They are not assumed to predict laptop latency.
Parameter counts exclude the shared project heatmap decoder unless noted.
Expected VRAM and local cost are planning ranges to be replaced by Stage A
measurements on the RTX 5060 Laptop GPU.

| ID | Architecture | Family; publication | Params; FLOPs/MACs | Native features at 256 | Pretraining and weights | License | Expected batch-16 FP32 VRAM; local cost |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A0 | HRNet-W18-Small-v2 | High-resolution CNN; HRNet 2019 / small-v2 model | 5.25M local encoder; 2.42G at 224 reported for classifier | strides 2/4/8/16/32 | ImageNet-1k; timm weights available | MIT model card | **1.86 GB measured**; 12-13 min/modality for current Stage C |
| A1 | EfficientNet-B0 | MBConv CNN; ICML 2019; same family used by UE-Pose | 5.3M classifier; 0.39G at 224 | strides 2/4/8/16/32 in timm | ImageNet-1k; timm/torchvision weights available | Apache-2.0 in timm | 1-2 GB; low |
| A2 | TinyViT-5M | Compact hierarchical transformer; ECCV 2022 | 5.4M; 1.3G at 224 | strides 4/8/16/32 | practical checkpoint: IN-22k CLIP distillation then IN-1k; an IN-1k-only checkpoint also exists | MIT official repository | 1.5-2.5 GB; low-medium |
| A3 | SegFormer MiT-B0 encoder | Hierarchical dense-prediction transformer; NeurIPS 2021 | 3.8M encoder; full B0 8.4G at 512 in paper | strides 4/8/16/32 | ImageNet-1k encoder weights available | NVIDIA non-commercial research license | 1.5-2.5 GB; medium because external integration is required |
| A4 | SwinV2-T window8-256 | Shifted-window hierarchical transformer; CVPR 2022 | 28M; 5.9G at 256 | strides 4/8/16/32 | ImageNet-1k 256; official/timm weights available | MIT | 3-6 GB; high, likely gradient accumulation |
| A5 | MobileViTv2-1.0 | CNN-locality plus separable self-attention hybrid; TMLR 2023 | 4.9M reported; about 1.8G at 256 | timm strides 2/4/8/16/32 | ImageNet-1k; Apple/timm weights available | Apple sample-code license; acknowledge subcomponent terms | 1-2 GB; low |
| A6 | EfficientViT-M2 | Cascaded-group-attention hybrid; CVPR 2023 | 4.0M; about 0.20G at 224 | timm exposes coarse stages; exact usable strides measured in A | ImageNet-1k; Microsoft/timm weights available | MIT official repository | 1-2 GB; low, but decoder risk is high |
| A7 | ViTPose-S | Plain ViT specialized for heatmap pose; NeurIPS 2022 | about 24M; about 5.3G at 256x192 | single stride-16 token map plus deconvolution head | MAE/ImageNet and human-pose weights exist; checkpoint choice is a domain confound | official repository license requires recorded review | 3-6 GB; high and integration-heavy |
| A8 | DINOv3 ViT-S/16 | Self-supervised foundation ViT; 2025 | 21M; roughly 12G reported at native setup | single stride-16 map; intermediate layers share resolution | LVD-1689M; gated weights require acceptance | custom DINOv3 license; **do not download before explicit scope review** | 2-5 GB frozen, substantially more full-tune; high |
| A9 | STP prompt plus frozen image backbone | Event-specific spatiotemporal prompt/transformer; ICCV 2025 | 2.2M trainable prompt reported, plus frozen backbone | multi-scale dense event stack into chosen backbone | event pretraining/checkpoint path must be verified | repository/checkpoint license must be verified | unknown; high; **not contract-compatible** |

## Candidate matrix: scientific value and challenge

Ratings are hypotheses, not results. “Event” here means the current accumulated
three-channel event frame unless explicitly marked temporal.

| ID | RGB / event / localization suitability | Scientific reason to include | Strongest expected advantage | Falsifier or confound | Stage disposition |
| --- | --- | --- | --- | --- | --- |
| A0 | high / demonstrated / high | Reference high-resolution CNN and floor every challenger must explain | Persistent fine spatial features | Older ImageNet-1k pretraining may understate modern CNN potential | Retain through every stage; do not rerun Stage C unless code equivalence changes |
| A1 | high / medium / medium-high | Tests whether an efficient conventional CNN and UE-Pose-like pyramid make attention unnecessary | Parameter/FLOP efficiency and stable transfer | A result mixes backbone with decoder; UE-Pose gains may come from uncertainty, not EfficientNet | Stage A/B; Stage C only if competent or materially cheaper |
| A2 | high / medium-high / high | Closest parameter-envelope test of hierarchical token mixing | Global context without losing a pyramid | Practical weights have stronger distillation/data than HRNet | **Primary compact transformer**; run both practical and IN-1k-only initialization if accessible |
| A3 | high / medium / high | Dense prediction is its native task; tests overlap-patch transformer bias | Multi-scale high-resolution features and light decoder | Official stack is external/non-commercial and older; integration may dominate | Stage A license/interface audit; B/C only if reproducible without copying incompatible code |
| A4 | high / medium-high / high | Tests mature shifted-window attention at 256 and provides an upper-capacity transformer control | Strong spatial hierarchy and exact 256 checkpoint | 5x HRNet capacity; a win is not family-causal and dual-stream cost may be excessive | Stage A; B only if memory safe; at most one Stage C seed |
| A5 | high / medium / medium | Tests whether explicit convolutional locality plus light global context is the best small-data compromise | Near-HRNet size and low deployment cost | May behave mainly like a CNN; feature quality for precise heatmaps is unproven | Stage A/B; advance if tiny overfit and feature resolution pass |
| A6 | high / medium / low-medium until measured | Tests the 2026 spacecraft EfficientViT direction under our own protocol | Extremely low nominal compute | Coarse timm features may force a large decoder and erase its efficiency/localization value | Stage A is decisive; reject if no usable stride-4/8 feature or decoder dominates params |
| A7 | high / low-medium / very high for human pose | Tests whether a plain ViT specialized for heatmaps beats pyramidal biases | Direct evidence that plain tokens can localize keypoints | Human anatomy pretraining, 24M capacity, and rectangular/native head are major confounds | Interface/VRAM Stage A only; B/C only if a clean 256-square, project-head path exists |
| A8 | very high practical / unknown / medium without neck | Tests whether massive self-supervised pretraining overwhelms architectural/locality disadvantages | Strong transferable dense features with frozen backbone | Pretraining scale dominates; stride 16; license/dual-use review; gated access | Documentation/static local instantiation only; no weights or Stage B without explicit license clearance |
| A9 | not RGB / high for raw temporal events / potentially high | Tests the specific hypothesis that event data prefer temporal, event-native bias | Preserves polarity/sparsity/time rather than treating events as RGB | Invalid under the frozen accumulated-frame contract; task/domain mismatch | Record as the first post-core raw-event candidate; do not train in current matrix |

Architectures audited but not promoted include FastPose-ViT, SPACE-HOP, PVSPE,
SO-PEN, PAID-ViT, FastViT, Hiera, TinyNeXt, LowFormer, and C-RADIO. They either
change the pose problem, duplicate a question already represented above, lack a
clean pretrained/reproducible path, or exceed the bounded compute envelope.
This is an evidence-based cut, not a claim that they are inferior models.

## Pretraining analysis

The exploration has two separately named comparisons:

1. **Practical pretrained comparison.** Use the strongest legally accessible,
   published checkpoint for each candidate. This answers which model we should
   actually use, but never supports “transformers beat CNNs.”
2. **Reduced-confound comparison.** For HRNet, EfficientNet, TinyViT,
   MobileViTv2, SegFormer, and SwinV2 where possible, prefer supervised
   ImageNet-1k initialization and the same project head. Training recipes and
   original augmentations still differ, so this is closer to controlled, not
   perfectly causal.

Training all models from scratch is not a useful equalizer under this small-data
budget: it replaces a pretraining confound with a data-starvation experiment.
If TinyViT's IN-22k-distilled checkpoint beats HRNet but its IN-1k-only checkpoint
does not, the supported conclusion is a pretraining/architecture package win,
not a transformer-family win.

## Staged experiment hierarchy

### Stage 0 — cache gate before compute interpretation

Build one immutable, versioned uint8 crop-cache shard from the existing 1.2x
square crop. Compare uncached versus cached complete HRNet train steps using the
same indices, FP32 batch 16, warm-up, and timing protocol. Expand toward the
estimated 13.80 GiB full synthetic cache only if median end-to-end throughput is
at least 2x and decoded tensors/crop metadata match. Raw ZIPs remain immutable.

**Result:** pass. A bit-exact 256-sample shard increased median complete-step
throughput from 18.36 to 89.50 samples/s (4.87x) across RGB/event and three
repeats. The 96.39 MiB shard reduced data wait from roughly 0.70-0.72 seconds to
0.016-0.020 seconds per batch. Full expansion is authorized but deferred until
Stage C survivors are known. Evidence:
`outputs/feasibility/crop-cache-benchmark.json`.

### Stage A — static feasibility

For every locally instantiable candidate, record:

- exact encoder/decoder/total/trainable parameters;
- source-reported and locally measured operations where available;
- feature shapes at 256x256 and heatmap-decoder compatibility;
- finite forward, loss, backward, and AdamW update;
- peak allocated/reserved VRAM, complete-step latency, throughput, and inference
  latency after warm-up;
- checkpoint URL/hash, pretraining, license, implementation source/version;
- estimated two-stream memory and checkpoint size.

Pass criteria: finite batch-1 and effective-batch-16 training, output exactly
`J x 64 x 64`, no change to crop/targets/PnP, peak allocation below 7.2 GiB or a
predeclared microbatch plan, and no unresolved license or weight provenance.
A single-scale model may use a minimal documented upsampling neck, but it is then
reported as a backbone-plus-neck package and excluded from a “same decoder” claim.

### Stage B — fixed tiny overfit

Run only serious candidates on the same 16 Cassini grouped-v1 examples used by
HRNet: both modalities, no augmentation, seed 249, 200 full-subset optimizer
updates, Gaussian 64x64 heatmaps, sigma 1.5, and the frozen decoder/PnP path.
Use effective batch 16 if microbatching is needed. Pass requires final decoded
RMSE <=5 px and at least 90% reduction from initialization for each modality.
One predeclared architecture-appropriate learning-rate fallback is allowed only
after diagnosing optimization failure; both attempts remain in the report.

### Stage C — short controlled convergence

Advance only candidates that pass A and B and answer a question not already
eliminated. Reuse the exact HRNet subset and contract: 1,024 Cassini grouped-v1
train, 256 validation, 256 input, 1.2x crop, heatmaps/PnP/metrics, seed 249,
effective batch 16, maximum 12 epochs, select minimum synthetic-validation
keypoint RMSE, minimum improvement 0.25 px, patience three.

The existing HRNet result is the reference and is not invalidated or silently
rerun. At most four new packages receive Stage C. A second synthetic seed is run
only for a candidate on the accuracy/efficiency Pareto frontier or for evidence
of an RGB/event rank reversal.

Required Stage C outputs are keypoint RMSE, PnP-valid fraction, median ARE,
median ATE, range-normalized translation error, FRESH 10-degree/10-mm success,
parameters, peak VRAM, training throughput, inference latency, curves, and
paired block-bootstrap intervals over the eight grouped validation blocks.
Frame bootstrap is not presented as sequence uncertainty.

## Go/no-go and convergence rules

A model advances beyond Stage C if it is competent in at least one modality and
provides one of the following relative to HRNet:

- at least 10% improvement in median ARE or median ATE without a greater than
  10% regression in the other pose metric;
- at least 25% reduction in total parameters, peak VRAM, or end-to-end time with
  no greater than 10% regression in both pose metrics; or
- a statistically/operationally meaningful RGB-versus-event rank reversal that
  justifies testing heterogeneous encoders.

Do not promote a model solely for lower keypoint RMSE if PnP pose worsens. Do not
promote a larger model solely for a small point-estimate win whose paired block
interval includes a practically important loss. A null result is still useful:
it supports HRNet as the efficiency/accuracy choice under this regime.

After unimodal evidence, test `HRNet|HRNet`, `T|T`, `HRNet|T`, and `T|HRNet`
only if the best transformer/hybrid is competent for both modalities and either
(a) improves an event pose metric by 10%, or (b) changes rank across modalities.
Use the same fusion module and decoder in all four. This factorial is not run to
fill a table.

## Compute and storage envelope

- Stage 0 shard: about 0.1-0.5 GiB; full two-modality uint8 cache: 13.80 GiB.
- Stage A: approximately 2-4 local GPU hours including integration/debugging;
  candidate failures stop immediately and are preserved.
- Stage B: approximately 10-30 minutes per two-modality candidate; at most six
  candidates (roughly 1-3 local GPU hours).
- Stage C: current HRNet evidence is about 26 minutes for both modalities on the
  1,024/256 screen. Budget 0.5-1.5 hours per candidate before cache benefit; at
  most four candidates (roughly 2-6 GPU hours).
- Temporary checkpoints are ignored and bounded below roughly 2 GiB total by
  retaining only each run's current/best state. Lightweight JSON and plots are
  committed under `outputs/`.

No cloud GPU should be rented for architecture discovery. Reconsider the USD
10-20 option only after the cache result, final candidate survivors, exact
commands/configs, seeds, stopping rules, resume smoke test, measured GPU hours,
storage transfer, and compatible PyTorch/CUDA image are frozen.

## Sources accessed 2026-09-27

- FRESH publisher record: https://doi.org/10.1016/j.ast.2025.111039
- FastPose-ViT official WACV paper: https://openaccess.thecvf.com/content/WACV2026/html/Ancey_FastPose-ViT_A_Vision_Transformer_for_Real-Time_Spacecraft_Pose_Estimation_WACV_2026_paper.html
- SPACE-HOP official CVPRW paper: https://openaccess.thecvf.com/content/CVPR2026W/AI4Space/html/Dutta_SPACE-HOP_Spacecraft_6-DoF_Pose_Estimation_using_Embedding-Predictive_Pretraining_and_Hopf_CVPRW_2026_paper.html
- STP official ICCV paper: https://openaccess.thecvf.com/content/ICCV2025/papers/Liang_Efficient_Event_Camera_Data_Pretraining_with_Adaptive_Prompt_Fusion_ICCV_2025_paper.pdf
- TESPEC official ICCV paper: https://openaccess.thecvf.com/content/ICCV2025/html/Mohammadi_TESPEC_Temporally-Enhanced_Self-Supervised_Pretraining_for_Event_Cameras_ICCV_2025_paper.html
- DINOv3 official repository/model table: https://github.com/facebookresearch/dinov3
- TinyViT official repository/model zoo: https://github.com/microsoft/Cream/tree/main/TinyViT
- Swin Transformer official repository/model table: https://github.com/microsoft/Swin-Transformer
- SegFormer official paper and repository: https://papers.nips.cc/paper/2021/hash/64f1f27bf1b4ec22924fd0acb550c235-Abstract.html and https://github.com/NVlabs/SegFormer
- MobileViTv2 official paper/repository: https://openreview.net/forum?id=tBl4yBEjKiD and https://github.com/apple/ml-cvnets
- EfficientViT (Microsoft) paper/repository: https://arxiv.org/abs/2305.07027 and https://github.com/microsoft/Cream/tree/main/EfficientViT
- ViTPose official paper/repository: https://arxiv.org/abs/2204.12484 and https://github.com/ViTAE-Transformer/ViTPose
- EfficientNet official paper: https://proceedings.mlr.press/v97/tan19a.html
- A3DKS publisher record: https://www.sciencedirect.com/science/article/abs/pii/S0952197626013576
- PVSPE publisher record: https://www.sciencedirect.com/science/article/pii/S0273117724004368
- PAID-ViT preprint: https://arxiv.org/abs/2606.11619
