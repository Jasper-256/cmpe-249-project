# Research and Experimentation Roadmap

**Project:** Saturation-Gated Attention Fusion for Event-RGB Spacecraft Pose
Estimation Under Harsh Lighting  
**Roadmap date:** September 27, 2026  
**Primary dataset:** FRESH  
**Core protocol:** synthetic-only training and model selection; one frozen
evaluation on real sequences

## Executive decision

The project is feasible for a semester if it is run as a controlled fusion
study, not as a simultaneous attempt to invent a new backbone, pose solver,
event representation, and domain-adaptation method. The strongest contribution
is a causal comparison: whether an explicit patch-level RGB clipping signal adds
value beyond parameter count and generic learned reliability.

The recommended minimum viable system still uses compact encoders, one frozen
landmark/pose contract, and a matched saturation-gate comparison. The encoder
family is now deliberately unresolved: HRNet-W18-Small-v2 is a verified
reference, while compact CNN, hierarchical-transformer, hybrid, and
event-specific alternatives pass through a bounded synthetic-only architecture
gate before the fusion protocol is frozen.

## Current evidence that changes the plan

- The FRESH paper and dataset establish aligned event-RGB harsh-light captures
  and a learning-free RANSAC fusion reference, not a matched proof that learned
  attention fails. See the [paper](https://doi.org/10.1016/j.ast.2025.111039),
  [author repository](https://github.com/mohsij/space-event-rgb-fusion), and
  [dataset record](https://zenodo.org/records/15861758).
- UE-Pose is now the closest same-dataset competitor. It uses per-keypoint
  evidential uncertainty throughout fusion and geometry, so any SOTA claim needs
  an equivalent protocol comparison. See the [publisher record](https://doi.org/10.1016/j.asr.2026.08.069)
  and [author repository](https://github.com/JEFFLzx/UE-Pose).
- The inspected UE-Pose implementation already contains soft inverse-variance
  fusion, hard disagreement/outlier gates, topology filtering, and
  uncertainty-weighted pose refinement. The proposed work therefore tests the
  *incremental information in explicit clipping*, not the novelty of gating.
- The official UE-Pose repository currently lacks a top-level license and a
  reproducible dependency file, normalizes uncertainty using medians computed
  over the full test sequence, and uses annotation-provided crop boxes in its
  dataset class. These facts make an author-faithful reproduction informative
  but not automatically protocol-equivalent to our frozen, non-transductive
  conditional-pose study.
- Sparse gated RGB-event fusion and newer per-pixel multimodal gates show that
  gating itself is crowded. The defensible novelty is the physical clipping
  constraint plus matched interventions. See [SNGAF](https://doi.org/10.3390/rs17173112)
  and [GatedFusion-Net](https://doi.org/10.1016/j.inffus.2025.103986).
- The current local GPU is an RTX 5060 Laptop GPU with 8 GB VRAM. Memory claims
  in the proposal are therefore treated as hypotheses until measured on both
  team devices.
- The competent HRNet reference reaches 7.62/9.16 px RGB/event keypoint RMSE,
  100% valid PnP for both, 3.88/4.00 degree median ARE, and 0.0169/0.0193 m
  median ATE on the frozen Cassini 1,024/256 synthetic screen. These results are
  preserved but do not establish architectural optimality.
- Recent spacecraft work makes a bounded transformer question defensible:
  FastPose-ViT and SPACE-HOP demonstrate viable ViT pose representations;
  A3DKS uses EfficientViT for spacecraft keypoints; and PVSPE/SO-PEN use
  pyramid/window transformers. Their heads and protocols differ, so they
  motivate rather than answer our controlled comparison.
- Recent event pretraining (STP and TESPEC, ICCV 2025) indicates that event data
  can benefit from temporal/event-native inductive bias. Those methods require
  event streams or recurrent histograms and therefore remain a separately named
  raw-event extension, not a silent replacement for the accumulated-frame input.

## Bounded architecture selection before fusion

The architecture question is:

> Under the fixed FRESH crop, split, heatmaps, PnP, and metrics, which compact
> architecture package is practically strongest, and is there evidence that RGB
> and accumulated-event images prefer different packages?

The maximum-ten audit covers HRNet-W18-Small-v2, EfficientNet-B0,
TinyViT-5M, SegFormer MiT-B0, SwinV2-T, MobileViTv2-1.0, EfficientViT-M2,
ViTPose-S, DINOv3 ViT-S/16, and an STP-style event prompt. Not all ten are
training candidates: DINOv3 is gated/custom-licensed, STP changes the input
contract, and ViTPose/SegFormer require separate integration review. See
`docs/transformer-architecture-audit.md` for the complete matrix and sources.

The staged decision is:

1. Benchmark one versioned uint8 crop-cache shard; expand only at >=2x
   complete-step throughput.
2. Stage A profiles static shapes, shared-head compatibility, finite
   forward/backward, parameters, VRAM, latency, throughput, weights, and license.
3. Stage B repeats the frozen 16-example, 200-update tiny-overfit gate.
4. Stage C admits at most four challengers to the same 1,024/256, seed-249,
   12-epoch checkpoint-selected screen that produced the HRNet reference.
5. Test heterogeneous RGB/event encoders only after a competent candidate shows
   a >=10% event-pose improvement or a meaningful modality rank reversal.

Practical-pretrained and reduced-pretraining-confound conclusions are reported
separately. A model with stronger data/distillation may be the right engineering
choice without proving that its architecture family is causally superior.

## Precise hypothesis and gate specification

Before implementing the gate, run a synthetic-only viability screen. If fewer
than 10% of sampled target crops contain a patch with at least 1% clipped pixels
at an 8-bit threshold of 254, natural synthetic FRESH does not contain enough
support to train the proposed input mechanism. This falsifies the immediate S1
training plan; it does not justify looking at real clipping statistics or adding
post-hoc augmentation. Any shared saturation-augmentation experiment is a newly
named protocol applied identically to all learned comparators.

For each RGB patch `p`, compute the clipping ratio before normalization:

```text
s_p = mean_pixel( max_channel(x_rgb) >= tau_high )
```

Keep underexposure (`min/max luminance <= tau_low`), flare proxies, and event
density as logged covariates, not inputs to the primary gate. This keeps the
headline intervention identifiable.

For attention head `h`, define RGB trust as:

```text
g_p,h = sigmoid(b_h - softplus(a_h) * s_p)
```

`softplus(a_h)` makes RGB trust monotonically non-increasing with measured
clipping. Apply `log(g + eps)` and `log(1-g + eps)` as modality biases to paired
RGB/event attention logits, or use the same weights in a paired-token residual.
The exact insertion point must be frozen after synthetic pilot studies.

Required matched interventions:

1. **Unrestricted attention:** content-conditioned fusion with no saturation
   constraint.
2. **Parameter-matched no-physics control:** identical gate module and parameter
   count with a constant input; this tests whether the physical input adds value.
3. **Fixed physical gate:** a validation-selected deterministic mapping from
   clipping ratio to trust; no learned gate parameters.
4. **Shuffled-input intervention:** permute `s_p` across patches/frames only at
   evaluation while preserving its marginal distribution.
5. **Gate bypass:** replace learned gate output by 0.5 to measure prediction
   dependence, not just attractive visualizations.

Add an event-quality-aware gate only as a separately named extension. Mixing it
into the primary model would change the hypothesis.

## Twelve-week execution plan

| Phase | Weeks | Deliverable | Exit criterion |
| --- | ---: | --- | --- |
| 0. Reproducible foundation | 1 | Locked environment, repository conventions, source registry, hardware report | Clean install and CUDA smoke test on both machines |
| 1. Dataset contract | 1-2 | Download manifest, checksum audit, paired sample browser, geometry/unit tests | Random samples pass timestamp, alignment, projection, pose-frame, and intrinsics checks |
| 2. Benchmark harness | 2-3 | ARE/ATE/native-threshold metrics, sequence aggregation, bootstrap, profiler | Metrics match hand-worked fixtures and one author-reported evaluation path |
| 3. Baseline and architecture evidence | 3-4 | Pinned FRESH geometric baseline, verified HRNet reference, cache gate, and bounded A/B/C architecture screen | Reproduction delta is explained; preprocessing is frozen; shared or modality-specific encoder decision is evidence-backed |
| 4. Fusion MVP | 4-5 | Unrestricted attention, monotone saturation gate, fixed gate, matched control using the selected encoder policy | Forward/backward tests pass; exact parameter accounting is generated |
| 5. One-object pilot | 5-6 | Memory/time scaling study and synthetic learning curves | No leakage; stable optimization; full experiment cost is estimated from measurements |
| 6. Protocol freeze | 6 | Signed experiment matrix and hashes | No design choice depends on real-test outcomes |
| 7. Core training | 7-9 | Central variants across three seeds; secondary variants as budget permits | All runs complete or failures are documented without cherry-picking |
| 8. Frozen real evaluation | 9-10 | Per-sequence predictions and paired uncertainty analysis | Real set evaluated once with frozen checkpoints and thresholds |
| 9. Stress/mechanism study | 10-11 | Saturation shuffle, bypass, clipping bins, low-motion/event-sparsity tests | Conclusions distinguish causal intervention from descriptive diagnostics |
| 10. Synthesis | 11-12 | Reproducible tables, figures, limitations, report/demo | Every claim traces to an artifact, config, seed, and statistical comparison |

## Experiment matrix and priority

| Priority | Variant | Purpose | Seeds |
| --- | --- | --- | ---: |
| P0 | RGB only | Single-modality competence and saturation failure | 1 pilot, then 3 |
| P0 | Event only | Complementarity and low-motion failure | 1 pilot, then 3 |
| P0 | Unrestricted attention | Central learned-fusion comparator | 3-5 |
| P0 | Learned monotone saturation gate | Proposed method | 3-5 |
| P0 | Parameter-matched constant-input gate | Capacity/control comparison | 3 |
| P0 | Fixed saturation gate | Is gate learning necessary? | 1 |
| P0 | Shuffled saturation and gate bypass | Causal dependence at evaluation | All proposed checkpoints |
| P0 | Published FRESH geometric fusion | Established same-dataset reference | Deterministic/reported repeats |
| P0 | HRNet-W18-Small-v2 unimodal reference | Verified high-resolution CNN reference; preserved result | Completed seed-249 screen |
| P0 | Bounded architecture screen | Challenge the shared-CNN assumption before fusion | One screening seed; repeat only Pareto survivors |
| P1 | UE-Pose author-faithful reproduction | Closest uncertainty-driven competitor; preserves disclosed transductive normalization | As published, if legally and technically reproducible |
| P1 | Non-transductive uncertainty fusion | Does learned reliability make clipping unnecessary under our frozen protocol? | 1 pilot, then 3 only if central |
| P1 | Mean/additive fusion | Low-complexity fusion sanity check | 1-3 |
| P1 | Attention regularization control | Separate regularization from physical signal | 3 |
| P2 | Event-quality-aware gate | Joint sensor reliability extension | Only after core result |
| P2 | Alternate event representation | Representation sensitivity | Only after core result |

Do not spend three-seed compute on a variant that fails basic synthetic
competence. Predeclare the failure rule and still report the pilot result.

## Evaluation and statistics

Primary outcomes:

- Rotation error in degrees using the geodesic angle on `SO(3)`.
- Translation Euclidean error in meters, plus FRESH's published 10 mm success
  threshold and rotation success below 10 degrees.
- A separately labeled, dimensionless range-normalized translation error.
- Synthetic-to-real gap for each absolute metric, reported with both endpoints.

The raw JSON translations are in millimetre-scale model coordinates (for
example, a synthetic depth of `1500` corresponds to 1.5 m). Convert translation
vectors to metres exactly once before metric computation. Report a controlled
shared/oracle-crop study separately from any end-to-end predicted-box pipeline.

The author-provided synthetic `test.json` is not used for project model
selection. It holds out regularly numbered frames while immediately adjacent
training frames differ by roughly one degree. A targeted audit found 172 of 411
validation crops meeting the project's near-duplicate proxy and a median nearest
64-bit dHash distance of zero. Project variants therefore use
`fresh-synthetic-grouped-v1`: the union of both synthetic JSONs, eight
80-frame validation blocks per object, and exclusion from training of every pose
within five geodesic rotation degrees of validation. Author-faithful baseline
reproductions retain the published partition and are not mixed with project
results.

Statistical unit and uncertainty:

- Aggregate frame predictions within each complete real sequence, then use a
  **paired sequence-level bootstrap** for model differences. Never bootstrap
  individual correlated frames as if independent.
- Report 95% intervals, the paired effect distribution, and all 24 per-sequence
  values. Treat 24 sequences as limited evidence, not a population guarantee.
- Report between-seed mean, standard deviation, and individual seed results
  separately. Bootstrap uncertainty does not replace training-seed variation.
- Correct or clearly label exploratory multiplicity when slicing many lighting,
  range, target, and motion subgroups.

Mechanistic and failure diagnostics:

- Gate response versus clipping ratio with calibration bins and confidence
  intervals.
- Prediction change under shuffled saturation and gate bypass.
- Attention entropy and effective rank as descriptive diagnostics only.
- Errors stratified by target, sequence, clipping severity, connected saturated
  area, range, event count/density, and motion proxy.
- Worst-case galleries with RGB, event representation, gate map, keypoints,
  pose overlay, and a written failure taxonomy.

## Compute plan and go/no-go rules

Profile each GPU independently with batch sizes 1, 2, 4, and 8 at mixed
precision. Record peak allocated/reserved VRAM, examples/s, optimizer-step time,
validation time, checkpoint size, and host-to-device overhead after warm-up.

Use pilot measurements to estimate:

```text
total_gpu_hours = variants * seeds * epochs * measured_epoch_seconds / 3600
```

Go/no-go rules:

- If the model cannot train with batch size 2 below 90% of available VRAM,
  reduce depth/width consistently across all learned variants before the protocol
  freeze; use gradient accumulation only after measuring its cost.
- If one-object overfit cannot drive the training objective down, stop and debug
  geometry/labels before any large run.
- If RGB/event single-modality models fail synthetic competence, do not interpret
  fusion results.
- If the published evaluation path cannot be reproduced, report the discrepancy
  and compare under the project's single verified harness; do not silently mix
  metric definitions.
- If three central seeds exceed the available budget, retain three seeds for the
  two central learned models and reduce secondary ablations first.

## Repository and artifact layout

```text
configs/              Hydra experiment, data, model, and trainer configs
src/saga_pose/         Data, geometry, models, losses, evaluation, profiling
tests/                 Unit, contract, and small integration tests
scripts/               Dataset audit and environment/report entry points
docs/                  Proposal, research log, protocol, and final analyses
data/                  Ignored local link or manifests only; never raw data in Git
outputs/               Committed lightweight reports/figures; no checkpoints or bulky logs
external/              Ignored, commit-pinned baseline checkouts
```

Every result row should be regenerable from a machine-readable run manifest and
per-sequence prediction file. Figures and tables should be generated, not edited
manually.

## Immediate next actions

1. Benchmark one immutable uint8 crop-cache shard and expand only if it doubles
   complete-step throughput without changing samples.
2. Run Stage A on locally supported candidates and preserve every rejection.
3. Run Stage B on at most six serious candidates and Stage C on at most four.
4. Converge on a shared or modality-specific encoder from pose, efficiency,
   uncertainty, and pretraining evidence—not keypoint RMSE alone.
5. Complete the published FRESH evaluation-path discrepancy audit.
6. Freeze the gate insertion point and training protocol using synthetic
   validation only, then execute the P0 fusion matrix.

## Scope reserved for follow-up work

Temporal tracking, test-time adaptation, SPEED+/SPARK transfer, raw-event
representation search, learned event-quality gating, and deployment optimization
are valuable but do not belong in the core experiment until the P0 matrix is
complete.
