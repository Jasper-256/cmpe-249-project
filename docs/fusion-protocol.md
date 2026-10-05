# Frozen Fusion Protocol (G4) — `main_v1`

**Frozen:** 2026-10-05, before any fusion-matrix training and before any real
evaluation. Changing any item below creates a new, separately named protocol.

## Data contract (crop-v2)

| Item | Frozen value |
| --- | --- |
| Synthetic split | `fresh-synthetic-grouped-v1` (union of author JSONs; 8×80-frame validation blocks per object; 5° rotation guard) |
| Crop | Annotation-box (shared/oracle) center-preserving square, 1.2× padding |
| Off-frame fill | RGB 0; events at the format neutral (127 synthetic, 128 real) |
| Resize | Pillow BILINEAR (antialiased) to 256×256; half-pixel geometry |
| Real labels | Zenodo annotations, with `satty-4-far` replaced by the authors' SHA-256-pinned correction |
| Real slices | Author `harsh-and-slow-frames.json`, inclusive frame ranges, overlapping subsets |
| Units | Annotation millimetres converted to metres once, in the metric |

## Clipping signal

- **Threshold:** a pixel is clipped when `max(R, G, B) >= 254` on 8-bit values.
- **Location:** native-resolution crop, before resize and normalization.
- **Patch reduction:** `cv2.INTER_AREA` average of the binary mask onto a 16×16 grid, giving `s ∈ [0, 1]^(16×16)`.
- **Per pyramid level:** nearest upsampling when the level is finer than 16×16, average pooling when coarser.

## Protocols

| Name | Training RGB / clipping | Selection | Notes |
| --- | --- | --- | --- |
| `natural` | Unmodified; `s ≡ 0` | Natural synthetic validation | Clipping slopes receive no gradient |
| `tss` | TSS-v1 (`TargetedSaturationConfig()`, seed 249, p = 0.5, deterministic per sample) | TSS-corrupted synthetic validation | Event frames unchanged |

Real evaluation always uses natural real RGB and measured real clipping.

## Event representations

- `raw`: the grey frame.
- `activity`: `|x − neutral| ≥ 32`.
- `magnitude`: `min(1, |x − neutral| / 64)`.

All three are computed at native resolution before resize. `activity` and
`magnitude` coincide on synthetic data, so an `activity` checkpoint is also
scored with the `magnitude` real encoding.

## Model

- **Encoders:** two `hrnet_w18_small_v2.ms_in1k` `features_only` encoders (RGB and event), separate weights, ImageNet normalization. The backbone may change only through the Stage C screen rule (`docs/transformer-architecture-audit.md`), applied identically to every variant.
- **Fusion insertion point:** every pyramid level (strides 2, 4, 8, 16, 32), before the shared 1×1 lateral projections.
- **Fusion outputs:** gated rules output `g·rgb + (1−g)·event` per channel and location. `concat` uses a 1×1 convolution over `[rgb, event]`.
- **Decoder:** nearest-neighbour resampling of the projected levels to 64×64, summed, then conv3×3, GroupNorm, GELU and conv1×1 to J heatmaps. J = 18 for Cassini and Satty, 26 for SOHO.
- **Fusion init:** fusion modules are built after the decoder inside a forked RNG seeded from the run seed, so decoder initialization is identical across rules.

| Rule | Gate | Init |
| --- | --- | --- |
| `mean` | 0.5 | — |
| `concat` | (1×1 conv mixer) | default |
| `content_gate` | `σ(W[rgb,event] + c)` | W = 0, c = 0 (starts as mean) |
| `clip_gate` | `σ(b − softplus(a)·s)` | softplus(a) = 5, b = 0 |
| `clip_gate_constant` | `σ(b − softplus(a)·0)` | same parameters as `clip_gate` |
| `clip_gate_fixed` | `0.5·(1 − [s ≥ 0.01])` | no parameters |
| `content_clip_gate` | `σ(W[rgb,event] + c − softplus(a)·s)` | W = 0, softplus(a) = 5 |
| `clip_gate_signed` | `σ(b + w·s)`, w unconstrained | w = −5 |

Monotonicity (non-increasing in `s`), parameter matching, the fixed-gate
extremes, intervention marginals, and initialization isolation are unit-tested
in `tests/test_fusion.py`.

## Training and selection (identical for every variant)

- **Models:** one per object, as in FRESH.
- **Training set:** every 4th frame of the grouped-v1 training membership.
- **Optimizer:** AdamW, LR 1e-3, weight decay 1e-4, batch 16, FP32.
- **Schedule:** 300 warm-up steps, then cosine decay over 15 epochs.
- **Objective:** visibility-weighted MSE on Gaussian heatmaps (σ = 1.5, 64×64).
- **Augmentation:** none.
- **Checkpoint selection:** minimum synthetic-validation keypoint RMSE under the run's protocol, with a 0.05 px minimum improvement and patience 4.
- **Pose recovery:** EPnP followed by LM refinement. PnP failures receive the FRESH default pose (identity rotation, zero translation) and count in all primary metrics.

## Interventions

- **Clipping interventions:** clipping shuffle (within-sample permutation of patch values) and clipping zero (bypass).
- **Gate bypass:** every gate set to 0.5. Not applied to `mean` or `concat`, which have no learned gate.
- **Where:** on TSS synthetic validation for every fusion run, and on real data after G6.

## Statistics and reporting

- **Primary real metrics:** sequence-mean ARE (degrees) and ATE (metres) over 24 sequences, pooling the three per-object models. Also reported: frame-level means and medians, 10°/10 mm success rates, and PnP validity.
- **Uncertainty:** paired sequence bootstrap (10,000 resamples, seed 249) for each pre-registered comparison in `configs/experiments/main_v1_comparisons.yaml`, overall and on the harsh and low-motion slices. Seed variation (seeds 249–251, central TSS rules) is reported separately.
- **Generalization gaps:** real minus natural-synthetic-validation frame means, always reported with both endpoints.
- **Interpretation rules:** a natural-protocol clipping-gate effect on real data is an untrained-prior effect. The natural `clip_gate` versus `clip_gate_constant` difference is the nondeterminism noise floor.
- **Evaluation discipline:** each checkpoint is evaluated once on real data, and the evaluator refuses re-evaluation.
