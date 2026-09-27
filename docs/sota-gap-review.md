# SOTA, Gap, and Falsification Review

**Review date:** 2026-09-23  
**Scope:** FRESH event–RGB spacecraft pose estimation under synthetic-only
training and frozen real evaluation

## Bottom-line research claim

The defensible contribution is not a new fusion gate. FRESH already combines
per-modality uncertainty with geometric fusion, and UE-Pose already routes
individual keypoints using evidential uncertainty and outlier gates. The open,
testable question is narrower:

> Does an explicit local RGB clipping measurement add predictive value beyond
> generic learned fusion and learned uncertainty, under an otherwise frozen
> conditional-pose protocol?

A result is useful if it identifies when the clipping prior helps, is redundant,
or fails because the event stream is unreliable or the synthetic training data
does not contain the relevant signal.

## Answers to the adversarial questions

### Who has done the closest work?

1. [UE-Pose](https://doi.org/10.1016/j.asr.2026.08.069) is the closest
   same-dataset method. It predicts per-keypoint evidential uncertainty and uses
   it for soft fusion, hard modality selection, topology validation, weighted
   refinement, and pose-candidate selection.
2. [FRESH](https://doi.org/10.1016/j.ast.2025.111039) is the canonical benchmark
   and geometric baseline. It combines separately learned keypoints, dropout
   uncertainty, cross-modal RANSAC, consistency checks, and temporal box logic.
3. The 2025 cross-modal self-attention spacecraft method is architecturally
   close but was evaluated on a different synthetic dataset/protocol. Its
   headline result is not directly rank-comparable with FRESH.

### What strongest baseline could make the method unnecessary?

UE-Pose, or a non-transductive uncertainty-guided keypoint fusion implemented in
our harness, can make clipping unnecessary if learned uncertainty routes the
modalities at least as accurately and efficiently. Within the feature-fusion
family, unrestricted attention is the strongest matched baseline. A constant
input gate alone is only a capacity control and is not a strong alternative.

### What would falsify the hypothesis?

- Natural synthetic target crops rarely activate the clipping input, so the gate
  cannot learn the claimed relationship without changing the data distribution.
- The proposed model is competent on synthetic validation but does not improve
  frozen real ARE/ATE, or paired intervals rule out a practically useful gain.
- A matched no-physics or unrestricted model performs equally well, indicating
  capacity/regularization rather than clipping caused any gain.
- Shuffling clipping spatially or bypassing the learned gate leaves predictions
  unchanged.
- Improvements occur only where event quality is high and reverse during low
  motion, contradicting any unconditional “trust events when RGB clips” rule.

### Which confounds invalidate the comparison?

- Different crop sources, detectors, temporal box smoothing, pose heads,
  backbones, augmentations, training budgets, or checkpoint rules.
- Treating annotation millimetres as metres, or comparing relative translation
  error with absolute ATE.
- Selecting clipping thresholds, augmentation, checkpoints, or severity bins
  after inspecting the 24 real sequences.
- Using full-test-set statistics to calibrate uncertainty in one method but not
  another without labeling the method transductive.
- Reporting frame-wise confidence intervals despite strong within-sequence
  correlation.
- Interpreting a smaller sim-to-real gap as a win when synthetic competence is
  worse.

### What is the cheapest decisive experiment this week?

Run a deterministic, target-crop synthetic audit before training:

1. Stratify samples across three objects and train/validation roles.
2. Measure clipping at 250, 254, and 255 before normalization, before and after
   resize, globally and on a 16-by-16 patch grid.
3. Measure paired event activity and zero-copy decode throughput.
4. Stop the primary learned-gate strategy if fewer than 10% of crops contain a
   patch with at least 1% clipping at threshold 254.
5. In parallel, verify pose projection, units, intrinsics scaling, pair existence,
   and split overlap for every annotation.

This costs minutes rather than GPU-days and directly tests whether the proposed
input signal exists in the allowed training domain.

### If the metric does not improve, is there still a result?

Yes, if competence and interventions are established. A null or negative result
can show that learned uncertainty subsumes clipping, that synthetic FRESH lacks
the necessary exposure variation, that resize destroys the signal, or that event
failure dominates exactly when a one-sided gate would switch modalities. Those
are actionable boundaries of validity, not failed engineering demonstrations.

## Scope and feasibility decision

The core study uses a shared crop and shared keypoint/pose path to isolate fusion
on the 8 GB RTX 5060 Laptop GPU. It reports this as conditional pose estimation.
End-to-end detector/temporal reproduction and the author-faithful UE-Pose run are
P1 comparisons. Raw-event representation search, temporal smoothing, and other
datasets stay parked until the central matched comparison passes its gates.

The primary model is not implemented until the data-contract and clipping-signal
screens pass. This prevents architecture work from obscuring a falsified input
assumption.

## Initial decision (2026-09-23)

The natural-data clipping screen failed its go criterion decisively: zero of 384
stratified synthetic target crops contained a pixel at or above 250, and the
largest observed channel value was 233. Therefore the original learned-gate
strategy is stopped before training. The project proceeds with a separately
labeled, shared synthetic-clipping augmentation study while preserving this
negative result. The augmentation study must not be described as evidence that
unmodified FRESH synthetic data contains the physical cue.
