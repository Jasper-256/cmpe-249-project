# Transformer architecture audit

**Decision date:** 2026-09-26  
**Scope:** synthetic-only architecture selection for the frozen FRESH
keypoint-plus-PnP protocol  
**Status:** decision-ready; no real FRESH sequence was inspected

## Decision in one sentence

Do not train a large transformer from scratch and do not replace the current
HRNet control merely because a newer foundation model exists. First finish the
bounded HRNet convergence test. If it remains incompetent, test exactly one
compact challenger: ImageNet-22k-distilled **TinyViT-5M** with the same heatmap
decoder and PnP path. Keep DINOv3 ViT-S/16 as a later frozen/adapter study only
if its gated weights and license are acceptable.

## Question this audit answers

The relevant question is not “what is the newest transformer?” It is:

> Which pretrained encoder gives spatially precise keypoints for RGB and
> event-frame inputs on one 8 GB GPU without changing the crop, pose head,
> solver, selection metric, or scientific comparison?

That distinction matters. FastPose-ViT and SPACE-HOP are recent spacecraft pose
papers, but both change the pose formulation. Using either as the core model
would confound backbone quality with direct pose regression, apparent-rotation
correction, or SO(3) classification. They are important comparators and possible
stretch studies, not drop-in encoders for the central fusion hypothesis.

## Constraints that dominate the choice

- The current task needs dense, spatially aligned features for 2D landmark
  heatmaps, not just a strong global classification token.
- Every learned fusion variant must retain the same backbone scale, decoder,
  PnP path, crop, split, and selection rule.
- Two modality streams and fusion must fit on the local GPU. A backbone that
  only fits as one frozen stream is not a fair central dual-encoder choice.
- Synthetic training has 48,820 labeled frames across three objects, but the
  first gate uses only 1,024/256 Cassini samples. This is not enough to justify
  foundation-model pretraining from scratch.
- FRESH supplies pre-rendered event frames in the core protocol. Methods whose
  main advantage requires raw event timing must remain stretch work.
- The real FRESH sequences are final held-out evaluation data and cannot select
  the transformer, resolution, tuning policy, or checkpoint.

## Current evidence

### Recent task-specific work

| Work | What it already does | Implication here |
| --- | --- | --- |
| FastPose-ViT (WACV 2026) | Uses a ViT to regress spacecraft pose directly from a detected crop and maps the crop-relative output back through projective geometry; it also reports edge deployment. | Strong evidence that pretrained ViTs are viable for spacecraft pose, but its direct head and SPEED protocol make it an invalid drop-in central baseline for our fixed keypoint/PnP study. |
| SPACE-HOP (CVPRW 2026) | Uses JEPA-style masked pretraining and classifies rotation on a Hopf-fibration SO(3) grid with learned offsets. | A meaningful future self-supervised comparison, but it changes both representation learning and pose output. Treat as P2, not as a backbone swap. |
| STP (ICCV 2025) | Transfers an image-pretrained model to sparse event data with a 2.2M-parameter spatiotemporal prompt; the paper reports about one tenth the pretraining parameters and one third the epochs of prior approaches on its tasks. | Supports frozen or lightly adapted image-pretrained features for events. It argues against independently training a full event transformer from scratch. Its temporal prompt cannot be reproduced faithfully from one accumulated FRESH event frame. |
| EventDAM / Depth AnyEvent (ICCV 2025) | Transfer dense RGB foundation-model knowledge to sparse event depth through feature mixing, distillation, or proxy labels. | Cross-modal transfer is plausible, but the evidence is from depth, not spacecraft pose. Adapter/distillation belongs after a competent core event baseline, not before it. |

### Candidate encoders

Parameter counts and feature shapes below were measured locally with the locked
`timm` installation using `features_only=True`; they are not paper-reported
pose results.

| Candidate | Local encoder parameters | Local pyramid at 256 px | Pretraining/access | Decision |
| --- | ---: | --- | --- | --- |
| HRNet-W18-Small-v2 | 5.25M | 64/128/256/512/1024 channels at strides 2/4/8/16/32 | ImageNet-1k, available through timm | Current spatial control. Finish its bounded convergence run. |
| TinyViT-5M | 5.07M | 64/128/160/320 at strides 4/8/16/32 | IN-22k distillation then IN-1k fine-tuning; weights available through timm/Microsoft | **Only P0 transformer challenger.** Nearly parameter-matched, hierarchical, and likely dual-stream feasible. |
| TinyNeXt-M (ICCV 2025) | 2.3M reported for classification; 3.3M with its detection system | Multi-stage hybrid; not available in the locked timm catalog | Official code and checkpoints exist, but the repository has only 15 commits and its reported IN-1k top-1 is 75.3% | Watchlist, not P0. Newer and cheaper, but weaker pretraining and a new external code path make it a poor first test of representation quality. |
| LowFormer-E1 (IJCV 2026) | Not reported in the repository model table | Edge-GPU convolutional transformer | Standalone external implementation; checkpoint distributed through Dropbox | Deployment watchlist only. Current evidence emphasizes latency, not spatial spacecraft keypoints, and it is absent from the locked toolchain. |
| DINOv3 ViT-S/16 | 21.59M | three selected 384-channel layers, all stride 16 | LVD-1689M; gated weight request and custom DINOv3 license | P1 frozen or adapter-only representation study. Not parameter-matched and spatially coarse without a neck. |
| Hiera-T | 27.14M | 96/192/384/768 at strides 4/8/16/32 | MAE plus IN-1k; weights CC BY-NC 4.0; official repository still lists training scripts as coming soon | Reject for P0: more than five times the current encoder and a less convenient reproducibility path. |
| SwinV2-T | 27.58M | 96/192/384/768 at strides 4/8/16/32 | ImageNet pretrained, readily available | Reject for P0: mature but not parameter- or memory-matched; offers no specific evidence advantage over TinyViT for this gate. |
| C-RADIOv4 | SO400M/H variants | dense stride-16 features | Released January 2026; foundation-model distillation from SigLIP2-g, DINOv3-7B, and SAM3 | Reject for the central matrix: far outside the dual-stream memory/parameter budget. RADIO1D is interesting, but official pretrained checkpoints are not released. |

TinyViT is not the newest model in the table. It is the most defensible test of
whether transformer features help, because its 5M scale closely matches HRNet,
it exposes a hierarchy for landmark localization, and its pretrained weights do
not require us to reproduce large-scale distillation. The comparison still
cannot be called perfectly capacity-matched until total decoder and trainable
parameter counts are measured.

It is also **not a causal CNN-versus-transformer experiment**: the proposed
TinyViT checkpoint saw ImageNet-22k distillation before ImageNet-1k fine-tuning,
whereas the HRNet checkpoint is ImageNet-1k supervised. Any gain mixes
architecture and pretraining data/teacher effects. That is acceptable for
selecting a competent common backbone, but the report must not attribute the
gain to self-attention. A causal architecture claim would require a matched
pretraining comparison (for example, the available TinyViT ImageNet-1k-only
weights) and is outside the P0 fusion question.

## Strongest baseline that could make a transformer unnecessary

The strongest immediate threat is the current pretrained HRNet heatmap model.
It preserves high-resolution features by design, has already passed the RGB and
event-frame tiny-overfit test, and is close to the representation used by prior
FRESH/UE-Pose work. If checkpoint-selected HRNet satisfies the frozen synthetic
competence thresholds, a transformer is not required to answer the saturation
gate question. It can be evaluated later as an architecture robustness study.

The strongest literature threat to the overall proposal remains UE-Pose, not a
new transformer. UE-Pose already performs uncertainty-aware per-keypoint fusion
and geometric rejection. The central experiment must therefore show value from
the explicit clipping input beyond learned reliability, independent of encoder
fashion.

Conversely, a strong transformer could make the proposed gate unnecessary. A
pretrained encoder plus unrestricted cross-modal attention may already encode
clipping, local contrast loss, and event reliability from appearance. That is a
falsification threat, not a reason to give only the proposed method a stronger
backbone. If a transformer becomes the reportable encoder, RGB-only,
event-only, unrestricted attention, parameter-matched no-physics control, and
the saturation gate must all use that same encoder and tuning policy. A
“transformer only for our method” comparison would not answer the research
question.

A second threat is catastrophic forgetting: fully fine-tuning a strong
foundation model on narrow synthetic imagery may destroy the invariances that
would help on real data. This cannot be selected by repeatedly checking FRESH
real sequences. If DINOv3 reaches P1, freeze-versus-late-block adaptation must be
chosen on synthetic validation and then locked before the one real evaluation.

## Predeclared TinyViT challenger

Run this only if either HRNet modality fails the bounded convergence gate.

1. Use `tiny_vit_5m_224.dist_in22k_ft_in1k` from the locked timm version.
2. Confirm the pretrained model can consume 256-by-256 inputs without positional
   or window-shape mismatch. If not, use 224 by 224 for **both** HRNet and
   TinyViT in a separately labeled architecture screen; do not compare 224 and
   256 results as if only the backbone changed.
3. Replace only the feature encoder. Retain the 64-channel decoder, Gaussian
   target definition, landmark ordering, grouped-v1 samples, crop, optimizer,
   seed, validation checkpoint rule, competence thresholds, and PnP evaluator.
4. Use at most 12 epochs with the same minimum-improvement and patience rule as
   the HRNet convergence run. One seed is sufficient for this gate.
5. Log total/trainable/encoder/decoder parameters, peak VRAM, training
   throughput, batch latency, and validation learning curves.

Advance TinyViT only if both modalities meet all frozen competence thresholds.
If it improves landmark RMSE but not ARE/ATE, inspect per-keypoint error and PnP
conditioning rather than declaring the transformer superior. If it fails, stop
the backbone search: the next experiment should test supervision or geometric
conditioning, not a sequence of larger encoders.

## Train, fine-tune, freeze, or adapt?

For this project the order should be:

1. **Fine-tune a compact pretrained hierarchy.** This is the TinyViT P0 path.
   It retains a strong initialization while allowing the spatial features to
   adapt to spacecraft landmarks and the rendered event-frame domain.
2. **Freeze then adapt a foundation encoder.** If a later DINOv3-S study is
   justified, first freeze it and train only a small dense neck or low-rank/
   prompt adapter. Only unfreeze late blocks after a frozen-feature baseline
   and a memory profile exist.
3. **Use RGB-to-event transfer only as a separate ablation.** A shared RGB
   initialization, distillation, or event adapter could be valuable, but it
   changes the event training prior and must not silently replace the ordinary
   event-only baseline.
4. **Do not pretrain from scratch.** Reproducing ImageNet-22k/LVD-scale
   representation learning on one laptop GPU is infeasible and would consume
   the project budget without isolating the clipping-gate hypothesis.

The DINOv3 license specifically requires acceptance and restricts prohibited
end uses, including activities subject to ITAR. Because spacecraft pose can be
dual-use, no DINOv3 weights should be downloaded or used until the user has
reviewed and accepted the license and confirmed that this academic project is
within scope. TinyViT's timm model card lists Apache-2.0, so it has the cleaner
first-study path.

## Compute plan before a full matrix

During the current FP32 batch-16 HRNet pilot, `nvidia-smi` reports roughly 3 GB
of total device memory in use on an 8.15 GB RTX 5060 Laptop GPU; the prior
script-level PyTorch peak allocation was 1.86 GB. Utilization is bursty because
archive decoding and deterministic preprocessing run on CPU. The next compute
gate should be a short, accuracy-neutral throughput screen rather than
immediately enlarging the model:

- profile batch 16 and 32 for FP32 and BF16 autocast after confirming
  `torch.cuda.is_bf16_supported()`;
- measure complete train-step time, loader wait, peak VRAM, and numerical
  finiteness, not forward-only throughput;
- retain the same precision for every method in a claimed comparison;
- validate that the selected synthetic metric does not materially change before
  adopting mixed precision; and
- consider an evictable one-object RAM/cache layer only if profiling confirms
  input starvation. Raw archives remain immutable and no full extraction is
  justified.

Current PyTorch 2.14 documentation recommends `torch.autocast` for mixed
precision and `torch.amp.GradScaler` for FP16 training. BF16 normally does not
need loss scaling, but that choice must be verified on this exact model rather
than assumed. `torch.compile` and custom attention kernels stay out of scope
until an ordinary eager run is competent and measured.

The screen subsequently confirmed that direction. All trials were finite;
FP32/BF16 batch-16 end-to-end throughput was 23.8/23.6 samples/s for RGB and
25.0/25.5 for event. BF16 nearly halved PyTorch peak allocation (1.73 to 0.97
GiB) but did not remove the dominant 0.47–0.51 s data wait. Batch 32 was slower.
The project therefore retains FP32 batch 16 and prioritizes one bounded crop
cache benchmark over precision or batch-size tuning.

A one-time USD 10–20 rented-GPU budget is available, but it should not fund
architecture search. Revisit current provider pricing only after the dual-stream
fusion implementation passes unit, one-batch, tiny-overfit, and one-object
gates. At that point the launch decision must include measured local GPU-hours,
peak VRAM, a compatible CUDA/PyTorch image, dataset staging time and license,
checkpoint/resume verification, the exact seed/config matrix, expected wall
time, and a hard cost ceiling. Synthetic training can move to rented compute;
keeping the frozen real evaluation local minimizes data transfer and accidental
test-set exposure.

## Falsification and null-result interpretation

The transformer hypothesis is falsified for this project if the parameter-scale
TinyViT model, under the controlled protocol, fails to improve competence or
compute efficiency over HRNet. A null result remains useful: it would show that
newer token-mixing capacity is not the limiting factor under FRESH synthetic
supervision and would protect the fusion study from an unnecessary architecture
confound.

For DINOv3, the falsifying test is a frozen ViT-S/16 encoder plus a small shared
decoder/adapter. If strong pretrained features do not improve synthetic
landmark/PnP metrics, full fine-tuning is not automatically justified. Possible
causes include stride-16 spatial coarseness and spacecraft-domain mismatch. Full
DINOv3 training from scratch is out of scope: it would test compute scale, not
the proposed physical gate.

## Sources accessed 2026-09-26

- DINOv3 official repository and model table:
  https://github.com/facebookresearch/dinov3
- DINOv3 license:
  https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md
- Hiera official repository:
  https://github.com/facebookresearch/hiera
- TinyViT official Microsoft implementation and model zoo:
  https://github.com/microsoft/Cream/tree/main/TinyViT
- timm TinyViT-5M pretrained model card and feature-map interface:
  https://huggingface.co/timm/tiny_vit_5m_224.dist_in22k_ft_in1k
- TinyNeXt official implementation (ICCV 2025):
  https://github.com/yuffeenn/TinyNeXt
- LowFormer official implementation and 2026 edge-GPU extension:
  https://github.com/altair199797/lowformer
- C-RADIO official NVIDIA repository:
  https://github.com/NVlabs/RADIO
- FastPose-ViT, WACV 2026:
  https://openaccess.thecvf.com/content/WACV2026/html/Ancey_FastPose-ViT_A_Vision_Transformer_for_Real-Time_Spacecraft_Pose_Estimation_WACV_2026_paper.html
- SPACE-HOP, CVPR Workshops 2026:
  https://openaccess.thecvf.com/content/CVPR2026W/AI4Space/html/Dutta_SPACE-HOP_Spacecraft_6-DoF_Pose_Estimation_using_Embedding-Predictive_Pretraining_and_Hopf_CVPRW_2026_paper.html
- Efficient Event Camera Data Pretraining with Adaptive Prompt Fusion, ICCV
  2025:
  https://openaccess.thecvf.com/content/ICCV2025/html/Liang_Efficient_Event_Camera_Data_Pretraining_with_Adaptive_Prompt_Fusion_ICCV_2025_paper.html
- EventDAM, ICCV 2025:
  https://openaccess.thecvf.com/content/ICCV2025/html/Zhu_Depth_Any_Event_Stream_Enhancing_Event-based_Monocular_Depth_Estimation_via_ICCV_2025_paper.html
- Depth AnyEvent, ICCV 2025:
  https://openaccess.thecvf.com/content/ICCV2025/html/Bartolomei_Depth_AnyEvent_A_Cross-Modal_Distillation_Paradigm_for_Event-Based_Monocular_Depth_ICCV_2025_paper.html
- timm feature extraction documentation:
  https://huggingface.co/docs/timm/feature_extraction
- timm HRNet-W18-Small-v2 model card:
  https://huggingface.co/timm/hrnet_w18_small_v2.ms_in1k
- PyTorch 2.14 automatic mixed precision recipe:
  https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html
