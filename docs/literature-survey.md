# Literature & State-of-the-Art Survey

**Project:** Saturation-Gated Attention Fusion for Event–RGB Spacecraft Pose Estimation Under Harsh Lighting

**Course:** CMPE 249: Intelligent Autonomous Systems, Fall 2026

**Team:** Advait Shinde and Jasper Morgal

**Survey date:** September 8, 2026

## Scope and selection

This survey covers eight papers published from October 2023 through 2026, within the preceding three years. It prioritizes spacecraft pose estimation, event–RGB fusion, synthetic-to-real generalization, and efficient attention mechanisms. Conference or journal publication years are used; earlier preprint dates are identified where relevant. Sources are original papers, publisher records, institutional repositories, and author-maintained repositories.

The research question is whether a compact gate based on local RGB saturation can improve generalization to real harsh-lighting conditions compared with unconstrained learned fusion. The papers below provide direct competitors, alternative approaches to domain shift, and relevant fusion designs. Reported results come from the cited studies; none have been independently reproduced for this project. Project implications are our interpretation of those results.

## Comparison overview

| # | Paper / model | Publication | Research focus | Role in this project |
| --- | --- | --- | --- | --- |
| 1 | Event–RGB fusion / FRESH | Aerospace Science and Technology, 2026; preprint 2025 | Spacecraft pose through geometric fusion | Primary dataset and RANSAC reference |
| 2 | Cross-modal monocular–neuromorphic fusion | Aerospace Science and Technology, 2025 | CNN–Transformer spacecraft fusion | Direct learned-attention predecessor |
| 3 | UE-Pose | Advances in Space Research, 2026 | Uncertainty-driven spacecraft fusion | Recent competitor using local reliability |
| 4 | Test-Time Certifiable Self-Supervision | IROS, 2024 | Event-based spacecraft domain adaptation | Alternative response to real sensor degradation |
| 5 | FastPose-ViT | WACV, 2026; preprint 2025 | Efficient monocular spacecraft pose | Backbone and pose-recovery reference |
| 6 | Predicted-View Synthesis adaptation | IEEE TAES, 2024 | Temporal adaptation for spacecraft pose | Generalization reference and possible extension |
| 7 | SDSTrack | CVPR, 2024 | Efficient multimodal tracking | Adapter and modality-failure controls |
| 8 | Orthogonal High-Rank Augmentation | ICCV, October 2023 | Regularized RGB–event attention | Attention regularization and diagnostic reference |

Publication details and source links appear in each entry below. Tracking results and results from different spacecraft datasets are not a shared leaderboard: they use different outputs, metrics, data, and adaptation protocols.

## Paper summaries

### 1. Event-RGB Fusion for Spacecraft Pose Estimation Under Harsh Lighting

**Authors:** Mohsi Jawaid, Marcus Märtens, and Tat-Jun Chin.

**Publication:** *Aerospace Science and Technology*, volume 168, article 111039, January 2026; first arXiv release July 2025. [Publisher record](https://doi.org/10.1016/j.ast.2025.111039).

**Method and findings.** FRESH combines aligned event–RGB data with RANSAC fusion of learned keypoints, distance checks, and dropout uncertainty. The paper reports better real-sequence success than individual channels without test-time learning. [Full paper](https://arxiv.org/html/2507.05698v3).

**Limitations and project relevance.** Reproduce the complete method and ablate plain RANSAC; low motion can weaken the event channel. Resources: [author repository](https://github.com/mohsij/space-event-rgb-fusion) and [dataset](https://zenodo.org/records/15861758).

### 2. Cross-Modal Fusion of Monocular Images and Neuromorphic Streams for 6D Pose Estimation of Non-Cooperative Targets

**Authors:** Yishi Wang, Michele Maestrini, Zexu Zhang, Mauro Massari, and Pierluigi Di Lizia.

**Publication:** *Aerospace Science and Technology*, volume 163, article 110338, 2025. [Publisher record](https://doi.org/10.1016/j.ast.2025.110338).

**Method and findings.** This work combines CNN feature extraction with Transformer attention to fuse monocular images and event streams. It introduces the synthetic Event-Monocular-Spacecraft dataset. The institutional abstract reports an 87% high-precision estimation rate on that dataset and improvements over both single-sensor variants. It also reports a separate real-world evaluation with 0.0535 m translation error and 3.082° angular error; its cited comparator has worse translation but better angular error. [Institutional abstract](https://re.public.polimi.it/handle/11311/1291205).

**Limitations and project relevance.** The accessible abstract does not establish equivalence to FRESH's real event–RGB harsh-lighting protocol or explain the high-precision threshold. Those results cannot be compared directly with FRESH. Nevertheless, attention-based event–RGB spacecraft fusion already exists: the proposed contribution must concern the saturation constraint and its measured generalization benefit. This summary uses the publisher preview and institutional abstract because full text was restricted.

### 3. UE-Pose: Uncertainty-Driven Cross-Modal Fusion of Event and RGB Imagery for Spacecraft Pose Estimation

**Authors:** Zhaoxiong Li, Shihao Han, Haodong Yan, Dong Le, Yonghe Zhang, Zhencai Zhu, and Guopeng Ding.

**Publication:** *Advances in Space Research*, 2026; DOI: 10.1016/j.asr.2026.08.069. [Publisher record and abstract](https://www.sciencedirect.com/science/article/pii/S0273117726012202).

**Method and findings.** UE-Pose estimates uncertainty for individual keypoints using evidential learning in one forward pass. An EfficientNet-B0 encoder and a Space-Aware Feature Pyramid decoder support keypoint prediction. Uncertainty guides cross-modal fusion, outlier rejection, structural consistency checks, and pose selection/refinement. The publisher abstract reports competitive accuracy on FRESH with reduced computational cost, but does not provide sufficient numerical results to establish a ranking against the original FRESH method. [Publisher abstract](https://www.sciencedirect.com/science/article/pii/S0273117726012202).

**Limitations and project relevance.** This is a particularly close competitor because it already adjusts fusion according to local reliability. Our distinction would be an explicit saturation signal acting on patch features, versus learned uncertainty acting on keypoints. A comparison must verify equivalent splits, adaptation rules, and compute measurements. The [author-linked repository](https://github.com/JEFFLzx/UE-Pose) contains implementation files. This entry is based on the accessible publisher preview; its complete experimental tables were not available for verification.

### 4. Test-Time Certifiable Self-Supervision to Bridge the Sim2Real Gap in Event-Based Satellite Pose Estimation

**Authors:** Mohsi Jawaid, Rajat Talak, Yasir Latif, Luca Carlone, and Tat-Jun Chin.

**Publication:** IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2024. [Paper and publication status](https://arxiv.org/abs/2409.06240).

**Method and findings.** The method corrects event-based pose predictions by aligning a projected satellite point cloud with observed events. A certifier selects sufficiently consistent corrections before they are used for self-supervised model updates. The authors report improvements over established test-time adaptation approaches. They also document that strong directional lighting can produce spurious events and uneven event densities, leaving a synthetic-to-real gap even for event cameras. [Paper](https://arxiv.org/abs/2409.06240).

**Limitations and project relevance.** Certification and optimization add work at inference, and adaptation changes model parameters using target-domain observations. This differs from the proposal's fixed-model evaluation. It motivates event-quality checks and cautions against treating events as universally reliable whenever RGB is saturated. If reproduced, results should appear in a separate adaptation-enabled comparison rather than being combined with models frozen before real testing.

### 5. FastPose-ViT: A Vision Transformer for Real-Time Spacecraft Pose Estimation

**Authors:** Pierre Ancey, Andrew Price, Saqib Javed, and Mathieu Salzmann.

**Publication:** IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2026; preprint December 2025. [Conference paper](https://openaccess.thecvf.com/content/WACV2026/papers/Ancey_FastPose-ViT_A_Vision_Transformer_for_Real-Time_Spacecraft_Pose_Estimation_WACV_2026_paper.pdf).

**Method and findings.** FastPose-ViT processes a detected spacecraft crop and predicts intermediate pose variables. Projective geometry converts crop-relative translation and apparent rotation into the full-image pose, avoiding a PnP stage. The authors report accuracy competitive with PnP approaches on SPEED and better than other non-PnP methods. On a Jetson Orin Nano, the deployed pipeline achieves approximately 75 ms sequential latency and up to 33 FPS with concurrent stage scheduling. [Paper abstract](https://arxiv.org/abs/2512.09792).

**Limitations and project relevance.** This is a monocular architecture, so its results do not establish performance for event–RGB fusion or FRESH. It provides a concrete reference for the proposal's intermediate-variable pose head and edge-compute evaluation. Latency and pipelined throughput must be reported separately. The authors release [code](https://github.com/PierreAncey/FastPose-ViT); the proposed fusion model would still require its own memory and runtime measurements.

### 6. Test-Time Adaptation for Keypoint-Based Spacecraft Pose Estimation Based on Predicted-View Synthesis

**Authors:** Juan Ignacio Bravo Pérez-Villar, Álvaro García-Martín, Jesús Bescós, and Juan C. SanMiguel.

**Publication:** *IEEE Transactions on Aerospace and Electronic Systems*, 2024; DOI: 10.1109/TAES.2024.3410956. [Publication record](https://arxiv.org/abs/2410.04298).

**Method and findings.** The model predicts poses from adjacent spacecraft images and reconstructs one view from the other. Reconstruction error supplies test-time supervision, while a keypoint-structure loss discourages degenerate solutions. On SHIRT's synthetic-to-Lightbox evaluation, Table II reports combined pose scores decreasing from 0.614 to 0.083 on ROE1 and from 2.660 to 0.200 on ROE2 after adaptation; lower is better. [Full paper, Table II](https://arxiv.org/html/2410.04298v1).

**Limitations and project relevance.** The approach uses temporal image pairs and target-domain optimization. It therefore belongs outside the current single-pair, frozen-model scope. Its results show why a survey of generalization methods must distinguish improved fusion from adaptation using real observations. It is a possible later extension, with extra compute and temporal requirements. [Author code](https://github.com/JotaBravo/spacecraft-tta).

### 7. SDSTrack: Self-Distillation Symmetric Adapter Learning for Multi-Modal Visual Object Tracking

**Authors:** Xiaojun Hou et al.

**Publication:** IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024. [Paper and author list](https://arxiv.org/abs/2403.16002).

**Method and findings.** SDSTrack uses lightweight adapters to transfer an RGB tracker to other modalities and fuse features symmetrically. Complementary patch masking and self-distillation train the model to tolerate missing information. On the RGB–event VisEvent benchmark, the paper reports 76.7% precision and 59.7% success. Its adaptation ablation trains 14.79 million parameters out of 107.80 million total, illustrating the distinction between trainable parameters and full model size. [Full paper, Section 4 and Table 6](https://arxiv.org/html/2403.16002v2).

**Limitations and project relevance.** Its output is a tracked bounding box, not spacecraft position and orientation, and its benchmarks do not establish synthetic-to-real spacecraft performance. It nevertheless supplies a useful alternative explanation for gains: efficient adaptation and modality masking may improve robustness without explicit saturation modeling. An adapter or masking control would help isolate the gate's contribution. [Author code](https://github.com/hoqolo/SDSTrack).

### 8. Cross-Modal Orthogonal High-Rank Augmentation for RGB-Event Transformer-Trackers

**Authors:** Zhiyu Zhu, Junhui Hou, and Dapeng Oliver Wu.

**Publication:** IEEE/CVF International Conference on Computer Vision (ICCV), October 2023. The conference publication falls within the survey window; its preprint first appeared in July 2023. [Conference record](https://openaccess.thecvf.com/content/ICCV2023/html/Zhu_Cross-Modal_Orthogonal_High-Rank_Augmentation_for_RGB-Event_Transformer-Trackers_ICCV_2023_paper.html).

**Method and findings.** The authors introduce cross-modal token masking to encourage interaction between RGB and event features in pretrained Transformers. An orthogonal high-rank loss regularizes attention and stabilizes learning under masking. Experiments report gains in tracking precision and success when these augmentations are applied to both one-stream and two-stream trackers. [Paper](https://arxiv.org/abs/2307.04129).

**Limitations and project relevance.** These are tracking experiments, not evidence that a particular attention rank will improve spacecraft pose generalization. The work motivates a regularized-attention comparator and makes attention rank a relevant diagnostic. It also shows that a low parameter count and a low attention-matrix rank are different concepts. The saturation-gated model should measure rank and entropy without presuming that either must decrease. [Author code](https://github.com/ZHU-Zhiyu/High-Rank_RGB-Event_Tracker).

## SOTA assessment and implications for the proposal

### What the evidence establishes

This survey does not establish an uncontested current SOTA. UE-Pose warrants a matched comparison before ranking the original FRESH pipeline. [UE-Pose publisher record](https://www.sciencedirect.com/science/article/pii/S0273117726012202).

FRESH argues that attention risks synthetic overfitting; it does not demonstrate a matched attention baseline's failure. [FRESH, Sections 1 and 5.3](https://arxiv.org/html/2507.05698v3). Describing the earlier monocular–neuromorphic paper as entirely synthetic-only also conflicts with its reported real-world evaluation. [Institutional abstract](https://re.public.polimi.it/handle/11311/1291205).

**Our interpretation:** the defensible contribution is a controlled test of a physically informed fusion constraint. The hypothesis is that local saturation information improves real-domain accuracy and generalization beyond what can be achieved by parameter reduction or regularization alone. This survey does not establish that the proposed gate is novel in every detail or that it will outperform existing methods.

### Evaluation decisions supported by the survey

1. **Match the original benchmark metrics.** FRESH reports translation distance in meters and separate success rates below 10 mm and 10°. [FRESH, Section 5.2](https://arxiv.org/html/2507.05698v3). Report these alongside mean errors and generalization gaps. Label range-normalized translation error separately: it is dimensionless.

2. **Isolate the fusion change.** Keep data splits, event representation, preprocessing, backbone, pose head, training budget, and augmentation consistent across the learned fusion variants. Compare the proposed gate against unrestricted attention, a parameter-matched learned gate without saturation input, a fixed saturation gate, and an attention-regularization control. Report full parameters, trainable parameters, peak memory, and measured latency separately. These are proposed experimental controls, not results already established by this survey.

3. **Include both sensor failure modes.** Evaluate harsh-lighting and low-motion subsets separately. Add controlled event dropout or sparsity tests; RGB saturation alone cannot certify event quality. This follows from the event-sensor degradation documented in the certifiable self-supervision work. [Event-based adaptation paper](https://arxiv.org/abs/2409.06240).

4. **Keep adaptation protocols explicit.** Reserve the principal comparison for models fixed before real evaluation. Report any use of unlabeled real images or temporal adaptation separately. The two adaptation papers provide useful context, but answer a different experimental question from the proposed fixed-model study.

5. **Measure both absolute error and the gap.** A smaller synthetic-to-real gap can result from worse synthetic performance. Retain the proposal's synthetic competence check and report real error directly. Bootstrap over complete real sequences to account for correlated frames. Treat attention entropy, effective rank, and gate visualizations as descriptive evidence; use gate removal or shuffled saturation inputs to test whether the physical signal actually contributes.

The proposed semester study can therefore focus on one concrete question: **does a compact saturation-informed gate improve real harsh-lighting spacecraft pose estimation under a controlled synthetic-training protocol?** The eight papers above establish the relevant baselines and design alternatives while leaving that experiment open.
