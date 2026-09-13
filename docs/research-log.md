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

