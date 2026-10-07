# Architecture and Experiment Design

Diagrams render on GitHub (Mermaid). Exact specifications are in
[`fusion-protocol.md`](fusion-protocol.md); code paths are linked per block.

## 1. End-to-end pipeline

```mermaid
flowchart LR
    subgraph data["Data contract (crop-v2)"]
        Z["FRESH archives<br/>synthetic.zip, real.zip.00x<br/>(zero-copy reads)"]
        A["Annotation-box crop<br/>1.2x square, padded<br/>RGB 0 / event neutral"]
        T["TSS-v1 specular saturation<br/>(protocol 2, RGB only,<br/>deterministic per sample)"]
        C["Clipping grid s<br/>max(RGB) >= 254,<br/>16x16 area average"]
        E["Event encodings<br/>raw / activity / magnitude"]
        K["uint8 memory-mapped cache<br/>per object and role"]
        Z --> A --> T --> C
        A --> E
        C --> K
        E --> K
        T --> K
    end
    subgraph model["Per-object model"]
        R["RGB encoder<br/>HRNet-W18-Small-v2"]
        V["Event encoder<br/>HRNet-W18-Small-v2"]
        F["Per-level fusion<br/>(5 pyramid levels)"]
        D["Shared decoder<br/>1x1 laterals, nearest to 64x64,<br/>conv head"]
        H["J heatmaps<br/>(18 or 26 keypoints)"]
        R --> F
        V --> F
        F --> D --> H
    end
    subgraph pose["Pose and evaluation"]
        P["Argmax + quarter-pixel decode"]
        N["EPnP + LM refinement<br/>(failure = FRESH default pose)"]
        M["ARE / ATE / 10°-10 mm success<br/>sequence bootstrap, slices"]
        P --> N --> M
    end
    K --> R
    K --> V
    K -- "s" --> F
    H --> P
```

Code: `src/saga_pose/data/` (archives, dataset, augment, representations,
cache, real), `src/saga_pose/models/` (heatmaps, fusion),
`src/saga_pose/evaluation/` (pose, metrics, bootstrap),
`src/saga_pose/training/engine.py`.

## 2. Fusion at one pyramid level

```mermaid
flowchart TB
    r["RGB features r<br/>(B, C, H, W)"]
    e["Event features e<br/>(B, C, H, W)"]
    s["Clipping grid s<br/>resampled to H x W"]
    g{"Gate rule g(r, e, s)"}
    o["Fused = g * r + (1 - g) * e"]
    r --> g
    e --> g
    s --> g
    r --> o
    e --> o
    g --> o
```

| Rule | Gate | Fusion parameters | Uses s |
| --- | --- | ---: | :---: |
| `mean` | 0.5 | 0 | no |
| `concat` | 1x1 conv over [r, e] (no gate) | 2.80 M | no |
| `content_gate` | σ(W[r,e] + c) | 2.80 M | no |
| `clip_gate` (proposed) | σ(b − softplus(a)·s), monotone | 3,968 | yes |
| `clip_gate_constant` (control) | σ(b − softplus(a)·0) | 3,968 | no |
| `clip_gate_fixed` | 0.5·(1 − [s ≥ 0.01]) | 0 | yes |
| `content_clip_gate` | σ(W[r,e] + c − softplus(a)·s) | 2.80 M | yes |
| `clip_gate_signed` (ablation) | σ(b + w·s), free sign | 3,968 | yes |

Interventions at evaluation: shuffle s within each sample, zero s, or force
every gate to 0.5.

## 3. Experimental design

```mermaid
flowchart LR
    subgraph train["Synthetic only"]
        S["grouped-v1 split<br/>(5° rotation guard)"]
        PN["Protocol natural"]
        PT["Protocol TSS-v1"]
        U["Unimodal baselines<br/>RGB, event raw / activity"]
        FR["7 fusion rules<br/>+ signed, slope 0.1,<br/>raw events, TSS 0.25"]
        SE["Seeds 249 / 250 / 251<br/>(central TSS rules)"]
        SEL["Checkpoint selection:<br/>min synthetic-val RMSE"]
        S --> PN
        S --> PT
        PN --> U
        PT --> U
        PN --> FR
        PT --> FR
        FR --> SE
        U --> SEL
        SE --> SEL
    end
    subgraph test["Frozen real test (once)"]
        RT["24 real sequences<br/>(corrected satty-4-far)"]
        IV["Interventions:<br/>shuffle / zero / bypass"]
        ST["Paired sequence bootstrap,<br/>primary test + Holm family,<br/>harsh and low-motion slices"]
        RT --> IV --> ST
    end
    SEL -- "frozen checkpoints" --> RT
```

Primary pre-registered comparison: `content_clip_gate` vs `content_gate`
(TSS-trained), co-primary sequence-mean ARE and ATE on the real test.
