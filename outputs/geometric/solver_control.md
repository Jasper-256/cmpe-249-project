RANSAC threshold 4 px (frozen). Pooled over three objects; synthetic validation.

| method [validation] | EPnP ARE mean / med | EPnP ATE mean mm | RANSAC ARE mean / med | RANSAC ATE mean mm | RANSAC 10°/10mm |
|---|---|---|---|---|---|
| u-rgb-natural [natural val] | 3.70 / 2.43 | 16.8 | 2.45 / 1.83 | 8.3 | 0.724 |
| u-event-activity [natural val] | 9.06 / 4.37 | 38.2 | 7.52 / 3.34 | 19.8 | 0.547 |
| f-clip_gate-natural-activity [natural val] | 3.44 / 2.38 | 17.0 | 2.43 / 1.78 | 9.7 | 0.721 |
| f-clip_gate_constant-natural-activity [natural val] | 3.44 / 2.38 | 17.0 | 2.43 / 1.78 | 9.7 | 0.721 |
| f-clip_gate_fixed-natural-activity [natural val] | 3.27 / 2.25 | 15.5 | 2.32 / 1.80 | 8.6 | 0.743 |
| f-concat-natural-activity [natural val] | 3.18 / 2.00 | 12.9 | 2.06 / 1.50 | 7.2 | 0.802 |
| f-content_clip_gate-natural-activity [natural val] | 3.15 / 2.20 | 15.1 | 2.21 / 1.71 | 8.5 | 0.739 |
| f-content_gate-natural-activity [natural val] | 3.15 / 2.20 | 15.1 | 2.21 / 1.71 | 8.5 | 0.739 |
| f-mean-natural-activity [natural val] | 3.27 / 2.25 | 15.5 | 2.32 / 1.80 | 8.6 | 0.743 |
| g-ransac-natural (RGB+event keypoints) [natural val] | — | — | 3.26 / 1.84 | 11.2 | 0.722 |
| u-rgb-tss [tss val] | 13.72 / 4.90 | 61.8 | 12.02 / 3.30 | 27.5 | 0.515 |
| u-event-activity [tss val] | 9.06 / 4.37 | 38.2 | 7.52 / 3.34 | 19.8 | 0.547 |
| f-clip_gate-tss-activity [tss val] | 8.16 / 3.69 | 38.7 | 5.75 / 2.31 | 15.0 | 0.635 |
| f-clip_gate_constant-tss-activity [tss val] | 8.81 / 3.39 | 44.6 | 6.09 / 2.36 | 16.6 | 0.626 |
| f-clip_gate_fixed-tss-activity [tss val] | 9.03 / 3.86 | 40.3 | 6.44 / 2.44 | 18.0 | 0.631 |
| f-clip_gate_signed-tss-activity [tss val] | 7.86 / 3.12 | 34.9 | 5.78 / 2.04 | 15.6 | 0.689 |
| f-concat-tss-activity [tss val] | 8.81 / 2.96 | 39.8 | 5.14 / 2.18 | 13.3 | 0.669 |
| f-content_clip_gate-tss-activity [tss val] | 7.45 / 3.39 | 37.0 | 5.59 / 2.47 | 14.6 | 0.598 |
| f-content_gate-tss-activity [tss val] | 7.45 / 3.02 | 34.5 | 5.26 / 2.07 | 12.7 | 0.686 |
| f-mean-tss-activity [tss val] | 5.84 / 3.03 | 27.6 | 4.65 / 2.11 | 15.0 | 0.646 |
| g-ransac-tss (RGB+event keypoints) [tss val] | — | — | 5.08 / 2.65 | 14.3 | 0.633 |
