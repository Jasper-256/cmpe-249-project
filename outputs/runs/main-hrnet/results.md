# Main matrix — synthetic validation (11 complete methods)

Pooled over the three per-object models (frame-weighted means; PnP failures
counted with the FRESH default pose). `own` = validation under the training
protocol; `nat`/`tss` = natural / TSS-corrupted validation. Interventions are
keypoint RMSE (px) on TSS validation. Gate = mean RGB trust at the stride-16
level on TSS validation (clipped / unclipped patches). Real data not used.

| method | protocol | nat RMSE | nat ARE° | nat ATE m | nat 10°/10mm | tss RMSE | tss ARE° | tss ATE m | tss 10°/10mm | shuffle | zero | bypass | gate clip/clean | fusion params | last-epoch? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| u-event-activity | natural | 11.19 | 9.06 | 0.0382 | 0.424 | 11.19 | 9.06 | 0.0382 | 0.424 | — | — | — | — / — | — | no | 1.48 |
| u-event-raw | natural | 12.69 | 9.65 | 0.0325 | 0.473 | 12.69 | 9.65 | 0.0325 | 0.473 | — | — | — | — / — | — | no | 1.01 |
| u-rgb-natural | natural | 6.78 | 3.70 | 0.0168 | 0.645 | 33.02 | 34.52 | 0.2281 | 0.359 | — | — | — | — / — | — | yes | 0.28 |
| f-clip_gate-tss-activity | tss | 8.73 | 4.82 | 0.0234 | 0.540 | 12.09 | 8.16 | 0.0387 | 0.480 | 12.77 | 12.81 | 12.69 | 0.117 / 0.502 | 3968 | no | 0.32 |
| f-clip_gate_constant-tss-activity | tss | 10.30 | 6.45 | 0.0345 | 0.545 | 12.34 | 8.81 | 0.0446 | 0.484 | 12.34 | 12.34 | 12.24 | 0.497 / 0.497 | 3968 | no | 1.47 |
| f-clip_gate_fixed-tss-activity | tss | 8.60 | 5.20 | 0.0222 | 0.523 | 12.41 | 9.03 | 0.0403 | 0.460 | 13.01 | 13.29 | 13.29 | 0.000 / 0.500 | 0 | no | 0.44 |
| f-concat-tss-activity | tss | 9.23 | 6.36 | 0.0275 | 0.586 | 11.54 | 8.81 | 0.0398 | 0.528 | 11.54 | 11.54 | — | — / — | 2795456 | no | 0.27 |
| f-content_clip_gate-tss-activity | tss | 7.82 | 4.27 | 0.0215 | 0.522 | 11.07 | 7.45 | 0.0370 | 0.462 | 11.60 | 11.66 | 12.01 | 0.133 / 0.506 | 2797440 | no | 0.86 |
| f-content_gate-tss-activity | tss | 10.06 | 5.81 | 0.0313 | 0.593 | 11.84 | 7.45 | 0.0345 | 0.532 | 11.84 | 11.84 | 12.79 | 0.471 / 0.491 | 2795456 | no | 0.35 |
| f-mean-tss-activity | tss | 8.86 | 4.93 | 0.0232 | 0.580 | 10.09 | 5.84 | 0.0276 | 0.532 | 10.09 | 10.09 | — | 0.500 / 0.500 | 0 | no | 1.60 |
| u-rgb-tss | tss | 8.53 | 5.36 | 0.0220 | 0.487 | 13.94 | 13.72 | 0.0618 | 0.382 | — | — | — | — / — | — | no | 0.89 |
