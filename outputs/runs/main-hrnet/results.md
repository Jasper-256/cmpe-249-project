# Main matrix — synthetic validation (6 complete methods)

Pooled over the three per-object models (frame-weighted means; PnP failures
counted with the FRESH default pose). `own` = validation under the training
protocol; `nat`/`tss` = natural / TSS-corrupted validation. Interventions are
keypoint RMSE (px) on TSS validation. Gate = mean RGB trust at the stride-16
level on TSS validation (clipped / unclipped patches). Real data not used.

| method | protocol | nat RMSE | nat ARE° | nat ATE m | nat 10°/10mm | tss RMSE | tss ARE° | tss ATE m | tss 10°/10mm | shuffle | zero | bypass | gate clip/clean | fusion params | last-epoch? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| u-event-activity | natural | 11.19 | 9.06 | 0.0382 | 0.424 | 11.19 | 9.06 | 0.0382 | 0.424 | — | — | — | — / — | — | no |
| u-event-raw | natural | 12.69 | 9.65 | 0.0325 | 0.473 | 12.69 | 9.65 | 0.0325 | 0.473 | — | — | — | — / — | — | no |
| u-rgb-natural | natural | 6.78 | 3.70 | 0.0168 | 0.645 | 33.02 | 34.52 | 0.2281 | 0.359 | — | — | — | — / — | — | yes |
| f-content_clip_gate-tss-activity | tss | 7.82 | 4.27 | 0.0215 | 0.522 | 11.07 | 7.45 | 0.0370 | 0.462 | 11.60 | 11.66 | 12.01 | 0.133 / 0.506 | 2797440 | no |
| f-content_gate-tss-activity | tss | 10.06 | 5.81 | 0.0313 | 0.593 | 11.84 | 7.45 | 0.0345 | 0.532 | 11.84 | 11.84 | 12.79 | 0.471 / 0.491 | 2795456 | no |
| u-rgb-tss | tss | 8.53 | 5.36 | 0.0220 | 0.487 | 13.94 | 13.72 | 0.0618 | 0.382 | — | — | — | — / — | — | no |
