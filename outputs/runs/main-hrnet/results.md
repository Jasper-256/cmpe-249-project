# Main matrix — synthetic validation (2 complete methods)

Pooled over the three per-object models (frame-weighted means; PnP failures
counted with the FRESH default pose). `own` = validation under the training
protocol; `nat`/`tss` = natural / TSS-corrupted validation. Interventions are
keypoint RMSE (px) on TSS validation. Gate = mean RGB trust at the stride-16
level on TSS validation (clipped / unclipped patches). Real data not used.

| method | protocol | nat RMSE | nat ARE° | nat ATE m | nat 10°/10mm | tss RMSE | tss ARE° | tss ATE m | tss 10°/10mm | shuffle | zero | bypass | gate clip/clean | fusion params | last-epoch? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| u-rgb-natural | natural | 6.78 | 3.70 | 0.0168 | 0.645 | 33.02 | 34.52 | 0.2281 | 0.359 | — | — | — | — / — | — | yes |
| u-rgb-tss | tss | 8.53 | 5.36 | 0.0220 | 0.487 | 13.94 | 13.72 | 0.0618 | 0.382 | — | — | — | — / — | — | no |
