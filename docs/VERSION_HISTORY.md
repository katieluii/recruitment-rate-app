# Version history: trial duration predictor

The public record of what each version of the predictor was. The portfolio page
(kl-portfolio `TrialPredictorVersions.tsx`) quotes only the tables; the narrative lives
here since 2026-08-31 (Katie's call: the page carries results, the repo carries history).

**Naming note (2026-08-31): the version labelled `v3.1` everywhere before this date is
now presented as `v4` on the portfolio page.** Nothing about the model changed with the
rename. Ledger configs, experiment names and code comments still use the v3.x lineage;
this file is the mapping.

## v1: per-phase random forest (archived)
One RandomForest per phase, scored on a single random split. Two defects found in
retrospective validation: no baseline had ever been recorded (scored honestly against a
per-therapeutic-area median lookup, v1 was 2.9× worse), and it used the completion year
of the very trial being predicted, which is target leakage. The iteration table's v1 column shows
the recipe with the leak removed, which flatters it; as deployed it was far worse.

## v2: leak removed; LightGBM quantile + conformal intervals
Data-layer fixes (real site counts, no leaked year), therapeutic-area target encoding,
LightGBM on a log target, conformal prediction intervals replacing v1's decorative
rmse-scaled band.

## v3: two-stage duration
Recruiting window and follow-up predicted separately and summed. Justified by the data:
the components are near-uncorrelated (r = +0.03) and follow-up is where the
therapeutic-area signal lives. Also yields the planning-useful split itself.

## v4 (formerly v3.1): full corpus, retuned, L2 point head. LIVE.
Three changes, no new architecture:
- **Training data**: an API cap of 5,000 records had the model training on 2,024 of
  17,092 eligible Phase 3 trials; removing the cap and retuning added roughly 0.10 R² on
  the like-for-like fold.
- **Point estimate**: the quantile head fits the median while R² rewards the conditional
  mean; an L2 point head added ~0.038 R² on Phase 1 and 0.027 on Phase 3 and reduced
  error on three of four phases.
- **Prediction intervals**: unchanged; they still come from the quantile heads.

Post-v4 accuracy work is tracked in the ledger rather than as version bumps: notably the
horizon fold (2026-08-04), IPCW parity + the served derived rate (2026-08-30), and
`ipcw_scope="total"` (2026-08-30).

## v5 (2026-08-31): forest point inside the calibrated two-stage band. LIVE.
Refit on the mature fold for a single results table, the v1 random forest (leak removed, on
the current feature pipeline) out-scored every LightGBM version on R² for every phase. A
raw-target LightGBM mean head did not close the gap and a single-stage LightGBM (v2) scored
level with v4, so the edge is the learner, not the architecture. v5 therefore takes the
forest's point estimate, puts a split-conformal band around it shaped by the forest's own tree
spread (scaled to 0.80 coverage on the calibration slice), and keeps the two-stage model for the
recruiting / follow-up split (rescaled to sum to the forest total), which the served rate derives
from. Recentring the two-stage band on the forest point was tried first: calibrated, but 33-41%
wider than v4; the forest-shaped band is 15-20% narrower than v4 at the same coverage. The forest
is refit on all training rows after the band is calibrated on the 80% slice; the test fold
confirms the scale holds. `experiments.candidates.HybridForestPoint`,
`trainer.DURATION_MODEL = "hybrid"`, `HYBRID_BAND = "forest"`, config
`hybrid_rf_refit_fband_ipcw_total` (P1HV at the default 0.80 target).

Mature fold, v5 vs v4, MAE (mo) / R² / coverage:

| phase | v5 | v4 |
|---|---|---|
| P1HV | 3.16 / 0.452 / 0.80 (width 7.9 mo) | 3.26 / 0.331 / 0.79 (9.6) |
| P1 | 7.80 / 0.668 / 0.79 (21.8) | 7.72 / 0.648 / 0.79 (27.2) |
| P2 | 9.70 / 0.456 / 0.78 (27.6) | 9.54 / 0.425 / 0.79 (33.2) |
| P3 | 9.90 / 0.467 / 0.79 (28.8) | 10.03 / 0.399 / 0.78 (34.0) |

## Mature-fold ladder (train <2018, test 2018-2020; every version refit on today's corpus)

| model | P1HV MAE / R² | P1 | P2 | P3 |
|---|---|---|---|---|
| baseline, TA-median lookup | 4.47 / -0.172 | 11.96 / 0.214 | 12.20 / 0.068 | 12.96 / -0.009 |
| v1 random forest | 3.16 / 0.452 | 7.80 / 0.668 | 9.70 / 0.456 | 9.90 / 0.467 |
| v2 | 3.27 / 0.343 | 7.65 / 0.649 | 9.51 / 0.429 | 10.13 / 0.396 |
| v3 | 3.27 / 0.324 | 7.76 / 0.630 | 9.51 / 0.403 | 10.14 / 0.367 |
| v4 | 3.26 / 0.331 | 7.72 / 0.648 | 9.54 / 0.425 | 10.03 / 0.399 |
| v5 (live) | 3.16 / 0.452 | 7.80 / 0.668 | 9.70 / 0.456 | 9.90 / 0.467 |

v1 fails the coverage gate on P2 and P3 (its band was the rmse-scaled one); v5 shares its
point estimate and passes everywhere. The standalone v1 row trains on 100% of the fold while
the conformal models train on 80%, which is why v5 refits its forest after calibration.

## Iteration benchmark (fixed fold: train <2021, test 2021+, completed trials only)

| measure | v1 | v2 | v3 | v4 |
|---|---|---|---|---|
| Phase 1 HV, R² | 0.420 | 0.371 | 0.368 | 0.370 |
| Phase 1, R² | 0.454 | 0.511 | 0.517 | 0.555 |
| Phase 2, R² | 0.193 | 0.330 | 0.343 | 0.345 |
| Phase 3, R² | 0.127 | 0.337 | 0.345 | 0.372 |
| Phase 1 HV, MAE (mo) | 2.87 | 2.66 | 2.64 | 2.67 |
| Phase 1, MAE (mo) | 5.96 | 5.35 | 5.25 | 5.00 |
| Phase 2, MAE (mo) | 8.30 | 7.16 | 7.01 | 6.95 |
| Phase 3, MAE (mo) | 8.75 | 7.06 | 6.99 | 6.71 |
| Phase 1 HV, RMSE (d) | 134 | 140 | 140 | 140 |
| Phase 1, RMSE (d) | 250 | 237 | 235 | 226 |
| Phase 2, RMSE (d) | 311 | 284 | 281 | 280 |
| Phase 3, RMSE (d) | 328 | 286 | 284 | 278 |

This fold exists to compare versions. It is optimistic in absolute terms: the registry
holds completed trials only, so a 2021+ test window cannot yet contain a six-to-eight-year
trial and is biased toward short ones. The numbers to quote are the horizon-fold rows in
`experiments/published_metrics.json` (README/RESULTS marker blocks).

## Page prose archived from the portfolio (removed 2026-08-31)

- **Why the iteration benchmark is optimistic**: completed-trials-only registry; a 2021+
  test window cannot include trials that take six to eight years; biased toward
  shorter-duration trials; the mature-outcome holdout (2018–2020 starts) is the honest one.
- **Limitations (constraint to consequence)**: endpoints not accrual histories to recruitment
  reconstructed, not measured; sites treated as open all window to late-opening sites count
  as day-one; uniform accrual assumed to real accrual is S-shaped; follow-up estimated to
  assumptions propagate into the rate; trial-wide average to no single site would observe it.
- **What would close the gap**: more features, other model classes, per-indication models
  and AACT were tested and closed; site-activation and per-patient enrolment dates are
  simply absent from CT.gov/AACT; CRO/CTMS or commercial sources (e.g. Citeline) would
  close it. The remaining limitation is primarily the data, not the model.
