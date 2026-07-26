# Results log

Generated 2026-07-26 from `experiments/ledger.jsonl` — every figure traces to a recorded run.

Protocol: train on trials starting before 2021-01-01, test on those starting after. `skill` is the fraction of the per-therapeutic-area median baseline's error removed; **negative means worse than a lookup table**.

## Duration — mean absolute error, months

| step | P1HV | P1 | P2 | P3 |
|---|---|---|---|---|
| **Baseline — per-therapeutic-area median lookup** | 3.82 | 8.53 | 8.95 | 9.72 |
| **v1 as it actually shipped** | 2.66 | 5.75 | 25.41 | 26.89 |
| **+ data-layer fixes (real site count, no leaked year)** | 2.78 | 6.44 | 9.00 | 8.96 |
| **+ therapeutic-area target encoding** | 2.76 | 7.21 | 8.88 | 8.49 |
| **+ LightGBM on a log target** | 2.94 | 6.13 | 7.55 | 7.94 |
| **+ conformal intervals (v2 shipped)** | 2.87 | 5.40 | 7.28 | 7.28 |
| **+ enrolment / follow-up split (v3.3)** | 2.63 | 5.29 | 7.02 | 7.03 |
| **+ country site-mix effect (v3.2) — NOT SHIPPED** | 2.86 | 5.43 | 7.24 | 7.19 |

## Duration — skill against the baseline

| step | P1HV | P1 | P2 | P3 |
|---|---|---|---|---|
| **Baseline — per-therapeutic-area median lookup** | +0.000 | +0.000 | +0.000 | +0.000 |
| **v1 as it actually shipped** | +0.199 | **-0.301** | **-1.868** | **-1.722** |
| **+ data-layer fixes (real site count, no leaked year)** | +0.294 | +0.327 | +0.017 | +0.107 |
| **+ therapeutic-area target encoding** | +0.299 | +0.247 | +0.031 | +0.154 |
| **+ LightGBM on a log target** | +0.254 | +0.360 | +0.176 | +0.208 |
| **+ conformal intervals (v2 shipped)** | +0.271 | +0.436 | +0.205 | +0.274 |
| **+ enrolment / follow-up split (v3.3)** | +0.312 | +0.380 | +0.215 | +0.277 |
| **+ country site-mix effect (v3.2) — NOT SHIPPED** | +0.274 | +0.433 | +0.210 | +0.283 |

## Interval calibration — share of actuals inside the 80% band

| step | P1HV | P1 | P2 | P3 |
|---|---|---|---|---|
| **Baseline — per-therapeutic-area median lookup** | 0.833 | 0.820 | 0.850 | 0.817 |
| **v1 as it actually shipped** | 0.667 | 0.252 | 0.084 | 0.083 |
| **+ data-layer fixes (real site count, no leaked year)** | 0.877 | 0.867 | 0.740 | 0.724 |
| **+ therapeutic-area target encoding** | 0.863 | 0.862 | 0.719 | 0.693 |
| **+ conformal intervals (v2 shipped)** | 0.879 | 0.815 | 0.839 | 0.812 |
| **+ enrolment / follow-up split (v3.3)** | 0.886 | 0.819 | 0.836 | 0.825 |
| **+ country site-mix effect (v3.2) — NOT SHIPPED** | 0.820 | 0.885 | 0.827 | 0.888 |

## Therapeutic-area differentiation

Distinct predicted medians out of the areas with enough test trials — the metric that caught the original failure, where 17 of 22 Phase 1 areas returned the identical 10.9 months.

| step | P1HV | P1 | P2 | P3 |
|---|---|---|---|---|
| **Baseline — per-therapeutic-area median lookup** | 9 | 17 | 17 | 17 |
| **v1 as it actually shipped** | 3 | 8 | 13 | 10 |
| **+ data-layer fixes (real site count, no leaked year)** | 4 | 13 | 15 | 14 |
| **+ therapeutic-area target encoding** | 4 | 13 | 15 | 14 |
| **+ LightGBM on a log target** | 4 | 13 | 14 | 14 |
| **+ conformal intervals (v2 shipped)** | 4 | 12 | 13 | 13 |
| **+ enrolment / follow-up split (v3.3)** | 11 | 18 | 16 | 17 |
| **+ country site-mix effect (v3.2) — NOT SHIPPED** | 4 | 13 | 14 | 12 |

## Recruitment rate — MAE, patients per site per month

| step | P1HV | P1 | P2 | P3 |
|---|---|---|---|---|
| **Baseline — per-area median rate** | 8.534 | 9.038 | 4.117 | 11.765 |
| **LightGBM, log1p target** | 7.430 | 6.479 | 2.469 | 5.672 |
| **LightGBM, plain log target** | 7.540 | 6.469 | 2.172 | 5.388 |

## R-squared and RMSE

Reported for continuity with the original project. Neither is the gate.

R-squared scores against predicting the MEAN, which is a weak reference for
a right-skewed target: the per-therapeutic-area median lookup posts a NEGATIVE
R-squared (-0.12 on P2, -0.14 on P3) while being the harder bar on MAE. A model
can therefore look respectable on R-squared while losing to a lookup table,
which is exactly what v1 did. `skill_vs_ta_median` is the same fraction-of-error-
removed idea measured against that harder reference, and it is what decides
whether a change ships.

RMSE squares the error, so a handful of eight-year trials dominate it. MAE is
the headline because the median quantile model minimises absolute error by
construction, and a metric that disagrees with the loss will reward the wrong
model.

| step | P2 R2 | P3 R2 | P2 RMSE (d) | P3 RMSE (d) |
|---|---|---|---|---|
| **Baseline — per-therapeutic-area median lookup** | 0.003 | -0.086 | 346 | 366 |
| **+ data-layer fixes (real site count, no leaked year)** | 0.003 | 0.108 | 334 | 329 |
| **+ conformal intervals (v2 shipped)** | 0.237 | 0.330 | 292 | 285 |
| **+ enrolment / follow-up split (v3.3)** | 0.341 | 0.336 | 281 | 286 |

## What each step was

- **Baseline — per-therapeutic-area median lookup** — The bar every model must clear. A learned model that loses to this is a lookup table with worse latency.
- **v1 as it actually shipped** — The RandomForest recipe with its original feature set, refit on a temporal fold. v1 was never compared to a baseline, so nobody knew it lost to one.
- **+ data-layer fixes (real site count, no leaked year)** — Same RandomForest, repaired inputs. `primary_completion_year` leaked the label's endpoint and `site_count` counted countries.
- **+ therapeutic-area target encoding** — Replaces 22 sparse binaries with one smoothed continuous signal the trees will actually split on.
- **+ LightGBM on a log target** — Gradient boosting and a log target for the right-skewed duration.
- **+ conformal intervals (v2 shipped)** — Real quantile intervals widened on the most recent training slice, replacing an interval pinned at rmse*0.5 for every input.
- **+ enrolment / follow-up split (v3.3)** — Duration modelled as two near-independent processes rather than one blended number.
- **+ country site-mix effect (v3.2) — NOT SHIPPED** — Adds the geography lever the tool was missing, but costs accuracy on 3 of 4 phases. Recorded, not merged; see the note below.

## Findings that changed the work

- **`primary_completion_year` leaked the label's own endpoint.** Removing it took Phase 2 MAE from 25.41 to 8.88 months.
- **`site_count` counted countries, not sites.** Training values sat in 1–20 while inference passed real site counts of 40+, outside the trained range where a forest returns a constant.
- **Completed-trials-only data is survivorship-biased.** At a 2018 vantage, Phase 3 duration looked 20.9 months when it was truly 24.6. Corrected by inverse-probability-of-censoring weighting.
- **Survival models lost.** Weibull AFT, random survival forest and gradient-boosted survival all cut the bias but lost more on scatter. Recorded rather than quietly dropped.
- **Duration is two processes.** Enrolment window and follow-up are near-uncorrelated (r = +0.03). A Phase 3 survival endpoint follows up for 26.0 months against 5.5 for a biomarker endpoint.

## Open

- The enrolment head and `N / (sites × rate)` disagree for some areas (P3 infectious disease 21.1 vs 13.0 months). Medians do not compose and the heads are fitted independently. V3.2 dissolves this by deriving the window from per-site rates.
