# Version history — trial duration predictor

The public record of what each version of the predictor was. The portfolio page
(kl-portfolio `TrialPredictorVersions.tsx`) quotes only the tables; the narrative lives
here since 2026-08-31 (Katie's call — the page carries results, the repo carries history).

**Naming note (2026-08-31): the version labelled `v3.1` everywhere before this date is
now presented as `v4` on the portfolio page.** Nothing about the model changed with the
rename. Ledger configs, experiment names and code comments still use the v3.x lineage;
this file is the mapping.

## v1 — per-phase random forest (archived)
One RandomForest per phase, scored on a single random split. Two defects found in
retrospective validation: no baseline had ever been recorded (scored honestly against a
per-therapeutic-area median lookup, v1 was 2.9× worse), and it used the completion year
of the very trial being predicted — target leakage. The iteration table's v1 column shows
the recipe with the leak removed, which flatters it; as deployed it was far worse.

## v2 — leak removed; LightGBM quantile + conformal intervals
Data-layer fixes (real site counts, no leaked year), therapeutic-area target encoding,
LightGBM on a log target, conformal prediction intervals replacing v1's decorative
rmse-scaled band.

## v3 — two-stage duration
Recruiting window and follow-up predicted separately and summed. Justified by the data:
the components are near-uncorrelated (r = +0.03) and follow-up is where the
therapeutic-area signal lives. Also yields the planning-useful split itself.

## v4 (formerly v3.1) — full corpus, retuned, L2 point head. LIVE.
Three changes, no new architecture:
- **Training data** — an API cap of 5,000 records had the model training on 2,024 of
  17,092 eligible Phase 3 trials; removing the cap and retuning added roughly 0.10 R² on
  the like-for-like fold.
- **Point estimate** — the quantile head fits the median while R² rewards the conditional
  mean; an L2 point head added ~0.038 R² on Phase 1 and 0.027 on Phase 3 and reduced
  error on three of four phases.
- **Prediction intervals** — unchanged; they still come from the quantile heads.

Post-v4 accuracy work is tracked in the ledger rather than as version bumps — notably the
horizon fold (2026-08-04), IPCW parity + the served derived rate (2026-08-30), and
`ipcw_scope="total"` (2026-08-30).

## Iteration benchmark (fixed fold: train <2021, test 2021+, completed trials only)

| measure | v1 | v2 | v3 | v4 |
|---|---|---|---|---|
| Phase 1 HV — R² | 0.420 | 0.371 | 0.368 | 0.370 |
| Phase 1 — R² | 0.454 | 0.511 | 0.517 | 0.555 |
| Phase 2 — R² | 0.193 | 0.330 | 0.343 | 0.345 |
| Phase 3 — R² | 0.127 | 0.337 | 0.345 | 0.372 |
| Phase 1 HV — MAE (mo) | 2.87 | 2.66 | 2.64 | 2.67 |
| Phase 1 — MAE (mo) | 5.96 | 5.35 | 5.25 | 5.00 |
| Phase 2 — MAE (mo) | 8.30 | 7.16 | 7.01 | 6.95 |
| Phase 3 — MAE (mo) | 8.75 | 7.06 | 6.99 | 6.71 |
| Phase 1 HV — RMSE (d) | 134 | 140 | 140 | 140 |
| Phase 1 — RMSE (d) | 250 | 237 | 235 | 226 |
| Phase 2 — RMSE (d) | 311 | 284 | 281 | 280 |
| Phase 3 — RMSE (d) | 328 | 286 | 284 | 278 |

This fold exists to compare versions. It is optimistic in absolute terms: the registry
holds completed trials only, so a 2021+ test window cannot yet contain a six-to-eight-year
trial and is biased toward short ones. The numbers to quote are the horizon-fold rows in
`experiments/published_metrics.json` (README/RESULTS marker blocks).

## Page prose archived from the portfolio (removed 2026-08-31)

- **Why the iteration benchmark is optimistic**: completed-trials-only registry; a 2021+
  test window cannot include trials that take six to eight years; biased toward
  shorter-duration trials; the mature-outcome holdout (2018–2020 starts) is the honest one.
- **Limitations (constraint → consequence)**: endpoints not accrual histories → recruitment
  reconstructed, not measured; sites treated as open all window → late-opening sites count
  as day-one; uniform accrual assumed → real accrual is S-shaped; follow-up estimated →
  assumptions propagate into the rate; trial-wide average → no single site would observe it.
- **What would close the gap**: more features, other model classes, per-indication models
  and AACT were tested and closed; site-activation and per-patient enrolment dates are
  simply absent from CT.gov/AACT; CRO/CTMS or commercial sources (e.g. Citeline) would
  close it. The remaining limitation is primarily the data, not the model.
