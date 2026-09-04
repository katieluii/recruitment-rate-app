# Direct recruitment-rate validation

The released recruitment-rate head estimates patients per centre per month from
ClinicalTrials.gov record histories. Its target is:

`final ACTUAL enrolment / (initiated centres × recorded recruiting months)`

This is quality Tier B. It improves on the retired duration-derived proxy because
the denominator ends when the study leaves recruiting rather than at primary
completion. It still assumes listed/initiated centres were active for the whole
interval. Therefore the result is an estimated planning rate, not an observation
of each centre's performance.

## Temporal holdout

Models train on trials starting before 2021-01-01 and are tested on trials
starting from 2021-01-01 through 2022-12-31. The baseline is the training-fold
therapeutic-area median.

| Phase | usable targets | train | test | MAE | median AE | log MAE | within 2× | median factor error | interval coverage | skill vs baseline |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P1 | 1,107 | 602 | 285 | 1.7885 | 0.4892 | 0.5275 | 0.723 | 1.541× | 0.818 | +0.285 |
| P2 | 1,280 | 727 | 354 | 3.1387 | 0.1620 | 0.5726 | 0.703 | 1.547× | 0.884 | +0.251 |
| P3 | 1,248 | 757 | 299 | 3.7444 | 0.2826 | 0.5334 | 0.706 | 1.590× | 0.829 | +0.462 |

All three heads clear the predeclared gates: positive skill against the
therapeutic-area median and 0.75–0.90 coverage for the nominal 80% interval.
P1HV has no released rate head.

## Quality reference

Tier A integrates dated facility-level recruiting-status snapshots and was
available for 23 of 30 records in the detailed feasibility sample. Paired Tier
A/B cases confirm that Tier B is an assumption-bearing proxy rather than an
equivalent site-month measurement. The UI and API surface that limitation with
every rate.

Reproduce the acquisition and scoring with:

```bash
python -m experiments.recruitment_history_pilot --mode summary --min-start-year 2017
python -m experiments.eval_history_rate
```

The acquisition is cached and resumable. The model artifacts record the target
definition, quality tier, training count, validation metrics and validation
window in each phase's `metadata.json`.
