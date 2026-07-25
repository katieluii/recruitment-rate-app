# Experiment harness

## The rule

No change to `backend/` ships without a ledger row showing it helped. A change that
does not improve MAE or TA differentiation on the temporal holdout gets reverted,
not rationalised.

## Running

```bash
python -m experiments.run --config all --split temporal --cutoff 2021-01-01
python -m experiments.run --config v1_recipe --phases P2,P3
python -m experiments.run --config all --split random          # v1's own protocol
```

Results append to `ledger.jsonl` (append-only, one record per run — never
read-modify-write it). Readable reports land in `reports/`.

## Why a temporal split

The model's job is to predict a trial that has not run yet. v1 was scored with
`train_test_split(random_state=42)`, which let it see 2023 trials while predicting
2019 ones. The gap between the two protocols is not academic — see below.

## Metrics

| metric | what it answers |
|---|---|
| `mae_months` | typical error, in the unit the user reads |
| `skill_vs_ta_median` | fraction of the baseline's error removed. **Negative = worse than a lookup table.** |
| `ta_spread_ratio` | predicted between-TA spread ÷ true spread. 0 = every area gets the same answer. >1 = over-dispersed. |
| `ta_rank_corr` | Spearman on TA medians — does it get the *order* right (Oncology slow, Dermatology fast)? |
| `ta_n_distinct` | distinct predicted TA medians. v1 shipped with 5 of 22 on P1. |
| `interval_coverage` | fraction of actuals inside the 80% interval. Should be ~0.80. |

## Baselines

`ta_median` is **the bar** — per-therapeutic-area median duration from the training
fold. Anything that loses to it is a lookup table with extra steps.

## Phase 1 result (2026-07-25, temporal cutoff 2021-01-01)

| config | P1HV | P1 | P2 | P3 |
|---|---|---|---|---|
| `ta_median` MAE (the bar) | 3.32 | 4.42 | 8.86 | 9.88 |
| `v1_recipe` MAE | 2.66 | 5.75 | **25.41** | **26.89** |
| skill vs bar | +0.20 | −0.30 | **−1.87** | **−1.72** |

The shipped recipe, evaluated honestly, is ~2.9× worse than a median lookup table
on Phase 2 and Phase 3. `v1_shipped` (the deployed pickle) scores better here only
because its training set overlaps the test period — those numbers are not comparable
and are recorded for the record, not for comparison.

**Cause, isolated:** `primary_completion_year`. It is the label's own endpoint. In the
pre-2021 training fold, high completion year correlates with long duration; every test
trial has a high completion year, so the model predicts uniformly long — bias +25.3
months on P2. Dropping that single feature:

| | with `primary_completion_year` | without |
|---|---|---|
| P2 MAE | 25.41 mo (bias +25.30) | **8.88 mo** (bias +4.05) |
| P3 MAE | 26.89 mo (bias +26.83) | **9.59 mo** (bias +5.81) |

Note what this does *not* fix: at 8.88 vs the 8.86 bar, the model still adds nothing
over the lookup table. Removing the leak stops it being actively harmful; earning its
existence is Phases 2–4.
