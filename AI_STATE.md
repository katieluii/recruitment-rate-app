# Project Memory State

## Current Context

WSi trial recruitment/duration predictor, live on Railway and verified serving the
current models on 2026-08-05 (n_train asserted per phase, not inferred from a
successful response). The four levers in `docs/OPEN_LEVERS.md` are all resolved and
none raised R2.

**The corpus was roughly half its true size until 2026-08-04.** `parse_dates` called
`pd.to_datetime` without a format against a column holding BOTH `2015-10` and
`2022-10-21`; pandas inferred one format from the first value and the `dropna` two
lines later deleted everything that did not match. Fixed with `format="ISO8601"`.
Every number recorded before that date was computed on roughly half the data, chosen
by an era-correlated criterion nobody selected. Do not compare a pre-2026-08-04
ledger row with a later one; re-measure instead.

**There is no absolute R2 gate any more (Katie, 2026-08-04).** The 0.70 bar was
unreachable from this feature set and was retired. R2 and RMSE are OPTIMISATION
TARGETS - R2 up, RMSE down - and the bar is each phase's own best recorded value,
held by `experiments/leaderboard.py`. Two absolute gates remain and both currently
pass: `skill_vs_ta_median > 0` (does the model beat a per-therapeutic-area median
lookup table, which decides whether it deserves to exist) and interval coverage
within 0.75-0.90.

**Scoring moved to the horizon fold**: train <2018, test 2018-2020, where trials
have had 5.4-8.6 years to finish against a corpus whose p95 duration is 5.9. The old
2021+ fold could not contain a long trial and rewarded any change that merely
predicted shorter. `--split temporal` reproduces pre-2026-08-04 rows; the two folds
are NOT comparable and the leaderboard refuses to mix them.

Current bar to beat, horizon fold, `two_stage_l2`:

| phase | R2 | RMSE (days) |
|---|---|---|
| P1 | 0.6292 | 357.1 |
| P2 | 0.4025 | 413.5 |
| P3 | 0.3631 | 438.7 |
| P1HV | 0.3187 | 177.1 |

Still settled, do not re-propose: per-indication stratified models, phase-purity
contamination, AACT as a second source, per-site enrolment, country recruitment
speed. Nothing ships without a row in `experiments/ledger.jsonl`.

Environment note for a machine that has not run this repo: there is no committed
venv and the parquet cache is gitignored. Bootstrap is
`/usr/bin/python3 -m venv .venv` (3.9.6), `pip install -r requirements.txt`, and
`brew install libomp` — LightGBM will not import without libomp. All seven caches
exist locally (completed for four phases, ongoing for P1/P2/P3); a cold machine needs
a fetch, one phase per run. Train from them with
`python -m scripts.train_models --use-cache`, which avoids twelve back-to-back CT.gov
fetches and fits the model on exactly the corpus the harness measured.

Railway AUTO-DEPLOYS from `main`. A push is a deploy.

## Completed

- `7770104` CT.gov fetch fails closed. A 429 mid-pagination used to break out of the
  loop and return a partial corpus that `dataset.load_raw` cached as complete;
  observed live returning 18,793 of 24,989 P1 studies. Now retries with backoff,
  raises rather than returning short, and checks the response's own `totalCount`.
  Also captures `startDateStruct.type` as `start_date_type`.
- `df5546e` Lever 1 (MIN_ENROL_FRACTION floor) closed, no gain, defaults unchanged.
  Sweeping 0.0/0.1/0.25/0.4 spans 0.009 R2; dropping the clipped rows costs 0.144 R2
  against a random-slice placebo's 0.016/0.013; down-weighting is monotone in the
  weight, so the optimum is 1.0. Cause: the components sum to the label by
  construction (verified to 1e-14), so the floor moves the enrolment/follow-up split
  and cannot move what is scored. Apparatus kept: `clip_policy`, `min_enrol_fraction`,
  `clip_weight`, `clip_seed` on `TwoStageDuration`, all defaulting to old behaviour.
- `780942d` Lever 4. `PredictRequest` sets `extra: forbid`, so a request sending
  `site_count` instead of `num_sites` now returns 422 instead of a 200 computed
  without any site count. Frontend checked first — `frontend/js/app.js` is the only
  POST client and sends exactly the five valid fields. Verified on both paths:
  `/api/predict` returns 200 with `n_train` 7713 for a valid payload and 422 for the
  typo. `tests/test_predict_request_strict.py` asserts the rejection, and the control
  was mutation-checked against a copy of the model without the config.
- `ef3924d` Lever 3. The observation-horizon bias is real and larger than the doc
  expected: the longest trial in each test start-year sits within ~0.2 years of that
  year's available horizon, and model bias runs -1.27 months at the longest horizon
  to +2.03 at the shortest. Both proposed fixes rejected — the horizon feature is
  `start_year` in disguise (excluded by name in `pipeline.py`) and is 0 for any trial
  quoted at design time; horizon-matched training gains +0.0753 R2 on the truncated
  fold and loses -0.0806 on an untruncated one, worsening bias from -2.14 to -4.50
  months. `experiments/horizon_bias.py` and `experiments/horizon_disproof.py`
  regenerate both tables.
- Lever 2 resolved inside `780942d`/`7770104`: `start_date_type` is captured but must
  not become a feature. A completed trial has started, so the corpus holds no
  ESTIMATED start dates — the field is ACTUAL/UNKNOWN, constant across the entire
  2021+ test fold (3011/3011) and 0.66% non-constant in training.
- `b85e89a` Endpoint rules fixed. 893 of 1,006 EVENT_COMPOSITE trials were safety
  trials (100% on P1/P1HV): the composite clause `incidence of .{0,40}events?`
  matched "Incidence of treatment-emergent adverse events" and is rule 2 of 10 while
  SAFETY is rule 7. Found by blind agreement-checking the LLM classifier against
  trials the regex had ALREADY labelled, before trusting it on abstentions —
  EVENT_COMPOSITE came back 0/12 and the model was right. Fixed with a negative
  lookahead; genuine composites still classify. Agreement 75.8% -> 80.8%.
- `b85e89a` Endpoint-profile layer added: most common primary-endpoint COMBINATIONS
  per phase x indication, precomputed to `endpoint_profiles.json` beside
  `analytics.json`. Trials list 4.15 primary measures collapsing to 1.17 archetypes;
  21.8% carry more than one, and the combination matters beyond its parts (P1
  oncology RESPONSE+SAFETY 44.9 months against SAFETY 33.7 and PK_PD 8.0).
- `a96ccc3` P1 retrained against the fixed rules; `lifelines` pinned in
  requirements.txt, which trainer needs for IPCW and which serving never imports.
  Isolated accuracy effect of the endpoint fix: R2 0.5549 -> 0.5545, i.e. none. The
  mislabel was CONSISTENT, so the trees used EVENT_COMPOSITE as a proxy for "safety
  trial" — wrong name, same information. The fix earns its place in what a human
  reads, not in what the model scores.
- `d599512` Date-parsing fix, above. Because every recovered row is an older trial
  they all land in TRAIN and the 2021+ test folds are unchanged, making this a clean
  more-data experiment: P1 5,757 -> 12,418 train, R2 0.5545 -> 0.5714; P1HV 2,041 ->
  5,355, 0.3791 -> 0.3483; P2 5,540 -> 13,940, 0.3460 -> 0.3261; P3 5,356 -> 13,395,
  0.3722 -> 0.3729. More data helped one phase and hurt two, which falsifies the
  repo's standing claim that more training data is the most reliable lever here.
- `1d625b0` All four phases retrained on the recovered corpus: P1 5,757 -> 15,429
  train rows, P1HV 2,041 -> 7,416, P2 5,540 -> 16,126, P3 5,356 -> 14,959. Served
  predictions moved down almost everywhere (P3 Cardiovascular -9.9 months, P2
  Oncology -6.6, most others -2 to 0, P3 Oncology +2.8), which was checked against
  the corpus rather than assumed: the recovered mass is 2000-2010 trials running a
  median 18.0 (P2) and 19.1 (P3) months against 19.9 and 21.8 for 2015-20.
- The first attempt at that retrain silently trained P2 and P3 with NO censoring
  correction — training from cache with no ongoing cohort made the loader return an
  empty frame and `_censoring_frame` caught it into one buried warning. Both were
  refetched and retrained, all four verified via `heads.duration.ipcw_applied` in
  metadata.json, and the loader now raises with the fix command instead of
  downgrading the model in silence.
- `1622bad` The 0.70 gate retired and scoring moved to the horizon fold, per the
  Current Context above. Bias by start year now prints on every run with a
  corr(year, bias) figure. Two bugs in that instrumentation were found by reading its
  own first output: the leaderboard compared every row against itself because the run
  wrote to the ledger before reading the bar back, and the bias note printed
  "strongly negative means..." under a +0.970 correlation. Both fixed and covered by
  rejection tests.
- Deployed and verified live 2026-08-05: all four phases serving the retrained models
  with n_train asserted per phase (7,416 / 15,429 / 16,126 / 14,959), predictions
  matching local exactly, and the unknown-field 422 still holding.
- Full suite green: 53 passed.

## Known Issues

- The model UNDER-PREDICTS in every test year on every phase, on the honest fold: P1
  by 2.8-3.5 months, P2 by 2.5-4.8, P3 by 2.3-5.9, P1HV by 1.3-3.0. This is a
  systematic optimism bias in a planning tool and it was invisible under a single R2
  number. It is the strongest open lead and it is cheaper to chase than new signal.
- EVERY measurement recorded before 2026-08-04 was computed on roughly half the
  corpus, and every measurement before 2026-08-04's gate change used the truncated
  2021+ fold. The structural findings survive because they are constructions rather
  than measurements (lever 1's components summing to the label; lever 3's horizon
  feature being `start_year` in disguise). The NUMBERS do not — lever 1's R2 curve and
  lever 3's +0.075/-0.081 disproof both need re-measuring on the horizon fold before
  being quoted again.
- The endpoint classifier abstains on 21.0% of P2 and 33.5% of P3, so UNCLASSIFIED is
  the largest single profile for both (4,222 and 4,343 trials). Blind agreement
  between the LLM classifier and the regex is 80.8% overall but only 50% for
  EVENT_COMPOSITE and EVENT_RATE and 58% for BIOMARKER; those three should not be
  trusted from an LLM label without the same case-by-case look that found the
  EVENT_COMPOSITE bug. `scripts/classify_endpoints_llm.py --mode resolve` is written
  and validated but has NOT been run on the abstentions.
- A strict top-3 endpoint profile hides the most decision-relevant oncology case: on
  the full corpus RESPONSE+SAFETY is 4th (n=330, 44.7 months) behind UNCLASSIFIED
  (n=344, 26.0), while running ~12 months longer than SAFETY alone. Consider ranking
  by decision value or excluding UNCLASSIFIED from the three.
- Lever 1's real defect stands and no R2 can see it: for ~18% of rows the
  enrolment/follow-up split is set by the floor constant, and `predict_components`
  surfaces it as a planning output. Unscoreable — the true split is exactly what the
  registry does not publish. Do not re-test it against R2.
- The API narrows the endpoint representation: `inference.py` sets exactly ONE
  `endpoint_has_*` flag while the model trains on a multi-hot set, so a
  RESPONSE+SAFETY trial cannot be expressed. Specced in `SPECS.md`, not built.
- `experiments/run.py` records `n_train` as the pre-filter split size, so the `l3_*`
  ledger rows overstate what those models trained on.
- `SiteMixRequest` still accepts and drops unknown fields — the defect lever 4 fixed
  on `PredictRequest`. No frontend caller.

## Exact Next Steps

1. Chase the under-prediction bias above. Start by checking whether it is uniform or
   concentrated in long trials — `python -m experiments.horizon_bias --phase P1`
   reports bias per start year, and the per-therapeutic-area table in the run report
   splits it by area. A calibration or a recency weight are the obvious candidates,
   and either is cheaper than finding new signal.
2. Re-measure levers 1 and 3 on the horizon fold and the recovered corpus before
   either result is quoted again:
   `python -m experiments.run --phases P1 --config two_stage_l2,l1_drop_clipped,l1_drop_random,l3_horizon_5y`
3. Run the training-window / recency-weight experiment. The recovered rows are all
   OLD and they all land in TRAIN; more of them helped P1 and hurt P1HV and P2, which
   is era drift rather than a data-volume effect.
4. Build the outstanding UI work, specced in `SPECS.md`: the endpoint-profile route,
   multi-archetype predict, and live enrollment/sites sliders. Not started.
5. Decide whether `SiteMixRequest` should also forbid extra fields.

Git identity for this repo is `dev <dev@localhost>` with no Claude trailers.
