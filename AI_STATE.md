# Project Memory State

## Current Context

WSi trial recruitment/duration predictor. v3.1 is live on Railway with an L2 point
head. The four levers in `docs/OPEN_LEVERS.md` are now all resolved and none of them
raised R2; the gate (R2 >= 0.70 in `experiments/metrics.py`) still fails on every
phase — P1 0.555, P1HV 0.370, P2 0.345, P3 0.372 as of the 2026-07-27 retrain.

Working protocol is unchanged: nothing ships without a row in
`experiments/ledger.jsonl`, R2 is the gate rather than a reported figure, and
per-indication stratified models, phase-purity contamination, AACT-as-a-second-source,
per-site enrolment and country recruitment speed are all settled as rejected or
non-identifiable — do not re-propose them.

Environment note for a machine that has not run this repo: there is no committed
venv and the parquet cache is gitignored. Bootstrap is
`/usr/bin/python3 -m venv .venv` (3.9.6), `pip install -r requirements.txt`, and
`brew install libomp` — LightGBM will not import without libomp. Only the P1 cache
(`data/cache/EARLY_PHASE1_PHASE1.parquet`, completed + ongoing) exists locally;
P2 and P3 need a fetch, one phase per run.

All measurements this session are P1 only, temporal split, cutoff 2021-01-01.

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
- Full suite green: 22 passed.

## Known Issues

- The R2 gate is computed on a fold whose target is truncated by observation horizon,
  and that fold rewards under-prediction. It ranked two models in opposite directions
  from an honest fold by 0.16 R2, so any future lever that shortens predictions will
  collect an unearned gain. Needs Katie's decision, written up at the end of
  `docs/OPEN_LEVERS.md` §3. Not changed here.
- Lever 1 leaves a real defect that R2 cannot see: for ~18% of P1 rows the
  enrolment/follow-up split is set by the floor constant, and `predict_components`
  surfaces that split as a planning output. It is unscoreable — the true split is the
  quantity the registry does not publish. Do not re-test it against R2.
- Lever 1 and lever 3 are measured on P1 only. The sum-to-label identity behind
  lever 1's flat sweep is construction and holds for every phase; the drop/weight and
  horizon magnitudes are not confirmed for P1HV/P2/P3.
- `experiments/run.py` records `n_train` as the pre-filter split size, so the `l3_*`
  ledger rows overstate what those models actually trained on. Real kept counts are in
  the run log (5.0 yr keeps 91.1%).
- `SiteMixRequest` in `backend/routes/site_rates.py` still accepts and drops unknown
  fields — the same defect lever 4 fixed on `PredictRequest`. Not changed, as it was
  outside the lever's scope and has no frontend caller.
- Lever 4 changes API behaviour for any external caller sending extra keys. Committed
  but NOT deployed; the live Railway app still silently drops them.

## Exact Next Steps

1. Decide the gate question in `docs/OPEN_LEVERS.md` §3 — either score the gate on a
   horizon-adequate window (train <2018 / test 2018-2020) or keep the 2021+ fold and
   report bias-by-start-year beside R2. This blocks interpreting any future lever.
2. Decide whether to deploy lever 4's 422 to Railway. Verify after deploy by asserting
   on `n_train` in the response as well as the prediction, and send `num_sites`.
3. Optional, cheap: confirm lever 1 and lever 3 on P2 and P3. Needs one cache fetch per
   phase — `python -m experiments.run --phases P2 --config two_stage_l2,l1_drop_clipped,l1_drop_random,l3_horizon_5y`.
   Retrain ONE phase per run; fetching all four back to back has rate-limited CT.gov twice.
4. Decide whether `SiteMixRequest` should also forbid extra fields.

Git identity for this repo is `dev <dev@localhost>` with no Claude trailers.
