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
held by `experiments/leaderboard.py`. Two absolute gates remain and both pass as of
2026-08-29 (P1HV coverage FAILED 0.729 < 0.75 from 2026-08-04 until then; see below):
`skill_vs_ta_median > 0` (does the model beat a per-therapeutic-area median
lookup table, which decides whether it deserves to exist) and interval coverage
within 0.75-0.90.

**2026-08-29 — published figures moved to the horizon fold; P1HV recalibrated.** The
cc-exchange "circular validation" audit found every PUBLIC surface (README table, RESULTS.md,
`provenance.py`'s hardcoded `"0.82-0.89"`, the portfolio `TrialPredictorVersions.tsx`) still
quoting the 2021+ fold, while the horizon fold (rows 323-330) was worse and P1HV failed its
coverage gate. Now:
- `experiments/publish_metrics.py` is the ONE source: it selects the shipped row per phase from
  the ledger, writes `experiments/published_metrics.json`, and fills the marked blocks in
  README.md and RESULTS.md (`--check` exits 1 if either is stale). `provenance.py` reads the
  JSON at request time — a missing file degrades to "not measured", never to a remembered number.
- P1HV: three recalibration configs run on the horizon fold (rows 332-334). `coverage=0.85`
  clears the gate (0.795, +1.6 mo width, MAE 3.27, R² 0.324); `calib_frac=0.3` alone did not
  (0.746). `trainer.COVERAGE_TARGET = {"P1HV": 0.85}`; artifact retrained on a fresh CT.gov pull
  (n_fit 7467, IPCW applied, band scale 1.25). `inference.py` now labels `confidence_pct` from the
  artifact's `coverage_nominal` (was a constant 80 — an 85% band would have read as 80%).
- ~~Still open: IPCW parity; rate head unpublished~~ → done 2026-08-30, below.

**2026-08-30 (S318) — the published figures now score the SHIPPED model.** Every eval config had
passed no `censoring_frame`, while `trainer.train_phase` reweights the duration head with IPCW —
so every published number measured an unweighted model that was not the one serving. Now:
- `two_stage_l2_ipcw` (P1/P2/P3) and `two_stage_l2_cov85_ipcw` (P1HV, `trainer.COVERAGE_TARGET`)
  build their frame with `trainer.build_censoring_frame` — extracted from `_censoring_frame` so the
  harness calls the SAME function — via `experiments.dataset.load_censoring_frame`. NOT
  `load_clean_censored`: that HV-filters (it feeds the survival models), the trainer does not, so
  P1 and P1HV share one censoring frame. That is the shipped behaviour and the harness mirrors it.
- Parity rows, horizon fold, all gates pass: P1HV 3.26 mo / R² 0.333 / cov 0.79 @ 0.85 nominal
  (row 350); P1 7.71 / 0.645 / 0.78 (336); P2 9.52 / 0.423 / 0.79 (339); P3 10.09 / 0.392 / 0.77
  (340). R² up 0.010–0.028 on the unweighted rows; MAE level. The log proves the weights fired
  (`P2 IPCW weights: min 0.87 max 2.79`), and each row records `ipcw_applied`.
- `publish_metrics` REFUSES a duration row whose `ipcw_applied` is not True (exit 2, row named;
  fixture `tests/fixtures/ledger_parity.jsonl`, 5 tests). `latest()` keys on target — the newer
  rate-head `ta_median` row was able to shadow the duration baseline. Rows carry the band's own
  nominal (P1HV no longer labelled "0.80 nominal"); docs show `achieved (nominal)`.
- Rate head, horizon fold, `lgbm_rate` (= the shipped `ConformalQuantileModel(transform="log")`):
  P1 / P2 / P3 pass (MAE 11.70 / 3.76 / 10.52 patients·site⁻¹·month⁻¹, skill +0.19 / +0.30 / +0.30);
  **P1HV failed coverage 0.744 < 0.75** (skill +0.09) — Katie's call, same evening: recalibrate.
  `lgbm_rate_cov85` covers 0.800 at MAE unchanged (row 352); `trainer.RATE_COVERAGE_TARGET =
  {"P1HV": 0.85}`, `publish_metrics.RATE_SHIPPED` per phase, and a test binds the two to each other
  (with its rejecting case). Rate block lives in `published_metrics_rate` (README, RESULTS) and
  `published_metrics.json["rate"]`. All eight gates pass.
- **Partial retrain is now a thing:** `python -m scripts.train_models --phase P1HV --use-cache
  --heads rate` rewrote only the rate pickles and `metadata.json["heads"]["rate"]`; the duration
  block (n_fit 7467, band_scale 1.2482, IPCW true) and the corpus keys carried forward byte-for-byte
  [verified: metadata diff before/after]. Rate n_fit is 7409 (cache corpus) beside duration's 7467
  (fresh pull) — recorded, not a defect. The API's single `confidence_pct` reads the DURATION head's
  nominal; keep a phase's two heads on the same target or that label lies about the rate band.
- `data/cache/EARLY_PHASE1_PHASE1.ongoing.parquet` fetched 2026-08-29 (4,551 rows) — P1/P1HV had no
  ongoing cache before; `--use-cache` training for those phases would have stopped loudly.
- **The rate head is a DARK SEAT, and the published rate figure now scores what ships.** Verifying
  the P1HV rate recalibration on Railway showed the served rate band unchanged (2.12–40.53 before
  and after) — because since Task 13 (`6a20fd5`) `inference.py` DERIVES the rate from the duration
  head's enrolment window (enrollment / (sites × window), band inverted from the duration band) and
  the rate head reaches a response only as `recruitment_rate_crosscheck`, a point with no band. All
  four artifacts are two-stage, so the `elif` that serves the head is dead. Katie's ruling: measure
  what ships. `DerivedRate` (`experiments/candidates.py`) mirrors the derivation line for line;
  `derived_rate_ipcw` / `derived_rate_cov85_ipcw` on the horizon fold, `--target recruitment_rate`:
  P1HV 21.04 / cov 0.85@0.85, P1 11.73 / 0.84, P2 3.68 / 0.82, P3 10.58 / 0.81 — all gates pass
  (rows P1HV=362, P1=366, P2=367, P3=368). `publish_metrics.RATE_SHIPPED` names these (parity-gated — the served rate is the
  IPCW duration head inverted); the standalone head is `RATE_HEAD_SHIPPED`, published as a labelled
  cross-check. The consistency test now binds RATE_SHIPPED to the DURATION coverage target. The
  P1HV rate-head recalibration stays (it is what the cross-check point is fitted from) but changes
  nothing a caller sees. Zero test rows hit the 0.0-month-window fallback the wrapper cannot mirror.
- (a) The leak, MEASURED and closed (Katie: measure, don't ship): `two_stage_l2_ipcw_vantage` /
  `_cov85_ipcw_vantage` re-censor the frame at 2018-01-01 (`dataset.load_vantage_censoring_frame`,
  via `censoring_backtest.apply_retrospective_censoring`; P2 frame 20,479 → 12,024 rows). R² moves
  ≤0.006 and not in one direction — P1 0.6443 vs 0.6448, P2 0.4244 vs 0.4228, P3 0.3895 vs 0.3915,
  P1HV 0.3274 vs 0.3331 (rows 353-356). Parity stays the published set; the vantage configs are
  measurement-only and the ledger says so.
- Flagged, NOT done: (b) the enrol stage looks its IPCW weight up at enrol-months against a KM
  over TOTAL duration (`quantile_model._ipcw_weights`, shipped behaviour, questionable); (c) P3's
  test fold moved 1709→1706 rows on the same cache file — some date-relative filter in `clean()`,
  unverified.

**Scoring moved to the horizon fold**: train <2018, test 2018-2020, where trials
have had 5.4-8.6 years to finish against a corpus whose p95 duration is 5.9. The old
2021+ fold could not contain a long trial and rewarded any change that merely
predicted shorter. `--split temporal` reproduces pre-2026-08-04 rows; the two folds
are NOT comparable and the leaderboard refuses to mix them.

Current bar to beat, horizon fold, `two_stage_l2_ipcw` (P1HV: `two_stage_l2_cov85_ipcw`) —
the parity rows, 2026-08-30. A candidate must ALSO pass a censoring frame or it is not comparable:

| phase | R2 | RMSE (days) |
|---|---|---|
| P1 | 0.6448 | 349.9 |
| P2 | 0.4228 | 408.1 |
| P3 | 0.3915 | 426.9 |
| P1HV | 0.3331 | 175.3 |

Still settled, do not re-propose: per-indication stratified models, phase-purity
contamination, AACT as a second source, per-site enrolment, country recruitment
speed. Nothing ships without a row in `experiments/ledger.jsonl`.

Environment note for a machine that has not run this repo: there is no committed
venv and the parquet cache is gitignored. Bootstrap is
`/usr/bin/python3 -m venv .venv` (3.9.6), `pip install -r requirements.txt`, and
`brew install libomp` — LightGBM will not import without libomp. All six caches
exist locally (completed: EARLY_PHASE1_PHASE1 / PHASE2 / PHASE3, shared by P1+P1HV;
ongoing for the same three — the P1 one only since 2026-08-29); a cold machine needs
a fetch, one phase per run. Train from them with
`python -m scripts.train_models --use-cache`, which avoids twelve back-to-back CT.gov
fetches and fits the model on exactly the corpus the harness measured.

Railway AUTO-DEPLOYS from `main`. A push is a deploy. HEAD 7bee96b is deployed and
verified live (n_train asserted per phase, dropdown gone, sliders arming, eligibility
parameter answering, guard firing in the sparse tail).

WS21 (`../clinical-trial-analyst`) imports WSi IN-PROCESS from this checkout and pins it
at a commit; `check_pin()` warns and never fails. It is pinned at 883a69d5 and has
delivered both contracts to `analyst/artifacts/` — `endpoint_combinations.json` and
`eligibility_clusters.json`. WSi does not read them yet.

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
- THE CEILING. Three independent probes now agree: the endpoint classifier fix (893
  trials relabelled) moved R2 +0.0004; a full permissive-to-restrictive eligibility swap
  moves 0.6 months against a 14.7-month observed cluster spread; WS21's event-driven flag
  shows 11.2 months observed within-cell and +0.16 MAE as a feature. Large real
  differences, negligible model gains, each redundant with something the model already
  holds. The duration model is near the ceiling of what design-time registry fields can
  tell it, which argues the under-prediction above is a CALIBRATION problem rather than a
  missing-feature one. Test that before extracting more fields.
- WS21's event-driven result is NOT gate-eligible as measured: their baselines are P1
  6.22 / P2 8.48 / P3 9.44 MAE against WSi's horizon-fold 7.78 / 9.46 / 10.22. Different
  fold or corpus. It must be re-run in WSi's harness, and measured against BIAS as well as
  MAE — a feature that fixes systematic under-prediction can leave MAE flat.
- The sites slider is near-inert on duration: 0.3 months across its whole trained range
  against enrolment's 11.5. It does move the recruitment rate (0.497 to 0.226 pt/site/mo
  between 57 and 200 sites). Keep / relabel / drop / show the rate instead — undecided.
- `co_primary` is not derivable from the registry, so SCHEMA_TO_MODEL_MAP's instruction to
  filter endpoints on primary + co_primary is partly unimplementable; use primary only.
  WS21 also caches no secondary outcome text, only a count, so key secondaries need a
  refetch decision.
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

0. (Done 2026-08-30, S318: IPCW parity; rate figure = the SERVED derived rate, head as cross-check;
   P1HV rate head recalibrated to 0.85; vantage leak measured at ≤0.006 R² and closed. Nothing open.)

1. Chase the under-prediction bias — the largest open problem, and the ceiling finding
   above suggests it is calibration rather than missing signal. Start with
   `python -m experiments.horizon_bias --phase P1` to see whether it is uniform or
   concentrated in long trials, then test a calibration or recency weight.
2. Re-measure levers 1 and 3 on the horizon fold and recovered corpus before either is
   quoted again:
   `python -m experiments.run --phases P1 --config two_stage_l2,l1_drop_clipped,l1_drop_random,l3_horizon_5y`
3. Run the training-window / recency-weight experiment. Recovered rows are all OLD and all
   land in TRAIN; more of them helped P1 (+0.017) and hurt P1HV (-0.031) and P2 (-0.020).
4. When WS21 hands over the event-driven regex, add it to WSi's own preprocessing and run
   it through this harness on the horizon fold, reporting bias by start year alongside MAE
   and R2. Only retrain if it beats the recorded best.
5. Wire the read side for WS21's two contract files, then build the endpoint and
   eligibility panels under the agreed presentation rules: ALWAYS re-run the prediction on
   selection, show the eligibility contribution explicitly, and put WS21's observed cluster
   median beside WSi's prediction labelled observed vs predicted.
6. Decide whether `SiteMixRequest` should also forbid extra fields.

Awaiting Katie, not actionable here: the WS21 deterministic-vs-LLM extraction project
shape; the sites slider question; who builds the endpoint/eligibility panels.

Git identity for this repo is `dev <dev@localhost>` with no Claude trailers.
