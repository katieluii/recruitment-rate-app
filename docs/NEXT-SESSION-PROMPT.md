# Next-session prompt — WSi recruitment/duration predictor

Start a session from `~/Projects/ws_professional/recruitment_rate_app` and paste the
block below. Last updated 2026-08-05.

---

Read `AI_STATE.md` first, then `docs/OPEN_LEVERS.md` for the detail behind it. The
numbers in both are measured — treat them as findings, not as claims to re-derive.

Where things stand: the app is live on Railway and verified serving the current
models. All four levers in OPEN_LEVERS.md are closed. Two things changed underneath
everything on 2026-08-04, and both matter more than any individual result:

**The corpus was roughly half its true size until then.** `parse_dates` parsed a
column holding both `2015-10` and `2022-10-21` without specifying a format; pandas
inferred one from the first value and the `dropna` two lines later silently deleted
the rest. Which half died depended on the order the API returned. Fixed, corpora
roughly doubled, all four models retrained. Do NOT compare a pre-2026-08-04 ledger
row against a later one — re-measure instead.

**There is no absolute R2 gate any more.** The 0.70 bar was unreachable from this
feature set and was retired by Katie. R2 and RMSE are optimisation targets — R2 up,
RMSE down — and the bar is each phase's own best recorded value, held by
`experiments/leaderboard.py` and printed on every run. Two absolute gates remain and
both pass: `skill_vs_ta_median > 0` and interval coverage in 0.75-0.90.

Scoring runs on the horizon fold by default (train <2018, test 2018-2020), because
the old 2021+ fold was truncated by observation horizon and rewarded any change that
merely predicted shorter. Current bar: P1 0.6292, P2 0.4025, P3 0.3631, P1HV 0.3187.

**Start here.** On the honest fold the model under-predicts in EVERY test year on
EVERY phase — P1 by 2.8-3.5 months, P2 by 2.5-4.8, P3 by 2.3-5.9. That is systematic
optimism in a planning tool, it was invisible under a single R2 number, and a
calibration or recency weight is far cheaper than finding new signal. Step 1 is to
find out whether the bias is uniform or concentrated in long trials.

Ground rules that are already established, so do not relitigate them:
- Per-site enrolment does not exist in CT.gov or AACT. Country recruitment speed is
  not identifiable. Both settled — see the memory node.
- Per-indication stratified models, phase-purity contamination and AACT-as-a-second-
  source were tested and rejected. Do not re-propose them.
- The four levers are closed, each with a measured negative result and a recorded
  reason. Do not re-propose the MIN_ENROL_FRACTION sweep, the `startDateStruct.type`
  feature, or horizon-matched training.
- Nothing ships without a row in `experiments/ledger.jsonl`, and a result is only
  comparable within one split. Do not mix horizon-fold and temporal-fold numbers.
- Retrain from the local cache — `python -m scripts.train_models --use-cache`. It
  avoids twelve back-to-back CT.gov fetches and fits on exactly the corpus the
  harness measured. If a cache is cold, fetch ONE phase per run; fetching all four
  back to back has rate-limited CT.gov twice.
- This environment needs `brew install libomp` or LightGBM will not import.
- Railway AUTO-DEPLOYS from `main`. A push is a deploy. Verify one by asserting on
  `n_train` in the response as well as the prediction, and send `num_sites` — unknown
  fields now return 422 rather than being dropped.
- Git identity for this repo is `dev <dev@localhost>` with no Claude trailers.

Two habits that earned their place this session, both of which caught real defects
that passing tests and clean logs did not: assert on the artifact rather than the log
(`heads.duration.ipcw_applied` in metadata.json caught two phases trained with no
censoring correction), and when a check disagrees with the code, find out which one
is wrong before believing either (a 0/12 agreement score turned out to be the regex
being wrong, not the model).
