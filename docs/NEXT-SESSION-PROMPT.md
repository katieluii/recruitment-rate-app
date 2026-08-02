# Next-session prompt — WSi recruitment/duration predictor

**Superseded 2026-08-03.** The prompt this file used to hold pointed at four open
levers in `docs/OPEN_LEVERS.md` and told the next session to start with lever 1.
All four are now resolved and none of them raised R2. Pasting the old text would
send a session to redo finished work, so it is gone rather than kept below.

Start a session from `~/Projects/ws_professional/recruitment_rate_app` and paste
the block below.

---

Read `AI_STATE.md` first, then `docs/OPEN_LEVERS.md` for the detail behind it.
Both were written on 2026-08-03 and the numbers in them are measured — treat them
as findings, not as claims to re-derive.

Where things stand: v3.1 is live on Railway with an L2 point head, R2 is
0.555 / 0.370 / 0.345 / 0.372 for P1 / P1HV / P2 / P3 against a 0.70 gate that
still fails everywhere, and all four levers in OPEN_LEVERS.md are closed. Three
were rejected on measurement; the fourth was a correctness fix and shipped to the
repo but not to Railway.

The blocking item is not a model change. It is a decision about the GATE, written
up at the end of OPEN_LEVERS.md §3: the 0.70 R2 bar is computed on the 2021+ fold,
whose target is truncated by observation horizon, and that fold rewards
under-prediction. Horizon-matched training was shown to gain +0.075 R2 there while
losing -0.081 on an honest fold. Until that is settled, any new lever's R2 number
is hard to interpret, so settle it before running more experiments.

This environment needs `brew install libomp` or LightGBM will not import, and only
the P1 cache exists locally — P2 and P3 need a fetch, one phase per run.

Ground rules that are already established, so do not relitigate them:
- R2 is the gate, not a reported figure. Do not move the bar to make numbers green.
  Note this is NOT the same as the §3 question, which is about which FOLD the bar is
  computed on.
- Per-site enrolment does not exist in CT.gov or AACT. Country recruitment speed is
  not identifiable. Both are settled — see the memory node.
- Per-indication stratified models, phase-purity contamination and AACT-as-a-second-
  source were all tested and rejected. Do not re-propose them.
- The four levers are closed. Do not re-propose the MIN_ENROL_FRACTION sweep, the
  `startDateStruct.type` feature, or horizon matching — each has a measured negative
  result and a recorded reason.
- Retrain ONE phase per run (`--phase P1`). Fetching all four back to back has
  rate-limited CT.gov twice and left artifacts half-applied.
- Verify a deploy by asserting on `n_train` in the response as well as the
  prediction, and send `num_sites` — the API now returns 422 for unknown fields
  locally, but the deployed build still drops them silently until lever 4 ships.
- Git identity for this repo is `dev <dev@localhost>` with no Claude trailers.
