# Next-session prompt — WSi recruitment/duration predictor

Paste the block below into a fresh session started from
`~/Projects/ws_professional/recruitment_rate_app`.

---

Read `docs/OPEN_LEVERS.md` first — it lists four untested levers with the evidence
already gathered. Work them in order, and treat the numbers in it as measured, not
as claims to re-derive.

Where things stand: v3.1 is live on Railway with an L2 point head. R2 is
0.555 / 0.370 / 0.345 / 0.372 for P1 / P1HV / P2 / P3 against a 0.70 gate that
still fails everywhere. `experiments/` is the harness; nothing ships without a
`ledger.jsonl` row, and the gate lives in `experiments/metrics.py`.

Start with lever 1, the MIN_ENROL_FRACTION clip, because it is the only one with a
measured defect rather than a suspicion: roughly one training row in six has its
enrolment target set by the constant 0.25 rather than by data, and the clip fires
hardest on long-followup trials — the cohort the model already handles worst. Run
the three tests named in the doc as separate ledger rows: drop clipped rows from the
enrolment head, sweep the fraction, then down-weight instead of dropping. Report the
R2 curve before changing any default.

Then lever 2 (`startDateStruct.type` is fetched and discarded), which is a free
feature and the third instance of the same defect pattern that produced the two
most expensive bugs in this project. Lever 3 last; expect it to be small.

Lever 4 is a correctness fix, not accuracy — do it whenever, but check both
frontends still send `num_sites` before forbidding extra fields.

Ground rules that are already established, so do not relitigate them:
- R2 is the gate, not a reported figure. Do not move the bar to make numbers green.
- Per-site enrolment does not exist in CT.gov or AACT. Country recruitment speed is
  not identifiable. Both are settled — see the memory node.
- Per-indication stratified models, phase-purity contamination and AACT-as-a-second-
  source were all tested and rejected. Do not re-propose them.
- Retrain ONE phase per run (`--phase P1`). Fetching all four back to back has
  rate-limited CT.gov twice and left artifacts half-applied.
- Verify a deploy by asserting on `n_train` in the response as well as the
  prediction, and send `num_sites` — the API silently drops unknown fields.
- Git identity for this repo is `dev <dev@localhost>` with no Claude trailers.
