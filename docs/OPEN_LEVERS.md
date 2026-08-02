# Untested levers

Four things that had never been tried or measured. Written down because they were
being carried in conversation, which is not a place they survive.

Ordered by expected value, not by effort. Every one gets a ledger row, and anything
that does not improve R2 on the temporal fold gets reverted rather than argued for.

**Status:** 1 closed (no gain, 2026-08-03) · 2, 3, 4 open.

## 1. The enrolment label is partly fabricated by a constant — CLOSED, no gain

**Measured 2026-08-03 on P1 (temporal, cutoff 2021-01-01). Every intervention lost.
The floor stays at 0.25.** Ledger rows `l1_*`; the section below is the original
hypothesis, kept because the reasoning in it was sound and the result was not
predictable from it.

| config | R2 | MAE (mo) |
|---|---|---|
| `two_stage_l2` — the 0.25 floor, kept (control) | **0.5549** | 4.99 |
| `l1_frac_025` — same thing via the new parameter | 0.5549 | 4.99 |
| sweep 0.0 / 0.1 / 0.4 | 0.5504 / 0.5519 / 0.5460 | 4.97 / 4.97 / 5.05 |
| drop clipped rows, enrolment head | 0.4111 | 5.76 |
| drop clipped rows, both heads | 0.3106 | 6.10 |
| down-weight clipped 0.1 / 0.25 / 0.5 | 0.4585 / 0.4977 / 0.5269 | 5.53 / 5.31 / 5.14 |

Three findings, in the order they change the conclusion:

**The floor cannot move the duration label. It only moves the split.** The two
components are `enrol = clip(total - fu)` and `fu = total - enrol`, so they sum to
`total` by construction — verified at max error 1e-14 across fractions 0.0 to 0.9.
The sweep is therefore not a test of label correctness at all: whatever the floor
does, the thing the model is scored on is unchanged, and the only way R2 can move
is through how learnable each half becomes. That is why the whole sweep spans
0.009 R2. No fraction was ever going to win, and this is the reason.

**The clipped rows carry MORE signal than an average row, not less.** Dropping them
cost 0.144 R2. A placebo dropping an equally large RANDOM slice (two seeds) cost
0.016 and 0.013 — about a tenth as much. So the loss is the rows themselves, not
the 18% sample-size cut, and the doc's own decision rule ("if it falls, those rows
were carrying signal") fires. Down-weighting reproduces this monotonically: R2
rises with the weight all the way to 1.0, i.e. the optimum is to leave them alone.

**Why: the clip fires on SHORT trials, not the long-follow-up ones assumed below.**
On P1 the clipped rows have a median total duration of 4.3 months against 12.3 for
the rest, and a median follow-up estimate of 4.0 against 1.1 — the follow-up
estimate swallows the whole span, leaving a raw window of −0.1 months. Removing
them removes the fast end of the distribution, which is a systematic slice, which
is why it costs an order of magnitude more than a random cut of the same size.

The area-level claim below does survive: Oncology (23.2% clipped) and Haematology
(22.0%) are hit two to three times harder than Urology (4.0%) or Immunology (6.6%).
Both facts are true at once — the clipped set is bimodal.

**What is NOT closed.** The component split is still partly fabricated for one row
in six, and `predict_components` surfaces it as a planning output ("months to last
patient in"). That defect is real, it is simply invisible to a duration-R2 gate,
and it cannot be scored — the true split is exactly the quantity the registry does
not publish. Do not re-test it against R2; it will keep coming back flat.

Measured share on the floor, this corpus: P1 18.3%, P1HV 9.1%, 15.2% before the
HV split — which reconciles with the 15.9% recorded below.

### Original hypothesis (2026-08-02)

`cleaner.py:210` derives the recruiting window as `total - followup`, then clips it:

```python
return (total - fu).clip(lower=MIN_ENROL_FRACTION * total)   # MIN_ENROL_FRACTION = 0.25
```

Measured share of rows sitting exactly ON that floor:

| phase | rows on the floor |
|---|---|
| P1 | 15.9% |
| P2 | 18.3% |
| P3 | 16.6% |

For roughly one row in six, the enrolment-stage target is not a measurement. It is
0.25 x duration, a constant chosen without evidence. The two-stage model then fits
its enrolment head against that, and the follow-up head against the remainder.

Worth knowing: 0.25 was never tuned, and the clip fires precisely where the estimated
follow-up is long relative to total duration, which is exactly the oncology-style
trial the model is worst at. The bias is not spread evenly across the corpus.

What to test, cheapest first:
- Drop clipped rows from the enrolment head's training set and refit. If R2 rises,
  the constant was injecting noise; if it falls, those rows were carrying signal.
- Sweep MIN_ENROL_FRACTION across 0.0 / 0.1 / 0.25 / 0.4 and read the R2 curve.
- Weight clipped rows down rather than dropping them.

## 2. `startDateStruct.type` is fetched and discarded

`ct_api_client.py:173` parses `startDateStruct` for its date and drops the `type`
field, which is ACTUAL or ESTIMATED. A trial whose start date is still an estimate
is a different object from one that has actually begun, and the flag is free.

This is the same shape as the two defects that cost the most so far: the
`site_count` train/serve mismatch, and enrolment being ACTUAL in training while a
user supplies an ESTIMATE at design time. Both were a field whose *type* mattered
and was thrown away. This is the third instance of that pattern.

## 3. Observation-horizon matching

The corpus is completed trials only, so every row survived long enough to finish.
IPCW corrects the duration target for this, but the FEATURES are never matched on
observation horizon: a 2015 trial and a 2021 trial have had very different windows
in which to complete, and the model sees no marker of that.

Test: add years-since-start-to-data-vintage as a feature, or restrict the training
fold to trials whose horizon matches the test fold's. Expect a small effect. Listed
because it is the only remaining structural bias that has been named but not measured.

## 4. The API silently ignores unknown fields

Not an accuracy lever, a correctness one. `PredictRequest` accepts extra fields and
drops them, so a request sending `site_count` instead of `num_sites` returns HTTP 200
with a prediction made without any site count at all. That is indistinguishable from
a correct response.

This cost real debugging time: a local-vs-deployed comparison sent the wrong field
name and appeared to show a stale deployment, off by up to 3.7 months across three
phases. The deployment was current.

Fix is one line, `model_config = {"extra": "forbid"}`, but it turns a silent default
into a 422 for any existing caller sending extra keys. Check the frontends first.
Both currently send `num_sites` correctly.
