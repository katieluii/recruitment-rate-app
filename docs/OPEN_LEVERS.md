# Untested levers

Four things that have never been tried or measured. Written down because they were
being carried in conversation, which is not a place they survive.

Ordered by expected value, not by effort. Every one gets a ledger row, and anything
that does not improve R2 on the temporal fold gets reverted rather than argued for.

## 1. The enrolment label is partly fabricated by a constant

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
