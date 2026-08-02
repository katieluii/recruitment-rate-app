# Untested levers

Four things that had never been tried or measured. Written down because they were
being carried in conversation, which is not a place they survive.

Ordered by expected value, not by effort. Every one gets a ledger row, and anything
that does not improve R2 on the temporal fold gets reverted rather than argued for.

**Status (2026-08-03):** all four resolved. 1 closed, no gain · 2 captured, not a feature · 3 bias confirmed, both fixes rejected, gate question raised for Katie · 4 fixed and tested.

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

## 2. `startDateStruct.type` is fetched and discarded — CAPTURED, not a feature

**Measured 2026-08-03. The field is now captured (`start_date_type`), and must NOT
be wired as a model feature. It does not carry the distinction the section below
assumes.**

A completed trial has, by definition, started, so the completed corpus never holds
an ESTIMATED start date. The values it actually takes are ACTUAL and UNKNOWN — the
latter meaning the registry record omits the `type` key, which is a
record-completeness marker, not a has-it-begun marker:

| frame | ACTUAL | ESTIMATED | UNKNOWN |
|---|---|---|---|
| P1 raw completed studies | 13,351 | 0 | 10,114 |
| P1 modelling frame, after cleaning | 8,730 | 0 | 38 |
| P1 temporal TRAIN fold (pre-2021) | 5,719 | 0 | 38 |
| P1 temporal TEST fold (2021+) | 3,011 | 0 | **0** |

The feature is constant across the entire test fold, so it cannot change a single
scored prediction; in training it is 0.66% non-constant. Note also the attrition:
of 10,114 raw UNKNOWN-start records only 38 survive cleaning, so the field is
largely a marker for records the cleaner already excludes.

Wiring it anyway would ADD an instance of the defect pattern rather than close one.
The column is ~constant in training and would take its value at serve time from a
user whose trial has not started — a field meaning one thing in training and
another in deployment, which is the exact shape of the `site_count` and enrolment
bugs.

**Where the premise does hold, checked:** the ONGOING cohort has 150 ESTIMATED-start
trials (all RECRUITING or ENROLLING_BY_INVITATION — a trial recruiting before its
start date is registry noise in its own right). 69 reach the censoring frame, 1.7%
of its 4,015 censored rows, where elapsed-time-so-far is measured from a projected
start. Checked for the failure that would matter: none are future-dated and none
produce a negative or zero elapsed time (min 3 days). No correction needed today;
the reason it is harmless is that the projections have all since passed, which is
a property of the vintage rather than of the code.

### Original hypothesis (2026-08-02)

`ct_api_client.py:173` parses `startDateStruct` for its date and drops the `type`
field, which is ACTUAL or ESTIMATED. A trial whose start date is still an estimate
is a different object from one that has actually begun, and the flag is free.

This is the same shape as the two defects that cost the most so far: the
`site_count` train/serve mismatch, and enrolment being ACTUAL in training while a
user supplies an ESTIMATE at design time. Both were a field whose *type* mattered
and was thrown away. This is the third instance of that pattern.

## 3. Observation-horizon matching — BIAS CONFIRMED, both fixes rejected

**Measured 2026-08-03 on P1. The bias named below is real and larger than "small".
Neither proposed fix survives. The more important finding is about the GATE, not
the model — see the last block.**

**The test fold is hard-truncated by observation horizon.** The corpus holds
completed trials, so a trial appears only if it finished by the data vintage. The
longest trial in each test start-year sits within ~0.2 years of the horizon that
year had:

| test start year | n | horizon available | longest trial observed | true median |
|---|---|---|---|---|
| 2021 | 836 | 4.93 yr | 5.14 yr | 12.73 mo |
| 2022 | 788 | 3.97 yr | 4.24 yr | 9.51 mo |
| 2023 | 606 | 2.97 yr | 3.21 yr | 9.00 mo |
| 2024 | 490 | 1.95 yr | 2.11 yr | 6.19 mo |
| 2025 | 272 | 1.06 yr | 1.34 yr | 3.68 mo |
| 2026 | 19 | 0.27 yr | 0.28 yr | 1.68 mo |

The falling median is an artifact of the window, not a trend in trial duration.
The model's bias tracks it: −1.27 months where the horizon is longest, +2.03 where
it is shortest, `corr(horizon, signed error) = −0.135` over 3,011 rows. So the
model over-predicts precisely where the fold cannot hold a long trial.

**Fix A — a horizon feature — is the excluded calendar feature wearing a hat.**
`years since start to data vintage` is `−start_date` plus a constant, i.e. a
monotone transform of start year, which `pipeline.py` excludes by name after
`primary_completion_year` cost +25 months of bias on P2. It also has no value at
serve time: a trial being quoted has not started, so its horizon is 0, outside
every value in training, where a tree is flat. Not tested; it is the known defect.

**Fix B — matching the training fold to the test horizon — is metric-fitting.**
Dropping training trials longer than the cut does raise R2 on the 2021+ fold, and
the gain is not a sample-size effect (lever 1's placebo puts a random cut of this
size at roughly −0.008):

| training cut | rows kept | R2 on 2021+ | MAE |
|---|---|---|---|
| none (control) | 5,757 | 0.5549 | 4.99 |
| ≤ 5.0 yr | 91.1% | **0.6026** | 4.77 |
| ≤ 3.6 yr | 81.4% | 0.5936 | 4.79 |
| ≤ 3.0 yr | 76.1% | 0.5609 | 4.89 |
| ≤ 2.0 yr | 63.5% | 0.3710 | 5.58 |

Scored on a fold that is NOT truncated, it reverses. Training on starts before
2018, then scoring the same two models on an untruncated fold (2018–2020 starts,
horizon 5.4–8.6 years against a corpus whose p95 duration is 5.9) and on the
truncated one:

| training set | untruncated fold R2 | truncated fold R2 |
|---|---|---|
| full | 0.6334 | 0.4883 |
| matched ≤5 yr | 0.5528 | 0.5636 |
| **change from matching** | **−0.0806** | **+0.0753** |

Equal and opposite. Matching also drives bias on the untruncated fold from −2.14
to −4.50 months: it makes the model worse, by 4.5 months, at exactly the long
trials a planner most needs warning about, and is paid for that in R2 by a fold
that structurally cannot contain them. Rejected.

**What this says about the gate — for Katie, not for me to change.** The 0.70 R2
bar is computed on the 2021+ fold, whose target is truncated by observation
horizon, and that fold rewards under-prediction. It ranked these two models in
opposite directions from an honest fold, by 0.16 R2. Any future lever that shortens
predictions will collect a gain here that it has not earned. Two options, both
cheap: score the gate on a horizon-adequate window (train <2018 / test 2018–2020,
which costs 3,011 test rows and buys an uncapped target), or keep the current fold
and report bias-by-start-year beside R2 so truncation-fitting is visible when it
happens. `experiments/horizon_bias.py` and `experiments/horizon_disproof.py`
produce both tables.

### Original hypothesis (2026-08-02)

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
