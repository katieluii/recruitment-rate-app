# WS21's extraction schema, mapped onto what WSi's model can consume

Written 2026-08-06 by the WSi session, against the endpoint_v1.0 /
eligibility_v1.0 schema in `docs/EXTRACTION_SCHEMA_v1.md`.

**Provenance correction (2026-08-06):** that schema was authored by KATIE and
supplied to the WSi session. An earlier WSi note attributed it to the WS21
session, which had never seen it — WS21 flagged the error rather than letting
silence read as assent. Nothing below is agreed on WS21's side until they say so.

Companion to `docs/WS21_CONTRACT.md`.

The schema's three-layer split — source / normalized / derived — is the right
shape, and it enforces something this project learned expensively: a derived
judgement that travels as a fact eventually gets used as one. Everything below
assumes that split holds.

The governing constraint from WSi's side is narrower than the schema:

**The fitted models consume 13 eligibility features and 11 endpoint archetypes.
Nothing else reaches a prediction without a retrain.** That is not an argument
against the richer schema — it is an argument about which parts are inputs today
and which are candidate features tomorrow, and the two should not be confused.

## 1. The collision to fix first: constraints vs bullets

**Applies only to constraint-level extraction, which is NOT what WS21 ships
today.** WS21 confirmed its `n_inclusion_criteria` already comes from a bullet
count, so nothing currently in flight is affected. This is for whoever builds
against the schema.

`count_criteria` (features.py:215) counts BULLETS — "one bullet per line" under
the Inclusion/Exclusion heading. The schema's atomic unit is one CONSTRAINT.

    "ECOG performance status 0-1 and ANC >= 1.5 x 10^9/L"
      -> 1 bullet   (what n_inclusion_criteria counts in training)
      -> 2 criterion records (what the schema emits)

Counting criterion records would inflate `n_inclusion_criteria` and
`n_exclusion_criteria` against every value the model was fitted on. The schema's
decomposition is the better representation; it is simply a different variable.

**Fix:** keep `source.criterion_order` as the bullet index and derive the model
features by counting DISTINCT `criterion_order` values per section, not records.
That preserves the schema's decomposition and the feature's meaning at once.

## 2. What maps to a model input today, with no retrain

| model feature | derive from | note |
|---|---|---|
| `n_inclusion_criteria` | distinct `criterion_order` where `criterion_section = inclusion` | bullets, not constraints — see above |
| `n_exclusion_criteria` | same, `exclusion` | |
| `criteria_chars` | `len()` of the FULL raw eligibility text | not truncated in training; do not cut it |
| `crit_*` (ten markers) | regex over the criteria text **truncated to 4,000 chars** | the markers are the only thing the 4,000 cap touches |
| `endpoint_archetypes` | `normalized.domain` + `normalized.concept` → ARCHETYPES | mapping below |

**The `crit_*` markers should stay regex-derived, not schema-derived.** The
schema's `category` enum is a better description of a criterion, but it is a
different variable from `crit_prior_therapy`, and the model learned the regex.
Mapping `category = prior_therapy` onto `crit_prior_therapy` would look
equivalent and would not be. Run WSi's `marker_frame()` over the same truncated
text and send its output; use `category` for clustering and display.

### Endpoint archetype mapping

ARCHETYPES is closed and the model has never seen anything else. Suggested
deterministic map from the schema's richer fields:

| schema | archetype |
|---|---|
| `concept.abbreviation` in {OS, PFS, DFS, EFS, RFS} or `variable_type = time_to_event` with a mortality/progression event | `SURVIVAL` |
| `concept` is a MACE / composite of death+hospitalisation | `EVENT_COMPOSITE` |
| `aggregation = incidence_rate`, or a per-period event rate | `EVENT_RATE` |
| `concept.abbreviation` in {ORR, pCR, DCR, CBR} or `domain = efficacy` with `variable_type = binary` response | `RESPONSE` |
| `concept` is seroconversion / titre | `IMMUNOGENICITY` |
| `instrument.instrument_type = questionnaire` or `clinical_scale`, `domain != patient_reported_outcome` | `CLINICAL_SCORE` |
| `domain = biomarker`, or a laboratory analyte | `BIOMARKER` |
| `domain in {patient_reported_outcome, quality_of_life}` | `PRO` |
| `domain = safety_tolerability` | `SAFETY` |
| `domain in {pharmacokinetics, pharmacodynamics}` | `PK_PD` |
| no confident map | `UNKNOWN` — emit it, do not guess |

`hierarchy` is the field that decides WHICH endpoints go into the archetype set:
use `primary` and `co_primary` only. WSi's training flags are built from primary
outcomes alone, so including key secondaries would light flags the model never
saw lit for those trials.

## 3. What cannot be a feature without a retrain — and which are worth one

Everything else in the schema is descriptive as far as the current models are
concerned. Four fields are worth retraining FOR, in rough order of expected
value against the problem WSi actually has:

**`event_definition.is_trial_event_driven` — the strongest candidate in the
schema.** An event-driven trial ends when N events accrue, not on a calendar. It
is a direct mechanism for duration, and WSi has no feature that captures it: the
model currently infers follow-up from parsed timeframe text, which cannot tell a
36-month fixed follow-up from "until 380 PFS events". If this field is populated
reliably, it is the first thing to test.

**`timeframe.maximum_duration_iso8601` — a cleaner version of a feature WSi
already has.** `followup_months` is parsed from free-text timeframes and fails
to parse on about half of trials, falling back to the endpoint-archetype median.
A normalized ISO8601 duration would replace an imputed value with a measured one
on the single feature that drives the follow-up half of the two-stage model.
This is the lowest-risk improvement in the list because it slots into an
existing feature rather than adding one.

**`assessment.assessor = independent_central_review` / `adjudicated`** — central
imaging review and event adjudication add real calendar time between last
patient and readout, and nothing in the current feature set sees them.

**`screening.computability` and `screening_burden`** — plausibly the recruitment
half's missing signal. A criterion needing a genomic test screens slower than one
readable from structured EHR. Aggregated per trial (e.g. worst-case
computability, count of not_computable criteria), this is a recruitment-friction
feature WSi has never had.

Deliberately NOT worth a retrain: `analysis.*`, `estimand.*`,
`multiplicity_role`. They describe inferential design, which is real and matters
for reading a trial, but has no plausible mechanism on how long it takes to run.

## 4. The honest ceiling on all of this

WSi's model responds to a full permissive-to-restrictive eligibility swap by
**0.6 months** on P3 oncology. WS21 measures the OBSERVED spread across clusters
at **14.7 months**. The gap is roughly 25x.

That means the schema's value to prediction is almost entirely as **retrain
input**, not as a better pipe for the existing 13 features. Sending richer,
cleaner values through the current parameter will not move the number, because
the current features are weak — not because they are badly derived.

So the sequence that makes sense:

1. Extract the MVP on a SAMPLE, not the full corpus — a few thousand trials
   across the four phases is enough to test a feature.
2. Test the four candidates above against the horizon fold, one ledger row each.
3. Retrain only on what earns it.
4. Only then build a UI that implies eligibility moves duration.

Extracting the full schema across ~54,000 trials before knowing whether any of
it predicts anything would be the expensive order to do this in.

## 5. On the MVP as proposed

Endorsed as written, with one addition: put `is_trial_event_driven` in the
endpoint MVP as a nullable field. It is the highest-value candidate feature in
the schema and it is cheap to capture at extraction time, whereas backfilling it
later means re-reading every outcome record.

The instinct behind leaving estimands, multiplicity, censoring and
restrictiveness as nullable enrichment is right, and matches the rule that has
caught the most defects in this project: a confident-looking value that was
never measured is worse than an absent one, because absence is visible.
