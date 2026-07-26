# WS21 — Clinical Trial Analyst · new-session kickoff prompt

Copy everything below the line into a fresh session started from
`~/Projects/ws_professional/recruitment_rate_app`.

---

Start WS21, a new workstream: a **Clinical Trial Analyst agent** built on top of
WSi (`~/Projects/ws_professional/recruitment_rate_app`). WSi stays what it is —
the recruitment/duration predictor. WS21 is the analyst layer above it.

## The three functions

1. **Trial duration** — reuse WSi's existing prediction. Do not rebuild it.
2. **Enrolment range** — for a given indication and phase, a *suggested range*,
   **stratified by trial design**. Not a free-text input.
3. **Trial design** — defined as the combination of:
   - **(a) inclusion/exclusion criteria** that characterise the patient
     population for trials in this indication and phase, and
   - **(b) the endpoints actually used** for this indication and phase.

## Why this exists — the UI critique that prompted it

WSi's current form asks the user to supply what they came to find out:

- The **"Target Enrolment"** free-text box. Enrolment is the single biggest
  driver of the duration prediction, so asking for it means the user provides
  most of the answer — and at design stage they often do not know it.
- The **"Primary Endpoint type"** dropdown exposes WSi's internal 11-value
  archetype vocabulary (SURVIVAL, BIOMARKER, …). For a real question like
  "P3 NSCLC" the useful answer is the concrete endpoint set actually used —
  OS, PFS, ORR with observed frequencies — not a taxonomy bucket. That
  vocabulary was built for the model and leaked into the interface.

## Start here

A **clustering pass**: what are the main clusters of **endpoints** per indication
/ therapeutic area per phase? Then the same for **inclusion/exclusion criteria**,
which define the patient population. Those clusters ARE the trial-design strata
that the enrolment range is conditioned on.

## What already exists in WSi — reuse, do not rebuild

- `criteria_text` (raw eligibility text, 4000 chars, retained S263),
  `n_inclusion_criteria`, `n_exclusion_criteria`
- `primary_outcome_measures`, `primary_outcome_timeframes` (endpoint text)
- `backend/preprocessing/endpoints.py` — 11-value archetype classifier,
  deterministic rules, 8.6/21.0/23.9% abstention on P1/P2/P3
- `experiments/` — temporal-split harness, median baselines, append-only
  `ledger.jsonl`, `progress.py --md` → `RESULTS.md`
- `backend/models/provenance.py` — Atlas WS9-schema provenance
- ~4,500 completed industry trials per phase, cached as parquet in `data/cache/`

## Working rules carried over from WSi — these were expensive to learn

- **No change ships without a ledger row.** The harness decides, not intuition.
  Two of four planned v3 items were built, measured, and disabled on evidence.
- **No per-site enrolment exists** in ClinicalTrials.gov or AACT.
- **Country recruitment speed is NOT identifiable.** A multi-country trial
  reports one enrolment window shared by every participating country, and
  Eastern Europe runs single-country trials 0.8% of the time against the US's
  43.6%. Do not re-attempt a country league table.
- **Per-site rate is largely arithmetic** — log(rate) on log(site_count) has a
  slope near −1. Prefer the enrolment window.
- **Completed-trials-only data is survivorship-biased.** At a 2018 vantage, P3
  duration looked 20.9 months when it was truly 24.6.
- **Medians do not compose.** Median-of-a-ratio ≠ ratio-of-medians; this is why
  all-defaults scenarios disagree with real-trial scenarios.
- Git identity for this WS repo: `dev <dev@localhost>`, no Claude trailers.
- Railway auto-deploys `main`. The `railway` CLI in this directory is linked to
  the WRONG project (`ws14-shared-kb`) — do not `railway up` here.

## One thing to decide first

WS21 is a separate workstream but currently has no folder. Decide whether it
lives in its own repo or as a package inside `recruitment_rate_app`. It consumes
WSi's data layer heavily, which argues for the latter; it is a different product
with a different audience, which argues for the former.

## Open item inherited from S263

`main` in `recruitment_rate_app` is **one commit ahead of origin and unpushed**
(`2afb716`, task 13 — recruitment rate derived from the enrolment window so the
two can never contradict). Pushing auto-deploys Railway. Katie wanted to review
the local app before it goes live.
