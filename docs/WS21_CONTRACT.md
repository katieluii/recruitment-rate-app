# What WSi needs from WS21's clustering

Written 2026-08-05 by the WSi session. Companion to `docs/WSI_CHANGES_FOR_WS21.md`,
which covers what moved underneath you; this one covers what WSi wants back.

## The division, settled 2026-08-05

WS21 owns the **endpoint clustering** and the **eligibility-criteria clustering**.
WSi owns the **duration prediction** and stays the thing that predicts it.

WSi does not want to compute the clusters and WS21 does not need to predict duration.
The join is: for a chosen phase x therapeutic area, WS21 says which endpoint
combinations and which eligibility profiles actually occur; the user picks one; WSi
predicts the duration conditioned on that pick.

## What WSi can already do with a selection

**`/api/predict` now accepts an endpoint COMBINATION** (shipped 2026-08-05):

```json
{"phase": "P1", "therapeutic_area": "Oncology/Solid Tumours",
 "enrollment": 60, "num_sites": 10,
 "endpoint_archetypes": ["RESPONSE", "SAFETY"]}
```

It sets one `endpoint_has_*` flag per archetype, matching how the model was trained.
The old single `endpoint_archetype` field still works; sending both is a 422. Values
outside `ARCHETYPES` are a 422 — the closed vocabulary is enforced at the door.

Before this, `inference.py` set exactly ONE flag, so a RESPONSE+SAFETY trial was
unrequestable and came back as whichever half was sent. That is fixed and is what
makes "predict for this endpoint combo" possible at all.

## Contract 1 — endpoint combinations

Per `(phase, therapeutic_area)`, the combinations that actually occur, most frequent
first.

```json
{
  "P1": {
    "Oncology/Solid Tumours": [
      {"archetypes": ["SAFETY"],              "label": "SAFETY",
       "n": 2101, "share": 0.452, "median_months": 33.0},
      {"archetypes": ["RESPONSE", "SAFETY"],  "label": "RESPONSE + SAFETY",
       "n": 330,  "share": 0.071, "median_months": 44.7}
    ]
  }
}
```

Requirements, each of which exists because of something already hit:

- **`archetypes` must be drawn strictly from WSi's `ARCHETYPES`**, and is the field
  WSi passes straight to `endpoint_archetypes`. A free-form label cannot be sent to
  the model — it would be a category it never trained on, and `/predict` will 422 it.
- **Order within `archetypes` matters slightly.** WSi sets the categorical
  `endpoint_archetype` from the FIRST element, matching `classify_primary`'s
  first-parseable convention in training. Put the endpoint you consider primary first.
- **Combination, not single label.** Trials list 4.15 primary outcome measures that
  collapse to 1.17 distinct archetypes, and 21.8% carry more than one. The unit that
  occurs is the set.
- **Do not rank purely by frequency.** On the full corpus P1 oncology ranks SAFETY
  (2,101), PK_PD (455), UNKNOWN (344), and only then RESPONSE+SAFETY (330) — the
  case that runs ~12 months longer than SAFETY alone. A strict top-3 by count buries
  the most decision-relevant profile behind the unclassified bucket. Rank by decision
  value, or exclude UNKNOWN from the three and say that you did.
- **Keep UNKNOWN visible with its real count.** It is 21.0% of P2 and 33.5% of P3
  and hiding it would inflate everything else's share. WSi will render it as a
  coverage caveat, not drop it.
- **State the cell size and say when you fell back.** WSi's own implementation uses a
  30-trial floor and falls back phase-wide with a flag; anything similar is fine as
  long as the fallback is reported rather than silent.

WSi has a working implementation at `backend/analytics/endpoint_profiles.py`
(`build_profiles(df)`), already producing exactly this shape. **Import it through your
bridge rather than reimplementing**, or take it over — WSi has no use for it beyond
serving, and it is yours under the ownership split.

## Contract 2 — eligibility-criteria clusters

**This is the one with a hard technical constraint, and it is not obvious.**

A cluster LABEL cannot change a WSi prediction. Eligibility reaches the duration model
through exactly 13 features, and `criteria_text` is NOT one of them — it is built by
`build_features` and then dropped, because the fitted preprocessor has no text block
(verified against `models/artifacts/P2/enrolment_point.pkl`: transformer blocks are
`cat`, `num`, `bin`, `ta_target`, `remainder` — no `crit_text`).

So each cluster must carry the feature values that DEFINE it, not just a name:

```json
{
  "P3": {
    "Oncology/Solid Tumours": [
      {"cluster_id": "onc-p3-biomarker-selected",
       "label": "Biomarker-selected, heavily pre-treated",
       "n": 412, "share": 0.31,
       "observed_median_months": 41.2,
       "features": {
         "n_inclusion_criteria": 9,
         "n_exclusion_criteria": 17,
         "criteria_chars": 3400,
         "crit_biomarker_required": 1,
         "crit_prior_therapy": 1,
         "crit_treatment_naive": 0,
         "crit_performance_status": 1,
         "crit_organ_function": 1,
         "crit_washout": 1,
         "crit_contraception": 1,
         "crit_pregnancy_excluded": 1,
         "crit_comorbidity_excluded": 1,
         "crit_hospitalised": 0
       },
       "exemplar_nct_ids": ["NCT01234567", "NCT02345678"]}
    ]
  }
}
```

- **All 13 keys, every time.** `n_inclusion_criteria`, `n_exclusion_criteria`,
  `criteria_chars`, and the ten `crit_*` markers named in
  `backend/preprocessing/text_features.py:CRITERIA_MARKERS`. A missing key falls back
  to the phase default and the cluster silently stops being distinguishable.
- **Cluster centroid or median, your choice, but say which.** WSi feeds these values
  in as the trial's characteristics; they need to represent the cluster, not one member.
- **The `crit_*` markers are binary 0/1** as WSi computes them. If your clustering
  works on a continuous score, threshold it and state the threshold.
- **`exemplar_nct_ids` are for the read surface**, so a user can check the cluster is
  real. Not consumed by the model.
- **`observed_median_months` is your ground truth, not a prediction.** WSi will show
  it beside its own prediction; where they disagree sharply, that disagreement is
  worth surfacing rather than hiding.

Honest caveat on what this buys: the eligibility features are real model inputs but
not dominant ones. Do not expect a cluster switch to move duration the way phase or
therapeutic area does. Worth measuring the effect size on a few clusters before
building a UI that implies it is large.

## Contract 3 — provenance, so staleness is detectable

Every file you emit should carry:

```json
{"_meta": {"wsi_commit": "0fc0b5c...", "corpus_rows": {"P1": 15429, "P2": 16126},
           "generated": "2026-08-05", "classifier_version": "post-EVENT_COMPOSITE-fix"}}
```

This exists because both of us just got caught by it. WS21 pins WSi at a commit and
`check_pin()` warns without failing, so six commits of changed inputs passed under a
live session behind one log line. And every cluster computed before 2026-08-04 used
roughly half the corpus and a classifier that put 893 safety trials in
`EVENT_COMPOSITE`. A stamped file lets either side notice; an unstamped one does not.

## Delivery

A JSON file per contract, or one file with both, anywhere WSi can read it — the
sibling checkout is fine, `models/artifacts/<phase>/` is fine if you would rather it
travel with the artifacts. Tell WSi the path and it will wire the read side.

## What WSi will build on top

1. The user picks phase and therapeutic area, presses Predict.
2. WSi shows your endpoint combinations and eligibility clusters for that cell.
3. The user picks one of each; WSi predicts duration conditioned on the selection,
   passing `endpoint_archetypes` and the cluster's 13 feature values.
4. Enrolment and sites remain live sliders, bounded by the model's trained range via
   `/api/input-ranges` so dragging cannot walk into extrapolation.

Steps 1 and 4 are built. Steps 2 and 3 are waiting on these two contracts.
