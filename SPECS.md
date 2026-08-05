# Specs

Testable acceptance criteria for non-trivial changes, written before the code so
the intent survives the session that had it. Newest first.

---

## 2026-08-03 — Endpoint profiles and live inputs

> **OWNERSHIP CORRECTED 2026-08-05. Most of this spec belongs to WS21, not WSi.**
> The dividing test: does it feed the duration prediction? The endpoint CLASSIFIER
> does — it supplies model features AND, through `cleaner.py`'s follow-up imputation,
> partly defines the training target — so it stays here. The endpoint PROFILES layer
> and the interactive UI do not; nothing in WSi's model or preprocessing reads
> `endpoint_profiles.json`. Both move to WS21, which `WS21_KICKOFF.md` and the
> tracker card already assign them to. See `docs/WSI_CHANGES_FOR_WS21.md`.
>
> **Still WSi's, and still unbuilt:** the multi-archetype predict fix. `inference.py`
> sets exactly one `endpoint_has_*` flag while the model trains on a multi-hot set,
> so a RESPONSE+SAFETY trial is unexpressable through `/predict`. That is a WSi model
> bug regardless of who builds the interface on top.

**Why.** Two problems in one screen. The endpoint dropdown asks the user to supply
the single strongest driver of duration after phase, which is the thing they came
to find out; and the API accepts only ONE endpoint archetype while the model was
trained on a multi-hot flag set, so the second most common Phase 1 oncology
configuration — RESPONSE+SAFETY, 250 trials, median 46.1 months — cannot be
expressed and comes back as 38.6 or 33.8 months depending on which half is sent.

### In scope

**Endpoint profiles replace the dropdown**

- WHEN a phase and therapeutic area are submitted, the system SHALL return the
  three most common primary-endpoint COMBINATIONS observed in that cell, each with
  its trial count, share, and median observed duration.
- The system SHALL predict using the most common combination and SHALL label which
  combination the prediction used.
- WHERE a cell holds fewer than 30 trials, the system SHALL fall back to the
  phase-wide profiles and SHALL mark the response as a fallback.
- The combination vocabulary SHALL be drawn strictly from `ARCHETYPES`; an
  out-of-vocabulary archetype SHALL be rejected, never admitted as a new category.

**The API stops narrowing the endpoint representation**

- The system SHALL accept a LIST of endpoint archetypes and SHALL set one
  `endpoint_has_*` flag per supplied archetype.
- The system SHALL continue to accept the existing single `endpoint_archetype`
  field, so callers written against the current API keep working.
- WHEN both fields are supplied, the system SHALL reject the request rather than
  silently preferring one.
- The categorical `endpoint_archetype` feature SHALL be set from the FIRST element
  of the list, matching `classify_primary`'s first-parseable convention in training.

**Enrollment and sites become live**

- Target enrollment and number of sites SHALL be sliders, and the outputs SHALL
  update on drag without a button press.
- Phase and therapeutic area SHALL remain dropdowns behind an explicit
  "Predict duration" action.
- Slider input SHALL be debounced so a drag issues at most one in-flight request.
- WHERE a slider moves outside the trained feature range, the existing
  extrapolation warning SHALL still surface.

### Out of scope

- Secondary outcomes stay unclassified. They are counted, not archetyped, so a
  secondary OS endpoint remains invisible to the model. Adding them is a feature
  change requiring a retrain per phase, and is deliberately NOT part of this.
- No retraining. Profiles are computed from the corpus and shipped as an artifact
  alongside `analytics.json` and `site_priors.json`; the fitted models are untouched.
- The R2 gate and the horizon question from `docs/OPEN_LEVERS.md` §3 are unaffected.

### Verification

- `endpoint_has_*` flag count in a served row equals the number of archetypes sent.
- A request for P1 / Oncology with `[RESPONSE, SAFETY]` returns a longer duration
  than the same request with `[RESPONSE]` or `[SAFETY]` alone, matching the
  direction of the observed medians.
- Sending both `endpoint_archetype` and `endpoint_archetypes` returns 422.
- Sending an archetype outside `ARCHETYPES` returns 422.
- Dragging a slider updates the number with no button press and issues one request.
