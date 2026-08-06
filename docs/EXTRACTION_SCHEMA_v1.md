# Trial extraction schema v1 — endpoints and eligibility criteria

**Provenance: authored by Katie, 2026-08-06, supplied to the WSi session.** It did
NOT originate in the WS21 session, and an earlier WSi note wrongly attributed it
there. Recorded here as the canonical copy so both sessions work from one text.

Companion documents:
- `docs/SCHEMA_TO_MODEL_MAP.md` — which fields WSi's model can consume today,
  which need a retrain, and the one collision to avoid.
- `docs/WS21_CONTRACT.md` — the aggregate contracts (endpoint combinations,
  eligibility clusters) that sit above this per-record schema.

## The governing principle

> Extract what the source states, normalize what can be normalized reliably, and
> explicitly label everything else as derived.

Three layers, never flattened into one object:

1. **Source** — exact text and provenance.
2. **Normalized** — controlled fields used for comparison, clustering, filtering.
3. **Derived** — analyst or algorithmic judgements: restrictiveness, patient
   relevance, recruitment burden.

This preserves auditability while preventing subjective judgements from
contaminating factual extraction. ClinicalTrials.gov gives each outcome a title,
description, type and timeframe; the schema retains those and normalizes further.

---

## 1. Endpoint schema

```jsonc
{
  "endpoint_id": "EP-001",
  "schema_version": "endpoint_v1.0",

  "source": {
    "source_type": "registry",
    // registry | protocol | sap | publication | regulatory_document | other
    "source_id": "NCT00000000",
    "document_version": null,
    "document_date": null,
    "locator": {"section": "Primary Outcome Measures", "page": null, "table": null},
    "title_raw": "Progression-free survival",
    "description_raw": "Time from randomisation to...",
    "timeframe_raw": "From randomisation until progression or death, up to 36 months"
  },

  "normalized": {
    "hierarchy": "primary",
    // primary | co_primary | key_secondary | secondary |
    // exploratory | other_prespecified | post_hoc | unknown

    "domain": "efficacy",
    // efficacy | safety_tolerability | pharmacokinetics | pharmacodynamics |
    // biomarker | patient_reported_outcome | quality_of_life | resource_use |
    // other | unknown

    "concept": {
      "canonical_label": "progression-free survival",
      "abbreviation": "PFS",
      "ontology_system": "NCI_THESAURUS",
      "ontology_code": null,
      "other_text": null
    },

    "variable_type": "time_to_event",
    // time_to_event | binary | continuous | ordinal | count |
    // recurrent_event | composite | exposure | other | unknown

    "aggregation": "time_to_event",
    // time_to_event | proportion | incidence_rate | mean | median |
    // change_from_baseline | percent_change | slope | area_under_curve |
    // maximum | minimum | count | score | other | unknown

    "directionality": "higher_better",
    // higher_better | lower_better | context_dependent | not_applicable | unknown

    "instrument": {
      "name": "RECIST", "version": "1.1", "instrument_type": "response_criteria"
      // response_criteria | questionnaire | laboratory_test | imaging_method |
      // clinical_scale | wearable | clinician_assessment | other |
      // not_applicable | unknown
    },

    "assessment": {
      "assessor": "independent_central_review",
      // investigator | independent_central_review |
      // blinded_independent_central_review | patient | caregiver |
      // laboratory | device | mixed | unknown
      "blinded": true,
      "adjudicated": true
    },

    "timeframe": {
      "start_anchor": "randomization",
      // screening | baseline | randomization | first_dose | surgery |
      // treatment_end | disease_progression | discharge | other | unknown
      "end_anchor": "event_or_max_followup",
      // fixed_timepoint | treatment_end | event | event_or_max_followup |
      // study_completion | other | unknown
      "minimum_duration_iso8601": null,
      "maximum_duration_iso8601": "P36M",
      "assessment_points_iso8601": [],
      "ongoing_through_study": false,
      "parse_status": "complete"       // complete | partial | failed
    },

    "event_definition": {
      "is_event_based": true,
      "is_trial_event_driven": null,
      "qualifying_events": ["disease_progression", "death"],
      "target_event_count": null,
      "competing_risks_reported": false,
      "censoring_rule_summary": null
    },

    "analysis": {
      "analysis_population": "intention_to_treat",
      // intention_to_treat | modified_intention_to_treat | per_protocol |
      // safety_population | efficacy_evaluable | pharmacokinetic_population |
      // other | not_reported
      "treatment_effect_measure": "hazard_ratio",
      // hazard_ratio | risk_ratio | odds_ratio | rate_ratio |
      // difference_in_means | difference_in_medians |
      // difference_in_proportions | ratio_of_means | descriptive_only |
      // other | not_reported
      "comparison_type": "between_arm",
      // between_arm | within_arm | single_arm_benchmark | dose_response |
      // descriptive | other | unknown
      "multiplicity_role": "primary_family",
      // primary_family | hierarchical_test | alpha_controlled_secondary |
      // nominal_secondary | exploratory | not_reported | not_applicable
      "estimand": {
        "population": null,
        "treatment_condition": null,
        "variable": "progression-free survival",
        "population_level_summary": "hazard_ratio",
        "intercurrent_event_strategy": "not_reported"
        // treatment_policy | hypothetical | composite | while_on_treatment |
        // principal_stratum | mixed | not_reported
      }
    }
  },

  "derived": {
    "endpoint_family": "survival",
    // survival | response | symptoms | function | safety | tolerability |
    // pk | pd | biomarker | quality_of_life | healthcare_utilization | other
    "clinical_relevance": "direct_clinical_benefit",
    // direct_clinical_benefit | surrogate | intermediate_clinical_endpoint |
    // mechanistic | safety_only | unclear
    "endpoint_maturity": null,
    "analyst_notes": null,
    "derivation_method": null
  },

  "quality": {
    "extraction_confidence": 0.96,
    "normalization_confidence": 0.91,
    "review_status": "machine_extracted",
    // machine_extracted | analyst_reviewed | adjudicated | rejected
    "ambiguity_flags": [],
    "normalizer_version": "endpoint_normalizer_v1.0",
    "vocabulary_version": "clinical_schema_vocab_2026_08"
  }
}
```

### Design choices that carry the weight

**`hierarchy` is separate from `domain`.** A primary endpoint may be efficacy,
safety or PK. Hierarchy is its inferential role; domain is what it measures.

**Concept is separate from statistical form.** Concept = progression-free
survival; variable_type = time_to_event; aggregation = time_to_event; effect
measure = hazard_ratio; instrument = RECIST 1.1; assessor = BICR. Collapsing
these into `endpoint_type: "PFS"` destroys comparability.

**Multiplicity is included but may be `not_reported`.** FDA distinguishes
endpoint ordering and multiplicity control: a nominal secondary is not an
alpha-controlled key secondary.

**Estimand fields are optional enrichment.** ICH E9(R1) separates population,
treatment condition, variable, intercurrent-event handling and population-level
summary. Registries rarely give all of it; protocols and SAPs may.

---

## 2. Eligibility criterion schema

**The atomic unit is one CONSTRAINT, not one source bullet.** "ECOG 0–1 and ANC
≥1.5 × 10⁹/L" becomes two criterion records joined by one AND logic group.

```jsonc
{
  "criterion_id": "EC-001",
  "schema_version": "eligibility_v1.0",

  "source": {
    "source_type": "registry",
    "source_id": "NCT00000000",
    "document_version": null,
    "document_date": null,
    "criterion_section": "inclusion",   // inclusion | exclusion | other | unknown
    "criterion_order": 7,
    "raw_text": "ECOG performance status of 0 or 1",
    "locator": {"section": "Eligibility Criteria", "page": null}
  },

  "normalized": {
    "summary": "Requires ECOG performance status 0–1",

    "eligibility_effect": "required",
    // required | prohibited | permitted | conditionally_permitted |
    // conditionally_required | unknown

    "category": "performance_status",
    // demographics | diagnosis | disease_subtype | disease_stage |
    // disease_severity | measurable_disease | biomarker | genetic_feature |
    // prior_therapy | treatment_response | treatment_resistance | washout |
    // concomitant_therapy | prohibited_therapy | laboratory | organ_function |
    // performance_status | comorbidity | infection | immunization |
    // reproductive | contraception | pregnancy | medication | procedure |
    // surgery | transplantation | imaging | physiological_measurement |
    // substance_use | lifestyle | geography | availability_and_compliance |
    // consent | investigator_judgment | other | unknown

    "constraint": {
      "concept": {
        "canonical_label": "ECOG performance status",
        "ontology_system": "NCI_THESAURUS",
        "ontology_code": null
      },
      "operator": "between_inclusive",
      // equal | not_equal | greater_than | greater_than_or_equal |
      // less_than | less_than_or_equal | between_inclusive | in_set |
      // not_in_set | present | absent | history_of | no_history_of |
      // confirmed | capable_of | other
      "value": null, "value_min": 0, "value_max": 1,
      "categorical_values": [],
      "unit_raw": null, "unit_normalized": null, "unit_system": null
    },

    "temporal_constraint": {
      "relation": "at_screening",
      // at_screening | at_baseline | current | prior_to_screening |
      // prior_to_randomization | prior_to_first_dose | after_event |
      // within_previous_period | at_least_period_since | ongoing |
      // lifetime_history | other | not_applicable | unknown
      "reference_event": "screening",
      "minimum_duration_iso8601": null,
      "maximum_duration_iso8601": null,
      "duration_raw": null
    },

    "applicability": {
      "population_scope": "all_participants",
      // all_participants | subgroup | sex_specific |
      // reproductive_potential_only | cohort_specific |
      // treatment_arm_specific | other
      "subgroup_description": null,
      "cohort_ids": []
    },

    "logic": {
      "atomic": true,
      "logic_group_id": null,
      "parent_criterion_id": null,
      "relation_to_siblings": null     // AND | OR | NOT | null
    },

    "requirement": {
      "strength": "explicit_mandatory",
      // explicit_mandatory | investigator_judgment |
      // operational_preference | unclear
      "waiver_status": "not_stated"
      // prohibited | permitted | not_stated | not_applicable
    },

    "screening": {
      "computability": "structured_ehr",
      // structured_ehr | derived_from_ehr | unstructured_chart_review |
      // laboratory_test_required | imaging_review_required |
      // genomic_test_required | clinician_judgment | patient_report |
      // external_documentation | not_computable | unknown
      "screening_burden": "low",        // low | moderate | high | unknown
      "data_domains": ["clinical_assessment"]
    }
  },

  "derived": {
    "restrictiveness": {"level": null, "score": null, "method": null,
                        "reference_population": null},
    "recruitment_impact": null,
    "site_feasibility_impact": null,
    "analyst_notes": null
  },

  "quality": {
    "extraction_confidence": 0.98,
    "normalization_confidence": 0.95,
    "review_status": "machine_extracted",
    "ambiguity_flags": [],
    "normalizer_version": "eligibility_normalizer_v1.0",
    "vocabulary_version": "clinical_schema_vocab_2026_08"
  }
}
```

### Three distinctions that must not be collapsed

**`criterion_section` is not semantic effect.** An inclusion bullet reading "No
prior treatment with a KRAS G12C inhibitor" is `criterion_section: inclusion`,
`eligibility_effect: prohibited`, `category: prior_therapy`. Querying on section
alone gives the wrong answer.

**No `hard_gate: true/false`.** Almost every stated eligibility requirement is
formally a gate; what varies is whether it depends on judgement. Use
`explicit_mandatory` (ANC ≥1.5), `investigator_judgment` ("adequate cardiac
function in the investigator's opinion"), `operational_preference` (site
preference, and only when documented outside formal criteria), `unclear`. Do not
infer "soft" from vague wording.

**Boolean logic must survive decomposition.** "No active brain metastases,
except treated stable lesions requiring no corticosteroids for ≥14 days" is a
tree, not three unrelated rows:

```json
{"operator": "OR", "children": [
  {"constraint": "brain metastases absent"},
  {"operator": "AND", "children": [
    {"constraint": "brain metastases treated"},
    {"constraint": "brain metastases stable"},
    {"constraint": "no corticosteroid requirement"},
    {"constraint": "duration >= P14D"}]}]}
```

Atoms may live in a table, but `logic_group_id`, parent-child links and AND/OR
operators are required to reconstruct meaning.

---

## 3. Controlled vocabulary policy

CDISC controlled terminology covers standard clinical concepts, units and
assessments, but is not a complete ontology for every analytical judgement, and
is versioned. Three vocabulary classes:

**A. Closed structural enums — reject out-of-vocabulary.**
`hierarchy`, `domain`, `variable_type`, `criterion_section`,
`eligibility_effect`, `operator`, logic operator, `review_status`. These define
how the schema functions.

**B. Extensible controlled concepts — do NOT reject the record.**
Endpoint concept, disease, biomarker, laboratory test, prior therapy, clinical
scale, event type. When unmapped, store it as unmapped:

```json
{"canonical_label": null, "ontology_code": null,
 "raw_value": "novel source term", "mapping_status": "unmapped"}
```

A novel biomarker should produce an unmapped concept, not a failed run.

**C. Derived analyst classifications — require method and version.**
Restrictiveness, recruitment impact, clinical relevance, surrogacy, screening
burden, endpoint maturity. Never appear as unqualified facts.

Every record carries `schema_version`, `vocabulary_version`,
`normalizer_version`.

---

## 4. Fields worth indexing for UI filters

Indexed columns, not nested-JSON queries.

**Endpoints:** `hierarchy` · `domain` · `concept.canonical_label` or code ·
`variable_type` · `instrument.name` · `assessment.assessor` · normalized max
duration · `event_definition.is_event_based` ·
`event_definition.is_trial_event_driven` · `analysis.analysis_population` ·
`analysis.treatment_effect_measure` · `analysis.multiplicity_role` · derived
`clinical_relevance`.

**Eligibility:** `criterion_section` · `eligibility_effect` · `category` ·
normalized concept/code · `constraint.operator` · normalized min/max · normalized
UCUM unit · normalized temporal duration · `requirement.strength` ·
`screening.computability` · `screening.screening_burden` · derived
restrictiveness · `quality.review_status`.

Raw text, normalized summaries and analyst notes should be full-text searchable
but not categorical filters.

---

## 5. MVP — required fields for the first implementation

**Endpoint:** `endpoint_id`, `hierarchy`, `domain`, `concept.canonical_label`,
`variable_type`, `instrument.name`/`version`, `assessor`, `timeframe_raw`,
`timeframe.start_anchor`, `timeframe.maximum_duration`, `is_event_based`, source
raw text, confidence, review status.

**Criterion:** `criterion_id`, `criterion_section`, `raw_text`, normalized
summary, `eligibility_effect`, `category`, concept, `operator`, value/range,
unit, temporal relation/duration, logic group, `requirement.strength`,
`computability`, confidence, review status.

Estimands, multiplicity, censoring, restrictiveness and recruitment impact stay
nullable enrichment. Requiring them immediately would generate large volumes of
confident-looking invented data.

**One WSi addition to the endpoint MVP:** carry
`event_definition.is_trial_event_driven` as nullable. It is the strongest
candidate feature in the schema for duration — a trial ending on event accrual
rather than a calendar is a direct duration mechanism, and nothing in WSi's
current feature set sees it. Cheap at extraction time; backfilling means
re-reading every outcome record.

---

## 6. Sequencing, given what WSi measured

WSi's model moves **0.6 months** across a full permissive-to-restrictive
eligibility swap (WS21 reproduced this independently at 0.7). The OBSERVED
spread across clusters is **14.7 months**. Roughly 25x.

So this schema's value to prediction is as **retrain input**, not as a better
pipe for the thirteen features WSi consumes today. Order that follows:

1. Extract the MVP on a SAMPLE — a few thousand trials across four phases.
2. Test the four candidate features against the horizon fold, one ledger row each
   (`is_trial_event_driven`, normalized max duration, `assessor`,
   `screening.computability` aggregated per trial).
3. Retrain only on what earns it.
4. Build the UI last.

Extracting across ~54,000 trials before knowing whether any of it predicts
anything is the expensive order.
