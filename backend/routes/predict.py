from __future__ import annotations
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.constants import PHASES, THERAPEUTIC_AREAS
from backend.models.inference import eligibility_fields, predict
from backend.preprocessing.endpoints import ARCHETYPES

router = APIRouter(tags=["predict"])


class PredictRequest(BaseModel):
    # Reject unknown fields instead of dropping them. A request sending
    # `site_count` rather than `num_sites` used to return HTTP 200 with a
    # prediction made without any site count, which is indistinguishable from a
    # correct answer: a local-vs-deployed comparison sent the wrong name and
    # appeared to show a stale deployment, off by up to 3.7 months across three
    # phases, when the deployment was current. A 422 costs one debugging minute;
    # a silent default cost an afternoon.
    model_config = {"extra": "forbid"}

    phase: str = Field(..., examples=["P2"])
    therapeutic_area: str = Field(..., examples=["Oncology/Solid Tumours"])
    enrollment: Optional[int] = Field(None, ge=1, le=50000)
    num_sites: Optional[int] = Field(None, ge=1, le=5000)
    drug_type: str = Field("DRUG", pattern="^(DRUG|BIOLOGICAL)$")
    region: str = Field("US")
    endpoint_archetype: Optional[str] = Field(
        None, examples=["SURVIVAL"],
        description=(
            "Primary endpoint type, single. The strongest single driver of "
            "duration after phase: on Phase 3 the median runs 39.6 months for a "
            "survival endpoint against 10.6 for immunogenicity. Kept for "
            "existing callers; prefer `endpoint_archetypes`."
        ),
    )
    eligibility_features: Optional[Dict[str, float]] = Field(
        None,
        examples=[{"n_inclusion_criteria": 9, "n_exclusion_criteria": 17,
                   "criteria_chars": 3400, "crit_biomarker_required": 1}],
        description=(
            "Patient-population characteristics as FEATURE VALUES. A cluster "
            "label cannot be sent: eligibility reaches the model through 13 "
            "numeric and binary features, and `criteria_text` is not among them "
            "— it is built and then dropped by the fitted preprocessor. Unknown "
            "keys are rejected rather than ignored. Effect size is modest; see "
            "docs/WS21_CONTRACT.md before building a UI that implies otherwise."
        ),
    )
    endpoint_archetypes: Optional[List[str]] = Field(
        None, examples=[["RESPONSE", "SAFETY"]],
        description=(
            "Every primary-endpoint archetype the trial carries. A trial has a "
            "COMBINATION, not one endpoint: 21.8% of trials light more than one "
            "flag, and the combination is not the sum of its parts — P1 oncology "
            "RESPONSE+SAFETY runs a median 44.7 months against 33.0 for SAFETY "
            "alone and 38.6 for RESPONSE alone. The model has always trained on "
            "a multi-hot flag set; sending a single value was the interface "
            "narrowing what the model could express."
        ),
    )


class PredictResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    phase_key: str
    phase_label: str
    therapeutic_area: str
    # Head A — total duration
    predicted_months: float
    lower_months: float
    upper_months: float
    predicted_days: float
    lower_days: float
    upper_days: float
    confidence_pct: int
    # Head B — recruitment rate
    recruitment_rate: Optional[float] = None
    recruitment_rate_lower: Optional[float] = None
    recruitment_rate_upper: Optional[float] = None
    rate_implied_total_months: Optional[float] = None
    recruitment_rate_crosscheck: Optional[float] = None
    estimated_recruitment_months: Optional[float] = None
    recruitment_rate_definition: Optional[str] = None
    recruitment_rate_target_quality: Optional[str] = None
    recruitment_rate_validation: Optional[dict] = None
    recruitment_rate_confidence_pct: Optional[int] = None
    recruitment_rate_n_train: Optional[int] = None
    enrollment_used: Optional[float] = None
    num_sites_used: Optional[float] = None
    rate_note: Optional[str] = None
    # Duration split into its two stages
    enrolment_months: Optional[float] = None
    followup_months: Optional[float] = None
    endpoint_archetypes_used: List[str] = []
    endpoint_source: str = "unknown"
    endpoint_profile_share: Optional[float] = None
    endpoint_profile_n: Optional[int] = None
    # Provenance
    model_used: str
    rmse_days: float
    n_train: int
    extrapolation_warnings: list[str] = []
    # Full working: sources, per-value derivation, input origins, gaps.
    provenance: Optional[dict] = None


# Rendered under the independently modelled rate figure.
_RATE_NOTE = (
    "Estimated from completed-trial record histories with actual enrollment and "
    "a recorded recruiting interval. Tier B assumes listed centres had the full "
    "interval to recruit; it is a planning benchmark, not observed centre performance."
)


@router.post("/predict", response_model=PredictResponse)
def post_predict(req: PredictRequest):
    if req.phase not in PHASES:
        raise HTTPException(422, f"Unknown phase '{req.phase}'. Valid: {list(PHASES)}")
    if req.therapeutic_area not in THERAPEUTIC_AREAS:
        raise HTTPException(422, f"Unknown therapeutic area '{req.therapeutic_area}'.")
    if req.endpoint_archetype and req.endpoint_archetypes:
        # Rejected rather than silently preferring one. Two callers disagreeing
        # about which field wins is the class of defect this API keeps hitting.
        raise HTTPException(
            422,
            "Send either `endpoint_archetype` or `endpoint_archetypes`, not both.",
        )

    archetypes = req.endpoint_archetypes or (
        [req.endpoint_archetype] if req.endpoint_archetype else [])
    unknown = [a for a in archetypes if a not in ARCHETYPES]
    if unknown:
        raise HTTPException(
            422,
            f"Unknown endpoint archetype(s) {unknown}. Valid: {list(ARCHETYPES)}",
        )

    if req.eligibility_features:
        allowed = set(eligibility_fields())
        unknown_keys = sorted(set(req.eligibility_features) - allowed)
        if unknown_keys:
            raise HTTPException(
                422,
                f"Unknown eligibility feature(s) {unknown_keys}. "
                f"Valid: {sorted(allowed)}",
            )

    try:
        result = predict(
            phase_key=req.phase,
            therapeutic_area=req.therapeutic_area,
            enrollment=req.enrollment,
            num_sites=req.num_sites,
            drug_type=req.drug_type,
            region=req.region,
            endpoint_archetypes=archetypes or None,
            eligibility_features=req.eligibility_features,
        )
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc))
    except Exception as exc:
        raise HTTPException(500, f"Prediction error: {exc}")

    return PredictResponse(
        phase_key=result.phase_key,
        phase_label=PHASES[result.phase_key]["label"],
        therapeutic_area=result.therapeutic_area,
        predicted_months=result.predicted_months,
        lower_months=result.lower_months,
        upper_months=result.upper_months,
        predicted_days=result.predicted_days,
        lower_days=result.lower_days,
        upper_days=result.upper_days,
        confidence_pct=result.confidence_pct,
        recruitment_rate=result.recruitment_rate,
        recruitment_rate_lower=result.recruitment_rate_lower,
        recruitment_rate_upper=result.recruitment_rate_upper,
        rate_implied_total_months=result.rate_implied_total_months,
        recruitment_rate_crosscheck=result.recruitment_rate_crosscheck,
        estimated_recruitment_months=result.estimated_recruitment_months,
        recruitment_rate_definition=result.recruitment_rate_definition,
        recruitment_rate_target_quality=result.recruitment_rate_target_quality,
        recruitment_rate_validation=result.recruitment_rate_validation,
        recruitment_rate_confidence_pct=result.recruitment_rate_confidence_pct,
        recruitment_rate_n_train=result.recruitment_rate_n_train,
        enrollment_used=result.enrollment_used,
        num_sites_used=result.num_sites_used,
        enrolment_months=result.enrolment_months,
        followup_months=result.followup_months,
        endpoint_archetypes_used=result.endpoint_archetypes_used,
        endpoint_source=result.endpoint_source,
        endpoint_profile_share=result.endpoint_profile_share,
        endpoint_profile_n=result.endpoint_profile_n,
        rate_note=_RATE_NOTE if result.recruitment_rate is not None else None,
        model_used=result.model_used,
        rmse_days=result.rmse_days,
        n_train=result.n_train,
        extrapolation_warnings=result.extrapolation_warnings,
        provenance=result.provenance,
    )


@router.get("/endpoint-archetypes")
def get_endpoint_archetypes():
    """Vocabulary for the optional endpoint_archetype input."""
    return {"archetypes": [a for a in ARCHETYPES if a != "UNKNOWN"]}
