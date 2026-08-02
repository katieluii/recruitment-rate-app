from __future__ import annotations
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.constants import PHASES, THERAPEUTIC_AREAS
from backend.models.inference import predict
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
            "Primary endpoint type. The strongest single driver of duration "
            "after phase: on Phase 3 the median runs 39.6 months for a survival "
            "endpoint against 10.6 for immunogenicity."
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
    rate_note: Optional[str] = None
    # Duration split into its two stages
    enrolment_months: Optional[float] = None
    followup_months: Optional[float] = None
    # Provenance
    model_used: str
    rmse_days: float
    n_train: int
    extrapolation_warnings: list[str] = []
    # Full working: sources, per-value derivation, input origins, gaps.
    provenance: Optional[dict] = None


_RATE_NOTE = (
    "Patients per site per month, modelled — no per-site enrolment is published "
    "in ClinicalTrials.gov or AACT. The denominator is the full "
    "start-to-primary-completion span, so trials with long follow-up have their "
    "rate understated and `rate_implied_total_months` reconstructs total "
    "duration rather than the enrolment period alone."
)


@router.post("/predict", response_model=PredictResponse)
def post_predict(req: PredictRequest):
    if req.phase not in PHASES:
        raise HTTPException(422, f"Unknown phase '{req.phase}'. Valid: {list(PHASES)}")
    if req.therapeutic_area not in THERAPEUTIC_AREAS:
        raise HTTPException(422, f"Unknown therapeutic area '{req.therapeutic_area}'.")
    if req.endpoint_archetype and req.endpoint_archetype not in ARCHETYPES:
        raise HTTPException(
            422,
            f"Unknown endpoint archetype '{req.endpoint_archetype}'. "
            f"Valid: {list(ARCHETYPES)}",
        )

    try:
        result = predict(
            phase_key=req.phase,
            therapeutic_area=req.therapeutic_area,
            enrollment=req.enrollment,
            num_sites=req.num_sites,
            drug_type=req.drug_type,
            region=req.region,
            endpoint_archetype=req.endpoint_archetype,
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
        enrolment_months=result.enrolment_months,
        followup_months=result.followup_months,
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
