from __future__ import annotations
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.constants import PHASES, THERAPEUTIC_AREAS
from backend.models.inference import predict

router = APIRouter(tags=["predict"])


class PredictRequest(BaseModel):
    phase: str = Field(..., examples=["P2"])
    therapeutic_area: str = Field(..., examples=["Oncology/Solid Tumours"])
    enrollment: Optional[int] = Field(None, ge=1, le=50000)
    num_sites: Optional[int] = Field(None, ge=1, le=5000)
    drug_type: str = Field("DRUG", pattern="^(DRUG|BIOLOGICAL)$")
    region: str = Field("US")


class PredictResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    phase_key: str
    phase_label: str
    therapeutic_area: str
    predicted_months: float
    lower_months: float
    upper_months: float
    predicted_days: float
    lower_days: float
    upper_days: float
    model_used: str
    rmse_days: float
    n_train: int
    confidence_pct: int


@router.post("/predict", response_model=PredictResponse)
def post_predict(req: PredictRequest):
    if req.phase not in PHASES:
        raise HTTPException(422, f"Unknown phase '{req.phase}'. Valid: {list(PHASES)}")
    if req.therapeutic_area not in THERAPEUTIC_AREAS:
        raise HTTPException(422, f"Unknown therapeutic area '{req.therapeutic_area}'.")

    try:
        result = predict(
            phase_key=req.phase,
            therapeutic_area=req.therapeutic_area,
            enrollment=req.enrollment,
            num_sites=req.num_sites,
            drug_type=req.drug_type,
            region=req.region,
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
        model_used=result.model_used,
        rmse_days=result.rmse_days,
        n_train=result.n_train,
        confidence_pct=result.confidence_pct,
    )
