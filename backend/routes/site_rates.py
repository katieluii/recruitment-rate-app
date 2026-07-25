from __future__ import annotations
"""Site-level recruitment endpoints."""
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from backend.analytics import site_rates as sr
from backend.constants import PHASES, THERAPEUTIC_AREAS
from backend.models import registry

router = APIRouter(tags=["site-rates"])

_CAVEAT = (
    "Modelled from trial-level rates, not observed per-site enrolment — "
    "neither ClinicalTrials.gov nor AACT publishes per-site counts. Facility "
    "figures are association, not attribution: a site that appears in fast "
    "trials scores well, which may reflect the trials it is chosen for."
)


def _priors(phase: str) -> dict:
    if phase not in PHASES:
        raise HTTPException(422, f"Unknown phase '{phase}'. Valid: {list(PHASES)}")
    entry = registry.load(phase)
    if entry is None:
        raise HTTPException(503, f"No trained artifacts for {phase}.")
    priors = entry.get("site_priors") or {}
    if not priors.get("countries"):
        raise HTTPException(503, f"No site priors for {phase}. Retrain to build them.")
    return priors


class SiteMixRequest(BaseModel):
    phase: str = Field(..., examples=["P3"])
    target_enrollment: int = Field(..., ge=1, le=100000)
    site_mix: dict[str, int] = Field(
        ..., examples=[{"United States": 40, "Poland": 20, "Japan": 10}],
        description="country → number of planned sites")
    therapeutic_area: Optional[str] = Field(None, examples=["Oncology/Solid Tumours"])


@router.post("/site-rates/simulate")
def post_simulate(req: SiteMixRequest):
    """Projected enrolment window for a proposed country/site mix."""
    if req.therapeutic_area and req.therapeutic_area not in THERAPEUTIC_AREAS:
        raise HTTPException(422, f"Unknown therapeutic area '{req.therapeutic_area}'.")
    priors = _priors(req.phase)
    try:
        result = sr.simulate_site_mix(
            priors, req.target_enrollment, req.site_mix, req.therapeutic_area)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    return {**result, "phase": req.phase, "caveat": _CAVEAT}


@router.get("/site-rates/countries")
def get_countries(
    phase: str = Query(..., examples=["P3"]),
    therapeutic_area: Optional[str] = Query(None),
    limit: int = Query(15, ge=1, le=100),
    min_trials: int = Query(10, ge=1),
):
    """Countries ranked by modelled recruitment rate."""
    priors = _priors(phase)
    return {
        "phase": phase,
        "therapeutic_area": therapeutic_area,
        "countries": sr.top_countries(priors, therapeutic_area, limit, min_trials),
        "caveat": _CAVEAT,
    }


@router.get("/site-rates/facilities")
def get_facilities(
    phase: str = Query(..., examples=["P3"]),
    country: Optional[str] = Query(None),
    limit: int = Query(25, ge=1, le=200),
):
    """Facility track record, ranked by number of trials participated in."""
    priors = _priors(phase)
    rows = [{"facility": k, **v} for k, v in (priors.get("facilities") or {}).items()]
    if country:
        rows = [r for r in rows if country in (r.get("countries") or [])]
    rows.sort(key=lambda r: -r["n_trials"])
    return {
        "phase": phase,
        "country": country,
        "facilities": rows[:limit],
        "caveat": _CAVEAT,
    }
