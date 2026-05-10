from fastapi import APIRouter, HTTPException, Query

from backend.constants import PHASES
from backend.models import registry

router = APIRouter(tags=["analytics"])


@router.get("/analytics")
def get_analytics(phase: str = Query("P2")):
    if phase not in PHASES:
        raise HTTPException(422, f"Unknown phase '{phase}'")

    entry = registry.load(phase)
    if entry is None:
        raise HTTPException(503, f"No trained model for {phase}. Run training first.")

    analytics = entry["analytics"]
    if not analytics:
        raise HTTPException(503, "Analytics data not yet computed.")

    # Sort by median duration descending
    rows = [
        {"therapeutic_area": ta, **stats}
        for ta, stats in analytics.items()
    ]
    rows.sort(key=lambda r: r["median"], reverse=True)
    return {"phase": phase, "phase_label": PHASES[phase]["label"], "data": rows}
