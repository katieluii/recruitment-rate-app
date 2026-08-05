from typing import Optional

from fastapi import APIRouter, HTTPException
from backend.constants import PHASES, THERAPEUTIC_AREAS
from backend.models import registry

router = APIRouter(tags=["meta"])


@router.get("/phases")
def get_phases():
    available = set(registry.available_phases())
    return [
        {
            "key": k,
            "label": v["label"],
            "trained": k in available,
        }
        for k, v in PHASES.items()
    ]


@router.get("/therapeutic-areas")
def get_therapeutic_areas():
    return THERAPEUTIC_AREAS


@router.get("/health")
def health():
    return {"status": "ok", "models": registry.load_all()}


@router.get("/input-ranges")
def get_input_ranges(phase: str = "P2",
                     therapeutic_area: Optional[str] = None):
    """Slider bounds and a sensible starting value for the two live inputs.

    Bounds come from the TRAINED range (p01-p99), not from the API's validation
    limits. `num_sites` accepts up to 5000, but a Phase 1 model has never seen a
    trial with 5000 sites and a tree is flat outside its trained range — that is
    the original site_count defect, where v1 trained on 1..20 and served 40. A
    slider that can only be dragged where the model has evidence prevents the
    user reaching that region by accident rather than warning them afterwards.

    The starting value is the therapeutic area's own median where one exists,
    because an oncology Phase 3 runs a median 89 sites and a dermatology Phase 3
    runs 33; opening both at the same number invents a trial neither resembles.
    """
    if phase not in PHASES:
        raise HTTPException(422, f"Unknown phase '{phase}'")
    entry = registry.load(phase)
    if entry is None:
        raise HTTPException(503, f"No trained model for {phase}.")

    ranges = entry.get("feature_ranges") or {}
    defaults = entry.get("feature_defaults") or {}
    by_ta = (entry.get("meta", {}).get("feature_defaults_by_ta") or {})
    ta_defaults = by_ta.get(therapeutic_area or "", {})

    out = {}
    for field, label in (("Enrollment", "enrollment"), ("site_count", "num_sites")):
        r = ranges.get(field) or {}
        lo = r.get("p01")
        hi = r.get("p99")
        if lo is None or hi is None:
            continue
        start = ta_defaults.get(field, defaults.get(field))
        lo, hi = max(1, int(round(lo))), max(2, int(round(hi)))
        out[label] = {
            "min": lo,
            "max": hi,
            "default": int(round(start)) if start is not None else int((lo + hi) / 2),
            "trained_min": int(round(r.get("min", lo))),
            "trained_max": int(round(r.get("max", hi))),
            "source": ("therapeutic_area_median" if field in ta_defaults
                       else "phase_median"),
        }
    return {"phase": phase, "therapeutic_area": therapeutic_area, "inputs": out}
