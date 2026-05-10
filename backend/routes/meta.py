from fastapi import APIRouter
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
