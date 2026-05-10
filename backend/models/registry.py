from __future__ import annotations
"""Load and cache trained model pipelines."""
import json
import logging
from pathlib import Path
from typing import Optional

import joblib

from backend.config import settings
from backend.constants import PHASES

log = logging.getLogger(__name__)

_cache: dict[str, dict] = {}


def _artifact_paths(phase_key: str) -> tuple[Path, Path, Path]:
    base = settings.models_dir / phase_key
    return base / "model.pkl", base / "metadata.json", base / "analytics.json"


def load(phase_key: str, force: bool = False) -> Optional[dict]:
    """Return dict with 'pipeline', 'rmse', 'n_train', 'analytics' or None."""
    if phase_key in _cache and not force:
        return _cache[phase_key]

    model_path, meta_path, analytics_path = _artifact_paths(phase_key)

    if not model_path.exists():
        log.warning("No model artifact for %s at %s", phase_key, model_path)
        return None

    pipeline = joblib.load(model_path)
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    analytics = json.loads(analytics_path.read_text()) if analytics_path.exists() else {}

    entry = {
        "pipeline": pipeline,
        "rmse": meta.get("rmse", 0.0),
        "n_train": meta.get("n_train", 0),
        "analytics": analytics,
    }
    _cache[phase_key] = entry
    log.info("Loaded model for %s (RMSE=%.1f, n=%d)", phase_key, entry["rmse"], entry["n_train"])
    return entry


def load_all() -> dict[str, bool]:
    return {k: (load(k) is not None) for k in PHASES}


def available_phases() -> list[str]:
    return [k for k in PHASES if (settings.models_dir / k / "model.pkl").exists()]
