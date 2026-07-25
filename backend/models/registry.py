from __future__ import annotations
"""Load and cache trained model artifacts.

Each phase directory holds two heads — `duration` and `rate` — with three
quantile models apiece, plus a metadata file carrying the conformal widening
constant, the training-set feature defaults and the trained feature ranges.
"""
import json
import logging
from pathlib import Path
from typing import Optional

import joblib

from backend.config import settings
from backend.constants import PHASES
from backend.models.quantile_model import QUANTILES, TRANSFORMS

log = logging.getLogger(__name__)

_cache: dict[str, dict] = {}

HEAD_NAMES = ("duration", "rate")


class LoadedHead:
    """A fitted head reassembled from disk, with the same predict surface as
    ConformalQuantileModel so callers do not care which they were handed."""

    def __init__(self, models: dict, qhat: float, transform: str):
        self.models = models
        self.qhat_ = qhat
        self.transform = transform
        self._fwd, self._inv, self._floor = TRANSFORMS[transform]

    def predict(self, X):
        import numpy as np
        return np.maximum(self._floor, self._inv(self.models[0.5].predict(X)))

    def predict_interval(self, X):
        import numpy as np
        lo = self.models[0.1].predict(X) - self.qhat_
        hi = self.models[0.9].predict(X) + self.qhat_
        lower = np.maximum(self._floor, self._inv(lo))
        upper = np.maximum(self._floor, self._inv(hi))
        return np.minimum(lower, upper), np.maximum(lower, upper)


def _load_head(base: Path, head: str, meta: dict) -> Optional[LoadedHead]:
    spec = (meta.get("heads") or {}).get(head)
    if not spec:
        return None
    models = {}
    for alpha in QUANTILES:
        path = base / f"{head}_q{int(alpha * 100)}.pkl"
        if not path.exists():
            log.warning("Missing %s — head '%s' unavailable", path, head)
            return None
        models[alpha] = joblib.load(path)
    return LoadedHead(models, spec.get("qhat", 0.0), spec.get("transform", "log1p"))


def load(phase_key: str, force: bool = False) -> Optional[dict]:
    """Return the loaded artifact bundle for a phase, or None if absent."""
    if phase_key in _cache and not force:
        return _cache[phase_key]

    base = settings.models_dir / phase_key
    meta_path = base / "metadata.json"
    if not meta_path.exists():
        log.warning("No metadata for %s at %s", phase_key, meta_path)
        return None

    meta = json.loads(meta_path.read_text())
    analytics_path = base / "analytics.json"
    analytics = json.loads(analytics_path.read_text()) if analytics_path.exists() else {}

    heads = {h: _load_head(base, h, meta) for h in HEAD_NAMES}
    heads = {k: v for k, v in heads.items() if v is not None}
    if not heads:
        log.warning("No usable heads for %s", phase_key)
        return None

    entry = {
        "heads": heads,
        "rmse": meta.get("rmse", 0.0),
        "n_train": meta.get("n_train", 0),
        "analytics": analytics,
        "feature_defaults": meta.get("feature_defaults", {}),
        "feature_ranges": meta.get("feature_ranges", {}),
        "meta": meta,
    }
    _cache[phase_key] = entry
    log.info("Loaded %s: heads=%s n_train=%d",
             phase_key, list(heads), entry["n_train"])
    return entry


def load_all() -> dict[str, bool]:
    return {k: (load(k) is not None) for k in PHASES}


def available_phases() -> list[str]:
    return [k for k in PHASES if (settings.models_dir / k / "metadata.json").exists()]
