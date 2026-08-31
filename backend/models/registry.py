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
from backend.models.quantile_model import (POINT_KEY, QUANTILES, TRANSFORMS,
                                           slot_name)

log = logging.getLogger(__name__)

_cache: dict[str, dict] = {}

HEAD_NAMES = ("duration", "rate")


class LoadedTwoStage:
    """Duration reassembled from its two stages.

    Presents the same predict/predict_interval surface as a single head, so
    inference does not have to know which shape it was handed.
    """

    def __init__(self, enrol: "LoadedHead", fu: "LoadedHead", band_scale: float):
        self.enrol = enrol
        self.fu = fu
        self.scale_ = band_scale

    def _raw_band(self, X):
        import numpy as np
        e_lo, e_hi = self.enrol.predict_interval(X)
        f_lo, f_hi = self.fu.predict_interval(X)
        e_mid, f_mid = self.enrol.predict(X), self.fu.predict(X)
        point = e_mid + f_mid
        # Quadrature, not addition: the stages are near-independent.
        lo_w = np.sqrt((e_mid - e_lo) ** 2 + (f_mid - f_lo) ** 2)
        hi_w = np.sqrt((e_hi - e_mid) ** 2 + (f_hi - f_mid) ** 2)
        return point, lo_w, hi_w

    def predict(self, X):
        import numpy as np
        return np.maximum(1.0, self.enrol.predict(X) + self.fu.predict(X))

    def predict_components(self, X):
        return self.enrol.predict(X), self.fu.predict(X)

    def predict_interval(self, X):
        import numpy as np
        point, lo_w, hi_w = self._raw_band(X)
        return (np.maximum(1.0, point - self.scale_ * lo_w),
                np.maximum(1.0, point + self.scale_ * hi_w))


class LoadedHybrid:
    """Forest point estimate inside the two-stage calibrated band (metadata kind
    "hybrid"). Mirrors experiments.candidates.HybridForestPoint: point from the
    forest, band = point ± scale × the two-stage raw half-widths, components =
    the two-stage split rescaled to the forest total."""

    def __init__(self, two: "LoadedTwoStage", forest, band_scale: float,
                 band_kind: str = "two_stage", forest_rmse: float = 0.0):
        self.two = two
        self.forest = forest
        self.scale_ = band_scale
        self.band_kind = band_kind
        self.forest_rmse = forest_rmse
        self.enrol, self.fu = two.enrol, two.fu

    def _spread(self, X):
        """Forest tree spread, floored at half the training rmse (V1Recipe.spread)."""
        import numpy as np
        rf = self.forest.named_steps["model"]
        Xt = self.forest.named_steps["pre"].transform(X)
        std = np.stack([t.predict(Xt) for t in rf.estimators_]).std(axis=0)
        return np.maximum(std, self.forest_rmse * 0.5)

    def predict(self, X):
        import numpy as np
        return np.maximum(1.0, self.forest.predict(X))

    def predict_interval(self, X):
        import numpy as np
        point = self.predict(X)
        if self.band_kind == "forest":
            s = self._spread(X); lo_w, hi_w = s, s
        else:
            _, lo_w, hi_w = self.two._raw_band(X)
        return (np.maximum(1.0, point - self.scale_ * lo_w),
                np.maximum(1.0, point + self.scale_ * hi_w))

    def predict_components(self, X):
        import numpy as np
        e, f = self.two.predict_components(X)
        k = self.predict(X) / np.maximum(1.0, e + f)
        return e * k, f * k


class LoadedHead:
    """A fitted head reassembled from disk, with the same predict surface as
    ConformalQuantileModel so callers do not care which they were handed."""

    def __init__(self, models: dict, qhat: float, transform: str,
                 point_transform: Optional[str] = None):
        self.models = models
        self.qhat_ = qhat
        self.transform = transform
        self._fwd, self._inv, self._floor = TRANSFORMS[transform]
        # The point head may have been fitted in a different target space from
        # the quantile heads (metadata "point_transform"; None = same space).
        # Inverting a raw-day head through expm1 would serve e^500 days.
        self.point_transform = point_transform or transform
        self._inv_point = TRANSFORMS[self.point_transform][1]

    def predict(self, X):
        import numpy as np
        # Prefer the L2 head. R2 is minimised by the conditional mean, and the
        # alpha=0.5 head fits the median, so serving the median would give away
        # the gain the L2 head was added for. Falls back to the median for
        # artifacts trained before the point head existed.
        head = self.models.get(POINT_KEY)
        if head is not None:
            return np.maximum(self._floor, self._inv_point(head.predict(X)))
        return np.maximum(self._floor, self._inv(self.models[0.5].predict(X)))

    def predict_interval(self, X):
        import numpy as np
        lo = self.models[0.1].predict(X) - self.qhat_
        hi = self.models[0.9].predict(X) + self.qhat_
        lower = np.maximum(self._floor, self._inv(lo))
        upper = np.maximum(self._floor, self._inv(hi))
        return np.minimum(lower, upper), np.maximum(lower, upper)


def _load_point(base: Path, prefix: str, models: dict) -> None:
    """Attach the L2 point head if this artifact has one. Optional by design:
    artifacts predating it stay loadable and serve from the median."""
    path = base / f"{prefix}_{POINT_KEY}.pkl"
    if path.exists():
        models[POINT_KEY] = joblib.load(path)


def _load_stage(base: Path, stage: str, transform: str,
                point_transform: Optional[str] = None) -> Optional[LoadedHead]:
    models = {}
    for alpha in QUANTILES:
        path = base / f"{stage}_{slot_name(alpha)}.pkl"
        if not path.exists():
            log.warning("Missing %s — stage '%s' unavailable", path, stage)
            return None
        models[alpha] = joblib.load(path)
    _load_point(base, stage, models)
    return LoadedHead(models, 0.0, transform, point_transform)


def _load_head(base: Path, head: str, meta: dict):
    spec = (meta.get("heads") or {}).get(head)
    if not spec:
        return None

    if spec.get("kind") == "two_stage":
        transform = spec.get("transform", "log1p")
        point_transform = spec.get("point_transform")
        enrol = _load_stage(base, "enrolment", transform, point_transform)
        fu = _load_stage(base, "followup", transform, point_transform)
        if enrol is None or fu is None:
            return None
        return LoadedTwoStage(enrol, fu, spec.get("band_scale", 1.0))
    if spec.get("kind") == "hybrid":
        transform = spec.get("transform", "log1p")
        point_transform = spec.get("point_transform")
        enrol = _load_stage(base, "enrolment", transform, point_transform)
        fu = _load_stage(base, "followup", transform, point_transform)
        forest_path = base / spec.get("point_model", "forest_point.pkl")
        if enrol is None or fu is None or not forest_path.exists():
            log.warning("Hybrid head for %s incomplete (forest at %s: %s)", base.name,
                        forest_path, forest_path.exists())
            return None
        two = LoadedTwoStage(enrol, fu, spec.get("two_stage_band_scale", 1.0))
        return LoadedHybrid(two, joblib.load(forest_path), spec.get("band_scale", 1.0),
                            spec.get("band_kind", "two_stage"), float(spec.get("forest_rmse", 0.0)))
    models = {}
    for alpha in QUANTILES:
        path = base / f"{head}_{slot_name(alpha)}.pkl"
        if not path.exists():
            log.warning("Missing %s — head '%s' unavailable", path, head)
            return None
        models[alpha] = joblib.load(path)
    _load_point(base, head, models)
    return LoadedHead(models, spec.get("qhat", 0.0), spec.get("transform", "log1p"),
                      spec.get("point_transform"))


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

    priors_path = base / "site_priors.json"
    site_priors = json.loads(priors_path.read_text()) if priors_path.exists() else {}

    # Most common primary-endpoint COMBINATIONS per therapeutic area. Absent on
    # artifacts built before 2026-08-03; the route reports the gap rather than
    # inventing a profile, so an un-regenerated phase is visible instead of silent.
    profiles_path = base / "endpoint_profiles.json"
    endpoint_profiles = (json.loads(profiles_path.read_text())
                         if profiles_path.exists() else {})

    entry = {
        "heads": heads,
        "site_priors": site_priors,
        "endpoint_profiles": endpoint_profiles,
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
