from __future__ import annotations
"""Prediction + uncertainty estimation."""
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from backend.constants import THERAPEUTIC_AREAS, REGIONS, PHASES
from backend.models import registry
from backend.preprocessing.pipeline import build_features

log = logging.getLogger(__name__)

_DAYS_PER_MONTH = 30.44


@dataclass
class Prediction:
    phase_key: str
    therapeutic_area: str
    predicted_days: float
    lower_days: float
    upper_days: float
    predicted_months: float
    lower_months: float
    upper_months: float
    model_used: str
    rmse_days: float
    n_train: int
    confidence_pct: int = 80
    extrapolation_warnings: list[str] = field(default_factory=list)


def _build_input_row(phase_key: str, therapeutic_area: str,
                     enrollment: Optional[int], num_sites: Optional[int],
                     drug_type: str, region: str,
                     followup_months: Optional[float] = None) -> pd.DataFrame:
    """Construct a single-row DataFrame that matches build_features output.

    Unspecified fields are left NaN so build_features and the fitted imputer
    fill them from the training distribution. v1 hardcoded them here
    (Allocation="RANDOMIZED", Masking="DOUBLE", primary_completion_year=2024),
    which meant the only thing varying between two requests for different
    therapeutic areas was 22 sparse binary columns.
    """
    hv = int(PHASES[phase_key]["hv"])

    # Pipe-separated conditions / countries strings to drive TA + region OHE
    conditions_str = therapeutic_area  # single area (no pipe)
    countries_str = region              # single region (no pipe, used in assign_region)

    row = {
        "conditions": conditions_str,
        "countries": countries_str,
        "brief_summary": "",
        "Phases": _phase_raw(phase_key),
        "Enrollment": enrollment,
        "site_count": num_sites,
        "country_count": 1,
        "followup_months": followup_months,
        "Drug_Type": drug_type,
        "is_hv": hv,
    }
    df = pd.DataFrame([row])
    X = build_features(df, phase_key)
    return _apply_defaults(X, phase_key)


def _apply_defaults(X: pd.DataFrame, phase_key: str) -> pd.DataFrame:
    """Fill unspecified features from the training-set defaults in the artifact."""
    entry = registry.load(phase_key) or {}
    defaults = entry.get("feature_defaults") or {}
    if not defaults:
        return X
    for col in X.columns:
        if col not in defaults or defaults[col] is None:
            continue
        if X[col].isna().all():
            X[col] = defaults[col]
        elif X[col].dtype == object:
            X[col] = X[col].replace("UNKNOWN", defaults[col])
    return X


def _check_extrapolation(X: pd.DataFrame, phase_key: str) -> list[str]:
    """Flag any numeric input outside the range the model was trained on.

    This is the guard for the failure that produced the flat predictions: the
    model was trained with site_count in 1..20 and served requests at 40+, far
    outside the trained range, where a forest returns a constant.
    """
    entry = registry.load(phase_key) or {}
    ranges = entry.get("feature_ranges") or {}
    warnings: list[str] = []
    for col, bounds in ranges.items():
        if col not in X.columns:
            continue
        val = X[col].iloc[0]
        if pd.isna(val) or not isinstance(val, (int, float, np.integer, np.floating)):
            continue
        if val < bounds["min"] or val > bounds["max"]:
            warnings.append(
                f"{col}={val:g} is outside the trained range "
                f"[{bounds['min']:g}, {bounds['max']:g}] — prediction is an "
                f"extrapolation and should be treated as indicative only"
            )
    return warnings


def _phase_raw(phase_key: str) -> str:
    mapping = {"P1HV": "PHASE1", "P1": "PHASE1", "P2": "PHASE2", "P3": "PHASE3"}
    return mapping[phase_key]


def predict(
    phase_key: str,
    therapeutic_area: str,
    enrollment: Optional[int] = None,
    num_sites: Optional[int] = None,
    drug_type: str = "DRUG",
    region: str = "US",
) -> Prediction:
    entry = registry.load(phase_key)
    if entry is None:
        raise FileNotFoundError(
            f"No trained model found for {phase_key}. "
            "Run scripts/train_models.py first."
        )

    pipeline = entry["pipeline"]
    rmse = entry["rmse"]

    X = _build_input_row(phase_key, therapeutic_area, enrollment, num_sites, drug_type, region)
    warnings = _check_extrapolation(X, phase_key)
    for w in warnings:
        log.warning("%s: %s", phase_key, w)

    # Point prediction
    pred_days = float(pipeline.predict(X)[0])

    # Uncertainty from individual tree predictions (RF tree variance)
    rf = pipeline.named_steps["model"]
    pre = pipeline.named_steps["pre"]
    X_transformed = pre.transform(X)
    tree_preds = np.array([t.predict(X_transformed)[0] for t in rf.estimators_])
    tree_std = float(tree_preds.std())

    # Use tree std if meaningful, else fall back to RMSE-based interval
    std_used = max(tree_std, rmse * 0.5)
    z = 1.28  # 80% CI
    lower = max(1.0, pred_days - z * std_used)
    upper = pred_days + z * std_used

    def to_months(d: float) -> float:
        return round(d / _DAYS_PER_MONTH, 1)

    return Prediction(
        phase_key=phase_key,
        therapeutic_area=therapeutic_area,
        predicted_days=round(pred_days, 1),
        lower_days=round(lower, 1),
        upper_days=round(upper, 1),
        predicted_months=to_months(pred_days),
        lower_months=to_months(lower),
        upper_months=to_months(upper),
        model_used="RandomForest",
        rmse_days=round(rmse, 1),
        n_train=entry["n_train"],
        extrapolation_warnings=warnings,
    )
