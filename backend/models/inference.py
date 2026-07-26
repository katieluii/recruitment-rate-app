from __future__ import annotations
"""Prediction: duration and recruitment rate."""
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from backend.constants import PHASES, REGIONS, THERAPEUTIC_AREAS
from backend.models import registry
from backend.preprocessing.pipeline import build_features

log = logging.getLogger(__name__)

_DAYS_PER_MONTH = 30.44


@dataclass
class Prediction:
    phase_key: str
    therapeutic_area: str
    # Head A — total duration, start to primary completion
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
    # Head B — recruitment rate, patients per site per month
    recruitment_rate: Optional[float] = None
    recruitment_rate_lower: Optional[float] = None
    recruitment_rate_upper: Optional[float] = None
    # NOT an enrolment window: the rate's denominator is the full
    # start → primary-completion span, so inverting it reconstructs total
    # duration for a trial of this size.
    rate_implied_total_months: Optional[float] = None
    rate_is_approximate: bool = True
    # The V3.3 split. Enrolment window is when the last patient is in; follow-up
    # is how long the endpoint then takes to read out. They are near-independent
    # processes, and only the first is something a site strategy can move.
    enrolment_months: Optional[float] = None
    followup_months: Optional[float] = None
    # How every number above was produced — see backend/models/provenance.py
    provenance: Optional[dict] = None


def _phase_raw(phase_key: str) -> str:
    return {"P1HV": "PHASE1", "P1": "PHASE1", "P2": "PHASE2", "P3": "PHASE3"}[phase_key]


def _build_input_row(phase_key: str, therapeutic_area: str,
                     enrollment: Optional[int], num_sites: Optional[int],
                     drug_type: str, region: str,
                     endpoint_archetype: Optional[str] = None,
                     followup_months: Optional[float] = None) -> pd.DataFrame:
    """Single-row frame matching build_features output.

    Unspecified fields are left NaN so they fall back to the training-set
    defaults stored in the artifact. v1 hardcoded them here — Allocation
    "RANDOMIZED", Masking "DOUBLE", primary_completion_year 2024 — so the only
    thing that varied between requests for two different therapeutic areas was
    22 sparse binary columns, and the model barely used them.
    """
    row = {
        "conditions": therapeutic_area,   # already a canonical TA label
        "countries": region,              # drives the region one-hot
        "brief_summary": "",
        "Phases": _phase_raw(phase_key),
        "Enrollment": enrollment,
        "site_count": num_sites,
        "country_count": 1,
        "followup_months": followup_months,
        "Drug_Type": drug_type,
        "is_hv": int(PHASES[phase_key]["hv"]),
    }
    if endpoint_archetype:
        row["primary_outcome_measures"] = ""
    df = pd.DataFrame([row])
    X = build_features(df, phase_key)
    if endpoint_archetype:
        X["endpoint_archetype"] = endpoint_archetype
        flag = f"endpoint_has_{endpoint_archetype}"
        if flag in X.columns:
            X[flag] = 1
    return _apply_defaults(X, phase_key, therapeutic_area)


def _apply_defaults(X: pd.DataFrame, phase_key: str,
                    therapeutic_area: str | None = None) -> pd.DataFrame:
    """Fill unspecified features, preferring the therapeutic area's own median.

    Asking for "a Phase 3 oncology trial" without giving an enrolment or site
    count should assume an oncology-shaped trial (median 502 patients across 89
    sites), not a phase-average one (334 across 32). Falling back to a single
    phase-wide median flattens two genuinely different trials before the model
    is consulted.
    """
    entry = registry.load(phase_key) or {}
    defaults = dict(entry.get("feature_defaults") or {})
    if not defaults:
        return X

    ta_defaults = (entry.get("meta", {}).get("feature_defaults_by_ta") or {})
    if therapeutic_area and therapeutic_area in ta_defaults:
        for col, val in ta_defaults[therapeutic_area].items():
            if col != "n":
                defaults[col] = val

    for col in X.columns:
        if col not in defaults or defaults[col] is None:
            continue
        if X[col].isna().all():
            X[col] = defaults[col]
        elif X[col].dtype == object:
            X[col] = X[col].replace("UNKNOWN", defaults[col])
    return X


def _check_extrapolation(X: pd.DataFrame, phase_key: str) -> list[str]:
    """Flag numeric inputs outside the range the model was trained on.

    The guard for the original failure: v1 trained site_count on 1..20 (it was
    really a country count) and served requests at 40+, where a forest returns
    a constant and every therapeutic area collapses onto the same answer.
    """
    entry = registry.load(phase_key) or {}
    ranges = entry.get("feature_ranges") or {}
    out: list[str] = []
    for col, bounds in ranges.items():
        if col not in X.columns:
            continue
        val = X[col].iloc[0]
        if pd.isna(val) or not isinstance(val, (int, float, np.integer, np.floating)):
            continue
        if val < bounds["min"] or val > bounds["max"]:
            out.append(
                f"{col}={val:g} is outside the trained range "
                f"[{bounds['min']:g}, {bounds['max']:g}] — this prediction is an "
                f"extrapolation and should be treated as indicative only"
            )
    return out


def predict(
    phase_key: str,
    therapeutic_area: str,
    enrollment: Optional[int] = None,
    num_sites: Optional[int] = None,
    drug_type: str = "DRUG",
    region: str = "US",
    endpoint_archetype: Optional[str] = None,
) -> Prediction:
    entry = registry.load(phase_key)
    if entry is None:
        raise FileNotFoundError(
            f"No trained model found for {phase_key}. "
            "Run scripts/train_models.py first."
        )
    heads = entry["heads"]
    if "duration" not in heads:
        raise FileNotFoundError(f"No duration head trained for {phase_key}.")

    X = _build_input_row(phase_key, therapeutic_area, enrollment, num_sites,
                         drug_type, region, endpoint_archetype)
    warnings = _check_extrapolation(X, phase_key)
    for w in warnings:
        log.warning("%s: %s", phase_key, w)

    duration = heads["duration"]
    pred_days = float(duration.predict(X)[0])
    lo_arr, hi_arr = duration.predict_interval(X)
    lower, upper = float(lo_arr[0]), float(hi_arr[0])

    enrol_m = fu_m = None
    if hasattr(duration, "predict_components"):
        e, f = duration.predict_components(X)
        enrol_m = round(float(e[0]) / _DAYS_PER_MONTH, 1)
        fu_m = round(float(f[0]) / _DAYS_PER_MONTH, 1)

    rate = rate_lo = rate_hi = implied_months = None
    if "rate" in heads:
        rate = float(heads["rate"].predict(X)[0])
        r_lo, r_hi = heads["rate"].predict_interval(X)
        rate_lo, rate_hi = float(r_lo[0]), float(r_hi[0])
        sites = num_sites or X["site_count"].iloc[0]
        target_n = enrollment or X["Enrollment"].iloc[0]
        if rate > 0 and sites and target_n:
            implied_months = round(float(target_n) / (float(sites) * rate), 1)

    def to_months(d: float) -> float:
        return round(d / _DAYS_PER_MONTH, 1)

    result = Prediction(
        phase_key=phase_key,
        therapeutic_area=therapeutic_area,
        predicted_days=round(pred_days, 1),
        lower_days=round(lower, 1),
        upper_days=round(upper, 1),
        predicted_months=to_months(pred_days),
        lower_months=to_months(lower),
        upper_months=to_months(upper),
        model_used="LightGBM conformalised quantile",
        rmse_days=round(entry.get("rmse", 0.0), 1),
        n_train=entry["n_train"],
        extrapolation_warnings=warnings,
        recruitment_rate=round(rate, 3) if rate is not None else None,
        recruitment_rate_lower=round(rate_lo, 3) if rate_lo is not None else None,
        recruitment_rate_upper=round(rate_hi, 3) if rate_hi is not None else None,
        rate_implied_total_months=implied_months,
        enrolment_months=enrol_m,
        followup_months=fu_m,
    )

    from backend.models import provenance as _prov

    result.provenance = _prov.build(
        phase_key, result,
        supplied={"enrollment": enrollment, "num_sites": num_sites,
                  "drug_type": drug_type, "region": region,
                  "endpoint_archetype": endpoint_archetype},
        therapeutic_area=therapeutic_area,
    )
    return result
