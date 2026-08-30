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
    #: The separately fitted rate model's answer, kept as a visible cross-check.
    recruitment_rate_crosscheck: Optional[float] = None
    # The V3.3 split. Enrolment window is when the last patient is in; follow-up
    # is how long the endpoint then takes to read out. They are near-independent
    # processes, and only the first is something a site strategy can move.
    enrolment_months: Optional[float] = None
    followup_months: Optional[float] = None
    # How every number above was produced — see backend/models/provenance.py
    provenance: Optional[dict] = None


def _phase_raw(phase_key: str) -> str:
    return {"P1HV": "PHASE1", "P1": "PHASE1", "P2": "PHASE2", "P3": "PHASE3"}[phase_key]


#: The eligibility features a caller may set, and the ONLY route by which a
#: patient-population description reaches the model. `criteria_text` is
#: deliberately absent: build_features produces it and the fitted preprocessor
#: drops it (no text block), so accepting it would imply an effect it cannot have.
ELIGIBILITY_NUMERIC = ("n_inclusion_criteria", "n_exclusion_criteria",
                       "criteria_chars")


def eligibility_fields() -> tuple:
    """Allowlist for `eligibility_features`, derived rather than restated so it
    cannot drift from the markers the pipeline actually builds."""
    from backend.preprocessing.text_features import CRITERIA_MARKERS

    return ELIGIBILITY_NUMERIC + tuple(f"crit_{n}" for n in CRITERIA_MARKERS)


def _build_input_row(phase_key: str, therapeutic_area: str,
                     enrollment: Optional[int], num_sites: Optional[int],
                     drug_type: str, region: str,
                     endpoint_archetypes: Optional[list] = None,
                     followup_months: Optional[float] = None,
                     eligibility_features: Optional[dict] = None) -> pd.DataFrame:
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
    if endpoint_archetypes:
        row["primary_outcome_measures"] = ""
    df = pd.DataFrame([row])
    X = build_features(df, phase_key)
    if endpoint_archetypes:
        # One flag per archetype. Setting exactly one was the interface being
        # narrower than the model: training rows carry a multi-hot set (21.8% of
        # trials light more than one), so a served row with a single flag sat in
        # a region the model rarely saw, and a RESPONSE+SAFETY trial could not be
        # asked about at all — it came back as whichever half was sent.
        #
        # The categorical takes the FIRST element, matching `classify_primary`'s
        # first-parseable convention in training rather than inventing a new one.
        X["endpoint_archetype"] = endpoint_archetypes[0]
        for archetype in endpoint_archetypes:
            flag = f"endpoint_has_{archetype}"
            if flag in X.columns:
                X[flag] = 1

    # Eligibility arrives as feature VALUES, not a label. The crit_* markers are
    # derived from criteria text inside build_features, so they are set after it
    # rather than through the row; the three numerics could go either way and are
    # set here too so one code path owns the whole block.
    if eligibility_features:
        for field, value in eligibility_features.items():
            if field in X.columns and value is not None:
                X[field] = value

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
    """Flag numeric inputs outside the range the model has real evidence for.

    The guard for the original failure: v1 trained site_count on 1..20 (it was
    really a country count) and served requests at 40+, where a forest returns
    a constant and every therapeutic area collapses onto the same answer.

    Bounds are the trained p01-p99, NOT min-max. Min-max lets a single outlier
    answer for the whole feature: one Phase 3 trial enrolled 7,702 patients at
    one site, which stretched `enrollment_per_site`'s "valid" band to
    [0.21, 7701.9] and let 123 patients-per-site pass without a word — a trial
    design that does not exist. That feature's p01-p99 band is [1.15, 1069] and
    was already being recorded and ignored. A control that cannot reject
    anything is indistinguishable from one that works.

    Values between p99 and max are real but rare, so they get a softer note
    rather than the same warning: the model HAS seen them, just barely.
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

        lo, hi = bounds.get("p01"), bounds.get("p99")
        if lo is None or hi is None:   # artifacts predating the p01/p99 record
            lo, hi = bounds["min"], bounds["max"]

        if val < bounds["min"] or val > bounds["max"]:
            out.append(
                f"{col}={val:g} is outside the trained range "
                f"[{bounds['min']:g}, {bounds['max']:g}]: extrapolated, indicative only"
            )
        elif val < lo or val > hi:
            out.append(
                f"{col}={val:g} is in the sparse tail (typical range "
                f"[{lo:g}, {hi:g}], under 2% of trials): indicative only"
            )
    return out


def predict(
    phase_key: str,
    therapeutic_area: str,
    enrollment: Optional[int] = None,
    num_sites: Optional[int] = None,
    drug_type: str = "DRUG",
    region: str = "US",
    endpoint_archetypes: Optional[list] = None,
    eligibility_features: Optional[dict] = None,
    endpoint_archetype: Optional[str] = None,
) -> Prediction:
    """Predict duration and recruitment rate for a described trial.

    `endpoint_archetype` (singular) is kept as an alias for the first element of
    `endpoint_archetypes`. It is not decoration: renaming the parameter broke a
    live WS21 session that calls this function in-process, and took its duration
    card down on every cell. In-process callers have no HTTP contract to shield
    them, so the Python signature is the contract.
    """
    if endpoint_archetype and not endpoint_archetypes:
        endpoint_archetypes = [endpoint_archetype]

    entry = registry.load(phase_key)
    if entry is None:
        raise FileNotFoundError(f"No trained model for {phase_key}.")
    heads = entry["heads"]
    if "duration" not in heads:
        raise FileNotFoundError(f"No duration head trained for {phase_key}.")

    X = _build_input_row(phase_key, therapeutic_area, enrollment, num_sites,
                         drug_type, region, endpoint_archetypes,
                         eligibility_features=eligibility_features)
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

    # The recruitment rate is DERIVED from the enrolment window rather than
    # predicted separately. Two independently fitted models produced two answers
    # to the same question, and in the all-defaults case they disagreed badly
    # (P3 infectious disease: 21.1 months from the enrolment head against 13.0
    # implied by the rate) because medians do not compose — the median of a ratio
    # is not the ratio of medians. On real trials the two agree closely (log
    # correlation +0.82 to +0.97) and are equally accurate, so nothing is lost by
    # collapsing them, and consistency becomes structural rather than hoped for.
    rate = rate_lo = rate_hi = implied_months = None
    rate_crosscheck = None
    sites = num_sites or X["site_count"].iloc[0]
    target_n = enrollment or X["Enrollment"].iloc[0]

    if enrol_m and sites and target_n and enrol_m > 0:
        denom = float(sites) * enrol_m
        rate = round(float(target_n) / denom, 3)
        # Bounds invert: a longer window means a slower rate.
        e_lo = max(float(lo_arr[0]) / _DAYS_PER_MONTH - fu_m, 0.1) if fu_m is not None else None
        e_hi = max(float(hi_arr[0]) / _DAYS_PER_MONTH - fu_m, 0.1) if fu_m is not None else None
        if e_lo and e_hi:
            rate_lo = round(float(target_n) / (float(sites) * max(e_hi, 0.1)), 3)
            rate_hi = round(float(target_n) / (float(sites) * max(e_lo, 0.1)), 3)
        implied_months = round(enrol_m, 1)

        # Keep the independently fitted head as a visible cross-check rather than
        # discarding it: a large gap is information, not something to hide.
        if "rate" in heads:
            rate_crosscheck = round(float(heads["rate"].predict(X)[0]), 3)
    elif "rate" in heads:
        rate = round(float(heads["rate"].predict(X)[0]), 3)
        r_lo, r_hi = heads["rate"].predict_interval(X)
        rate_lo, rate_hi = round(float(r_lo[0]), 3), round(float(r_hi[0]), 3)

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
        # The band's nominal target is whatever the artifact was calibrated to — P1HV
        # aims at 0.85 since 2026-08-29 — so the label must come from metadata, never
        # the dataclass default. A constant 80 on an 85% band is a mislabel.
        confidence_pct=int(round(100 * float(
            ((entry.get("meta", {}).get("heads", {}) or {}).get("duration", {}) or {})
            .get("coverage_nominal", 0.80)))),
        extrapolation_warnings=warnings,
        recruitment_rate=rate,
        recruitment_rate_lower=rate_lo,
        recruitment_rate_upper=rate_hi,
        recruitment_rate_crosscheck=rate_crosscheck,
        rate_implied_total_months=implied_months,
        enrolment_months=enrol_m,
        followup_months=fu_m,
    )

    from backend.models import provenance as _prov

    result.provenance = _prov.build(
        phase_key, result,
        supplied={"enrollment": enrollment, "num_sites": num_sites,
                  "drug_type": drug_type, "region": region,
                  "endpoint_archetype": (endpoint_archetypes[0]
                                         if endpoint_archetypes else None),
                  "endpoint_archetypes": endpoint_archetypes},
        therapeutic_area=therapeutic_area,
    )
    return result
