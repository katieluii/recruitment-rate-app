from __future__ import annotations
"""Scoring.

v1 reported a single number (RMSE in days) from a single random split, with no
baseline to compare against — so nobody could tell whether the model beat
predicting the per-therapeutic-area median. These metrics fix that, and add the
two that speak directly to the observed failure: does the model produce
DIFFERENT answers per therapeutic area, and does it get their ORDER right.
"""
import logging

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from backend.constants import THERAPEUTIC_AREAS
from backend.preprocessing.features import assign_therapeutic_area
from backend.preprocessing.pipeline import one_hot_pipe_col

log = logging.getLogger(__name__)

DAYS_PER_MONTH = 30.44
MIN_TA_ROWS = 5  # matches trainer._build_analytics; below this a median is noise


def ta_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Boolean mask per therapeutic area.

    A trial with conditions spanning two areas counts in BOTH, matching the
    convention in trainer._build_analytics so these numbers stay comparable
    with the per-TA medians already stored in models/artifacts/*/analytics.json.
    """
    ta = df["conditions"].apply(
        lambda c: c if c in THERAPEUTIC_AREAS else assign_therapeutic_area(c)
    )
    ohe = one_hot_pipe_col(pd.DataFrame({"Therapeutic_Area": ta}),
                           "Therapeutic_Area", THERAPEUTIC_AREAS)
    return {area: ohe[area] == 1 for area in THERAPEUTIC_AREAS}


# ── Point-accuracy ────────────────────────────────────────────────────────────

def point_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                  unit: str = "days") -> dict:
    """Point accuracy. `unit="days"` reports in months; `unit="raw"` reports in
    the target's own units (used by the recruitment-rate head, whose values are
    patients per site per month and would be meaningless divided by 30.44)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    with np.errstate(divide="ignore", invalid="ignore"):
        ape = np.abs(err) / np.where(y_true == 0, np.nan, y_true)

    scale = DAYS_PER_MONTH if unit == "days" else 1.0
    suffix = "months" if unit == "days" else "raw"

    # R-squared against predicting the mean. Reported for continuity with the
    # original project, NOT used as a gate: the reference it scores against is
    # the mean, which is a weak bar for a right-skewed target. `skill_vs_ta_median`
    # is the same idea against a much harder reference, and it is what decides
    # whether a change ships. A model can post a respectable R2 while losing to a
    # per-therapeutic-area median lookup, which is exactly what v1 did.
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None

    return {
        "n": int(len(y_true)),
        f"mae_{suffix}": round(float(np.mean(np.abs(err))) / scale, 4),
        "rmse_days": round(float(np.sqrt(np.mean(err ** 2))), 4),
        f"rmse_{suffix}": round(float(np.sqrt(np.mean(err ** 2))) / scale, 4),
        "mape_pct": round(float(np.nanmean(ape)) * 100, 1),
        f"median_ae_{suffix}": round(float(np.median(np.abs(err))) / scale, 4),
        f"bias_{suffix}": round(float(np.mean(err)) / scale, 4),
        "mse": round(float(np.mean(err ** 2)), 2),
        "r2": round(r2, 4) if r2 is not None else None,
    }


def per_ta_errors(df: pd.DataFrame, y_true: np.ndarray,
                  y_pred: np.ndarray, unit: str = "days") -> pd.DataFrame:
    """MAE and true-vs-predicted median per therapeutic area."""
    scale = DAYS_PER_MONTH if unit == "days" else 1.0
    rows = []
    for area, mask in ta_masks(df).items():
        m = mask.to_numpy()
        if m.sum() < MIN_TA_ROWS:
            continue
        t, p = np.asarray(y_true)[m], np.asarray(y_pred)[m]
        rows.append({
            "therapeutic_area": area,
            "n": int(m.sum()),
            "true_median_months": round(float(np.median(t)) / scale, 3),
            "pred_median_months": round(float(np.median(p)) / scale, 3),
            "mae_months": round(float(np.mean(np.abs(p - t))) / scale, 3),
        })
    if not rows:
        return pd.DataFrame(columns=["therapeutic_area", "n", "true_median_months",
                                     "pred_median_months", "mae_months"])
    return pd.DataFrame(rows).sort_values("true_median_months").reset_index(drop=True)


# ── The metrics that speak to the actual bug ─────────────────────────────────

def ta_differentiation(per_ta: pd.DataFrame) -> dict:
    """Does the model differentiate between therapeutic areas at all?

    spread_ratio  — predicted between-TA spread / true between-TA spread.
                    v1 shipped at roughly 0.19 / 0.20 / 0.06 for P1 / P2 / P3.
                    A model at 0.0 gives every area the same answer.
    rank_corr     — Spearman between predicted and true TA medians. Even with a
                    compressed spread, getting the ORDER right (Oncology slow,
                    Dermatology fast) is decision-useful. Compression without
                    order is not.
    n_distinct    — how many distinct predicted medians. 17 of 22 P1 areas
                    returned the identical 10.9 months in v1.
    """
    if len(per_ta) < 2:
        return {"ta_spread_ratio": None, "ta_rank_corr": None,
                "ta_n_distinct": None, "ta_n_areas": len(per_ta)}

    true_med = per_ta["true_median_months"].to_numpy()
    pred_med = per_ta["pred_median_months"].to_numpy()

    true_spread = float(true_med.max() - true_med.min())
    pred_spread = float(pred_med.max() - pred_med.min())
    ratio = pred_spread / true_spread if true_spread > 0 else None

    rho = spearmanr(true_med, pred_med).correlation
    if rho is not None and np.isnan(rho):
        rho = None

    return {
        "ta_spread_ratio": round(ratio, 3) if ratio is not None else None,
        "ta_true_spread_months": round(true_spread, 1),
        "ta_pred_spread_months": round(pred_spread, 1),
        "ta_rank_corr": round(float(rho), 3) if rho is not None else None,
        "ta_n_distinct": int(len(np.unique(np.round(pred_med, 1)))),
        "ta_n_areas": int(len(per_ta)),
    }


def interval_calibration(y_true: np.ndarray, lower: np.ndarray,
                         upper: np.ndarray, nominal: float = 0.80) -> dict:
    """An 80% interval should contain 80% of actuals. v1's did not — it pinned
    the half-width at rmse*0.5 (~6 months) for essentially every input."""
    y_true = np.asarray(y_true, dtype=float)
    inside = (y_true >= np.asarray(lower)) & (y_true <= np.asarray(upper))
    width = np.asarray(upper, dtype=float) - np.asarray(lower, dtype=float)
    return {
        "interval_nominal": nominal,
        "interval_coverage": round(float(inside.mean()), 3),
        "interval_coverage_gap": round(float(inside.mean()) - nominal, 3),
        "interval_mean_width_months": round(float(width.mean()) / DAYS_PER_MONTH, 2),
        "interval_width_cv": round(
            float(width.std() / width.mean()) if width.mean() else 0.0, 3
        ),  # ~0 means every interval is the same width, i.e. uninformative
    }


def concordance(y_true: np.ndarray, y_pred: np.ndarray,
                event_observed: np.ndarray) -> float | None:
    """Harrell's C-index — the fraction of comparable trial pairs ranked correctly.

    The metric the survival literature reports (the 2025 duration survey puts
    DeepSurv at 0.777, random survival forest 0.762, Cox/Weibull AFT 0.754).
    Unlike MAE it can score CENSORED rows: a trial known to have run at least 30
    months and counting is still comparable to one that finished in 12.
    0.5 is random ordering.
    """
    try:
        from lifelines.utils import concordance_index
    except ImportError:
        log.warning("lifelines not installed — skipping C-index")
        return None
    try:
        return round(float(concordance_index(
            np.asarray(y_true, dtype=float),
            np.asarray(y_pred, dtype=float),
            np.asarray(event_observed, dtype=int),
        )), 4)
    except Exception as exc:  # degenerate folds
        log.warning("C-index failed: %s", exc)
        return None


def skill_score(candidate_mae: float, baseline_mae: float) -> float | None:
    """Fraction of the baseline's error removed. Negative = worse than baseline."""
    if not baseline_mae:
        return None
    return round(1.0 - (candidate_mae / baseline_mae), 3)


def evaluate(df_test: pd.DataFrame, y_true, y_pred,
             lower=None, upper=None, unit: str = "days") -> dict:
    """Full metric bundle for one (phase, config) pair."""
    per_ta = per_ta_errors(df_test, y_true, y_pred, unit=unit)
    out = point_metrics(y_true, y_pred, unit=unit)
    out.update(ta_differentiation(per_ta))
    if lower is not None and upper is not None:
        out.update(interval_calibration(y_true, lower, upper))
        if unit != "days":
            out.pop("interval_mean_width_months", None)
    out["_per_ta"] = per_ta.to_dict(orient="records")
    return out
