from __future__ import annotations
"""Date parsing, duration computation, outlier removal."""
import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["Start Date", "Primary Completion Date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    df.dropna(subset=["Start Date", "Primary Completion Date"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def compute_duration(df: pd.DataFrame) -> pd.DataFrame:
    df["duration_days"] = (
        df["Primary Completion Date"] - df["Start Date"]
    ).dt.days
    # Kept for reporting and splitting only — NOT features. See pipeline._NUM_COLS.
    df["primary_completion_year"] = df["Primary Completion Date"].dt.year
    df["start_year"] = df["Start Date"].dt.year
    df = df[df["duration_days"] > 0].reset_index(drop=True)
    return df


def remove_outliers(df: pd.DataFrame, col: str, z_thresh: float = 3.0) -> pd.DataFrame:
    if col not in df.columns or df[col].isna().all():
        return df
    z = np.abs((df[col] - df[col].mean()) / df[col].std(ddof=1))
    before = len(df)
    df = df[z < z_thresh].reset_index(drop=True)
    log.debug("Outlier removal on %s: %d → %d rows", col, before, len(df))
    return df


def winsorise(df: pd.DataFrame, col: str,
              lower_q: float = 0.01, upper_q: float = 0.99) -> pd.DataFrame:
    """Clip extremes instead of deleting them.

    v1 dropped rows via a z-score filter and then a hard phase cap, which
    removed the long tail entirely — the model was then scored on a truncated
    distribution and could never predict a genuinely slow trial. Clipping keeps
    the row and its features while stopping one 12-year study from dominating
    the loss.
    """
    if col not in df.columns or df[col].isna().all():
        return df
    lo, hi = df[col].quantile([lower_q, upper_q])
    n_clipped = int(((df[col] < lo) | (df[col] > hi)).sum())
    df[col] = df[col].clip(lower=lo, upper=hi)
    log.debug("Winsorised %s at [%.1f, %.1f]: %d rows clipped", col, lo, hi, n_clipped)
    return df


def impute_medians(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    return df


def derive_eligibility(df: pd.DataFrame) -> pd.DataFrame:
    """Parse the raw eligibility strings into numeric features."""
    from backend.preprocessing.features import max_followup_months, parse_age_years

    if "min_age_raw" in df.columns:
        df["min_age_years"] = df["min_age_raw"].apply(parse_age_years)
    if "max_age_raw" in df.columns:
        df["max_age_years"] = df["max_age_raw"].apply(parse_age_years)
    if {"min_age_years", "max_age_years"}.issubset(df.columns):
        df["age_span_years"] = df["max_age_years"].fillna(100) - df["min_age_years"].fillna(0)
    if "primary_outcome_timeframes" in df.columns:
        df["followup_months"] = df["primary_outcome_timeframes"].apply(max_followup_months)
    return df


def clean(df: pd.DataFrame, phase_key: str) -> pd.DataFrame:
    df = parse_dates(df)
    df = compute_duration(df)
    df = impute_medians(df, [
        "Enrollment", "total_primary_outcomes",
        "total_secondary_outcomes", "number_of_arms",
    ])

    # site_count now arrives from the data layer as the real number of sites.
    # Fall back only for legacy frames that predate that change.
    if "site_count" not in df.columns:
        log.warning("site_count absent — falling back to country count (legacy frame)")
        df["site_count"] = (
            df.get("countries", pd.Series([""] * len(df), index=df.index))
            .fillna("")
            .str.count(r"\|")
            .add(1)
        )
    df["site_count"] = pd.to_numeric(df["site_count"], errors="coerce").fillna(0)
    # A completed industry trial with zero listed sites is a registry gap, not a
    # trial that ran nowhere; treat it as missing rather than as 0 sites.
    df.loc[df["site_count"] <= 0, "site_count"] = np.nan
    df["site_count"] = df["site_count"].fillna(df["site_count"].median())

    df = derive_eligibility(df)

    # Clip rather than delete: keeps the long tail's features in the training set.
    df = winsorise(df, "Enrollment")
    df = winsorise(df, "site_count")
    df = winsorise(df, "duration_days")

    df = compute_recruitment_rate(df)

    if len(df) < 10:
        raise ValueError(f"Too few rows after cleaning ({len(df)}) for phase {phase_key}")
    return df


def compute_recruitment_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Patients enrolled per site per month.

    APPROXIMATION, and it must be labelled as one wherever it is surfaced.
    The denominator is the full start → primary-completion window because the
    registry does not publish an enrolment-completion date, so any trial with a
    long follow-up tail — an oncology survival study above all — has its rate
    UNDERSTATED. It is a like-for-like comparator across trials of similar
    endpoint type, not an absolute enrolment speed.
    """
    months = df["duration_days"] / 30.44
    sites = pd.to_numeric(df["site_count"], errors="coerce")
    enrol = pd.to_numeric(df["Enrollment"], errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = enrol / (sites * months)
    df["recruitment_rate"] = rate.replace([np.inf, -np.inf], np.nan)
    df = winsorise(df, "recruitment_rate")
    return df
