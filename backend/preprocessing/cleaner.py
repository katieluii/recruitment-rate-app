from __future__ import annotations
"""Date parsing, duration computation, outlier removal."""
import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def parse_dates(df: pd.DataFrame, require_completion: bool = True) -> pd.DataFrame:
    for col in ["Start Date", "Primary Completion Date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    subset = ["Start Date", "Primary Completion Date"] if require_completion else ["Start Date"]
    df.dropna(subset=subset, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def compute_censored_duration(df: pd.DataFrame, as_of: pd.Timestamp | None = None
                              ) -> pd.DataFrame:
    """Duration and an event indicator, handling trials that have not finished.

    `event_observed = 1` — the trial finished and its primary completion date is
    ACTUAL, so `duration_days` is the real endpoint.

    `event_observed = 0` — the trial is still running. Its published primary
    completion date is the sponsor's ESTIMATE and is discarded; the duration is
    censored at (as_of − start), which says only "it has already run this long
    and is not done".

    Excluding these rows is what makes recent history look fast: among trials
    that started recently, only the quick ones have finished. Our own harness
    measured the effect — test-fold median recruitment rate ran 2.4-3.6x the
    training median — and the 2025 duration-prediction literature reports the
    same selection bias independently.
    """
    as_of = as_of or pd.Timestamp.today().normalize()

    ongoing = df.get("is_ongoing", pd.Series([0] * len(df), index=df.index)).fillna(0).astype(int)
    ctype = df.get("primary_completion_type",
                   pd.Series(["ACTUAL"] * len(df), index=df.index)).fillna("UNKNOWN")

    observed = (ongoing == 0) & (ctype != "ESTIMATED")

    observed_days = (df["Primary Completion Date"] - df["Start Date"]).dt.days
    censored_days = (as_of - df["Start Date"]).dt.days

    df["duration_days"] = np.where(observed, observed_days, censored_days)
    df["event_observed"] = observed.astype(int)
    df["primary_completion_year"] = df["Primary Completion Date"].dt.year
    df["start_year"] = df["Start Date"].dt.year

    df = df[df["duration_days"] > 0].reset_index(drop=True)
    n_cens = int((df["event_observed"] == 0).sum())
    log.info("Censoring: %d of %d rows are right-censored (%.1f%%)",
             n_cens, len(df), 100 * n_cens / max(len(df), 1))
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


def clean(df: pd.DataFrame, phase_key: str, censored: bool = False) -> pd.DataFrame:
    """Clean a raw frame.

    `censored=False` (default, what v2 ships) keeps only finished trials and
    treats every duration as observed. `censored=True` additionally admits
    ongoing trials as right-censored rows for the survival models — see
    compute_censored_duration.
    """
    if censored:
        df = parse_dates(df, require_completion=False)
        df = compute_censored_duration(df)
    else:
        df = parse_dates(df)
        df = compute_duration(df)
        df["event_observed"] = 1
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
    if not censored:
        # Never winsorise a censored duration. Clipping "has run at least 91
        # months and counting" down to a percentile turns a lower bound into a
        # false observation and corrupts the survival likelihood.
        df = winsorise(df, "duration_days")

    df = compute_recruitment_rate(df)

    if len(df) < 10:
        raise ValueError(f"Too few rows after cleaning ({len(df)}) for phase {phase_key}")
    return df


#: Floor on the recruiting window as a fraction of total duration, so a trial
#: whose follow-up estimate swallows its whole span cannot yield an infinite rate.
MIN_ENROL_FRACTION = 0.25


def _followup_months(df: pd.DataFrame) -> pd.Series:
    """Estimated follow-up in months, imputed the same way for every caller.

    Split out of `recruiting_months` so that `clipped_by_floor` can ask which
    rows the floor is about to fabricate WITHOUT re-deriving the imputation and
    letting two copies of it drift.
    """
    from backend.preprocessing.endpoints import add_endpoint_features

    if "endpoint_archetype" not in df.columns:
        df = add_endpoint_features(df.copy())

    fu = df.get("followup_months", pd.Series([np.nan] * len(df), index=df.index))
    by_arch = df.groupby("endpoint_archetype")["followup_months"].median()
    fu = fu.fillna(df["endpoint_archetype"].map(by_arch))
    return fu.fillna(fu.median()).fillna(0.0)


def clipped_by_floor(df: pd.DataFrame, min_fraction: float | None = None) -> pd.Series:
    """Boolean mask: rows whose recruiting window is SET BY the floor constant.

    For these rows `total - followup` fell below `min_fraction * total`, so the
    enrolment-stage target is not a measurement — it is `min_fraction x duration`.
    Used to drop or down-weight them rather than to fit against a constant.
    """
    frac = MIN_ENROL_FRACTION if min_fraction is None else min_fraction
    total = df["duration_days"] / 30.44
    return (total - _followup_months(df)) < (frac * total)


def recruiting_months(df: pd.DataFrame,
                      min_fraction: float | None = None) -> pd.Series:
    """Months actually spent recruiting: total duration minus follow-up.

    THE single definition of the denominator — `recruitment_grid` imports this
    rather than keeping its own copy, because two drifting definitions of "the
    rate" is exactly the kind of thing nobody notices until the numbers disagree.

    v2 divided by the full start → primary-completion span, which includes
    follow-up, so it measured recruitment speed diluted by however long patients
    were then watched. An oncology trial recruiting in 14 months and following
    for 24 scored as though recruitment took 38. Correcting this moves the Phase 3
    median from 0.455 to 0.737 patients/site/month — the old figure understated
    recruitment by roughly 40%, which would lead a planner to over-provision sites.

    Follow-up comes from the parsed primary-outcome time frame, and where that
    will not parse (about half of trials) from the median for that endpoint
    archetype — survival endpoints carry long follow-up, PK endpoints almost none.

    `min_fraction` overrides the floor for a single caller (the sweep in
    `experiments/`); production leaves it None so the module constant governs.
    """
    frac = MIN_ENROL_FRACTION if min_fraction is None else min_fraction
    total = df["duration_days"] / 30.44
    fu = _followup_months(df)
    return (total - fu).clip(lower=frac * total)


def compute_recruitment_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Patients enrolled per site per month over the RECRUITING window.

    Still an approximation and must be labelled as one wherever it is surfaced:
    the registry publishes no enrolment-completion date, so the recruiting window
    is inferred by subtracting an estimated follow-up rather than measured.
    """
    months = recruiting_months(df)
    sites = pd.to_numeric(df["site_count"], errors="coerce")
    enrol = pd.to_numeric(df["Enrollment"], errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = enrol / (sites * months)
    df["recruiting_months"] = months
    df["recruitment_rate"] = rate.replace([np.inf, -np.inf], np.nan)

    # A censored row's denominator is "time elapsed so far", not the trial's
    # length, and its Enrollment field is the TARGET rather than what has been
    # recruited. The resulting ratio is meaningless, so it is dropped rather
    # than fed to the rate head.
    if "event_observed" in df.columns:
        df.loc[df["event_observed"] == 0, "recruitment_rate"] = np.nan

    df = winsorise(df, "recruitment_rate")
    return df
