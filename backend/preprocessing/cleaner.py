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
    df["primary_completion_year"] = df["Primary Completion Date"].dt.year
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


def impute_medians(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    return df


def clean(df: pd.DataFrame, phase_key: str) -> pd.DataFrame:
    df = parse_dates(df)
    df = compute_duration(df)
    df = impute_medians(df, [
        "Enrollment", "total_primary_outcomes",
        "total_secondary_outcomes", "number_of_arms",
    ])
    df["site_count"] = (
        df.get("countries", pd.Series([""] * len(df), index=df.index))
        .fillna("")
        .str.count(r"\|")
        .add(1)
    )
    df = remove_outliers(df, "Enrollment")
    df = remove_outliers(df, "duration_days")

    # Phase-specific duration caps (from prior EDA)
    caps = {"P1HV": 600, "P1": 600, "P2": 2200, "P3": 3000}
    cap = caps.get(phase_key)
    if cap:
        df = df[df["duration_days"] <= cap].reset_index(drop=True)

    if len(df) < 10:
        raise ValueError(f"Too few rows after cleaning ({len(df)}) for phase {phase_key}")
    return df
