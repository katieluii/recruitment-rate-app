from __future__ import annotations
"""Fetch-once, reuse-many dataset cache.

Experiments must be reproducible and must not re-hit ClinicalTrials.gov on every
run. Raw study rows are cached to parquet keyed by the API phase set; the cache
is gitignored (see .gitignore) because it is derived data.
"""
import asyncio
import logging
from pathlib import Path

import pandas as pd

from backend.constants import PHASES

log = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent.parent / "data" / "cache"


def _cache_key(api_phases: list[str]) -> str:
    return "_".join(sorted(api_phases))


def cache_path(phase_key: str, cohort: str = "completed") -> Path:
    key = _cache_key(PHASES[phase_key]["api_phases"])
    suffix = "" if cohort == "completed" else f".{cohort}"
    return CACHE_DIR / f"{key}{suffix}.parquet"


def load_ongoing(phase_key: str, refresh: bool = False) -> pd.DataFrame:
    """Trials that are still running — right-censored observations.

    Fetched separately from the completed cohort because they need a different
    status filter and because their primary completion dates are ESTIMATED and
    must not be mistaken for outcomes.
    """
    import asyncio as _asyncio

    path = cache_path(phase_key, "ongoing")
    if path.exists() and not refresh:
        df = pd.read_parquet(path)
        log.info("Cache hit %s: %d ongoing rows", path.name, len(df))
        return df

    from backend.data.ct_api_client import (ONGOING_STATUSES, fetch_studies,
                                            flatten_study)

    log.info("Fetching ongoing trials for %s", phase_key)
    raw = _asyncio.run(fetch_studies(PHASES[phase_key]["api_phases"],
                                     statuses=ONGOING_STATUSES))
    rows = [r for s in raw if (r := flatten_study(s)) is not None]
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    log.info("Cached %d ongoing rows to %s", len(df), path.name)
    return df


def load_clean_censored(phase_key: str, refresh: bool = False) -> pd.DataFrame:
    """Completed + ongoing trials, with an `event_observed` indicator.

    This is the frame the survival models train on. The completed rows carry
    observed endpoints; the ongoing rows are censored at time-elapsed-so-far.
    """
    from backend.preprocessing.cleaner import clean

    completed = load_raw(phase_key, refresh=refresh)
    ongoing = load_ongoing(phase_key, refresh=refresh)

    for frame, flag in ((completed, 0), (ongoing, 1)):
        if "is_ongoing" not in frame.columns and len(frame):
            frame["is_ongoing"] = flag

    combined = pd.concat([completed, ongoing], ignore_index=True, sort=False)
    df = clean(combined, phase_key, censored=True)

    hv_flag = PHASES[phase_key]["hv"]
    if "is_hv" in df.columns:
        df = df[df["is_hv"] == int(hv_flag)].reset_index(drop=True)
    return df


def load_censoring_frame(phase_key: str, refresh: bool = False) -> pd.DataFrame | None:
    """The frame the SHIPPED duration head reweights against, built by the
    trainer's own `build_censoring_frame` — so measured == shipped by
    construction, not by a copy that can drift.

    Not the same as `load_clean_censored`: that applies the healthy-volunteer
    filter (it feeds the survival models), the trainer does not. Passing the
    filtered frame here would measure a model nobody serves.
    """
    from backend.models.trainer import build_censoring_frame

    return build_censoring_frame(load_raw(phase_key, refresh=refresh),
                                 load_ongoing(phase_key, refresh=refresh),
                                 phase_key)


def load_raw(phase_key: str, refresh: bool = False) -> pd.DataFrame:
    """Return the raw (pre-clean) study frame for a phase key.

    P1 and P1HV share an API phase set, so they share one cache file; the
    healthy-volunteer split happens downstream on the `is_hv` column.
    """
    path = cache_path(phase_key)
    if path.exists() and not refresh:
        df = pd.read_parquet(path)
        log.info("Cache hit %s: %d rows", path.name, len(df))
        return df

    from backend.data.data_layer import get_raw_dataframe

    log.info("Cache miss %s — fetching", path.name)
    df = asyncio.run(get_raw_dataframe(phase_key))
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    log.info("Cached %d rows to %s", len(df), path.name)
    return df


def load_clean(phase_key: str, refresh: bool = False) -> pd.DataFrame:
    """Raw → cleaned → HV-filtered, matching what trainer.train_phase does.

    Returns the full cleaned frame (not just X) so the caller still has
    `Start Date`, `nct_id` and the raw columns needed for temporal splitting
    and per-therapeutic-area reporting.
    """
    from backend.preprocessing.cleaner import clean

    raw = load_raw(phase_key, refresh=refresh)
    df = clean(raw, phase_key)

    hv_flag = PHASES[phase_key]["hv"]
    if "is_hv" in df.columns:
        df = df[df["is_hv"] == int(hv_flag)].reset_index(drop=True)
    return df


def refresh_all() -> None:
    for phase_key in PHASES:
        load_raw(phase_key, refresh=True)
