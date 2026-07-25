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


def cache_path(phase_key: str) -> Path:
    return CACHE_DIR / f"{_cache_key(PHASES[phase_key]['api_phases'])}.parquet"


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
