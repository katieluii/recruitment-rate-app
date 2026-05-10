from __future__ import annotations
"""Unified data access — routes to API or Postgres based on config."""
import asyncio
import logging

import pandas as pd

from backend.config import settings
from backend.constants import PHASES

log = logging.getLogger(__name__)


async def get_raw_dataframe(phase_key: str) -> pd.DataFrame:
    """
    Fetch raw trial data for a given phase key (P1HV, P1, P2, P3).
    Tries the configured source first, falls back to the other.
    """
    phase_info = PHASES[phase_key]
    api_phases = phase_info["api_phases"]

    if settings.data_source == "postgres":
        return await _from_postgres(api_phases)
    return await _from_api(api_phases)


async def _from_api(api_phases: list[str]) -> pd.DataFrame:
    from backend.data.ct_api_client import fetch_studies, flatten_study

    log.info("Fetching from ClinicalTrials.gov API: phases=%s", api_phases)
    raw = await fetch_studies(api_phases)
    rows = [r for s in raw if (r := flatten_study(s)) is not None]
    if not rows:
        raise ValueError("CT.gov API returned no usable studies.")
    return pd.DataFrame(rows)


async def _from_postgres(api_phases: list[str]) -> pd.DataFrame:
    from backend.data import postgres_client

    loop = asyncio.get_event_loop()
    df = await loop.run_in_executor(None, postgres_client.fetch_studies, api_phases)
    if df.empty:
        raise ValueError("PostgreSQL returned no usable studies.")
    return df
