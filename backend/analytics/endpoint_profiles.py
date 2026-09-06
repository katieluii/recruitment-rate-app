from __future__ import annotations
"""Most common primary-endpoint COMBINATIONS per phase and therapeutic area.

A trial does not have "an endpoint". P1 trials list 4.15 primary outcome measures
that collapse to 1.17 distinct archetypes, and 21.8% carry more than one — so the
unit that actually occurs is a combination, not a single label.

The combination matters beyond its parts. In P1 oncology, SAFETY alone runs a
median 33.8 months and RESPONSE alone 38.6, but RESPONSE+SAFETY runs 46.1 — the
pairing is longer than either half, which is signal the model's multi-hot flags
can use and a single-label interface throws away.

Built at artifact time and shipped as `endpoint_profiles.json`, matching how
`analytics.json` and `site_priors.json` already travel: the deployed app has no
corpus, so anything derived from it must be precomputed.
"""
import logging

import numpy as np
import pandas as pd

from backend.preprocessing.endpoints import ARCHETYPES, add_endpoint_features

log = logging.getLogger(__name__)

#: Below this, a cell's "most common combination" is one or two trials' accident.
MIN_CELL = 30

TOP_N = 3

_FLAGS = [f"endpoint_has_{a}" for a in ARCHETYPES if a != "UNKNOWN"]


def _combo_key(row) -> tuple[str, ...]:
    """The archetypes present on a trial, in ARCHETYPES order.

    Ordered by the vocabulary rather than by the registry's listing order so the
    same set always produces the same key — otherwise ORR+safety and safety+ORR
    count as two different profiles.
    """
    return tuple(a for a in ARCHETYPES
                 if a != "UNKNOWN" and row.get(f"endpoint_has_{a}", 0) == 1)


def _profiles_for(frame: pd.DataFrame) -> list[dict]:
    """Top-N combinations in one cell, most frequent first."""
    if not len(frame):
        return []

    combos = frame.apply(_combo_key, axis=1)
    months = frame["duration_days"] / 30.44

    out = []
    for combo, idx in combos.groupby(combos).groups.items():
        vals = months.loc[idx].dropna()
        if not len(vals):
            continue
        out.append({
            "archetypes": list(combo),
            # An empty tuple means the rules abstained on every primary outcome.
            # Kept rather than dropped: it is a real and common state (119 of the
            # P1 oncology trials), and hiding it would inflate the others' shares.
            "label": " + ".join(combo) if combo else "UNCLASSIFIED",
            "n": int(len(vals)),
            "share": round(float(len(vals) / len(frame)), 3),
            "median_months": round(float(vals.median()), 1),
        })

    out.sort(key=lambda r: (-r["n"], r["label"]))
    return out[:TOP_N]


def build_profiles(df: pd.DataFrame) -> dict:
    """{therapeutic_area: [profile, ...]} plus an "_phase" key for the fallback.

    Areas are masked with the same convention as `analytics.json` — a trial
    spanning two areas counts in BOTH — so the numbers stay comparable with the
    per-area medians already on that endpoint.
    """
    from experiments.metrics import ta_masks

    if "endpoint_archetype" not in df.columns:
        df = add_endpoint_features(df.copy())
    for flag in _FLAGS:
        if flag not in df.columns:
            df[flag] = 0

    profiles: dict[str, list[dict]] = {"_phase": _profiles_for(df)}

    for area, mask in ta_masks(df).items():
        sub = df[mask.to_numpy()]
        if len(sub) < MIN_CELL:
            log.info("endpoint profiles: %s has %d trials (< %d), falling back "
                     "to the phase-wide profile", area, len(sub), MIN_CELL)
            continue
        rows = _profiles_for(sub)
        if rows:
            profiles[area] = rows

    return profiles


def lookup(profiles: dict, therapeutic_area: str | None) -> tuple[list[dict], bool]:
    """Return (profiles, is_fallback) for an area, falling back phase-wide.

    The fallback is REPORTED rather than hidden: a user reading "the top endpoint
    profile for Phase 2 dermatology" deserves to know when they are actually being
    shown Phase 2 overall.
    """
    if therapeutic_area and therapeutic_area in profiles:
        return profiles[therapeutic_area], False
    return profiles.get("_phase", []), True
