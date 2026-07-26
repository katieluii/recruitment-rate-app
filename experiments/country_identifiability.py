from __future__ import annotations
"""Can country recruitment speed be identified from registry data at all?

    python -m experiments.country_identifiability

Run this before believing any country league table, including the one this
project publishes.

THE PROBLEM
A trial reports ONE enrolment window. A trial running in 20 countries reports
that same single number for all 20. There is no per-country enrolment split in
ClinicalTrials.gov or AACT, so within a multi-country trial there is nothing to
attribute — every participating country is handed an identical observation.

Clean evidence therefore comes only from SINGLE-COUNTRY trials, where the window
unambiguously belongs to one country. That would be fine if every country ran
some domestic-only trials. They do not, and the split is severe:

    United States    43.6% of its appearances are single-country
    China            73.2%
    Poland            0.4%   (3 of 813)
    Latvia, Ireland, Norway, Croatia, Lithuania    0%

So the comparison a planner actually wants — Eastern Europe against the US — has
no evidence path. Eastern Europe is essentially never observed alone, and in the
multi-country trials where it does appear, its window is shared with every other
participating country.

WHAT A MODEL DOES INSTEAD
It learns which KIND of trial each country appears in. The US looks fast because
it runs many small domestic trials, which have short windows; countries that only
ever join large global studies inherit those studies' long windows. That is a
statement about trial portfolios, not about recruitment speed, and it is why a
fitted model put the US fastest and China slowest — a ranking a clinical
operations lead recognises immediately as wrong.
"""
import logging
from collections import Counter

import numpy as np
import pandas as pd

from backend.analytics.site_rates import _parse_locations
from backend.preprocessing.cleaner import recruiting_months
from experiments.dataset import load_clean

log = logging.getLogger(__name__)

EASTERN_EUROPE = ["Poland", "Hungary", "Ukraine", "Bulgaria", "Czechia",
                  "Romania", "Serbia", "Slovakia", "Croatia", "Latvia",
                  "Lithuania", "Estonia"]


def _frame(phases=("P1", "P2", "P3")) -> pd.DataFrame:
    df = pd.concat([load_clean(p).assign(phase=p) for p in phases],
                   ignore_index=True)
    df["window"] = recruiting_months(df)
    locs = df.get("locations", pd.Series([""] * len(df), index=df.index))
    counts = [Counter(x[2] for x in _parse_locations(e) if x[2]) for e in locs]
    df["n_countries"] = [len(c) for c in counts]
    df["solo_country"] = [max(c, key=c.get) if len(c) == 1 else None for c in counts]
    df["_counts"] = counts
    return df


def solo_coverage(df: pd.DataFrame, min_appearances: int = 80) -> pd.DataFrame:
    """Share of each country's appearances that are single-country trials.

    This is the identifiability check. A country at 0% can never be observed
    apart from its co-participants, so no amount of modelling separates it.
    """
    recs = []
    for counts in df["_counts"]:
        solo = len(counts) == 1
        for c in counts:
            recs.append((c, solo))
    g = pd.DataFrame(recs, columns=["country", "solo"])
    t = g.groupby("country").agg(appearances=("solo", "size"), solo=("solo", "sum"))
    t["solo_pct"] = (100 * t["solo"] / t["appearances"]).round(1)
    return t[t["appearances"] >= min_appearances].sort_values("solo_pct")


def country_spread_by_geography(df: pd.DataFrame) -> pd.DataFrame:
    """How much of the window is explained by geographic spread alone."""
    rows = []
    for lo, hi, label in [(1, 1, "1 country"), (2, 3, "2-3"), (4, 8, "4-8"),
                          (9, 20, "9-20"), (21, 99, "21+")]:
        s = df[(df["n_countries"] >= lo) & (df["n_countries"] <= hi)]
        if len(s) < 20:
            continue
        rows.append({"countries": label, "n": len(s),
                     "median_window_mo": round(float(s["window"].median()), 1),
                     "median_sites": int(s["site_count"].median()),
                     "median_enrolment": int(pd.to_numeric(
                         s["Enrollment"], errors="coerce").median())})
    return pd.DataFrame(rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    df = _frame()

    print("1. GEOGRAPHIC SPREAD DRIVES THE WINDOW\n")
    print(country_spread_by_geography(df).to_string(index=False))
    print("\n   A country's share being 1.0 IMPLIES a single-country trial, so an")
    print("   encoder reading raw shares partly learns 'domestic trial = fast'.\n")

    print("2. WHICH COUNTRIES CAN BE OBSERVED ALONE\n")
    cov = solo_coverage(df)
    print(cov.head(8).to_string())
    print("   ...")
    print(cov.tail(4).to_string())

    ee = cov[cov.index.isin(EASTERN_EUROPE)]
    if not ee.empty:
        print(f"\n   Eastern Europe: {int(ee.appearances.sum())} appearances, "
              f"{int(ee.solo.sum())} solo "
              f"({100 * ee.solo.sum() / ee.appearances.sum():.1f}%)")
    if "United States" in cov.index:
        us = cov.loc["United States"]
        print(f"   United States : {int(us.appearances)} appearances, "
              f"{int(us.solo)} solo ({us.solo_pct:.1f}%)")

    print("\n3. VERDICT\n")
    print("   Country recruitment speed is NOT identifiable from this data.")
    print("   Countries that never run domestic-only trials have no observation")
    print("   that separates them from their co-participants, and the countries")
    print("   a planner most wants to compare sit on opposite sides of that line.")
    print("   Any country ranking built here reflects trial-portfolio mix.")


if __name__ == "__main__":
    main()
