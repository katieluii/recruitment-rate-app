from __future__ import annotations
"""Is "patients per site per month" measuring sites, or measuring arithmetic?

    python -m experiments.rate_confounding

Run before building anything further on the per-site rate. The answer decides
whether a Poisson-Gamma site layer has a real quantity to model.

THE PROBLEM
    rate = enrolment / (sites x recruiting months)

`sites` is in the denominator, so for a fixed enrolment the rate falls with site
count by construction. And the causal arrow runs the wrong way for a site
metric: sponsors add sites BECAUSE each site is slow, so site count and site
productivity are jointly chosen rather than independent.

If log(rate) regresses on log(site_count) with a slope near -1, the rate is
mostly a restatement of how many sites the sponsor picked, and a country ranking
built on it partly ranks which countries appear in small trials.
"""
import logging

import numpy as np
import pandas as pd

from backend.analytics.recruitment_grid import site_month_rate
from backend.analytics.site_rates import _parse_locations
from backend.preprocessing.cleaner import recruiting_months
from experiments.dataset import load_clean

log = logging.getLogger(__name__)

MIN_COUNTRY_ROWS = 40


def _prepared(phase: str) -> pd.DataFrame:
    df = load_clean(phase)
    df["rate"] = site_month_rate(df)
    df = df[df["rate"].notna()].copy()
    lo, hi = df["rate"].quantile([0.01, 0.99])
    df = df[(df["rate"] >= lo) & (df["rate"] <= hi)].copy()
    df["window"] = recruiting_months(df)
    df["log_k"] = np.log(df["site_count"].clip(lower=1))
    df["log_N"] = np.log(pd.to_numeric(df["Enrollment"], errors="coerce").clip(lower=1))
    df["log_rate"] = np.log(df["rate"].clip(lower=1e-4))
    df["log_window"] = np.log(df["window"].clip(lower=0.1))
    return df


def mechanical_share(phase: str) -> dict:
    """How much of the rate is explained by the site count alone."""
    df = _prepared(phase)
    slope = np.polyfit(df["log_k"], df["log_rate"], 1)[0]
    return {
        "phase": phase,
        "n": int(len(df)),
        "log_k_slope": round(float(slope), 3),
        "corr_sites_rate": round(float(np.corrcoef(df["log_k"], df["log_rate"])[0, 1]), 3),
        # The window is the alternative planning quantity; if it is much less
        # correlated with site count, it is the sounder thing to model.
        "corr_sites_window": round(
            float(np.corrcoef(df["log_k"], df["log_window"])[0, 1]), 3),
    }


def country_ranking_stability(phase: str) -> dict:
    """Does the country ranking survive controlling for trial size?"""
    df = _prepared(phase)
    rows = []
    locs = df.get("locations", pd.Series([""] * len(df), index=df.index))
    for i in range(len(df)):
        for c in {c for _, _, c in _parse_locations(locs.iloc[i]) if c}:
            rows.append((c, df["rate"].iloc[i], df["log_rate"].iloc[i],
                         df["log_k"].iloc[i], df["log_N"].iloc[i]))
    g = pd.DataFrame(rows, columns=["country", "rate", "log_rate", "log_k", "log_N"])
    g = g.groupby("country").filter(lambda x: len(x) >= MIN_COUNTRY_ROWS)
    if g.empty:
        return {"phase": phase, "rank_corr": None}

    X = np.vstack([g["log_k"], g["log_N"], np.ones(len(g))]).T
    beta = np.linalg.lstsq(X, g["log_rate"].to_numpy(), rcond=None)[0]
    g = g.assign(resid=g["log_rate"].to_numpy() - X @ beta)

    raw = g.groupby("country")["rate"].median()
    adj = g.groupby("country")["resid"].median()
    rho = raw.rank().corr(adj.rank(), method="spearman")

    table = pd.DataFrame({"raw_rate": raw.round(3), "size_adjusted": adj.round(3),
                          "n": g.groupby("country").size()}).sort_values(
        "raw_rate", ascending=False)
    return {"phase": phase, "n_countries": int(len(raw)),
            "rank_corr": round(float(rho), 3), "table": table}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("MECHANICAL SHARE — a slope near -1.0 means the rate is arithmetic\n")
    for ph in ("P1", "P2", "P3"):
        m = mechanical_share(ph)
        print(f"  {m['phase']}  n={m['n']:5d}  log(sites) slope {m['log_k_slope']:+.3f}"
              f"   corr(sites, rate) {m['corr_sites_rate']:+.3f}"
              f"   corr(sites, WINDOW) {m['corr_sites_window']:+.3f}")

    print("\nCOUNTRY RANKING STABILITY — rank correlation of raw vs size-adjusted\n")
    for ph in ("P2", "P3"):
        r = country_ranking_stability(ph)
        if r.get("rank_corr") is None:
            continue
        print(f"  {r['phase']}  {r['n_countries']} countries  "
              f"rank correlation {r['rank_corr']:+.3f}")
        print(r["table"].head(6).to_string())
        print()

    print("READ: a slope near -1 and a low rank correlation together mean the")
    print("per-site rate is mostly a restatement of the sponsor's site count, and")
    print("a country league table built on it is confounded by trial-size mix.")


if __name__ == "__main__":
    main()
