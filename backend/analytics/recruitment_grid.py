from __future__ import annotations
"""Expected recruitment rate per site per month, by country × area × phase.

The question this answers: *how fast will a site in this country recruit for this
indication at this phase?* — the number that decides where to put sites.

TWO CORRECTIONS OVER THE v2 RATE
--------------------------------
1. **Denominator.** v2 divided enrolment by the full start → primary-completion
   span, which includes follow-up. That is not recruitment speed; it is speed
   diluted by however long you then watch patients. Follow-up is subtracted here,
   read from the parsed primary-outcome time frame and imputed from the endpoint
   archetype's median where the time frame will not parse. It moves the P3
   median from 0.455 to 0.737 patients/site/month — the old figure understated
   recruitment by about 40%.

2. **Attribution.** A trial's rate is still assigned to every country it ran in,
   because per-country enrolment splits are not published. A multi-country trial
   contributes the same rate to each. This makes the grid a comparator across
   countries, not a measurement of any one of them.

WHAT IT STILL IS NOT
--------------------
Per-site truth. No public registry publishes how many patients a given site
enrolled. The facility figures are a track record — the rates of trials a site
took part in — which reflects the trials it gets chosen for as much as its own
performance.
"""
import logging
from collections import defaultdict

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

#: Shrink thin cells toward their parent so a country with 4 trials cannot
#: outrank one with 400 on noise alone.
SMOOTHING = 8.0
MIN_TRIALS_CELL = 3
#: Floor the enrolment window so a trial whose follow-up estimate swallows its
#: whole duration cannot produce an infinite rate.
MIN_ENROL_FRACTION = 0.25


def enrolment_months(df: pd.DataFrame) -> pd.Series:
    """Months spent recruiting, with follow-up removed.

    Delegates to the cleaner so there is exactly one definition of the
    denominator in the codebase.
    """
    from backend.preprocessing.cleaner import recruiting_months

    return recruiting_months(df)


def site_month_rate(df: pd.DataFrame) -> pd.Series:
    """Patients per site per month over the recruiting window."""
    enrol = pd.to_numeric(df["Enrollment"], errors="coerce")
    sites = pd.to_numeric(df["site_count"], errors="coerce")
    months = enrolment_months(df)
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = enrol / (sites * months)
    return rate.replace([np.inf, -np.inf], np.nan)


def size_adjusted_effect(df: pd.DataFrame) -> pd.Series:
    """Each trial's rate relative to what its SIZE alone would predict.

    The raw rate is largely arithmetic. Regressing log(rate) on log(site_count)
    gives a slope near -1 (-0.88 on P3, -1.40 on P1) because sites sit in the
    denominator and sponsors add sites precisely because sites are slow. Ranking
    countries on the raw number therefore partly ranks which countries appear in
    small trials: China tops the raw P3 table but sits BELOW average once trial
    size is controlled, and the raw and adjusted rankings correlate at only 0.29.

    This removes the mechanical component by regressing out trial size, and
    returns exp(residual) — a multiplier where 1.20 means "20% faster than a
    trial of this size and site count would normally manage".
    """
    rate = df["site_month_rate"] if "site_month_rate" in df.columns else site_month_rate(df)
    ok = rate.notna() & (rate > 0)
    out = pd.Series(np.nan, index=df.index, dtype=float)
    if ok.sum() < 50:
        return out

    log_rate = np.log(rate[ok].clip(lower=1e-4))
    log_k = np.log(pd.to_numeric(df.loc[ok, "site_count"], errors="coerce").clip(lower=1))
    log_n = np.log(pd.to_numeric(df.loc[ok, "Enrollment"], errors="coerce").clip(lower=1))
    X = np.vstack([log_k, log_n, np.ones(ok.sum())]).T
    beta = np.linalg.lstsq(X, log_rate.to_numpy(), rcond=None)[0]
    out.loc[ok] = np.exp(log_rate.to_numpy() - X @ beta)
    log.info("size adjustment: log(sites) %+.3f, log(enrolment) %+.3f",
             beta[0], beta[1])
    return out


def _shrink(values: np.ndarray, parent: float, smoothing: float = SMOOTHING) -> float:
    n = len(values)
    if n == 0:
        return parent
    return float((n * np.median(values) + smoothing * parent) / (n + smoothing))


def build_grid(frames: dict[str, pd.DataFrame]) -> dict:
    """frames: phase_key → cleaned frame. Returns the nested rate grid."""
    from backend.analytics.site_rates import _parse_locations, is_placeholder_facility
    from experiments.metrics import ta_masks

    grid: dict = {"phases": {}, "meta": {}}
    all_rates: list[float] = []

    for phase, raw in frames.items():
        df = raw.copy()
        df["site_month_rate"] = site_month_rate(df)
        df["enrol_months"] = enrolment_months(df)
        df = df[df["site_month_rate"].notna()].reset_index(drop=True)
        if df.empty:
            continue

        # Trim the extreme tail: a 2-site trial enrolling 535 patients yields
        # 18 patients/site/month, which is a data artefact, not a benchmark.
        lo, hi = df["site_month_rate"].quantile([0.01, 0.99])
        df["site_month_rate"] = df["site_month_rate"].clip(lo, hi)

        df["size_adj"] = size_adjusted_effect(df)
        rates = df["site_month_rate"].to_numpy(dtype=float)
        adj = df["size_adj"].to_numpy(dtype=float)
        all_rates.extend(rates.tolist())
        phase_rate = float(np.median(rates))

        masks = {a: m.to_numpy() for a, m in ta_masks(df).items()}
        ta_rate = {a: _shrink(rates[m], phase_rate)
                   for a, m in masks.items() if m.sum() >= MIN_TRIALS_CELL}

        cell: dict[tuple[str, str], list[float]] = defaultdict(list)
        cell_adj: dict[tuple[str, str], list[float]] = defaultdict(list)
        country_all: dict[str, list[float]] = defaultdict(list)
        country_adj: dict[str, list[float]] = defaultdict(list)
        fac: dict[str, dict] = defaultdict(
            lambda: {"rates": [], "countries": set(), "areas": set()})

        loc_col = df.get("locations", pd.Series([""] * len(df), index=df.index))
        for i, enc in enumerate(loc_col):
            parsed = _parse_locations(enc)
            countries = {c for _, _, c in parsed if c}
            areas = [a for a, m in masks.items() if m[i]]
            for c in countries:
                country_all[c].append(rates[i])
                if not np.isnan(adj[i]):
                    country_adj[c].append(adj[i])
                for a in areas:
                    cell[(c, a)].append(rates[i])
                    if not np.isnan(adj[i]):
                        cell_adj[(c, a)].append(adj[i])
            for f, _city, c in parsed:
                if not f or is_placeholder_facility(f):
                    continue
                rec = fac[f]
                rec["rates"].append(rates[i])
                if c:
                    rec["countries"].add(c)
                rec["areas"].update(areas[:2])

        grid["phases"][phase] = {
            "phase_rate": round(phase_rate, 4),
            "n_trials": int(len(df)),
            "median_enrol_months": round(float(df["enrol_months"].median()), 1),
            "median_total_months": round(float(df["duration_days"].median() / 30.44), 1),
            "by_area": {a: round(r, 4) for a, r in ta_rate.items()},
            "by_country": {
                c: {"rate": round(_shrink(np.array(v), phase_rate), 4),
                    "adj": round(_shrink(np.array(country_adj.get(c, [])), 1.0), 3),
                    "n_trials": len(v)}
                for c, v in country_all.items() if len(v) >= MIN_TRIALS_CELL
            },
            "by_country_area": {
                f"{c}||{a}": {
                    "rate": round(_shrink(np.array(v), ta_rate.get(a, phase_rate)), 4),
                    "adj": round(_shrink(np.array(cell_adj.get((c, a), [])), 1.0), 3),
                    "n_trials": len(v),
                }
                for (c, a), v in cell.items() if len(v) >= MIN_TRIALS_CELL
            },
            "facilities": {
                f: {"rate": round(_shrink(np.array(r["rates"]), phase_rate), 4),
                    "n_trials": len(r["rates"]),
                    "countries": sorted(r["countries"])[:3],
                    "areas": sorted(r["areas"])[:2]}
                for f, r in sorted(fac.items(), key=lambda kv: -len(kv[1]["rates"]))[:400]
                if len(r["rates"]) >= 5
            },
        }
        log.info("%s grid: %d countries, %d country×area cells, %d facilities",
                 phase, len(grid["phases"][phase]["by_country"]),
                 len(grid["phases"][phase]["by_country_area"]),
                 len(grid["phases"][phase]["facilities"]))

    grid["meta"] = {
        "global_rate": round(float(np.median(all_rates)), 4) if all_rates else None,
        "smoothing": SMOOTHING,
        "min_trials_cell": MIN_TRIALS_CELL,
        "definition": ("patients per site per month over the RECRUITING window "
                       "(total duration minus follow-up)"),
        "caveat": ("Trial-level rates attributed to every country the trial ran in. "
                   "No per-site or per-country enrolment split is published, so this "
                   "compares countries rather than measuring any one of them."),
    }
    return grid


def lookup(grid: dict, phase: str, country: str,
           area: str | None = None) -> dict | None:
    """Most specific rate available for a cell, with what it rests on."""
    p = grid.get("phases", {}).get(phase)
    if not p:
        return None
    if area:
        c = p["by_country_area"].get(f"{country}||{area}")
        if c:
            return {**c, "level": "country x area"}
    c = p["by_country"].get(country)
    if c:
        return {**c, "level": "country"}
    if area and area in p["by_area"]:
        return {"rate": p["by_area"][area], "n_trials": 0, "level": "area"}
    return {"rate": p["phase_rate"], "n_trials": 0, "level": "phase"}
