from __future__ import annotations
"""Site-level recruitment rates: country priors, facility track record, site-mix
simulation.

WHAT THIS IS NOT
================
Neither ClinicalTrials.gov nor AACT publishes per-site enrolment counts or
per-site activation dates. Observed site-level recruitment does not exist in the
free data. Everything here is therefore MODELLED from trial-level facts:

  * a country's rate is the distribution of TRIAL-level rates across trials that
    ran sites in that country, not the enrolment those sites actually delivered;
  * a facility's index is an ASSOCIATION — a site that appears in fast trials
    scores well, which may say more about the trials it is chosen for than about
    the site. It is a screening prior, not a causal estimate of site performance.

Both must be labelled as such wherever they are surfaced. Getting to observed
per-site performance needs CTMS or a commercial source (Citeline, TrialTrove).

The underlying rate is itself approximate: the denominator is the full
start → primary-completion window because no enrolment-completion date is
published, so trials with long follow-up have their rate understated. It is a
like-for-like comparator, not an absolute enrolment speed.
"""
import logging
import re
from collections import defaultdict

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

#: Most industry sponsors do not publish real site names — they register
#: placeholders. Left in, these dominate the index purely by volume: "Research
#: Site" appears in 9,676 Phase 3 trials and "GSK Investigational Site" in
#: 7,227. They are not institutions and ranking them tells you nothing about
#: where to run a trial, so they are excluded rather than shown.
_PLACEHOLDER_RE = re.compile(
    r"(investigat\w*|research|clinical|study|trial|local|"
    r"gsk|novartis|pfizer|msd|merck|lilly|astrazeneca|sanofi|roche|novo\s*nordisk)"
    r"[\s\-]*(site|centre|center|facility|institution)?\s*(number|no\.?|#)?\s*\d*$"
    r"|^(site|facility|centre|center|location|unit)\s*(number|no\.?|#)?\s*[\d\-]+$"
    r"|^\d+$",
    re.I,
)

#: Registry free-text that lands in the facility field but is not a site name.
_BOILERPLATE_PREFIXES = (
    "for additional information", "for more information", "please contact",
    "see contact", "refer to", "clinicaltrials.gov", "http",
)


def is_placeholder_facility(name: str) -> bool:
    """True for anonymised sponsor placeholders and registry boilerplate.

    Without this the index is dominated by strings that are not institutions:
    "Research Site" appears in 9,676 Phase 3 trials and a truncated
    "For additional information regarding investigati..." in 935.
    """
    n = (name or "").strip()
    if len(n) < 4:
        return True
    low = n.lower()
    if low.startswith(_BOILERPLATE_PREFIXES):
        return True
    return bool(_PLACEHOLDER_RE.search(n))

#: Shrinkage weight — a country with 3 trials should not outrank one with 300.
SMOOTHING = 10.0
MIN_TRIALS_COUNTRY = 3
MIN_TRIALS_FACILITY = 5
MAX_FACILITIES = 750  # keeps the artifact small; ranked by trial count


def _parse_locations(encoded: str | None) -> list[tuple[str, str, str]]:
    """'facility^city^country|...' → [(facility, city, country), ...]."""
    if not encoded or pd.isna(encoded):
        return []
    out = []
    for rec in str(encoded).split("|"):
        parts = rec.split("^")
        if len(parts) == 3:
            out.append((parts[0].strip(), parts[1].strip(), parts[2].strip()))
    return out


def _shrink(values: np.ndarray, fallback: float, smoothing: float = SMOOTHING) -> float:
    n = len(values)
    if n == 0:
        return fallback
    return float((n * np.median(values) + smoothing * fallback) / (n + smoothing))


def build_country_priors(df: pd.DataFrame) -> dict:
    """Recruitment-rate prior per country, and per country × therapeutic area.

    A trial contributes its rate to every country it ran in. Countries with few
    trials are shrunk toward the therapeutic-area rate, and areas with few
    trials toward the phase-level rate.
    """
    from experiments.metrics import ta_masks

    sub = df[df["recruitment_rate"].notna()].reset_index(drop=True)
    if sub.empty:
        return {"global": None, "by_country": {}, "by_country_ta": {}}

    rates = sub["recruitment_rate"].to_numpy(dtype=float)
    global_rate = float(np.median(rates))

    ta_rate: dict[str, float] = {}
    for area, mask in ta_masks(sub).items():
        m = mask.to_numpy()
        if m.sum() >= MIN_TRIALS_COUNTRY:
            ta_rate[area] = _shrink(rates[m], global_rate)

    country_rows: dict[str, list[float]] = defaultdict(list)
    country_ta_rows: dict[tuple[str, str], list[float]] = defaultdict(list)
    masks = {a: m.to_numpy() for a, m in ta_masks(sub).items()}

    for i, enc in enumerate(sub.get("locations", pd.Series([""] * len(sub)))):
        countries = {c for _, _, c in _parse_locations(enc) if c}
        if not countries:
            continue
        for country in countries:
            country_rows[country].append(rates[i])
            for area, m in masks.items():
                if m[i]:
                    country_ta_rows[(country, area)].append(rates[i])

    by_country = {
        c: {"rate": round(_shrink(np.array(v), global_rate), 4), "n_trials": len(v)}
        for c, v in country_rows.items() if len(v) >= MIN_TRIALS_COUNTRY
    }
    by_country_ta = {
        f"{c}||{a}": {
            "rate": round(_shrink(np.array(v), ta_rate.get(a, global_rate)), 4),
            "n_trials": len(v),
        }
        for (c, a), v in country_ta_rows.items() if len(v) >= MIN_TRIALS_COUNTRY
    }

    log.info("Country priors: %d countries, %d country x TA cells",
             len(by_country), len(by_country_ta))
    return {
        "global": round(global_rate, 4),
        "by_ta": {a: round(r, 4) for a, r in ta_rate.items()},
        "by_country": by_country,
        "by_country_ta": by_country_ta,
    }


def build_facility_index(df: pd.DataFrame) -> dict:
    """Per-facility track record.

    ASSOCIATION, NOT ATTRIBUTION. A facility inherits the rate of every trial it
    participated in; a site chosen for fast trials will look fast. Useful for
    screening a long site list, not for ranking site performance.
    """
    sub = df[df["recruitment_rate"].notna()].reset_index(drop=True)
    if sub.empty:
        return {}

    rates = sub["recruitment_rate"].to_numpy(dtype=float)
    global_rate = float(np.median(rates))

    rows: dict[str, dict] = defaultdict(
        lambda: {"rates": [], "countries": set(), "areas": set()})
    ta_col = sub["conditions"] if "conditions" in sub.columns else None

    for i, enc in enumerate(sub.get("locations", pd.Series([""] * len(sub)))):
        for facility, _city, country in _parse_locations(enc):
            if not facility or is_placeholder_facility(facility):
                continue
            rec = rows[facility]
            rec["rates"].append(rates[i])
            if country:
                rec["countries"].add(country)
            if ta_col is not None and isinstance(ta_col.iloc[i], str):
                rec["areas"].add(ta_col.iloc[i].split("|")[0][:40])

    index = {
        name: {
            "n_trials": len(rec["rates"]),
            "rate": round(_shrink(np.array(rec["rates"]), global_rate), 4),
            "countries": sorted(rec["countries"])[:5],
            "top_areas": sorted(rec["areas"])[:3],
        }
        for name, rec in rows.items() if len(rec["rates"]) >= MIN_TRIALS_FACILITY
    }
    top = sorted(index.items(), key=lambda kv: -kv[1]["n_trials"])[:MAX_FACILITIES]
    log.info("Facility index: %d facilities above n>=%d (kept top %d)",
             len(index), MIN_TRIALS_FACILITY, len(top))
    return dict(top)


def build_priors(df: pd.DataFrame) -> dict:
    return {
        "countries": build_country_priors(df),
        "facilities": build_facility_index(df),
        "caveat": (
            "Modelled from trial-level rates. No per-site enrolment counts exist "
            "in ClinicalTrials.gov or AACT. Facility figures are association, "
            "not attribution."
        ),
    }


# ── Simulation ────────────────────────────────────────────────────────────────

def country_rate(priors: dict, country: str,
                 therapeutic_area: str | None = None) -> tuple[float, int]:
    """Best available rate for a country, preferring the country × TA cell."""
    c = priors.get("countries", {})
    if therapeutic_area:
        cell = c.get("by_country_ta", {}).get(f"{country}||{therapeutic_area}")
        if cell:
            return cell["rate"], cell["n_trials"]
    cell = c.get("by_country", {}).get(country)
    if cell:
        return cell["rate"], cell["n_trials"]
    if therapeutic_area and therapeutic_area in c.get("by_ta", {}):
        return c["by_ta"][therapeutic_area], 0
    return c.get("global") or 0.0, 0


def simulate_site_mix(priors: dict, target_enrollment: int,
                      site_mix: dict[str, int],
                      therapeutic_area: str | None = None) -> dict:
    """Expected enrolment window for a proposed country → site-count mix.

    Returns the blended rate, the projected months to full enrolment, and the
    marginal value of one more site in each country — which is the number the
    decision actually turns on.
    """
    if not site_mix or target_enrollment <= 0:
        raise ValueError("site_mix must be non-empty and target_enrollment > 0")

    per_country = []
    total_capacity = 0.0  # patients per month across the whole network
    for country, n_sites in site_mix.items():
        if n_sites <= 0:
            continue
        rate, n_trials = country_rate(priors, country, therapeutic_area)
        capacity = rate * n_sites
        total_capacity += capacity
        per_country.append({
            "country": country,
            "sites": int(n_sites),
            "rate_per_site_month": round(rate, 4),
            "patients_per_month": round(capacity, 2),
            "evidence_n_trials": n_trials,
        })

    if total_capacity <= 0:
        raise ValueError("No usable rate for any country in the mix")

    months = target_enrollment / total_capacity

    # Marginal value: months saved by adding one site in each country.
    for row in per_country:
        new_capacity = total_capacity + row["rate_per_site_month"]
        row["months_saved_per_extra_site"] = round(
            months - (target_enrollment / new_capacity), 2)

    per_country.sort(key=lambda r: -r["months_saved_per_extra_site"])
    return {
        "target_enrollment": int(target_enrollment),
        "total_sites": int(sum(site_mix.values())),
        "blended_rate_per_site_month": round(total_capacity / sum(site_mix.values()), 4),
        "patients_per_month": round(total_capacity, 2),
        # NOT the enrolment window. The rate's denominator is the full
        # start → primary-completion span, because no enrolment-completion date
        # is published, so inverting it reconstructs TOTAL trial duration for a
        # trial of this size and mix. Calling it an enrolment window would
        # overstate how long recruitment itself takes.
        "projected_total_months": round(months, 1),
        "projected_months_basis": (
            "total start-to-primary-completion span, not the enrolment period "
            "alone; the underlying rate uses full trial duration as its "
            "denominator"
        ),
        "by_country": per_country,
        "basis": "modelled from trial-level rates; not observed per-site enrolment",
    }


def top_countries(priors: dict, therapeutic_area: str | None = None,
                  limit: int = 10, min_trials: int = 10) -> list[dict]:
    """Countries ranked by modelled rate, filtered to those with real evidence."""
    c = priors.get("countries", {})
    rows = []
    if therapeutic_area:
        for key, cell in c.get("by_country_ta", {}).items():
            country, area = key.split("||", 1)
            if area == therapeutic_area and cell["n_trials"] >= min_trials:
                rows.append({"country": country, **cell})
    if not rows:
        rows = [{"country": k, **v} for k, v in c.get("by_country", {}).items()
                if v["n_trials"] >= min_trials]
    rows.sort(key=lambda r: -r["rate"])
    return rows[:limit]
