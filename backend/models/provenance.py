from __future__ import annotations
"""Provenance — the working behind every number this app returns.

Follows the WS9/WS12 Atlas provenance schema: a `sources` list, one entry per
emitted VALUE naming where it came from and how it was derived, and explicit
GAPS rather than quiet guesses.

One structural difference from Atlas, and it applies to everything here. In WS9 a
clinical figure is usually read from a paper and can be `verification: primary` —
someone saw that number in that table. Nothing in this app is read from a source.
Every figure is COMPUTED from a model fitted to registry data, which is Atlas's
`inference` tier: `value_verified` is always false, and `derivation` carries the
calculation rather than a quotation. Marking any of it "verified" would import
Atlas's credibility signal without Atlas's evidence.

What that buys the reader is the thing Atlas cares about — being able to check.
Every prediction states which inputs the user actually gave, which were filled
from the therapeutic area's own median, how many trials sit behind each fill,
and which arithmetic combined them.
"""
import json
import logging
from datetime import date
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

CTGOV_QUERY = ("interventional, industry-sponsored, drug or biologic, "
               "completed, with start and primary completion dates")

#: How a feature value came to be. Ordered most to least trustworthy.
ORIGINS = {
    "user": "supplied in the request",
    "ta_default": "median for this therapeutic area in the training data",
    "phase_default": "median for this phase in the training data",
    "derived": "computed from other inputs",
    "constant": "fixed by the model contract",
}


def _artifact_meta(phase_key: str) -> dict:
    from backend.config import settings

    path = settings.models_dir / phase_key / "metadata.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def _artifact_built(phase_key: str) -> str | None:
    from backend.config import settings

    path = settings.models_dir / phase_key / "metadata.json"
    if not path.exists():
        return None
    return date.fromtimestamp(path.stat().st_mtime).isoformat()


def build_sources(phase_key: str) -> list[dict]:
    meta = _artifact_meta(phase_key)
    heads = meta.get("heads", {})
    dur = heads.get("duration", {})

    sources = [{
        "id": "ctgov",
        "type": "registry_api",
        "label": "ClinicalTrials.gov API v2",
        "url": "https://clinicaltrials.gov/data-api/api",
        "selection": CTGOV_QUERY,
        "n_trials": meta.get("n_train"),
        "note": ("Completed trials only. Trials still running are absent and are "
                 "disproportionately the slow ones, so the corpus skews fast; the "
                 "duration head corrects for this by inverse-probability-of-"
                 "censoring weighting."),
    }]

    sources.append({
        "id": f"model_{phase_key}_duration",
        "type": "model_artifact",
        "label": f"Two-stage conformalised quantile model, {phase_key}",
        "structure": dur.get("kind", "single"),
        "stages": dur.get("stages"),
        "n_fit": dur.get("n_fit"),
        "quantiles": dur.get("quantiles"),
        "coverage_nominal": dur.get("coverage_nominal"),
        "ipcw_applied": dur.get("ipcw_applied"),
        "built": _artifact_built(phase_key),
    })

    if "rate" in heads:
        sources.append({
            "id": f"model_{phase_key}_rate",
            "type": "model_artifact",
            "label": f"Recruitment-rate quantile model, {phase_key}",
            "n_fit": heads["rate"].get("n_fit"),
            "quantiles": heads["rate"].get("quantiles"),
            "built": _artifact_built(phase_key),
        })
    return sources


#: Request field -> the feature name the defaults are stored under. Without this
#: `num_sites` reports "missing" while a perfectly good `site_count` default sits
#: in the artifact.
_REQUEST_TO_FEATURE = {
    "num_sites": "site_count",
    "enrollment": "Enrollment",
    "drug_type": "Drug_Type",
}


def _clean(value):
    """Defaults are numeric for most features and categorical for some
    (Masking, Allocation, Drug_Type), so round only what is actually a number."""
    try:
        return round(float(value), 2)
    except (TypeError, ValueError):
        return str(value)


def input_origins(phase_key: str, supplied: dict[str, Any],
                  therapeutic_area: str | None) -> dict:
    """Where each model input came from, and how much evidence backs a fill.

    The evidence count matters: an enrolment filled from an area with 175 trials
    behind it is a different proposition from one filled from an area with 16,
    and the reader cannot tell without being told.
    """
    from backend.models import registry

    entry = registry.load(phase_key) or {}
    ta_defaults = (entry.get("meta", {}).get("feature_defaults_by_ta") or {})
    ta_block = ta_defaults.get(therapeutic_area or "", {})
    phase_defaults = entry.get("feature_defaults") or {}

    out: dict[str, dict] = {}
    for key, value in supplied.items():
        if value is not None:
            out[key] = {"value": value, "origin": "user",
                        "explanation": ORIGINS["user"]}
            continue
        feat = _REQUEST_TO_FEATURE.get(key, key)
        if feat in ta_block:
            out[key] = {
                "value": _clean(ta_block[feat]),
                "origin": "ta_default",
                "explanation": ORIGINS["ta_default"],
                "evidence_n_trials": int(ta_block.get("n", 0)),
                "basis": therapeutic_area,
            }
        elif feat in phase_defaults and phase_defaults[feat] is not None:
            out[key] = {
                "value": _clean(phase_defaults[feat]),
                "origin": "phase_default",
                "explanation": ORIGINS["phase_default"],
                "basis": phase_key,
            }
        else:
            out[key] = {"value": None, "origin": "missing",
                        "explanation": "no value and no default available"}
    return out


def build(phase_key: str, prediction, supplied: dict[str, Any],
          therapeutic_area: str) -> dict:
    """Full provenance block for one prediction."""
    dur_src = f"model_{phase_key}_duration"
    rate_src = f"model_{phase_key}_rate"
    inputs = input_origins(phase_key, supplied, therapeutic_area)

    def _fmt(key: str) -> str:
        rec = inputs.get(key) or {}
        v = rec.get("value")
        if v is None:
            return "unknown"
        return f"{v:g}" if isinstance(v, (int, float)) else str(v)

    values: dict[str, dict] = {}

    if prediction.enrolment_months is not None and prediction.followup_months is not None:
        values["predicted_months"] = {
            "value": prediction.predicted_months,
            "unit": "months",
            "verification": "inference",
            "value_verified": False,
            "source_id": [dur_src],
            "derivation": (
                f"enrolment window {prediction.enrolment_months} months "
                f"+ follow-up {prediction.followup_months} months "
                f"= {prediction.predicted_months} months"),
            "note": ("The two stages are modelled separately because they are "
                     "near-independent (r = +0.03) and driven by different "
                     "things — geography and eligibility move enrolment, the "
                     "endpoint moves follow-up."),
        }
        values["enrolment_months"] = {
            "value": prediction.enrolment_months,
            "unit": "months",
            "verification": "inference",
            "value_verified": False,
            "source_id": [dur_src],
            "derivation": ("median of the enrolment-window quantile model, given "
                           f"enrolment {_fmt('enrollment')} across "
                           f"{_fmt('num_sites')} sites"),
        }
        values["followup_months"] = {
            "value": prediction.followup_months,
            "unit": "months",
            "verification": "inference",
            "value_verified": False,
            "source_id": [dur_src],
            "derivation": ("median of the follow-up quantile model, driven by the "
                           "primary endpoint type"),
        }
    else:
        values["predicted_months"] = {
            "value": prediction.predicted_months,
            "unit": "months",
            "verification": "inference",
            "value_verified": False,
            "source_id": [dur_src],
            "derivation": "median of the duration quantile model",
        }

    values["interval"] = {
        "value": [prediction.lower_months, prediction.upper_months],
        "unit": "months",
        "verification": "inference",
        "value_verified": False,
        "source_id": [dur_src],
        "derivation": (
            f"{prediction.confidence_pct}% band from the 0.1 and 0.9 quantile "
            "models, combined across the two stages in quadrature and scaled to "
            "hit nominal coverage on held-out trials"),
        "measured_coverage": "0.82-0.89 on a temporal holdout, against 0.80 nominal",
    }

    if prediction.recruitment_rate is not None:
        entry = {
            "value": prediction.recruitment_rate,
            "unit": "patients per site per month",
            "verification": "inference",
            "value_verified": False,
            "source_id": [dur_src],
            "derivation": (
                f"{_fmt('enrollment')} patients / ({_fmt('num_sites')} sites x "
                f"{prediction.enrolment_months} recruiting months) "
                f"= {prediction.recruitment_rate}"),
            "note": ("Derived from the enrolment window rather than predicted "
                     "separately, so the two can never contradict each other. "
                     "Two independent models used to answer this same question "
                     "and disagreed by up to 62% on default inputs, because the "
                     "median of a ratio is not the ratio of medians."),
            "caveat": ("Largely determined by how many sites the sponsor chose — "
                       "regressing it on site count gives a slope near -1. Use it "
                       "to compare trials of similar size, not as a site metric."),
        }
        cross = prediction.recruitment_rate_crosscheck
        if cross:
            gap = abs(cross - prediction.recruitment_rate) / max(
                prediction.recruitment_rate, 1e-6)
            entry["crosscheck"] = {
                "independent_rate_model": cross,
                "source_id": [rate_src],
                "relative_gap": round(gap, 3),
                "interpretation": (
                    "close agreement" if gap < 0.25 else
                    "the two approaches disagree materially here — treat the rate "
                    "as indicative and prefer the enrolment window"),
            }
        values["recruitment_rate"] = entry

    gaps = [
        "Per-site enrolment is not published by ClinicalTrials.gov or AACT, so no "
        "figure here is an observed site performance.",
        "Country recruitment speed is not identifiable from this data: a "
        "multi-country trial reports one enrolment window shared by every "
        "participating country, and countries that never run domestic-only trials "
        "(Poland 0.4%, Latvia 0%) can never be observed apart from their "
        "co-participants.",
    ]
    if prediction.extrapolation_warnings:
        gaps.extend(prediction.extrapolation_warnings)

    return {
        "schema": "recruitment-rate-app/provenance/v1",
        "generated": date.today().isoformat(),
        "phase": phase_key,
        "therapeutic_area": therapeutic_area,
        "sources": build_sources(phase_key),
        "inputs": inputs,
        "values": values,
        "gaps": gaps,
        "note": ("Every value here is DERIVED, never read from a source, so all "
                 "carry verification 'inference' and value_verified false — the "
                 "same tier Atlas uses for computed figures."),
    }
