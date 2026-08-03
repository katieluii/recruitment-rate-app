from __future__ import annotations
"""Endpoint archetype classification.

Why this exists: `duration_days` fuses two different processes — how long it
takes to recruit, and how long you then follow patients. Oncology Phase 3 runs
a median 34 months not because it recruits slowly but because it waits for
survival events; Dermatology Phase 3 runs 15 months reading a skin score at
week 16. With no endpoint feature, one model over that blended target cannot
express the difference and regresses everything to the phase mean.

The classifier is DETERMINISTIC first. `ARCHETYPES` is a closed vocabulary —
nothing outside it may enter the feature set, including from the optional LLM
fallback, which is only consulted where the rules abstain and whose output is
validated against the vocabulary before use.
"""
import logging
import re

import pandas as pd

log = logging.getLogger(__name__)

#: Closed vocabulary. Anything not in this tuple is rejected, not coerced.
ARCHETYPES: tuple[str, ...] = (
    "SURVIVAL",         # OS, PFS, DFS, EFS — event-driven, longest follow-up
    "EVENT_COMPOSITE",  # MACE, hospitalisation, mortality composites
    "EVENT_RATE",       # annualised bleeding rate, seizure frequency, attack rate
    "RESPONSE",         # ORR, pCR, DCR, clinical remission, treatment success
    "IMMUNOGENICITY",   # seroconversion, seroprotection, antibody titre — vaccines
    "CLINICAL_SCORE",   # ACR20, PASI75, ADAS-Cog, EDSS, 6MWD
    "BIOMARKER",        # HbA1c, LDL, viral load, eGFR
    "PRO",              # patient-reported outcomes, quality of life
    "SAFETY",           # AE, TEAE, DLT, MTD, tolerability
    "PK_PD",            # Cmax, AUC, half-life, clearance
    "UNKNOWN",          # rules abstained
)

# Evaluated in order — first match wins, so the most duration-defining
# archetypes are tested before the generic ones. "score" would otherwise
# swallow ACR20 and ADAS-Cog alike, and almost every oncology endpoint also
# mentions safety somewhere.
_RULES: list[tuple[str, re.Pattern]] = [
    ("SURVIVAL", re.compile(
        r"\b(overall survival|progression[- ]free survival|disease[- ]free survival|"
        r"event[- ]free survival|recurrence[- ]free survival|relapse[- ]free survival|"
        r"metastasis[- ]free survival|failure[- ]free survival|"
        r"time to progression|time to treatment failure|time to death|"
        r"duration of survival|survival time|"
        r"\bos\b|\bpfs\b|\bdfs\b|\befs\b|\brfs\b|\bmfs\b|\bttp\b)", re.I)),

    ("EVENT_COMPOSITE", re.compile(
        r"\b(major adverse cardiac|major adverse cardiovascular|\bmace\b|"
        r"composite (?:end ?point|outcome)|"
        r"all[- ]cause mortality|cardiovascular death|hospitali[sz]ation for|"
        r"time to first (?:event|occurrence)|"
        # "incidence of X events" is a composite ONLY when X is not an adverse
        # event. Without the lookahead this clause swallowed "incidence of
        # treatment-emergent adverse events", the single most common safety
        # phrasing in the registry, and EVENT_COMPOSITE is evaluated before
        # SAFETY — so 893 of 1,006 EVENT_COMPOSITE trials were safety trials,
        # 100% of them on Phase 1. Measured 2026-08-03.
        r"incidence of (?!.{0,40}(?:adverse|\bAEs?\b|\bTEAEs?\b|\bSAEs?\b|"
        r"toxicit|reactions?))"
        r".{0,40}(?:events?|episodes?))", re.I)),

    ("EVENT_RATE", re.compile(
        r"\b(annuali[sz]ed .{0,25}rate|annual .{0,20}rate|"
        r"bleeding rate|\babr\b|attack rate|exacerbation rate|relapse rate|"
        r"seizure frequency|seizure[- ]free|migraine[- ]free|headache[- ]free|"
        r"(?:number|frequency|rate) of .{0,30}"
        r"(?:attacks?|episodes?|exacerbations?|seizures?|flares?|bleeds?|"
        r"infections?|recurrences?|relapses?) per|"
        r"per (?:month|year|patient[- ]year|100 patient))", re.I)),

    ("IMMUNOGENICITY", re.compile(
        r"\b(seroconversion|seroprotection|seropositivit|seronegativ|"
        r"\bspr\b|\bscr\b|geometric mean titer|geometric mean titre|\bgmt\b|"
        r"geometric mean concentration|\bgmc\b|"
        r"(?:neutrali[sz]ing|neutralisation|binding) antibod|"
        r"antibody (?:response|level|titer|titre|concentration)|"
        r"anti[- ](?:hbs|hav|drug) antibod|immunogenic)", re.I)),

    ("RESPONSE", re.compile(
        r"\b(objective response rate|overall response rate|best overall response|"
        r"complete response|partial response|pathologi(?:c|cal) complete response|"
        r"disease control rate|clinical benefit rate|duration of response|"
        r"tumou?r response|response rate|remission rate|cure rate|"
        r"clinical (?:response|remission|resolution|success|cure|improvement)|"
        r"treatment success|therapeutic success|"
        r"(?:achiev\w+|with|attain\w+) .{0,30}(?:remission|resolution|clearance|"
        r"response|success|cure)|"
        r"\borr\b|\bpcr\b|\bdcr\b|\bcbr\b|\bdor\b|recist)", re.I)),

    ("PK_PD", re.compile(
        r"\b(pharmacokinetic|pharmacodynamic|\bc ?max\b|\bt ?max\b|\bauc\b|"
        r"area under the (?:plasma |serum )?(?:concentration|curve)|"
        r"half[- ]life|\bt1/2\b|clearance|volume of distribution|"
        r"plasma concentration|serum concentration|bioavailability|bioequivalence)", re.I)),

    ("SAFETY", re.compile(
        r"\b(adverse events?|adverse reactions?|\bteaes?\b|\baes?\b|\bsaes?\b|"
        r"\baesis?\b|solicited .{0,20}reactions?|injection site reactions?|"
        r"dose[- ]limiting toxicit|\bdlt\b|maximum tolerated dose|\bmtd\b|"
        r"tolerabilit|safety and tolerabilit|number of participants with .{0,30}"
        r"(?:adverse|toxicit)|toxicit|treatment[- ]emergent)", re.I)),

    ("BIOMARKER", re.compile(
        r"\b(hba1c|glycated h(?:a)?emoglobin|\bldl\b|\bhdl\b|cholesterol|triglycerid|"
        r"viral load|\bhiv[- ]1 rna\b|sustained virologic|\begfr\b|creatinine|"
        r"h(?:a)?emoglobin|platelet count|neutrophil count|"
        r"blood pressure|\bfev1\b|forced expiratory|bone mineral density|"
        r"\bpsa\b|c[- ]reactive protein|\bcrp\b|serum level|plasma level|"
        r"body weight|body mass index|\bbmi\b|waist circumference|"
        r"parathyroid hormone|\bpth\b|vascular resistance|ejection fraction|"
        r"intraocular pressure|\biop\b|glucose|insulin|testosterone|estradiol|"
        r"(?:change|reduction|percent change) from baseline in\b)", re.I)),

    ("PRO", re.compile(
        r"\b(patient[- ]reported|quality of life|\bqol\b|\bhrqol\b|"
        r"\bsf[- ]?36\b|\beq[- ]?5d\b|questionnaire|patient global|"
        r"subject global|satisfaction|symptom diary|\bvas\b|"
        r"visual analog(?:ue)? scale)", re.I)),

    ("CLINICAL_SCORE", re.compile(
        r"\b(acr ?\d{2}|pasi ?\d{2}|easi ?\d{2}|\basas\b|\bdas28\b|"
        r"adas[- ]cog|\bmmse\b|\bcdr[- ]sb\b|\bedss\b|\bupdrs\b|\bpanss\b|"
        r"\bmadrs\b|\bham[- ]?d\b|\bymrs\b|6[- ]minute walk|\b6mwd\b|"
        r"\bnihss\b|\bmayo score\b|\bcdai\b|\bsledai\b|"
        r"(?:score|scale|index|questionnaire|assessment)\b)", re.I)),
]


def classify_measure(measure: str | None) -> str:
    """Classify a single outcome-measure string into the closed vocabulary."""
    if not measure or pd.isna(measure):
        return "UNKNOWN"
    text = str(measure)
    for archetype, pattern in _RULES:
        if pattern.search(text):
            return archetype
    return "UNKNOWN"


def classify_primary(measures_str: str | None) -> str:
    """Archetype of the trial's FIRST primary outcome — 'the' primary endpoint.

    Falls through to later primary outcomes only if the first abstains, so a
    trial whose first endpoint is unparseable is not silently labelled UNKNOWN
    when its second says 'overall survival'.
    """
    if not measures_str or pd.isna(measures_str):
        return "UNKNOWN"
    for measure in str(measures_str).split("|"):
        archetype = classify_measure(measure)
        if archetype != "UNKNOWN":
            return archetype
    return "UNKNOWN"


def archetype_flags(measures_str: str | None) -> dict[str, int]:
    """Binary presence flag per archetype across ALL primary outcomes.

    A trial can carry both a survival and a safety endpoint; the first-endpoint
    label alone would lose that. `endpoint_has_SURVIVAL` is the single most
    duration-predictive flag in the set.
    """
    found: set[str] = set()
    if measures_str and not pd.isna(measures_str):
        for measure in str(measures_str).split("|"):
            a = classify_measure(measure)
            if a != "UNKNOWN":
                found.add(a)
    return {
        f"endpoint_has_{a}": int(a in found)
        for a in ARCHETYPES if a != "UNKNOWN"
    }


def validate(label: str) -> str:
    """Gate for any externally-produced label (e.g. an LLM fallback).

    Out-of-vocabulary values are rejected to UNKNOWN rather than admitted —
    a free-form label entering the feature set would silently create a category
    the model has never seen at training time.
    """
    if label in ARCHETYPES:
        return label
    log.warning("Rejected out-of-vocabulary endpoint label %r", label)
    return "UNKNOWN"


def add_endpoint_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach `endpoint_archetype` and the per-archetype flags to a frame."""
    col = "primary_outcome_measures"
    if col not in df.columns:
        log.warning("%s absent — endpoint features will be UNKNOWN", col)
        df["endpoint_archetype"] = "UNKNOWN"
        for a in ARCHETYPES:
            if a != "UNKNOWN":
                df[f"endpoint_has_{a}"] = 0
        return df

    df["endpoint_archetype"] = df[col].apply(classify_primary)
    flags = pd.DataFrame(list(df[col].apply(archetype_flags)), index=df.index)
    for c in flags.columns:
        df[c] = flags[c]
    return df


# ── Optional LLM fallback ─────────────────────────────────────────────────────
#
# OFF by default (ENDPOINT_LLM_FALLBACK=1 to enable). The deterministic rules
# above abstain on 8.6% / 21.0% / 23.9% of P1 / P2 / P3, so the fallback is a
# refinement, not a dependency — the feature set is complete without it.
#
# Two constraints if it is switched on:
#   * it may only ASSIGN A LABEL FROM `ARCHETYPES` — every response goes through
#     validate() and an out-of-vocabulary answer is rejected to UNKNOWN, never
#     admitted as a new category the model has not been trained on;
#   * verdicts are cached by NCT id, so classification runs once per trial
#     rather than once per training run.

_LLM_CACHE_PATH = "data/cache/endpoint_llm_verdicts.json"


def _load_llm_cache() -> dict[str, str]:
    import json
    from pathlib import Path

    p = Path(_LLM_CACHE_PATH)
    return json.loads(p.read_text()) if p.exists() else {}


def _save_llm_cache(cache: dict[str, str]) -> None:
    import json
    from pathlib import Path

    p = Path(_LLM_CACHE_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cache, indent=2, sort_keys=True))


def llm_fallback_enabled() -> bool:
    import os

    return os.getenv("ENDPOINT_LLM_FALLBACK", "").strip() in {"1", "true", "yes"}


def resolve_abstentions(df: pd.DataFrame, classify_fn=None) -> pd.DataFrame:
    """Fill UNKNOWN archetypes via `classify_fn`, cached by NCT id.

    `classify_fn(measure_text) -> str` is injected rather than imported so the
    model call site stays outside this module and the whole path is testable
    without a network. No-ops unless ENDPOINT_LLM_FALLBACK is set.
    """
    if not llm_fallback_enabled() or classify_fn is None:
        return df
    if "endpoint_archetype" not in df.columns:
        df = add_endpoint_features(df)

    cache = _load_llm_cache()
    todo = df.index[df["endpoint_archetype"] == "UNKNOWN"]
    resolved = 0
    for idx in todo:
        nct = str(df.at[idx, "nct_id"]) if "nct_id" in df.columns else None
        if nct and nct in cache:
            label = cache[nct]
        else:
            measures = df.at[idx, "primary_outcome_measures"]
            if not measures or pd.isna(measures):
                continue
            label = validate(str(classify_fn(str(measures))).strip().upper())
            if nct:
                cache[nct] = label
        if label != "UNKNOWN":
            df.at[idx, "endpoint_archetype"] = label
            resolved += 1
    _save_llm_cache(cache)
    log.info("LLM fallback resolved %d of %d abstentions", resolved, len(todo))
    return df


def coverage_report(df: pd.DataFrame) -> pd.Series:
    """Share of trials per archetype — the abstention rate is the UNKNOWN row."""
    if "endpoint_archetype" not in df.columns:
        df = add_endpoint_features(df.copy())
    return df["endpoint_archetype"].value_counts(normalize=True).round(4)
