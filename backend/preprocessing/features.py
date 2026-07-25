from __future__ import annotations
"""Feature engineering: therapeutic area, region, phase normalisation."""
import re

import pandas as pd

from backend.constants import THERAPEUTIC_AREAS, REGIONS

# ── Therapeutic Area ──────────────────────────────────────────────────────────

_MESH_MAP: dict[str, str] = {
    "bacterial infections and mycoses": "Infectious Diseases",
    "virus diseases": "Infectious Diseases",
}

_KEYWORD_MAP: dict[str, list[str]] = {
    "Oncology/Solid Tumours": [
        "cancer", "melanoma", "tumor", "tumour", "carcinoma", "nsclc", "crc",
        "hcc", "neoplasm", "oncology", "carcinoid", "solid malignancies",
    ],
    "CNS/Neurology": [
        "alzheimer", "parkinson", "huntington", "stroke", "epilep", "migraine",
        "adhd", "seizure", "tinnitus", "glioma", "glioblastoma", "brain",
        "neuralgia", "neuropathy", "schizophrenia", "bipolar", "anxiety",
        "depression", "depressive disorder", "cognition", "insomnia", "autism",
        "multiple sclerosis", "amyotrophic lateral sclerosis", "spinal",
    ],
    "Immunology": [
        "rheumatoid arthritis", "lupus", "spondylitis", "gout",
        "graft versus host disease", "gvhd", "colitis", "inflamma",
        "inflammatory bowel disease", "celiac", "allergy", "allergic",
        "sjogren",
    ],
    "Cardiovascular": [
        "heart failure", "coronary", "atrial fibrillation", "cardiomyopathy",
        "hypertension", "cardiac", "atherosclerosis", "angina", "arrhythm",
        "myocardial", "artery", "cardiovascular",
    ],
    "Metabolic": [
        "diabetes", "obesity", "hyperlipidemia", "metabolic", "nash",
        "nonalcoholic steatohepatitis", "chronic kidney disease", "cholesterol",
        "prediabetes", "nafld", "dyslipid",
    ],
    "Infectious Diseases": [
        "hiv", "aids", "tuberculosis", "virus", "infecti", "herpes", "sepsis",
        "covid-19", "covid", "influenza", "malaria", "bacterial", "fungal",
        "sars-cov-2", "dengue",
    ],
    "Endocrinology": [
        "thyroid", "hypothyroidism", "endocrine", "adrenal", "pituitary",
        "hormone", "menopause", "cushing", "graves", "hashimoto",
    ],
    "Gastroenterology": [
        "irritable bowel", "ibs", "gerd", "gastritis", "gastroesophageal",
        "cirrhosis", "hepatitis", "crohn", "ulcerative colitis", "peptic",
        "esophagitis", "cholangitis", "liver", "hepatic", "constipation",
    ],
    "Haematology": [
        "leukemia", "leukaemia", "lymphoma", "anaemia", "anemia",
        "myelodysplastic", "hemophilia", "haemophilia", "thrombocytopenia",
        "sickle cell", "thalassemia", "myeloma", "neutropenia",
    ],
    "Respiratory": [
        "asthma", "copd", "pulmonary", "respiratory", "sinusitis",
        "bronchitis", "pah", "fibrosis", "lung", "bronchiectasis",
    ],
    "Urology/Male": [
        "benign prostatic", "bph", "overactive bladder", "renal impairment",
        "nephropathy", "chronic renal", "kidney disease",
    ],
    "Ophthalmology": [
        "glaucoma", "macular", "retinopathy", "vision", "ophthalmic",
        "conjunctivitis", "dry eye", "retinitis", "ocular",
    ],
    "Dermatology/Connective Tissue Diseases": [
        "dermatitis", "eczema", "psoriasis", "acne", "pruritus",
        "vitiligo", "alopecia", "rosacea", "dermato", "scleroderma",
        "urticaria", "skin",
    ],
    "Congenital/Hereditary/Rare Diseases": [
        "cystic fibrosis", "duchenne", "gaucher", "rare disease",
        "muscular dystrophy", "phenylketonuria", "pompe", "congenital",
    ],
    "Rheumatology": [
        "arthritis", "fractures", "osteoporosis", "joint replacement",
        "degenerative disc",
    ],
    "Chemically-Induced Disorders": [
        "substance abuse", "addiction", "opioid", "cocaine", "smoking",
        "alcohol use disorder",
    ],
    "Urology/Female Health": [
        "contraception", "uterine fibroids", "vaginal atrophy",
        "female infertility",
    ],
    "Pathological Conditions, Signs and Symptoms": [
        "postoperative pain", "low back pain", "lower back pain",
        "dyspepsia", "scar",
    ],
    "Stomatognathic Diseases": ["oral mucositis", "periodontal"],
    "Wounds and Injuries": ["wound", "ankle injuries"],
    "Otorhinolaryngology": ["meniere", "nasal congestion", "nasal polyposis"],
}


def map_condition_to_area(condition: str) -> str:
    norm = condition.lower().replace("'", "")
    mapped = _MESH_MAP.get(norm)
    if mapped:
        return mapped
    for area, keywords in _KEYWORD_MAP.items():
        for kw in keywords:
            if kw.lower() in norm:
                return area
    return "Other"


def assign_therapeutic_area(conditions_str: str | None) -> str:
    if not conditions_str or pd.isna(conditions_str):
        return "Other"
    conditions = [c.strip() for c in conditions_str.split("|") if c.strip()]
    areas = list({map_condition_to_area(c) for c in conditions})
    return "|".join(areas) if areas else "Other"


# ── Region ────────────────────────────────────────────────────────────────────

_REGION_KEYWORDS: dict[str, list[str]] = {
    "US": ["united states", "usa", "california", "texas", "new york",
           "boston", "chicago", "florida", "los angeles", "san francisco"],
    "North America (excluding US)": ["canada", "toronto", "vancouver", "montreal"],
    "Europe": ["germany", "france", "italy", "spain", "united kingdom", "uk",
               "england", "netherlands", "belgium", "sweden", "denmark",
               "norway", "finland", "poland", "greece", "switzerland"],
    "Asia": ["china", "japan", "india", "south korea", "singapore", "taiwan",
             "hong kong", "thailand", "malaysia"],
    "Australia/Oceania": ["australia", "sydney", "melbourne", "new zealand"],
    "Latin America": ["brazil", "argentina", "chile", "colombia", "mexico",
                      "peru", "venezuela"],
    "MENA": ["israel", "saudi arabia", "uae", "united arab emirates", "egypt",
             "jordan", "lebanon", "morocco"],
    "Sub-Saharan Africa": ["south africa", "nigeria", "kenya", "ethiopia", "ghana"],
}


def map_country_to_region(country: str) -> str:
    cl = country.lower()
    for region, keywords in _REGION_KEYWORDS.items():
        if any(kw in cl for kw in keywords):
            return region
    return "ROW"


def assign_region(countries_str: str | None) -> str:
    if not countries_str or pd.isna(countries_str):
        return "ROW"
    countries = [c.strip() for c in countries_str.split("|") if c.strip()]
    regions = list({map_country_to_region(c) for c in countries})
    return "|".join(regions) if regions else "ROW"


# ── Phase normalisation ───────────────────────────────────────────────────────

_PHASE_NUM_MAP = {
    "PHASE1": 1.0, "EARLY_PHASE1": 0.5,
    "PHASE2": 2.0, "PHASE3": 3.0,
    "PHASE1_PHASE2": 1.5, "PHASE2_PHASE3": 2.5,
    # legacy pipe-separated variants from older AACT exports
    "PHASE1|PHASE2": 1.5, "PHASE2|PHASE3": 2.5,
}


def normalise_phase(raw: str | None) -> float | None:
    if not raw or pd.isna(raw):
        return None
    return _PHASE_NUM_MAP.get(str(raw).upper().replace(" ", "_"))


# ── SAD / MAD (Phase 1 sub-classification) ────────────────────────────────────

def classify_sad_mad(summary: str | None) -> str:
    if not summary or pd.isna(summary):
        return "None"
    text = summary.lower()
    sad = "single ascending" in text or "single-ascending" in text
    mad = "multiple ascending" in text or "multiple-ascending" in text
    if sad and mad:
        return "SAD_MAD"
    if sad:
        return "SAD"
    if mad:
        return "MAD"
    return "None"


# ── Eligibility parsing ───────────────────────────────────────────────────────

_AGE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(year|month|week|day|hour|minute)", re.I)
_AGE_UNIT_YEARS = {
    "year": 1.0, "month": 1 / 12, "week": 1 / 52.18,
    "day": 1 / 365.25, "hour": 1 / 8766, "minute": 1 / 525960,
}


def parse_age_years(raw: str | None) -> float | None:
    """'18 Years' → 18.0, '6 Months' → 0.5, 'N/A' → None."""
    if not raw or pd.isna(raw):
        return None
    m = _AGE_RE.search(str(raw))
    if not m:
        return None
    return round(float(m.group(1)) * _AGE_UNIT_YEARS[m.group(2).lower()], 4)


def count_criteria(criteria: str | None, section: str) -> int:
    """Count bullets under the inclusion or exclusion heading.

    Eligibility text is free-form, but essentially every industry protocol uses
    'Inclusion Criteria:' / 'Exclusion Criteria:' headings with one bullet per
    line. Restrictiveness is a first-order driver of recruitment speed and is
    absent from the v1 feature set entirely.
    """
    if not criteria or pd.isna(criteria):
        return 0
    text = str(criteria)
    lowered = text.lower()
    start = lowered.find(f"{section.lower()} criteria")
    if start == -1:
        return 0
    other = "exclusion" if section.lower() == "inclusion" else "inclusion"
    end = lowered.find(f"{other} criteria", start + 1)
    block = text[start: end if end != -1 else len(text)]
    bullets = [
        ln.strip() for ln in block.splitlines()
        if len(ln.strip()) > 3 and not ln.strip().lower().startswith(f"{section.lower()} criteria")
    ]
    return len(bullets)


# ── Outcome time-frame parsing ────────────────────────────────────────────────

_TIMEFRAME_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(year|month|week|day)s?", re.I
)
_TF_UNIT_MONTHS = {"year": 12.0, "month": 1.0, "week": 1 / 4.345, "day": 1 / 30.44}


def parse_followup_months(timeframe: str | None) -> float | None:
    """Longest follow-up horizon mentioned in an outcome time frame, in months.

    'Up to 60 months' → 60.0; 'Baseline to Week 12' → 2.76. This is close to a
    direct read on how much of a trial's duration is follow-up rather than
    recruitment — the thing that makes Oncology Phase 3 run 38 months while
    Dermatology runs 9.5 — and v1 discarded it.
    """
    if not timeframe or pd.isna(timeframe):
        return None
    months = [
        float(v) * _TF_UNIT_MONTHS[u.lower()]
        for v, u in _TIMEFRAME_RE.findall(str(timeframe))
    ]
    return round(max(months), 2) if months else None


def max_followup_months(timeframes_str: str | None) -> float | None:
    """Max follow-up across a pipe-separated set of outcome time frames."""
    if not timeframes_str or pd.isna(timeframes_str):
        return None
    vals = [
        v for v in (parse_followup_months(t) for t in str(timeframes_str).split("|"))
        if v is not None
    ]
    return max(vals) if vals else None


# ── One-hot helpers ───────────────────────────────────────────────────────────

def one_hot_pipe_col(df: pd.DataFrame, col: str, categories: list[str]) -> pd.DataFrame:
    """Expand a pipe-separated column into binary indicator columns."""
    dummies = df[col].str.get_dummies(sep="|")
    for cat in categories:
        if cat not in dummies.columns:
            dummies[cat] = 0
    return dummies[categories]
