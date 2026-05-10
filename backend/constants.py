PHASES = {
    "P1HV": {"label": "Phase 1 – Healthy Volunteers", "api_phases": ["PHASE1", "EARLY_PHASE1"], "hv": True},
    "P1":   {"label": "Phase 1 – Patient",             "api_phases": ["PHASE1", "EARLY_PHASE1"], "hv": False},
    "P2":   {"label": "Phase 2",                       "api_phases": ["PHASE2"], "hv": False},
    "P3":   {"label": "Phase 3",                       "api_phases": ["PHASE3"], "hv": False},
}

THERAPEUTIC_AREAS = [
    "Cardiovascular",
    "CNS/Neurology",
    "Chemically-Induced Disorders",
    "Congenital/Hereditary/Rare Diseases",
    "Dermatology/Connective Tissue Diseases",
    "Endocrinology",
    "Gastroenterology",
    "Haematology",
    "Immunology",
    "Infectious Diseases",
    "Metabolic",
    "Oncology/Solid Tumours",
    "Ophthalmology",
    "Otorhinolaryngology",
    "Pathological Conditions, Signs and Symptoms",
    "Respiratory",
    "Rheumatology",
    "Stomatognathic Diseases",
    "Urology/Female Health",
    "Urology/Male",
    "Wounds and Injuries",
    "Other",
]

REGIONS = [
    "US",
    "North America (excluding US)",
    "Europe",
    "Asia",
    "Australia/Oceania",
    "Latin America",
    "MENA",
    "Sub-Saharan Africa",
    "ROW",
]

# Median defaults per phase used when individual features are not supplied
PHASE_DEFAULTS = {
    "P1HV": {"Enrollment": 24,  "site_count": 3,  "total_primary_outcomes": 3, "total_secondary_outcomes": 5,  "number_of_arms": 2},
    "P1":   {"Enrollment": 40,  "site_count": 5,  "total_primary_outcomes": 3, "total_secondary_outcomes": 5,  "number_of_arms": 2},
    "P2":   {"Enrollment": 120, "site_count": 15, "total_primary_outcomes": 2, "total_secondary_outcomes": 8,  "number_of_arms": 2},
    "P3":   {"Enrollment": 300, "site_count": 40, "total_primary_outcomes": 2, "total_secondary_outcomes": 10, "number_of_arms": 2},
}
