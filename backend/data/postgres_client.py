from __future__ import annotations
"""AACT PostgreSQL client (fallback data source)."""
import logging
import urllib.parse
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, text

from backend.config import settings

log = logging.getLogger(__name__)

_SQL = """
WITH agg_countries AS (
    SELECT nct_id, STRING_AGG(DISTINCT name, '|') AS countries
    FROM countries GROUP BY nct_id
),
agg_conditions AS (
    SELECT nct_id, STRING_AGG(DISTINCT name, '|') AS conditions
    FROM conditions GROUP BY nct_id
),
agg_interventions AS (
    SELECT nct_id,
           STRING_AGG(intervention_type, '|') AS intervention_types
    FROM interventions GROUP BY nct_id
),
agg_sponsors AS (
    SELECT nct_id, MAX(agency_class) AS agency_class
    FROM sponsors
    WHERE lead_or_collaborator = 'LEAD'
    GROUP BY nct_id
),
agg_primary_outcomes AS (
    SELECT nct_id,
           COUNT(*)                              AS total_primary_outcomes,
           STRING_AGG(measure, '|')              AS primary_outcome_measures,
           STRING_AGG(time_frame, '|')           AS primary_outcome_timeframes
    FROM design_outcomes WHERE outcome_type = 'Primary' GROUP BY nct_id
),
agg_secondary_outcomes AS (
    SELECT nct_id, COUNT(*) AS total_secondary_outcomes
    FROM design_outcomes WHERE outcome_type = 'Secondary' GROUP BY nct_id
),
-- Real site count + the site list itself, from the facilities table. The API
-- path takes this from contactsLocationsModule.locations; both must agree.
agg_facilities AS (
    SELECT nct_id,
           COUNT(*)                              AS site_count,
           COUNT(DISTINCT country)               AS country_count,
           STRING_AGG(
               COALESCE(name, '') || '^' || COALESCE(city, '') || '^' || COALESCE(country, ''),
               '|'
           )                                     AS locations
    FROM facilities GROUP BY nct_id
),
agg_arms AS (
    SELECT nct_id, COUNT(*) AS design_group_count
    FROM design_groups GROUP BY nct_id
),
agg_collaborators AS (
    SELECT nct_id, COUNT(*) AS n_collaborators
    FROM sponsors WHERE lead_or_collaborator = 'COLLABORATOR' GROUP BY nct_id
),
lead_sponsor AS (
    SELECT nct_id, MAX(name) AS lead_sponsor
    FROM sponsors WHERE lead_or_collaborator = 'LEAD' GROUP BY nct_id
),
brief AS (
    SELECT nct_id, description AS brief_summary FROM brief_summaries
)
SELECT
    s.nct_id,
    s.start_date         AS "Start Date",
    s.primary_completion_date AS "Primary Completion Date",
    s.phase              AS "Phases",
    s.enrollment         AS "Enrollment",
    COALESCE(s.number_of_arms, ag.design_group_count) AS number_of_arms,
    ei.healthy_volunteers,
    ei.gender            AS "Sex",
    ei.minimum_age       AS min_age_raw,
    ei.maximum_age       AS max_age_raw,
    ei.criteria,
    d.allocation,
    d.intervention_model,
    d.masking,
    d.primary_purpose,
    ac.countries,
    aco.conditions,
    COALESCE(af.site_count, 0)               AS site_count,
    COALESCE(af.country_count, 0)            AS country_count,
    af.locations,
    COALESCE(po.total_primary_outcomes, 0)   AS total_primary_outcomes,
    COALESCE(so.total_secondary_outcomes, 0) AS total_secondary_outcomes,
    po.primary_outcome_measures,
    po.primary_outcome_timeframes,
    COALESCE(acol.n_collaborators, 0)        AS n_collaborators,
    ls.lead_sponsor,
    b.brief_summary,
    sp.agency_class
FROM studies s
LEFT JOIN agg_countries ac       ON s.nct_id = ac.nct_id
LEFT JOIN agg_conditions aco     ON s.nct_id = aco.nct_id
LEFT JOIN agg_interventions ai   ON s.nct_id = ai.nct_id
LEFT JOIN agg_sponsors sp        ON s.nct_id = sp.nct_id
LEFT JOIN agg_primary_outcomes po ON s.nct_id = po.nct_id
LEFT JOIN agg_secondary_outcomes so ON s.nct_id = so.nct_id
LEFT JOIN agg_facilities af      ON s.nct_id = af.nct_id
LEFT JOIN agg_arms ag            ON s.nct_id = ag.nct_id
LEFT JOIN agg_collaborators acol ON s.nct_id = acol.nct_id
LEFT JOIN lead_sponsor ls        ON s.nct_id = ls.nct_id
LEFT JOIN designs d              ON s.nct_id = d.nct_id
LEFT JOIN eligibilities ei       ON s.nct_id = ei.nct_id
LEFT JOIN brief b                ON s.nct_id = b.nct_id
WHERE s.phase IN ({phase_filter})
  AND s.study_type = 'INTERVENTIONAL'
  AND s.overall_status = 'Completed'
  AND sp.agency_class = 'INDUSTRY'
  AND (ai.intervention_types LIKE '%%DRUG%%' OR ai.intervention_types LIKE '%%BIOLOGICAL%%')
"""


def _engine():
    user = urllib.parse.quote_plus(settings.db_user)
    pw = urllib.parse.quote_plus(settings.db_pass)
    url = f"postgresql://{user}:{pw}@{settings.db_host}:{settings.db_port}/{settings.db_name}"
    return create_engine(url)


def _derive_shared_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Bring the AACT frame up to the same schema the API path emits.

    AACT hands back the raw eligibility text; the API path already counts the
    criteria in flatten_study. Deriving them here keeps a single downstream
    contract so cleaner/pipeline never have to know which source they got.
    """
    from backend.preprocessing.features import count_criteria

    criteria = df.get("criteria", pd.Series([""] * len(df), index=df.index)).fillna("")
    df["criteria_chars"] = criteria.str.len()
    df["n_inclusion_criteria"] = criteria.apply(lambda c: count_criteria(c, "inclusion"))
    df["n_exclusion_criteria"] = criteria.apply(lambda c: count_criteria(c, "exclusion"))

    hv = df.get("healthy_volunteers", pd.Series([""] * len(df), index=df.index))
    df["is_hv"] = (
        hv.fillna("").astype(str).str.lower().str.contains("yes")
        | df.get("conditions", pd.Series([""] * len(df), index=df.index))
          .fillna("").str.lower().str.contains("healthy")
    ).astype(int)

    df["has_collaborators"] = (
        pd.to_numeric(df.get("n_collaborators", 0), errors="coerce").fillna(0) > 0
    ).astype(int)

    # Match the API path's Drug_Type, derived from intervention types.
    if "Drug_Type" not in df.columns:
        df["Drug_Type"] = "DRUG"

    rename = {"allocation": "Allocation", "intervention_model": "Intervention_Model",
              "masking": "Masking", "primary_purpose": "Primary_Purpose"}
    return df.rename(columns={k: v for k, v in rename.items() if k in df.columns})


def fetch_studies(phases: list[str]) -> pd.DataFrame:
    phase_filter = ", ".join(f"'{p}'" for p in phases)
    sql = _SQL.format(phase_filter=phase_filter)
    try:
        engine = _engine()
        with engine.connect() as conn:
            df = pd.read_sql(text(sql), conn)
        log.info("Postgres: fetched %d rows for phases %s", len(df), phases)
        return _derive_shared_columns(df)
    except Exception as exc:
        log.error("Postgres fetch failed: %s", exc)
        raise
