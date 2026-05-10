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
    SELECT nct_id, COUNT(*) AS total_primary_outcomes
    FROM design_outcomes WHERE outcome_type = 'Primary' GROUP BY nct_id
),
agg_secondary_outcomes AS (
    SELECT nct_id, COUNT(*) AS total_secondary_outcomes
    FROM design_outcomes WHERE outcome_type = 'Secondary' GROUP BY nct_id
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
    s.number_of_arms,
    ei.healthy_volunteers,
    d.allocation,
    d.intervention_model,
    d.masking,
    d.primary_purpose,
    ac.countries,
    aco.conditions,
    COALESCE(po.total_primary_outcomes, 0)   AS total_primary_outcomes,
    COALESCE(so.total_secondary_outcomes, 0) AS total_secondary_outcomes,
    b.brief_summary,
    sp.agency_class
FROM studies s
LEFT JOIN agg_countries ac       ON s.nct_id = ac.nct_id
LEFT JOIN agg_conditions aco     ON s.nct_id = aco.nct_id
LEFT JOIN agg_interventions ai   ON s.nct_id = ai.nct_id
LEFT JOIN agg_sponsors sp        ON s.nct_id = sp.nct_id
LEFT JOIN agg_primary_outcomes po ON s.nct_id = po.nct_id
LEFT JOIN agg_secondary_outcomes so ON s.nct_id = so.nct_id
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


def fetch_studies(phases: list[str]) -> pd.DataFrame:
    phase_filter = ", ".join(f"'{p}'" for p in phases)
    sql = _SQL.format(phase_filter=phase_filter)
    try:
        engine = _engine()
        with engine.connect() as conn:
            df = pd.read_sql(text(sql), conn)
        log.info("Postgres: fetched %d rows for phases %s", len(df), phases)
        return df
    except Exception as exc:
        log.error("Postgres fetch failed: %s", exc)
        raise
