from __future__ import annotations
"""ClinicalTrials.gov API v2 client."""
import asyncio
import logging
from typing import Any

import httpx

from backend.config import settings
from backend.preprocessing.features import count_criteria

log = logging.getLogger(__name__)

_FIELDS = ",".join([
    "protocolSection.identificationModule.nctId",
    "protocolSection.statusModule.startDateStruct",
    "protocolSection.statusModule.primaryCompletionDateStruct",
    "protocolSection.statusModule.overallStatus",
    "protocolSection.designModule.studyType",
    "protocolSection.designModule.phases",
    "protocolSection.designModule.enrollmentInfo",
    "protocolSection.designModule.designInfo",
    "protocolSection.conditionsModule.conditions",
    "protocolSection.armsInterventionsModule.interventions",
    "protocolSection.armsInterventionsModule.armGroups",
    "protocolSection.eligibilityModule.healthyVolunteers",
    "protocolSection.eligibilityModule.sex",
    "protocolSection.eligibilityModule.minimumAge",
    "protocolSection.eligibilityModule.maximumAge",
    "protocolSection.eligibilityModule.eligibilityCriteria",
    "protocolSection.sponsorCollaboratorsModule.leadSponsor",
    "protocolSection.sponsorCollaboratorsModule.collaborators",
    "protocolSection.outcomesModule.primaryOutcomes",
    "protocolSection.outcomesModule.secondaryOutcomes",
    "protocolSection.contactsLocationsModule.locations",
    "protocolSection.descriptionModule.briefSummary",
])


#: Trials that have finished and therefore have an ACTUAL primary completion date.
COMPLETED_STATUSES = ["COMPLETED"]

#: Trials still running. Their primary completion date is ESTIMATED — the
#: sponsor's projection, not an outcome — so it must never be used as an event
#: time. They enter the model as RIGHT-CENSORED observations: all we know is
#: that the trial has already lasted at least (today − start) and is not done.
#: Excluding them is what makes recent history look artificially fast, because
#: the only recent trials that have finished are the quick ones.
ONGOING_STATUSES = ["RECRUITING", "ACTIVE_NOT_RECRUITING", "ENROLLING_BY_INVITATION"]


async def fetch_studies(
    phases: list[str],
    max_records: int = 5000,
    statuses: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Return raw study dicts from the ClinicalTrials.gov v2 API.

    Uses comma-separated phase values and status filter only;
    industry/drug filtering is done in flatten_study() to avoid
    complex query.term syntax that varies across API versions.
    """
    # CT.gov v2 uses AREA[] query syntax for phase filtering
    phase_terms = " OR ".join(f"AREA[Phase]{p}" for p in phases)
    query_term = (
        f"({phase_terms}) "
        "AND AREA[StudyType]INTERVENTIONAL "
        "AND AREA[LeadSponsorClass]INDUSTRY"
    )
    params: dict[str, Any] = {
        "filter.overallStatus": "|".join(statuses or COMPLETED_STATUSES),
        "query.term": query_term,
        "fields": _FIELDS,
        "pageSize": min(settings.ct_api_page_size, 1000),
        "format": "json",
    }

    studies: list[dict] = []
    async with httpx.AsyncClient(timeout=60) as client:
        for page in range(settings.ct_api_max_pages):
            try:
                resp = await client.get(
                    f"{settings.ct_api_base}/studies",
                    params=params,
                )
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                body = exc.response.text[:500]
                log.error("CT.gov API HTTP %s on page %d: %s", exc.response.status_code, page, body)
                break
            except httpx.HTTPError as exc:
                log.error("CT.gov API error on page %d: %s", page, exc)
                break

            data = resp.json()
            page_studies = data.get("studies", [])
            studies.extend(page_studies)
            log.info("Page %d: +%d studies (total %d)", page, len(page_studies), len(studies))

            if len(studies) >= max_records:
                break

            next_token = data.get("nextPageToken")
            if not next_token:
                break
            params["pageToken"] = next_token

    return studies[:max_records]


def _safe_date(struct: dict | None) -> str | None:
    if not struct:
        return None
    return struct.get("date")


def _date_type(struct: dict | None) -> str:
    """'ACTUAL' or 'ESTIMATED'. Ongoing trials publish an ESTIMATED primary
    completion date; treating that as an observed outcome trains the model on
    the sponsor's projection rather than on what happened."""
    if not struct:
        return "UNKNOWN"
    return struct.get("type", "UNKNOWN")


def _intervention_type(interventions: list[dict]) -> str:
    types = {iv.get("type", "") for iv in interventions}
    if "BIOLOGICAL" in types:
        return "BIOLOGICAL"
    if "DRUG" in types:
        return "DRUG"
    return "OTHER"


def _encode_locations(locations: list[dict]) -> str:
    """Serialise sites as 'facility^city^country' records, pipe-separated.

    Retained in full for the site-level layer. v1 collapsed this to a distinct
    country list and then counted the countries as if they were sites.
    """
    out = []
    for loc in locations:
        out.append("^".join([
            (loc.get("facility") or "").replace("|", " ").replace("^", " "),
            (loc.get("city") or "").replace("|", " ").replace("^", " "),
            (loc.get("country") or "").replace("|", " ").replace("^", " "),
        ]))
    return "|".join(out)


def _outcome_field(outcomes: list[dict], key: str) -> str:
    return "|".join(
        str(o.get(key, "")).replace("|", " ") for o in outcomes if o.get(key)
    )


def flatten_study(study: dict) -> dict | None:
    """Flatten a raw API study dict to a plain row dict. Returns None if unusable."""
    proto = study.get("protocolSection", {})

    status = proto.get("statusModule", {})
    design = proto.get("designModule", {})
    design_info = design.get("designInfo", {})
    outcomes = proto.get("outcomesModule", {})
    contacts = proto.get("contactsLocationsModule", {})
    sponsor_mod = proto.get("sponsorCollaboratorsModule", {})
    eligibility = proto.get("eligibilityModule", {})

    nct_id = proto.get("identificationModule", {}).get("nctId")
    start_date = _safe_date(status.get("startDateStruct"))
    primary_date = _safe_date(status.get("primaryCompletionDateStruct"))
    overall_status = status.get("overallStatus", "")

    # An ongoing trial needs only a start date — its endpoint is unobserved and
    # will be censored downstream. A finished trial needs both.
    is_ongoing = overall_status in ONGOING_STATUSES
    if not start_date:
        return None
    if not primary_date and not is_ongoing:
        return None
    if sponsor_mod.get("leadSponsor", {}).get("class") != "INDUSTRY":
        return None

    interventions = proto.get("armsInterventionsModule", {}).get("interventions", [])
    drug_type = _intervention_type(interventions)
    if drug_type == "OTHER":
        return None

    locations = contacts.get("locations", [])
    countries = list({loc.get("country", "") for loc in locations if loc.get("country")})

    hv_raw = eligibility.get("healthyVolunteers", "")
    is_hv = isinstance(hv_raw, str) and "yes" in hv_raw.lower()

    conditions = proto.get("conditionsModule", {}).get("conditions", [])
    is_hv = is_hv or any("healthy" in c.lower() for c in conditions)

    primary_outcomes = outcomes.get("primaryOutcomes", [])
    criteria = eligibility.get("eligibilityCriteria", "") or ""
    arm_groups = proto.get("armsInterventionsModule", {}).get("armGroups", [])

    return {
        "nct_id": nct_id,
        "Start Date": start_date,
        "Primary Completion Date": primary_date,
        "overall_status": overall_status,
        "is_ongoing": int(is_ongoing),
        # ACTUAL vs ESTIMATED. Only an ACTUAL primary completion date is an
        # observed event; everything else is censored.
        "primary_completion_type": _date_type(status.get("primaryCompletionDateStruct")),
        "Phases": "|".join(design.get("phases", [])),
        "Enrollment": design.get("enrollmentInfo", {}).get("count"),
        "Drug_Type": drug_type,
        "Allocation": design_info.get("allocation"),
        "Intervention_Model": design_info.get("interventionModel"),
        "Masking": design_info.get("maskingInfo", {}).get("masking"),
        "Primary_Purpose": design_info.get("primaryPurpose"),
        # Real arm count from armGroups. v1 hardcoded None here, so the feature
        # imputed to a constant and carried no information.
        "number_of_arms": len(arm_groups) or None,
        "conditions": "|".join(conditions),
        "countries": "|".join(countries),
        # site_count is the number of SITES. v1 counted countries instead, which
        # put training values in 1..20 while inference passed real site counts
        # (P3 default 40) — far outside the trained range, where a forest is flat.
        "site_count": len(locations),
        "country_count": len(countries),
        "locations": _encode_locations(locations),
        "is_hv": int(is_hv),
        "Sex": eligibility.get("sex", "ALL"),
        "lead_sponsor": sponsor_mod.get("leadSponsor", {}).get("name", ""),
        "has_collaborators": int(bool(sponsor_mod.get("collaborators"))),
        "n_collaborators": len(sponsor_mod.get("collaborators", []) or []),
        "brief_summary": proto.get("descriptionModule", {}).get("briefSummary", ""),
        "total_primary_outcomes": len(primary_outcomes),
        "total_secondary_outcomes": len(outcomes.get("secondaryOutcomes", [])),
        # Endpoint text — the raw material for archetype classification and for
        # reading follow-up length straight off the protocol.
        "primary_outcome_measures": _outcome_field(primary_outcomes, "measure"),
        "primary_outcome_timeframes": _outcome_field(primary_outcomes, "timeFrame"),
        # Eligibility restrictiveness.
        "min_age_raw": eligibility.get("minimumAge", ""),
        "max_age_raw": eligibility.get("maximumAge", ""),
        "criteria_chars": len(criteria),
        # Retain the text itself, not just its length. TrialEnroll (2024, 31k
        # trials) shows the criteria WORDING carries signal that counts do not —
        # "prior systemic therapy" and "washout" restrict a population in ways a
        # bullet count cannot see. Truncated to keep the cache compact.
        "criteria_text": criteria[:4000],
        "n_inclusion_criteria": count_criteria(criteria, "inclusion"),
        "n_exclusion_criteria": count_criteria(criteria, "exclusion"),
    }
