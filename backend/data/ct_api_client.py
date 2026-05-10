from __future__ import annotations
"""ClinicalTrials.gov API v2 client."""
import asyncio
import logging
from typing import Any

import httpx

from backend.config import settings

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
    "protocolSection.eligibilityModule.healthyVolunteers",
    "protocolSection.eligibilityModule.sex",
    "protocolSection.sponsorCollaboratorsModule.leadSponsor",
    "protocolSection.sponsorCollaboratorsModule.collaborators",
    "protocolSection.outcomesModule.primaryOutcomes",
    "protocolSection.outcomesModule.secondaryOutcomes",
    "protocolSection.contactsLocationsModule.locations",
    "protocolSection.descriptionModule.briefSummary",
])


async def fetch_studies(
    phases: list[str],
    max_records: int = 5000,
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
        "filter.overallStatus": "COMPLETED",
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


def _intervention_type(interventions: list[dict]) -> str:
    types = {iv.get("type", "") for iv in interventions}
    if "BIOLOGICAL" in types:
        return "BIOLOGICAL"
    if "DRUG" in types:
        return "DRUG"
    return "OTHER"


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

    if not start_date or not primary_date:
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

    return {
        "nct_id": nct_id,
        "Start Date": start_date,
        "Primary Completion Date": primary_date,
        "Phases": "|".join(design.get("phases", [])),
        "Enrollment": design.get("enrollmentInfo", {}).get("count"),
        "Drug_Type": drug_type,
        "Allocation": design_info.get("allocation"),
        "Intervention_Model": design_info.get("interventionModel"),
        "Masking": design_info.get("maskingInfo", {}).get("masking"),
        "Primary_Purpose": design_info.get("primaryPurpose"),
        "number_of_arms": None,  # not exposed in API v2; imputed from phase defaults
        "conditions": "|".join(conditions),
        "countries": "|".join(countries),
        "is_hv": int(is_hv),
        "Sex": eligibility.get("sex", "ALL"),
        "has_collaborators": int(bool(sponsor_mod.get("collaborators"))),
        "brief_summary": proto.get("descriptionModule", {}).get("briefSummary", ""),
        "total_primary_outcomes": len(outcomes.get("primaryOutcomes", [])),
        "total_secondary_outcomes": len(outcomes.get("secondaryOutcomes", [])),
    }
