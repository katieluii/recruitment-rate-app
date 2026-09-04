from __future__ import annotations
"""ClinicalTrials.gov record-history targets for recruitment-rate modelling.

The public v2 API exposes only the latest record.  ClinicalTrials.gov's own
record-history interface exposes the dated versions behind the website.  This
module uses those snapshots offline to estimate achieved recruitment intensity:

    actual enrolled participants / recruiting centre-months

It deliberately does not run at prediction time.  The history interface is a
data-acquisition dependency used to build an auditable training target; serving
uses only the fitted artifacts.
"""

import asyncio
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any, Optional

import httpx


HISTORY_BASE = "https://clinicaltrials.gov/api/int/studies"
ACTIVE_SITE_STATUSES = {"RECRUITING", "ENROLLING_BY_INVITATION"}
ACTIVE_STUDY_STATUSES = {"RECRUITING", "ENROLLING_BY_INVITATION"}


@dataclass(frozen=True)
class SiteSnapshot:
    version: int
    recorded_date: str
    overall_status: str
    active_sites: int
    listed_sites: int


@dataclass(frozen=True)
class RecruitmentTarget:
    nct_id: str
    usable: bool
    quality_tier: Optional[str]
    exclusion_reason: Optional[str]
    enrollment_actual: Optional[int]
    recruitment_start: Optional[str]
    recruitment_end_proxy: Optional[str]
    recruiting_months: Optional[float]
    recruiting_centre_months: Optional[float]
    initiated_sites: Optional[int]
    recruitment_rate: Optional[float]
    initial_record_lag_days: Optional[int]
    active_site_coverage: Optional[float]
    snapshots: tuple[SiteSnapshot, ...]
    denominator_method: Optional[str]
    start_date_evidence: Optional[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parse_date(value: Any) -> Optional[date]:
    if not value:
        return None
    text = str(value)
    try:
        if len(text) == 7:
            text += "-01"
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _proto(payload: dict[str, Any]) -> dict[str, Any]:
    return payload.get("study", {}).get("protocolSection", {})


def _snapshot(version: int, recorded_date: str,
              payload: dict[str, Any]) -> SiteSnapshot:
    proto = _proto(payload)
    status = proto.get("statusModule", {}).get("overallStatus", "UNKNOWN")
    locations = proto.get("contactsLocationsModule", {}).get("locations", []) or []
    active = sum(1 for loc in locations
                 if loc.get("status") in ACTIVE_SITE_STATUSES)
    return SiteSnapshot(
        version=int(version),
        recorded_date=recorded_date,
        overall_status=status,
        active_sites=active,
        listed_sites=len(locations),
    )


def required_versions(history: dict[str, Any]) -> list[int]:
    """Versions needed to reconstruct status and centre-count change points."""
    changes = history.get("history", {}).get("changes", []) or []
    if not changes:
        return []
    selected = {int(changes[0]["version"]), int(changes[-1]["version"])}
    previous_status = None
    for change in changes:
        version = int(change["version"])
        status = change.get("status")
        labels = set(change.get("moduleLabels") or [])
        if "Contacts/Locations" in labels or status != previous_status:
            selected.add(version)
        previous_status = status
    return sorted(selected)


def _recorded_recruitment_end(changes: list[dict[str, Any]],
                              start: date) -> Optional[date]:
    saw_active = False
    for change in sorted(changes, key=lambda c: int(c["version"])):
        status = change.get("status")
        change_date = _parse_date(change.get("date"))
        if status in ACTIVE_STUDY_STATUSES:
            saw_active = True
        elif saw_active and change_date and change_date > start:
            return change_date
    return None


def _excluded(nct_id: str, reason: str,
              snapshots: list[SiteSnapshot] | None = None,
              **known: Any) -> RecruitmentTarget:
    return RecruitmentTarget(
        nct_id=nct_id, usable=False, quality_tier=None,
        exclusion_reason=reason, enrollment_actual=known.get("enrollment_actual"),
        recruitment_start=known.get("recruitment_start"),
        recruitment_end_proxy=known.get("recruitment_end_proxy"),
        recruiting_months=None, recruiting_centre_months=None,
        initiated_sites=known.get("initiated_sites"), recruitment_rate=None,
        initial_record_lag_days=known.get("initial_record_lag_days"),
        active_site_coverage=None, snapshots=tuple(snapshots or []),
        denominator_method=None, start_date_evidence=known.get("start_date_evidence"),
    )


def build_recruitment_target(nct_id: str, history: dict[str, Any],
                             versions: dict[int, dict[str, Any]]) -> RecruitmentTarget:
    """Build one quality-tiered historical PPCM target from registry snapshots.

    Tier A integrates the explicit count of actively recruiting facilities over
    time.  Tier B divides by the maximum listed site count over the recorded
    recruiting period.  Tier B is retained for coverage analysis but should not
    be mixed into a Tier-A model without an explicit sensitivity test.
    """
    changes = history.get("history", {}).get("changes", []) or []
    if not changes or not versions:
        return _excluded(nct_id, "no_record_history")

    change_by_version = {int(c["version"]): c for c in changes}
    final_version = max(versions)
    final_proto = _proto(versions[final_version])
    enrollment = final_proto.get("designModule", {}).get("enrollmentInfo", {})
    if enrollment.get("type") != "ACTUAL" or not enrollment.get("count"):
        return _excluded(nct_id, "final_enrollment_not_actual")
    enrollment_actual = int(enrollment["count"])

    start_struct = final_proto.get("statusModule", {}).get("startDateStruct", {})
    start = _parse_date(start_struct.get("date"))
    start_type = start_struct.get("type")
    if not start or start_type == "ESTIMATED":
        return _excluded(nct_id, "study_start_not_actual",
                         enrollment_actual=enrollment_actual)
    # Migrated legacy records often omit the date type entirely.  A terminal
    # record makes the start date usable as a proxy, but never Tier A evidence.
    start_evidence = "explicit_actual" if start_type == "ACTUAL" else "legacy_untyped"

    end = _recorded_recruitment_end(changes, start)
    if not end:
        return _excluded(
            nct_id, "no_recorded_recruitment_end", enrollment_actual=enrollment_actual,
            recruitment_start=start.isoformat(),
        )

    snapshots: list[SiteSnapshot] = []
    for version, payload in versions.items():
        change = change_by_version.get(int(version), {})
        if change.get("date"):
            snapshots.append(_snapshot(version, change["date"], payload))
    snapshots.sort(key=lambda s: (s.recorded_date, s.version))
    if not snapshots:
        return _excluded(nct_id, "no_dated_site_snapshots",
                         enrollment_actual=enrollment_actual,
                         recruitment_start=start.isoformat(),
                         recruitment_end_proxy=end.isoformat())

    initiated_sites = max((s.listed_sites for s in snapshots), default=0)
    first_record = _parse_date(snapshots[0].recorded_date)
    lag_days = (first_record - start).days if first_record else None
    total_days = (end - start).days
    if total_days < 30:
        return _excluded(nct_id, "recruiting_period_under_30_days", snapshots,
                         enrollment_actual=enrollment_actual,
                         recruitment_start=start.isoformat(),
                         recruitment_end_proxy=end.isoformat(),
                         initiated_sites=initiated_sites,
                         initial_record_lag_days=lag_days)

    # A version is effective on its recorded date.  Before the first recorded
    # version, its site state is back-cast to the actual first-participant date;
    # Tier A permits that only when the registration lag is <= 180 days.
    events: list[tuple[date, SiteSnapshot]] = []
    for snap in snapshots:
        snap_date = _parse_date(snap.recorded_date)
        if snap_date and snap_date < end:
            events.append((max(start, snap_date), snap))
    events.sort(key=lambda item: (item[0], item[1].version))
    if not events:
        return _excluded(nct_id, "no_site_snapshot_during_recruitment", snapshots,
                         enrollment_actual=enrollment_actual,
                         recruitment_start=start.isoformat(),
                         recruitment_end_proxy=end.isoformat(),
                         initiated_sites=initiated_sites,
                         initial_record_lag_days=lag_days)

    # Back-cast the earliest available snapshot to study start.  Multiple
    # updates on the same day collapse to the last submitted version.
    timeline: dict[date, SiteSnapshot] = {start: events[0][1]}
    for event_date, snap in events:
        timeline[event_date] = snap
    dates = sorted(timeline)
    centre_days = 0.0
    covered_days = 0
    for index, current_date in enumerate(dates):
        next_date = dates[index + 1] if index + 1 < len(dates) else end
        interval_days = max(0, (min(next_date, end) - current_date).days)
        if interval_days <= 0:
            continue
        active_sites = timeline[current_date].active_sites
        if active_sites > 0:
            centre_days += active_sites * interval_days
            covered_days += interval_days

    coverage = covered_days / total_days
    recruiting_months = total_days / 30.4375
    lag_ok = lag_days is not None and lag_days <= 180
    if (start_evidence == "explicit_actual" and lag_ok and coverage >= 0.90
            and centre_days > 0):
        tier = "A"
        centre_months = centre_days / 30.4375
        method = "integrated_active_recruiting_site_snapshots"
    elif initiated_sites > 0:
        tier = "B"
        centre_months = initiated_sites * recruiting_months
        method = "max_listed_sites_x_recorded_recruiting_period"
    else:
        return _excluded(nct_id, "no_usable_site_count", snapshots,
                         enrollment_actual=enrollment_actual,
                         recruitment_start=start.isoformat(),
                         recruitment_end_proxy=end.isoformat(),
                         initiated_sites=initiated_sites,
                         initial_record_lag_days=lag_days)

    rate = enrollment_actual / centre_months
    if not (rate > 0):
        return _excluded(nct_id, "non_positive_rate", snapshots,
                         enrollment_actual=enrollment_actual,
                         recruitment_start=start.isoformat(),
                         recruitment_end_proxy=end.isoformat(),
                         initiated_sites=initiated_sites,
                         initial_record_lag_days=lag_days)

    return RecruitmentTarget(
        nct_id=nct_id, usable=True, quality_tier=tier,
        exclusion_reason=None, enrollment_actual=enrollment_actual,
        recruitment_start=start.isoformat(), recruitment_end_proxy=end.isoformat(),
        recruiting_months=round(recruiting_months, 4),
        recruiting_centre_months=round(centre_months, 4),
        initiated_sites=initiated_sites, recruitment_rate=round(rate, 6),
        initial_record_lag_days=lag_days, active_site_coverage=round(coverage, 4),
        snapshots=tuple(snapshots), denominator_method=method,
        start_date_evidence=start_evidence,
    )


def build_summary_recruitment_target(nct_id: str,
                                     payload: dict[str, Any]) -> RecruitmentTarget:
    """Build the one-response Tier-B target used for the large feasibility set.

    ``?history=true`` contains both the version index and current study record.
    It therefore supports a conservative initiated-sites x overall-recruiting-
    period denominator without downloading every facility snapshot.  The result
    is always Tier B; detailed Tier A targets remain the validation reference.
    """
    changes = payload.get("history", {}).get("changes", []) or []
    proto = payload.get("study", {}).get("protocolSection", {})
    if not changes or not proto:
        return _excluded(nct_id, "no_record_history")

    enrollment = proto.get("designModule", {}).get("enrollmentInfo", {})
    if enrollment.get("type") != "ACTUAL" or not enrollment.get("count"):
        return _excluded(nct_id, "final_enrollment_not_actual")
    enrollment_actual = int(enrollment["count"])

    start_struct = proto.get("statusModule", {}).get("startDateStruct", {})
    start = _parse_date(start_struct.get("date"))
    start_type = start_struct.get("type")
    if not start or start_type == "ESTIMATED":
        return _excluded(nct_id, "study_start_not_actual",
                         enrollment_actual=enrollment_actual)
    start_evidence = "explicit_actual" if start_type == "ACTUAL" else "legacy_untyped"

    end = _recorded_recruitment_end(changes, start)
    if not end:
        return _excluded(nct_id, "no_recorded_recruitment_end",
                         enrollment_actual=enrollment_actual,
                         recruitment_start=start.isoformat(),
                         start_date_evidence=start_evidence)
    total_days = (end - start).days
    if total_days < 30:
        return _excluded(nct_id, "recruiting_period_under_30_days",
                         enrollment_actual=enrollment_actual,
                         recruitment_start=start.isoformat(),
                         recruitment_end_proxy=end.isoformat(),
                         start_date_evidence=start_evidence)

    locations = proto.get("contactsLocationsModule", {}).get("locations", []) or []
    initiated_sites = len(locations)
    if initiated_sites <= 0:
        return _excluded(nct_id, "no_usable_site_count",
                         enrollment_actual=enrollment_actual,
                         recruitment_start=start.isoformat(),
                         recruitment_end_proxy=end.isoformat(),
                         initiated_sites=initiated_sites,
                         start_date_evidence=start_evidence)

    recruiting_months = total_days / 30.4375
    centre_months = initiated_sites * recruiting_months
    first_record = _parse_date(changes[0].get("date"))
    lag_days = (first_record - start).days if first_record else None
    return RecruitmentTarget(
        nct_id=nct_id, usable=True, quality_tier="B", exclusion_reason=None,
        enrollment_actual=enrollment_actual, recruitment_start=start.isoformat(),
        recruitment_end_proxy=end.isoformat(),
        recruiting_months=round(recruiting_months, 4),
        recruiting_centre_months=round(centre_months, 4),
        initiated_sites=initiated_sites,
        recruitment_rate=round(enrollment_actual / centre_months, 6),
        initial_record_lag_days=lag_days, active_site_coverage=None,
        snapshots=tuple(),
        denominator_method="initiated_sites_x_recorded_recruiting_period",
        start_date_evidence=start_evidence,
    )


class TrialHistoryClient:
    """Small retrying async client for the record-history acquisition step."""

    def __init__(self, timeout: float = 45.0, retries: int = 4):
        self.timeout = timeout
        self.retries = retries

    async def _json(self, client: httpx.AsyncClient, url: str) -> dict[str, Any]:
        last: Optional[Exception] = None
        for attempt in range(self.retries + 1):
            try:
                response = await client.get(url)
                response.raise_for_status()
                return response.json()
            except (httpx.HTTPError, ValueError) as exc:
                last = exc
                if attempt == self.retries:
                    break
                await asyncio.sleep(1.5 * (2 ** attempt))
        assert last is not None
        raise last

    async def fetch_target(self, nct_id: str) -> RecruitmentTarget:
        async with httpx.AsyncClient(timeout=self.timeout,
                                     headers={"User-Agent": "WSi research audit"}) as client:
            history = await self._json(client, f"{HISTORY_BASE}/{nct_id}?history=true")
            numbers = required_versions(history)
            versions: dict[int, dict[str, Any]] = {}
            if not numbers:
                return build_recruitment_target(nct_id, history, versions)
            # Fetch the final record first.  Most exclusions can be decided from
            # it plus the lightweight history index, avoiding dozens of location
            # version requests for a trial that cannot yield a valid target.
            final_version = numbers[-1]
            versions[final_version] = await self._json(
                client, f"{HISTORY_BASE}/{nct_id}/history/{final_version}")
            preflight = build_recruitment_target(nct_id, history, versions)
            if preflight.exclusion_reason in {
                "final_enrollment_not_actual", "study_start_not_actual",
                "no_recorded_recruitment_end",
            }:
                return preflight
            # Sequential within a trial is intentionally polite; callers may run
            # a small number of trials concurrently.
            for version in numbers:
                if version == final_version:
                    continue
                versions[version] = await self._json(
                    client, f"{HISTORY_BASE}/{nct_id}/history/{version}")
            return build_recruitment_target(nct_id, history, versions)

    async def fetch_summary_target(
        self, nct_id: str, client: Optional[httpx.AsyncClient] = None,
    ) -> RecruitmentTarget:
        if client is not None:
            payload = await self._json(
                client, f"{HISTORY_BASE}/{nct_id}?history=true")
            return build_summary_recruitment_target(nct_id, payload)
        async with httpx.AsyncClient(timeout=self.timeout,
                                     headers={"User-Agent": "WSi research audit"}) as owned:
            payload = await self._json(
                owned, f"{HISTORY_BASE}/{nct_id}?history=true")
            return build_summary_recruitment_target(nct_id, payload)
