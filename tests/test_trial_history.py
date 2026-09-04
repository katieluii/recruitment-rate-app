from backend.data.trial_history import (build_recruitment_target,
                                        build_summary_recruitment_target,
                                        required_versions)


def _version(status, enrollment_type="ESTIMATED", enrollment=100,
             start_type="ACTUAL", sites=1):
    return {
        "study": {"protocolSection": {
            "statusModule": {
                "overallStatus": status,
                "startDateStruct": {"date": "2020-01-01", "type": start_type},
            },
            "designModule": {
                "enrollmentInfo": {"count": enrollment, "type": enrollment_type}
            },
            "contactsLocationsModule": {
                "locations": [
                    {"facility": f"Site {i}", "status": "RECRUITING"}
                    for i in range(sites)
                ]
            },
        }}
    }


def test_required_versions_selects_location_and_status_changes():
    history = {"history": {"changes": [
        {"version": 0, "date": "2020-01-02", "status": "RECRUITING",
         "moduleLabels": []},
        {"version": 1, "date": "2020-03-01", "status": "RECRUITING",
         "moduleLabels": ["Outcome Measures"]},
        {"version": 2, "date": "2020-06-01", "status": "RECRUITING",
         "moduleLabels": ["Contacts/Locations"]},
        {"version": 3, "date": "2021-01-01", "status": "COMPLETED",
         "moduleLabels": ["Study Status"]},
    ]}}
    assert required_versions(history) == [0, 2, 3]


def test_tier_a_integrates_recruiting_centre_months():
    history = {"history": {"changes": [
        {"version": 0, "date": "2020-01-02", "status": "RECRUITING",
         "moduleLabels": []},
        {"version": 1, "date": "2020-07-01", "status": "RECRUITING",
         "moduleLabels": ["Contacts/Locations"]},
        {"version": 2, "date": "2021-01-01", "status": "COMPLETED",
         "moduleLabels": ["Study Status"]},
    ]}}
    versions = {
        0: _version("RECRUITING", sites=1),
        1: _version("RECRUITING", sites=3),
        2: _version("COMPLETED", enrollment_type="ACTUAL", sites=3),
    }
    result = build_recruitment_target("NCT00000001", history, versions)
    assert result.usable
    assert result.quality_tier == "A"
    assert result.denominator_method == "integrated_active_recruiting_site_snapshots"
    assert result.start_date_evidence == "explicit_actual"
    # Roughly six months at one site plus six months at three sites = 24 site-months.
    assert 23.5 < result.recruiting_centre_months < 24.6
    assert 4.0 < result.recruitment_rate < 4.3


def test_tier_b_is_separate_when_registration_lag_is_large():
    history = {"history": {"changes": [
        {"version": 0, "date": "2020-08-01", "status": "RECRUITING",
         "moduleLabels": []},
        {"version": 1, "date": "2021-01-01", "status": "COMPLETED",
         "moduleLabels": ["Study Status"]},
    ]}}
    versions = {
        0: _version("RECRUITING", sites=2),
        1: _version("COMPLETED", enrollment_type="ACTUAL", sites=2),
    }
    result = build_recruitment_target("NCT00000002", history, versions)
    assert result.usable
    assert result.quality_tier == "B"
    assert result.denominator_method == "max_listed_sites_x_recorded_recruiting_period"


def test_estimated_final_enrollment_is_excluded():
    history = {"history": {"changes": [
        {"version": 0, "date": "2020-01-01", "status": "RECRUITING",
         "moduleLabels": []},
        {"version": 1, "date": "2021-01-01", "status": "COMPLETED",
         "moduleLabels": []},
    ]}}
    result = build_recruitment_target(
        "NCT00000003", history,
        {0: _version("RECRUITING"), 1: _version("COMPLETED")},
    )
    assert not result.usable
    assert result.exclusion_reason == "final_enrollment_not_actual"


def test_legacy_untyped_start_is_usable_only_as_tier_b():
    history = {"history": {"changes": [
        {"version": 0, "date": "2020-01-02", "status": "RECRUITING",
         "moduleLabels": []},
        {"version": 1, "date": "2021-01-01", "status": "COMPLETED",
         "moduleLabels": []},
    ]}}
    versions = {
        0: _version("RECRUITING", start_type=None, sites=2),
        1: _version("COMPLETED", enrollment_type="ACTUAL",
                    start_type=None, sites=2),
    }
    result = build_recruitment_target("NCT00000004", history, versions)
    assert result.usable
    assert result.quality_tier == "B"
    assert result.start_date_evidence == "legacy_untyped"


def test_summary_target_uses_one_response_tier_b_denominator():
    payload = _version("COMPLETED", enrollment_type="ACTUAL", enrollment=120,
                       sites=4)
    payload["history"] = {"changes": [
        {"version": 0, "date": "2020-01-02", "status": "RECRUITING"},
        {"version": 1, "date": "2021-01-01", "status": "COMPLETED"},
    ]}
    result = build_summary_recruitment_target("NCT00000005", payload)
    assert result.usable
    assert result.quality_tier == "B"
    assert result.initiated_sites == 4
    assert 47.5 < result.recruiting_centre_months < 48.5
    assert 2.45 < result.recruitment_rate < 2.55
