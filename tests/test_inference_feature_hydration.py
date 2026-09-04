import pytest

from backend.models.inference import _build_input_row


@pytest.mark.parametrize("phase", ["P1", "P2", "P3"])
def test_dependent_features_recomputed_after_defaults(phase):
    frame = _build_input_row(
        phase, "Oncology/Solid Tumours", None, None, "DRUG", "US", ["SAFETY"]
    )
    primary = frame["total_primary_outcomes"].iloc[0]
    secondary = frame["total_secondary_outcomes"].iloc[0]
    assert frame["outcomes_total"].iloc[0] == primary + secondary
    assert frame["outcomes_total"].iloc[0] > 0
    assert frame["enrollment_per_site"].iloc[0] == pytest.approx(
        frame["Enrollment"].iloc[0] / frame["site_count"].iloc[0]
    )


def test_explicit_endpoint_category_and_flags_agree():
    frame = _build_input_row(
        "P3", "Oncology/Solid Tumours", 500, 80, "DRUG", "US",
        ["SURVIVAL", "SAFETY"],
    )
    assert frame["endpoint_archetype"].iloc[0] == "SURVIVAL"
    assert frame["endpoint_has_SURVIVAL"].iloc[0] == 1
    assert frame["endpoint_has_SAFETY"].iloc[0] == 1
    assert frame["endpoint_has_RESPONSE"].iloc[0] == 0
