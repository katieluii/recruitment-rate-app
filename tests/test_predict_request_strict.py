"""The predict request must REJECT unknown fields, not drop them.

Written as a rejection test rather than an acceptance test on purpose: a control
that fails open looks identical to a control that works until the day it matters.
The interesting assertion is the 422, not the 200.
"""
import pytest
from pydantic import ValidationError

from backend.routes.predict import PredictRequest

VALID = {"phase": "P2", "therapeutic_area": "Oncology/Solid Tumours"}


def test_known_fields_still_accepted():
    """The five fields the frontend actually sends must all survive."""
    req = PredictRequest(**VALID, enrollment=200, num_sites=40,
                         drug_type="DRUG", region="US",
                         endpoint_archetype="SURVIVAL")
    assert req.num_sites == 40


def test_unknown_field_is_rejected():
    """`site_count` is the real misspelling that cost the debugging time."""
    with pytest.raises(ValidationError) as exc:
        PredictRequest(**VALID, site_count=40)
    assert "site_count" in str(exc.value)


@pytest.mark.parametrize("bad", ["sites", "num_site", "numSites", "enrollment_count"])
def test_near_miss_field_names_are_rejected(bad):
    with pytest.raises(ValidationError):
        PredictRequest(**VALID, **{bad: 40})


def test_no_site_count_leaks_through_as_a_silent_default():
    """The failure mode itself: a mistyped field name must not yield a request
    that validates cleanly with num_sites left unset."""
    try:
        req = PredictRequest(**VALID, site_count=40)
    except ValidationError:
        return  # correct behaviour
    pytest.fail(f"site_count was accepted and dropped; num_sites={req.num_sites}")
