"""The eligibility join, and the extrapolation guard that had gone inert.

Both are rejection tests. The guard in particular passed everything it was ever
shown before this change, which is indistinguishable from working.
"""
import pytest
from pydantic import ValidationError

from backend.models.inference import eligibility_fields
from backend.routes.predict import PredictRequest

VALID = {"phase": "P3", "therapeutic_area": "Oncology/Solid Tumours"}


def test_the_allowlist_is_the_thirteen_model_features():
    fields = eligibility_fields()
    assert len(fields) == 13
    assert "n_inclusion_criteria" in fields
    assert "crit_biomarker_required" in fields


def test_criteria_text_is_not_accepted():
    """It is built and then dropped by the fitted preprocessor, so accepting it
    would imply an effect it cannot have."""
    assert "criteria_text" not in eligibility_fields()


def test_eligibility_features_are_accepted_by_the_request_model():
    req = PredictRequest(**VALID, eligibility_features={"n_inclusion_criteria": 9})
    assert req.eligibility_features["n_inclusion_criteria"] == 9


def test_the_singular_endpoint_kwarg_still_works_in_process():
    """An in-process caller has no HTTP contract to shield it. Renaming this
    parameter took a live WS21 session's duration card down on every cell."""
    from backend.models.inference import predict
    import inspect

    params = inspect.signature(predict).parameters
    assert "endpoint_archetype" in params, "the singular alias must survive"
    assert "endpoint_archetypes" in params


def test_sending_both_endpoint_fields_is_rejected():
    from backend.routes.predict import post_predict
    from fastapi import HTTPException

    req = PredictRequest(**VALID, endpoint_archetype="SAFETY",
                         endpoint_archetypes=["RESPONSE"])
    with pytest.raises(HTTPException) as exc:
        post_predict(req)
    assert exc.value.status_code == 422


# ── the guard ────────────────────────────────────────────────────────────────

def _warnings_for(**kw):
    from backend.models.inference import predict
    return predict("P3", "Oncology/Solid Tumours", **kw).extrapolation_warnings


def test_guard_rejects_a_value_in_the_sparse_tail():
    """The case that motivated the change. Under min/max bounds this passed,
    because ONE training trial enrolled 7,702 patients at a single site and
    stretched the band to cover everything."""
    warnings = _warnings_for(enrollment=7000, num_sites=5)   # 1,400 per site
    assert any("enrollment_per_site" in w for w in warnings), warnings


def test_guard_stays_quiet_for_an_ordinary_trial():
    """A control that fires on everything is as useless as one that never does."""
    assert _warnings_for(enrollment=465, num_sites=57) == []


def test_guard_would_pass_the_sparse_value_under_the_old_min_max_bounds():
    """Pins WHY p01/p99 is used: the old bounds cannot reject the tail value."""
    from backend.models import registry

    bounds = registry.load("P3")["feature_ranges"]["enrollment_per_site"]
    assert bounds["min"] <= 1400 <= bounds["max"], "min/max would have passed it"
    assert not (bounds["p01"] <= 1400 <= bounds["p99"]), "p01/p99 must reject it"
