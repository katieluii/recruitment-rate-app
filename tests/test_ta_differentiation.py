from __future__ import annotations
"""The regression test for the bug that started the rework.

v1 returned near-identical durations for most therapeutic areas: 17 of 22 Phase 1
areas gave the identical 10.9 months, and predicted spread across areas was 1.3 /
4.4 / 1.6 months on P1 / P2 / P3 against true spreads of 7.0 / 21.5 / 28.4.

Nothing in the test suite would have caught that, because every prediction was
individually plausible. These assertions make it impossible to ship again.

    python -m pytest tests/ -v
"""
import numpy as np
import pytest

from backend.constants import THERAPEUTIC_AREAS
from backend.models import registry
from backend.models.inference import predict

PHASES_UNDER_TEST = ["P1", "P2", "P3"]

#: v1 shipped 5 / 6 / 9 distinct values out of 22 areas.
MIN_DISTINCT_PREDICTIONS = 12
#: v1's app-level spread was 1.3 / 4.4 / 1.6 months.
MIN_SPREAD_MONTHS = 4.0
#: Clinical ground truth: oncology and haematology Phase 3 run far longer than
#: dermatology or infectious disease. A model that inverts this is broken
#: regardless of its aggregate error.
SLOW_AREAS = ["Oncology/Solid Tumours", "Haematology"]
FAST_AREAS = ["Infectious Diseases", "Dermatology/Connective Tissue Diseases"]


def _predictions(phase: str) -> dict[str, float]:
    return {ta: predict(phase, ta).predicted_months for ta in THERAPEUTIC_AREAS}


@pytest.fixture(scope="module", autouse=True)
def _require_artifacts():
    if registry.load("P3") is None:
        pytest.skip("No trained artifacts — run scripts/train_models.py first")


@pytest.mark.parametrize("phase", PHASES_UNDER_TEST)
def test_areas_get_distinct_predictions(phase):
    """Therapeutic areas must not collapse onto the same answer."""
    preds = _predictions(phase)
    distinct = len(set(preds.values()))
    assert distinct >= MIN_DISTINCT_PREDICTIONS, (
        f"{phase}: only {distinct} distinct predictions across "
        f"{len(preds)} therapeutic areas — the v1 collapse has returned"
    )


@pytest.mark.parametrize("phase", PHASES_UNDER_TEST)
def test_prediction_spread_is_material(phase):
    values = list(_predictions(phase).values())
    spread = max(values) - min(values)
    assert spread >= MIN_SPREAD_MONTHS, (
        f"{phase}: spread across therapeutic areas is only {spread:.1f} months"
    )


def test_slow_areas_predicted_slower_than_fast_areas():
    """Ordering sanity, on Phase 3 where the real gap is largest (38 vs 9.5 mo)."""
    preds = _predictions("P3")
    slow = np.mean([preds[a] for a in SLOW_AREAS])
    fast = np.mean([preds[a] for a in FAST_AREAS])
    assert slow > fast, (
        f"P3: oncology/haematology predicted {slow:.1f} months but "
        f"dermatology/infectious-disease {fast:.1f} — ordering is inverted"
    )


@pytest.mark.parametrize("phase", PHASES_UNDER_TEST)
def test_intervals_are_not_constant_width(phase):
    """v1's interval half-width was pinned at rmse*0.5 for essentially every
    input, so uncertainty carried no information."""
    widths = []
    for ta in THERAPEUTIC_AREAS[:8]:
        p = predict(phase, ta)
        widths.append(p.upper_months - p.lower_months)
    assert np.std(widths) > 0.01, (
        f"{phase}: every prediction interval is the same width — "
        f"the interval is decorative"
    )


def test_extrapolation_is_flagged_not_silent():
    """Out-of-range inputs must warn rather than silently return a flat number.

    site_count far above the trained maximum is the exact shape of the original
    failure.
    """
    p = predict("P3", "Oncology/Solid Tumours", enrollment=400, num_sites=5000)
    assert p.extrapolation_warnings, (
        "site_count=5000 is far outside the trained range but produced no warning"
    )


@pytest.mark.parametrize("phase", PHASES_UNDER_TEST)
def test_recruitment_rate_head_present_and_plausible(phase):
    p = predict(phase, "Oncology/Solid Tumours")
    assert p.recruitment_rate is not None, f"{phase}: no recruitment rate returned"
    assert 0 < p.recruitment_rate < 100, (
        f"{phase}: implausible recruitment rate {p.recruitment_rate}"
    )
    assert p.recruitment_rate_lower <= p.recruitment_rate <= p.recruitment_rate_upper


@pytest.mark.parametrize("phase,expected", [
    ("P1", "SAFETY"), ("P2", "RESPONSE"), ("P3", "SURVIVAL"),
])
def test_oncology_default_uses_coherent_endpoint_profile(phase, expected):
    p = predict(phase, "Oncology/Solid Tumours")
    assert p.endpoint_archetypes_used == [expected]
    assert p.endpoint_source == "therapeutic_area_profile"
    assert p.endpoint_profile_n > 0


def test_rate_is_direct_history_model_and_scenario_is_separate():
    p = predict("P2", "Oncology/Solid Tumours", enrollment=100, num_sites=10)
    assert p.recruitment_rate_definition == (
        "actual enrollment / (initiated sites x recorded recruiting months)"
    )
    assert p.recruitment_rate_target_quality == "B"
    assert p.rate_implied_total_months is None
    assert p.estimated_recruitment_months == round(100 / (10 * p.recruitment_rate), 1)


def test_leaky_year_features_are_not_in_the_feature_set():
    """primary_completion_year is the label's own endpoint. Re-adding it took
    Phase 2 MAE from 8.9 to 25.4 months on a temporal holdout."""
    from backend.preprocessing.pipeline import _NUM_COLS

    for banned in ("primary_completion_year", "start_year"):
        assert banned not in _NUM_COLS, (
            f"{banned} is back in the feature set — see the note in pipeline.py"
        )


def test_point_metrics_reports_median_absolute_error():
    import numpy as np
    from experiments.metrics import point_metrics
    y = np.array([1.0, 1.0, 1.0, 100.0]); yhat = np.array([1.5, 0.5, 1.5, 1.0])
    m = point_metrics(y, yhat, unit="raw")
    assert m["mae_raw"] == 25.125 and m["medae_raw"] == 0.5  # one tail trial owns the mean
