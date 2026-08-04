"""The best-so-far bar that replaced the absolute R2 gate.

Written as rejection tests. A leaderboard that reports "level" for everything
looks identical to a working one in the run output — which is exactly what the
first version did, because the run appended its rows to the ledger before
reading the bar back and so compared every result against itself.
"""
import pytest

from experiments import leaderboard

TABLE = {"P1": {"r2": {"value": 0.60, "config": "prev", "ts": "t0"},
                "rmse_days": {"value": 300.0, "config": "prev", "ts": "t0"}}}


def _row(r2, rmse=300.0, phase="P1"):
    return {"phase": phase, "r2": r2, "rmse_days": rmse}


def test_a_better_r2_is_a_record():
    v = leaderboard.compare(_row(0.65), TABLE)
    assert v["r2"]["verdict"] == "record"
    assert v["r2"]["previous"] == 0.60


def test_a_worse_r2_is_a_regression_not_a_pass():
    """The failure that matters: a drop must NOT read as acceptable."""
    v = leaderboard.compare(_row(0.50), TABLE)
    assert v["r2"]["verdict"] == "regression"


def test_noise_sized_movement_is_level_in_both_directions():
    assert leaderboard.compare(_row(0.6001), TABLE)["r2"]["verdict"] == "level"
    assert leaderboard.compare(_row(0.5999), TABLE)["r2"]["verdict"] == "level"


def test_rmse_direction_is_inverted():
    """Lower RMSE is better; the same delta sign means the opposite verdict."""
    assert leaderboard.compare(_row(0.60, rmse=250.0), TABLE)["rmse_days"]["verdict"] == "record"
    assert leaderboard.compare(_row(0.60, rmse=350.0), TABLE)["rmse_days"]["verdict"] == "regression"


def test_an_unseen_phase_sets_the_first_record():
    v = leaderboard.compare(_row(0.1, phase="P9"), TABLE)
    assert v["r2"]["verdict"] == "record"
    assert v["r2"]["previous"] is None


def test_a_missing_metric_is_unknown_not_a_pass():
    """A run that measured nothing must not inherit a passing verdict."""
    v = leaderboard.compare({"phase": "P1", "r2": None, "rmse_days": None}, TABLE)
    assert v["r2"]["verdict"] == "unknown"


def test_comparing_against_an_identical_value_is_level_not_record():
    """Guards the self-comparison bug: if a run reads back its own row, every
    verdict collapses to this, so the test above for 'record' is what proves the
    mechanism is live."""
    assert leaderboard.compare(_row(0.60), TABLE)["r2"]["verdict"] == "level"


@pytest.mark.parametrize("metric", ["r2", "rmse_days"])
def test_every_objective_is_reported(metric):
    assert metric in leaderboard.compare(_row(0.62, 290.0), TABLE)
