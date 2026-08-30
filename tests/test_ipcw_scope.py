"""IPCW scope on the two-stage duration model.

Pins the 2026-08-30 finding: with the default `enrol` scope the enrolment stage looked its
censoring weight up at the ENROLMENT window against a KM over TOTAL duration, and the
follow-up stage trained unweighted. `total` computes one weight per trial from total
duration and hands it to both stages. These tests fit nothing — `_stage_weights` is pure.
"""
import numpy as np
import pandas as pd
import pytest

from backend.models.quantile_model import TwoStageDuration, ipcw_weights


def censoring_frame(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Long trials are the censored ones — the shape of the real corpus."""
    rng = np.random.default_rng(seed)
    dur = rng.lognormal(mean=6.5, sigma=0.6, size=n)          # ~665 days median
    censored = dur > np.quantile(dur, 0.7)
    return pd.DataFrame({"duration_days": dur, "event_observed": (~censored).astype(int)})


def test_ipcw_weights_rise_with_duration_and_are_normalised():
    frame = censoring_frame()
    t = np.array([100.0, 500.0, 1500.0, 3000.0])
    w = ipcw_weights(frame, t, weight_cap=10.0)
    assert w is not None
    assert np.all(np.diff(w) >= 0), "longer trials must be upweighted"
    assert w[0] < w[-1]
    full = ipcw_weights(frame, frame["duration_days"].to_numpy(), weight_cap=10.0)
    assert abs(full.mean() - 1.0) < 1e-9
    assert full.max() <= 10.0 / full.mean() * 1.0001


def test_ipcw_weights_none_when_nothing_is_censored():
    frame = censoring_frame(); frame["event_observed"] = 1
    assert ipcw_weights(frame, np.array([100.0, 900.0])) is None
    assert ipcw_weights(None, np.array([100.0])) is None


def test_total_scope_weights_both_stages_identically():
    frame = censoring_frame()
    m = TwoStageDuration("P2", censoring_frame=frame, ipcw_scope="total")
    total = np.array([200.0, 800.0, 2000.0, 4000.0])
    clipped = np.zeros(4, dtype=bool)
    e_w, f_w, e_keep, f_keep = m._stage_weights(total, clipped)
    assert e_w is not None and f_w is not None
    np.testing.assert_allclose(e_w, f_w)
    assert np.all(np.diff(e_w) >= 0)
    assert e_keep.all() and f_keep.all()
    assert m.ipcw_applied_ is True
    # The frame is deliberately NOT handed to the enrolment stage under total scope.
    assert m.enrol.censoring_frame is None


def test_enrol_scope_leaves_followup_unweighted():
    """The defect, pinned: under `enrol` the follow-up stage gets no weight at all,
    and the enrolment stage carries the frame (to look G up at the wrong quantity)."""
    frame = censoring_frame()
    m = TwoStageDuration("P2", censoring_frame=frame, ipcw_scope="enrol")
    e_w, f_w, _, _ = m._stage_weights(np.array([200.0, 800.0]), np.zeros(2, dtype=bool))
    assert e_w is None and f_w is None          # nothing at the two-stage level ...
    assert m.enrol.censoring_frame is frame      # ... the stage does its own (wrong-unit) lookup
    assert m.fu.censoring_frame is None


def test_total_scope_composes_with_clip_policies():
    frame = censoring_frame()
    total = np.array([200.0, 800.0, 2000.0, 4000.0])
    clipped = np.array([True, False, True, False])
    m = TwoStageDuration("P2", censoring_frame=frame, ipcw_scope="total",
                         clip_policy="weight", clip_weight=0.25, clip_scope="enrol")
    e_w, f_w, _, _ = m._stage_weights(total, clipped)
    base = ipcw_weights(frame, total)
    np.testing.assert_allclose(e_w, base * np.where(clipped, 0.25, 1.0))
    np.testing.assert_allclose(f_w, base)
    m = TwoStageDuration("P2", censoring_frame=frame, ipcw_scope="total", clip_policy="drop")
    e_w, f_w, e_keep, f_keep = m._stage_weights(total, clipped)
    assert e_keep.tolist() == [False, True, False, True] and f_keep.all()
    assert len(e_w) == 2 and len(f_w) == 4
    np.testing.assert_allclose(e_w, base[[1, 3]])


def test_unknown_scope_rejected():
    with pytest.raises(ValueError):
        TwoStageDuration("P2", ipcw_scope="both")
