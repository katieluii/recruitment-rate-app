"""The point head's target space is independent of the quantile heads' (2026-08-31)."""
import numpy as np
import pytest

from backend.models.quantile_model import ConformalQuantileModel, TwoStageDuration


def test_default_point_space_is_the_model_transform():
    m = ConformalQuantileModel("P2", transform="log1p")
    assert m._point_space == "log1p"
    np.testing.assert_allclose(m._inv_point(m._fwd_point([100.0, 900.0])), [100.0, 900.0])


def test_raw_point_space_is_identity_while_interval_space_stays_log():
    m = ConformalQuantileModel("P2", transform="log1p", point_transform="none")
    y = np.array([100.0, 900.0])
    np.testing.assert_allclose(m._fwd_point(y), y)
    np.testing.assert_allclose(m._inv_point(y), y)
    np.testing.assert_allclose(m._fwd(y), np.log1p(y))  # quantile heads untouched


def test_two_stage_threads_point_transform_to_both_stages():
    m = TwoStageDuration("P2", point_transform="none")
    assert m.enrol.point_transform == "none" and m.fu.point_transform == "none"
    m = TwoStageDuration("P2")
    assert m.enrol.point_transform is None and m.fu._point_space == "log1p"


def test_unknown_point_transform_rejected():
    with pytest.raises(ValueError):
        ConformalQuantileModel("P2", point_transform="sqrt")


class _Const:
    def __init__(self, v): self.v = v
    def predict(self, X): return np.full(len(X), self.v, dtype=float)


def test_registry_inverts_point_head_in_its_own_space():
    """A raw-day point head served through the log inverse would return expm1(days)."""
    from backend.models.registry import LoadedHead
    import pandas as pd
    X = pd.DataFrame({"a": [0, 0]})
    models = {0.1: _Const(1.0), 0.5: _Const(2.0), 0.9: _Const(3.0), "point": _Const(400.0)}
    raw = LoadedHead(models, 0.0, "log1p", point_transform="none")
    np.testing.assert_allclose(raw.predict(X), [400.0, 400.0])
    lo, hi = raw.predict_interval(X)
    np.testing.assert_allclose(lo, np.expm1(1.0)); np.testing.assert_allclose(hi, np.expm1(3.0))
    legacy = LoadedHead({**models, "point": _Const(3.0)}, 0.0, "log1p")  # no point_transform in metadata
    np.testing.assert_allclose(legacy.predict(X), np.expm1(3.0))


def test_point_head_receives_the_raw_target(monkeypatch):
    """Spy on the fit path: with point_transform='none' the point head must be handed
    duration in DAYS while the quantile heads get log1p(days). This is the scope bug
    (y_point undefined inside _fit_quantiles) that killed the first mature-fold run."""
    import pandas as pd
    from backend.models import quantile_model as qm
    seen = {}
    monkeypatch.setattr(qm, "build_features", lambda df, phase: pd.DataFrame({"x": np.arange(len(df), dtype=float)}))
    def fake_fit_quantiles(self, X, y, sample_weight=None, y_point=None):
        seen["y"], seen["y_point"] = np.asarray(y), None if y_point is None else np.asarray(y_point)
        return {}
    monkeypatch.setattr(qm.ConformalQuantileModel, "_fit_quantiles", fake_fit_quantiles)
    train = pd.DataFrame({"duration_days": [100.0, 400.0, 900.0], "Start Date": ["2015-01", "2016-01", "2017-01"]})
    qm.ConformalQuantileModel("P2", conformal=False, point_transform="none").fit(train, "duration_days")
    np.testing.assert_allclose(seen["y"], np.log1p([100.0, 400.0, 900.0]))
    np.testing.assert_allclose(seen["y_point"], [100.0, 400.0, 900.0])
    qm.ConformalQuantileModel("P2", conformal=False).fit(train, "duration_days")
    np.testing.assert_allclose(seen["y_point"], np.log1p([100.0, 400.0, 900.0]))  # default: same space
