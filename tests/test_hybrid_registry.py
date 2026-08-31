"""LoadedHybrid must reproduce HybridForestPoint's arithmetic from stub heads."""
import numpy as np
import pandas as pd

from backend.models.registry import LoadedHead, LoadedHybrid, LoadedTwoStage


class _Const:
    def __init__(self, v): self.v = v
    def predict(self, X): return np.full(len(X), self.v, dtype=float)


def _stage(q10, q50, q90, point):
    return LoadedHead({0.1: _Const(q10), 0.5: _Const(q50), 0.9: _Const(q90), "point": _Const(point)}, 0.0, "none")


def test_hybrid_point_band_and_rescaled_split():
    X = pd.DataFrame({"a": [0, 0]})
    enrol = _stage(100, 200, 300, 200)      # mid 200, ±100
    fu = _stage(300, 400, 500, 400)         # mid 400, ±100
    two = LoadedTwoStage(enrol, fu, band_scale=9.9)   # two-stage's own scale must NOT be used
    h = LoadedHybrid(two, _Const(900.0), band_scale=2.0)
    np.testing.assert_allclose(h.predict(X), 900.0)
    lo, hi = h.predict_interval(X)
    width = np.sqrt(100**2 + 100**2)         # quadrature of the two half-widths
    np.testing.assert_allclose(lo, 900 - 2.0 * width); np.testing.assert_allclose(hi, 900 + 2.0 * width)
    e, f = h.predict_components(X)
    np.testing.assert_allclose(e + f, 900.0)                 # sums to the forest total
    np.testing.assert_allclose(e / f, 200 / 400)             # two-stage ratio preserved


class _Forest:
    """Stub sklearn pipeline: named_steps pre/model with three trees."""
    class _Pre:
        def transform(self, X): return X
    class _Model:
        def __init__(self): self.estimators_ = [_Const(800.0), _Const(900.0), _Const(1000.0)]
    def __init__(self): self.named_steps = {"pre": self._Pre(), "model": self._Model()}
    def predict(self, X): return np.full(len(X), 900.0)


def test_forest_band_uses_tree_spread_floored_at_half_rmse():
    X = pd.DataFrame({"a": [0, 0]})
    two = LoadedTwoStage(_stage(100, 200, 300, 200), _stage(300, 400, 500, 400), band_scale=9.9)
    h = LoadedHybrid(two, _Forest(), band_scale=2.0, band_kind="forest", forest_rmse=100.0)
    spread = np.std([800.0, 900.0, 1000.0])               # 81.65 > rmse/2 = 50 -> tree spread wins
    lo, hi = h.predict_interval(X)
    np.testing.assert_allclose(lo, 900 - 2.0 * spread); np.testing.assert_allclose(hi, 900 + 2.0 * spread)
    h2 = LoadedHybrid(two, _Forest(), band_scale=1.0, band_kind="forest", forest_rmse=400.0)
    lo, hi = h2.predict_interval(X)                        # floor: rmse/2 = 200 > 81.65
    np.testing.assert_allclose(hi - lo, 400.0)
