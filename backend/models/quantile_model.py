from __future__ import annotations
"""Conformalised quantile gradient-boosting — the model behind both heads.

This lives in `backend/` rather than in the experiment harness so the thing
that is measured is literally the thing that ships. `experiments.candidates`
imports this class; there is no second implementation to drift out of sync.

Design notes, each of which is a fix for something the harness caught:

* Three LightGBM models per head at alpha 0.1 / 0.5 / 0.9. The median is the
  point prediction and the outer pair are the interval. v1 derived its interval
  from `max(tree_std, rmse * 0.5)`, which pinned the half-width near +/-6 months
  for every input and covered 8% of actuals on Phase 2.
* Quantiles are equivariant under a monotone transform, so quantiles fitted on
  log(y) exponentiate back to genuine quantiles of y.
* Transform per head: log1p for durations, plain log for the recruitment rate.
  The rate spans four orders of magnitude and is strictly positive; log1p barely
  separates values below 1, which left rate-head coverage at 0.19.
* Conformal widening calibrated on the MOST RECENT slice of the training data,
  not a random one — the corpus holds completed trials only, so recent history
  skews fast and a random calibration slice produces intervals that are too
  narrow for the period being predicted.
"""
import logging

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from backend.preprocessing.pipeline import build_features, make_preprocessor

log = logging.getLogger(__name__)

QUANTILES = (0.1, 0.5, 0.9)

TRANSFORMS = {
    "log1p": (np.log1p, np.expm1, 1.0),
    "log": (lambda y: np.log(np.maximum(y, 1e-6)), np.exp, 1e-4),
    "none": (lambda y: y, lambda y: y, 0.0),
}

DEFAULT_PARAMS = {
    "n_estimators": 600,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}


class ConformalQuantileModel:
    """Fit/predict with calibrated prediction intervals."""

    def __init__(self, phase_key: str, transform: str = "log1p",
                 params: dict | None = None, ta_target_encoding: bool = True,
                 conformal: bool = True, calib_frac: float = 0.2,
                 coverage: float = 0.80, calib_strategy: str = "recent"):
        if transform not in TRANSFORMS:
            raise ValueError(f"Unknown transform {transform!r}")
        self.phase_key = phase_key
        self.transform = transform
        self.params = dict(params or DEFAULT_PARAMS)
        self.ta_target_encoding = ta_target_encoding
        self.conformal = conformal
        self.calib_frac = calib_frac
        self.coverage = coverage
        self.calib_strategy = calib_strategy
        self.qhat_ = 0.0

    # ── transforms ───────────────────────────────────────────────────────────

    @property
    def _floor(self) -> float:
        return TRANSFORMS[self.transform][2]

    def _fwd(self, y):
        return TRANSFORMS[self.transform][0](np.asarray(y, dtype=float))

    def _inv(self, y):
        return TRANSFORMS[self.transform][1](np.asarray(y, dtype=float))

    # ── fitting ──────────────────────────────────────────────────────────────

    def _fit_quantiles(self, X: pd.DataFrame, y: np.ndarray) -> dict:
        import lightgbm as lgb

        models = {}
        for alpha in QUANTILES:
            pipe = Pipeline([
                ("pre", make_preprocessor(ta_target_encoding=self.ta_target_encoding)),
                ("model", lgb.LGBMRegressor(objective="quantile", alpha=alpha,
                                            **self.params)),
            ])
            pipe.fit(X, y)
            models[alpha] = pipe
        return models

    def fit(self, train: pd.DataFrame, target: str):
        train = train[train[target].notna()].reset_index(drop=True)
        X = build_features(train, self.phase_key)
        y = self._fwd(train[target].to_numpy(dtype=float))

        self.qhat_ = 0.0
        if not self.conformal or len(y) < 100:
            self.models = self._fit_quantiles(X, y)
            return self

        n_cal = max(30, int(len(y) * self.calib_frac))
        if self.calib_strategy == "recent" and "Start Date" in train.columns:
            order = np.argsort(pd.to_datetime(train["Start Date"]).to_numpy())
            cal_idx, tr_idx = order[-n_cal:], order[:-n_cal]
        else:
            from sklearn.model_selection import train_test_split
            tr_idx, cal_idx = train_test_split(
                np.arange(len(y)), test_size=self.calib_frac, random_state=42)

        self.models = self._fit_quantiles(X.iloc[tr_idx], y[tr_idx])

        Xc = X.iloc[cal_idx]
        lo = self.models[0.1].predict(Xc)
        hi = self.models[0.9].predict(Xc)
        scores = np.maximum(lo - y[cal_idx], y[cal_idx] - hi)
        n = len(scores)
        level = min(1.0, np.ceil((n + 1) * self.coverage) / n)
        self.qhat_ = float(np.quantile(scores, level, method="higher"))
        log.info("%s/%s conformal qhat=%.4f on n=%d calibration rows",
                 self.phase_key, target, self.qhat_, n)
        return self

    # ── prediction ───────────────────────────────────────────────────────────

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """X is already feature-built (see build_features)."""
        return np.maximum(self._floor, self._inv(self.models[0.5].predict(X)))

    def predict_interval(self, X: pd.DataFrame):
        lo = self.models[0.1].predict(X) - self.qhat_
        hi = self.models[0.9].predict(X) + self.qhat_
        lower = np.maximum(self._floor, self._inv(lo))
        upper = np.maximum(self._floor, self._inv(hi))
        # Independently fitted quantile models can cross on thin slices.
        return np.minimum(lower, upper), np.maximum(lower, upper)

    def predict_df(self, df: pd.DataFrame):
        """Convenience: build features from a raw frame, then predict."""
        X = build_features(df, self.phase_key)
        point = self.predict(X)
        lower, upper = self.predict_interval(X)
        return point, lower, upper
