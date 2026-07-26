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
                 coverage: float = 0.80, calib_strategy: str = "recent",
                 censoring_frame: pd.DataFrame | None = None,
                 weight_cap: float = 10.0, country_mix: bool = True):
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
        self.censoring_frame = censoring_frame
        self.weight_cap = weight_cap
        self.country_mix = country_mix
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

    def _fit_quantiles(self, X: pd.DataFrame, y: np.ndarray,
                       sample_weight: np.ndarray | None = None) -> dict:
        import lightgbm as lgb

        models = {}
        for alpha in QUANTILES:
            pipe = Pipeline([
                ("pre", make_preprocessor(
                    ta_target_encoding=self.ta_target_encoding,
                    country_mix=self.country_mix)),
                ("model", lgb.LGBMRegressor(objective="quantile", alpha=alpha,
                                            **self.params)),
            ])
            if sample_weight is None:
                pipe.fit(X, y)
            else:
                pipe.fit(X, y, model__sample_weight=sample_weight)
            models[alpha] = pipe
        return models

    # ── censoring correction ─────────────────────────────────────────────────

    def _ipcw_weights(self, train: pd.DataFrame, target: str) -> np.ndarray | None:
        """Inverse-probability-of-censoring weights for the training rows.

        The corpus holds completed trials, so the slow ones are systematically
        missing: a trial that started recently and has already finished is
        disproportionately a quick one. Measured by retrospective backtest —
        standing at a 2018 vantage and hiding what had not finished by then —
        Phase 3 duration LOOKED 20.9 months when it was truly 24.6.

        The fix estimates G(t), the probability of still being uncensored at t,
        with a reverse Kaplan-Meier over `censoring_frame`, then weights each
        completed trial by 1/G(T). Long trials are the ones censoring removes, so
        they are the ones upweighted, pushing the training set back toward the
        duration mix that would exist if every trial had been allowed to finish.

        Chosen over training a survival model directly: survival learners cut the
        bias too but lose more on scatter than they win, being weaker point
        predictors than gradient boosting. Backtest, MAE / bias in months:

            completed only   P2 11.75 / −4.57    P3 10.71 / −2.57
            IPCW             P2 11.70 / −2.61    P3 10.87 / −0.93
            survival (GBSA)  P2 12.53 / −1.27    P3 11.37 / +1.22
        """
        frame = self.censoring_frame
        if frame is None or "event_observed" not in frame.columns:
            return None
        if frame["event_observed"].nunique() < 2:
            log.info("%s: censoring frame has no censored rows — skipping IPCW",
                     self.phase_key)
            return None

        from lifelines import KaplanMeierFitter

        kmf = KaplanMeierFitter()
        # Flip the indicator: here the "event" is being censored.
        kmf.fit(frame["duration_days"].to_numpy(dtype=float),
                event_observed=1 - frame["event_observed"].to_numpy(dtype=int))

        t = train[target].to_numpy(dtype=float)
        g = np.clip(kmf.survival_function_at_times(t).to_numpy(),
                    1.0 / self.weight_cap, 1.0)
        w = np.clip(1.0 / g, 1.0, self.weight_cap)
        w = w / w.mean()  # hold the effective sample size steady
        log.info("%s IPCW weights: min %.2f max %.2f (%.0f%% of the frame censored)",
                 self.phase_key, w.min(), w.max(),
                 100 * (1 - frame["event_observed"].mean()))
        return w

    def fit(self, train: pd.DataFrame, target: str):
        train = train[train[target].notna()].reset_index(drop=True)
        X = build_features(train, self.phase_key)
        y = self._fwd(train[target].to_numpy(dtype=float))

        w = self._ipcw_weights(train, target)
        self.ipcw_applied_ = w is not None

        self.qhat_ = 0.0
        if not self.conformal or len(y) < 100:
            self.models = self._fit_quantiles(X, y, w)
            return self

        n_cal = max(30, int(len(y) * self.calib_frac))
        if self.calib_strategy == "recent" and "Start Date" in train.columns:
            order = np.argsort(pd.to_datetime(train["Start Date"]).to_numpy())
            cal_idx, tr_idx = order[-n_cal:], order[:-n_cal]
        else:
            from sklearn.model_selection import train_test_split
            tr_idx, cal_idx = train_test_split(
                np.arange(len(y)), test_size=self.calib_frac, random_state=42)

        self.models = self._fit_quantiles(
            X.iloc[tr_idx], y[tr_idx], None if w is None else w[tr_idx])

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


# ── Two-stage duration: enrolment window + follow-up ──────────────────────────

class TwoStageDuration:
    """Predict the recruiting window and the follow-up separately, then add them.

    Justified by the data rather than by taste: the two components are
    essentially UNCORRELATED (r = +0.03 on both P2 and P3) and split the variance
    roughly 60/40, so they are two processes wearing one number. Follow-up is
    where the therapeutic-area signal actually lives — a Phase 3 survival
    endpoint carries a 26.0 month median follow-up against 5.5 for a biomarker
    endpoint, while their recruiting windows are 11.1 and 13.5 months. Oncology
    is not slow to recruit; it is slow to finish. A single blended head has to
    infer that through therapeutic-area proxies, which is the inference it was
    failing to make.

    It also produces the split itself, which is the decision-useful part for
    trial planning: how long until the last patient is in, separately from how
    long until the endpoint reads out.
    """

    name = "two_stage_duration"

    def __init__(self, phase_key: str, params: dict | None = None,
                 calib_frac: float = 0.2, coverage: float = 0.80,
                 censoring_frame: pd.DataFrame | None = None,
                 country_mix: bool = True):
        self.phase_key = phase_key
        self.coverage = coverage
        self.calib_frac = calib_frac
        # Each stage fits its own 0.1/0.5/0.9 quantiles; the composite band is
        # assembled from both spreads and scaled once to hit nominal coverage.
        self.enrol = ConformalQuantileModel(
            phase_key, transform="log1p", params=params, conformal=False,
            censoring_frame=censoring_frame, country_mix=country_mix)
        # Follow-up is set by the protocol's endpoint, not by geography, so the
        # site mix is deliberately withheld from that stage.
        self.fu = ConformalQuantileModel(
            phase_key, transform="log1p", params=params, conformal=False,
            country_mix=False)
        self.scale_ = 1.0

    @staticmethod
    def _components(df: pd.DataFrame):
        from backend.preprocessing.cleaner import recruiting_months

        total = df["duration_days"] / 30.44
        enrol = recruiting_months(df)
        return enrol, (total - enrol).clip(lower=0.0)

    def _raw_band(self, test: pd.DataFrame):
        """Point estimate and one-sided half-widths, before scaling.

        The components are near-independent here, so their spreads combine in
        QUADRATURE rather than by simple addition — adding them outright would
        assume both stages always miss in the same direction.

        Crucially this keeps the band's shape: wide where the model is unsure,
        narrow where it is confident. An earlier cut applied one additive
        conformal shift instead, which gave every trial an identical width
        (CV 0.04) — the same decorative interval v1 shipped.
        """
        X = build_features(test, self.phase_key)
        e_lo, e_hi = self.enrol.predict_interval(X)
        f_lo, f_hi = self.fu.predict_interval(X)
        e_mid, f_mid = self.enrol.predict(X), self.fu.predict(X)
        point = e_mid + f_mid
        lo_w = np.sqrt((e_mid - e_lo) ** 2 + (f_mid - f_lo) ** 2)
        hi_w = np.sqrt((e_hi - e_mid) ** 2 + (f_hi - f_mid) ** 2)
        return point, lo_w, hi_w

    def fit(self, train: pd.DataFrame, target: str = "duration_days"):
        train = train[train[target].notna()].reset_index(drop=True)
        enrol, fu = self._components(train)

        # Calibrate on the most recent slice, matching the rest of the stack.
        n_cal = max(30, int(len(train) * self.calib_frac))
        order = np.argsort(pd.to_datetime(train["Start Date"]).to_numpy())
        cal_idx, tr_idx = order[-n_cal:], order[:-n_cal]

        tr = train.iloc[tr_idx].reset_index(drop=True)
        self.enrol.fit(tr.assign(_t=enrol.iloc[tr_idx].to_numpy() * 30.44), "_t")
        self.fu.fit(tr.assign(_t=fu.iloc[tr_idx].to_numpy() * 30.44), "_t")

        cal = train.iloc[cal_idx].reset_index(drop=True)
        y = cal[target].to_numpy(dtype=float)
        point, lo_w, hi_w = self._raw_band(cal)

        best, best_gap = 1.0, 9e9
        for g in np.linspace(0.3, 4.0, 200):
            cov = float((((point - g * lo_w) <= y) & (y <= (point + g * hi_w))).mean())
            gap = abs(cov - self.coverage)
            if gap < best_gap:
                best, best_gap = float(g), gap
        self.scale_ = best
        log.info("%s two-stage band scale %.2f (calibration gap %.3f)",
                 self.phase_key, self.scale_, best_gap)
        return self

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        X = build_features(test, self.phase_key)
        return np.maximum(1.0, self.enrol.predict(X) + self.fu.predict(X))

    def predict_components(self, test: pd.DataFrame):
        """Enrolment window and follow-up in days — the planning-useful split."""
        X = build_features(test, self.phase_key)
        return self.enrol.predict(X), self.fu.predict(X)

    def predict_interval(self, test: pd.DataFrame):
        point, lo_w, hi_w = self._raw_band(test)
        return (np.maximum(1.0, point - self.scale_ * lo_w),
                np.maximum(1.0, point + self.scale_ * hi_w))
