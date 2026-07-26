from __future__ import annotations
"""Candidate models, each wrapped in the same fit/predict interface as the
baselines so `run.py` can score them identically.

`V1Recipe` reproduces exactly what ships today — the same features
(backend.preprocessing.pipeline.build_features), the same RandomForest
hyperparameters (backend.models.trainer._RF_PARAMS) and the same interval
formula (backend.models.inference, max(tree_std, rmse*0.5) at z=1.28). Refitting
that recipe on the temporal training fold is the only fair way to compare it to
the baselines; the shipped pickle was fitted on a random split that included
trials from after the test period.
"""
import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from backend.preprocessing.pipeline import build_features, make_preprocessor

log = logging.getLogger(__name__)

_Z_80 = 1.28  # v1's 80% z-score

#: v1's RandomForest hyperparameters, frozen here. They no longer live in
#: backend/models/trainer.py — that module now trains the two LightGBM heads —
#: but the harness still needs them to reproduce the old recipe as a reference
#: row. Do not "update" these: their whole purpose is to stay v1.
_RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 15,
    "min_samples_leaf": 10,
    "max_features": 0.5,
    "random_state": 42,
    "n_jobs": -1,
}


class V1Recipe:
    """The shipped recipe, refit on whatever fold it is given.

    `ta_target_encoding` toggles the therapeutic-area target encoder so its
    contribution can be isolated in the ledger rather than assumed.
    """

    name = "v1_recipe"

    def __init__(self, phase_key: str, params: dict | None = None,
                 ta_target_encoding: bool = False):
        self.phase_key = phase_key
        self.params = dict(params or _RF_PARAMS)
        self.ta_target_encoding = ta_target_encoding

    def _X(self, df: pd.DataFrame) -> pd.DataFrame:
        return build_features(df, self.phase_key)

    def _pre(self):
        return make_preprocessor(ta_target_encoding=self.ta_target_encoding)

    def fit(self, train: pd.DataFrame, target: str = "duration_days"):
        X = self._X(train)
        y = train[target].to_numpy(dtype=float)
        self.pipe = Pipeline([
            ("pre", self._pre()),
            ("model", RandomForestRegressor(**self.params)),
        ])
        self.pipe.fit(X, y)

        # v1 derives its interval half-width from in-sample-split RMSE; reproduce
        # that by scoring on the training fold's own held-out portion.
        from sklearn.model_selection import train_test_split
        Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.2, random_state=42)
        probe = Pipeline([
            ("pre", self._pre()),
            ("model", RandomForestRegressor(**self.params)),
        ])
        probe.fit(Xa, ya)
        self.rmse = float(np.sqrt(np.mean((probe.predict(Xb) - yb) ** 2)))
        return self

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        return self.pipe.predict(self._X(test))

    def predict_interval(self, test: pd.DataFrame):
        X = self._X(test)
        rf = self.pipe.named_steps["model"]
        Xt = self.pipe.named_steps["pre"].transform(X)
        tree_preds = np.stack([t.predict(Xt) for t in rf.estimators_])
        tree_std = tree_preds.std(axis=0)
        std_used = np.maximum(tree_std, self.rmse * 0.5)
        point = self.pipe.predict(X)
        return (np.maximum(1.0, point - _Z_80 * std_used), point + _Z_80 * std_used)


from backend.models.quantile_model import DEFAULT_PARAMS, ConformalQuantileModel

#: Kept as an alias so ledger rows and configs keep their names.
LGBM_PARAMS = DEFAULT_PARAMS


class LGBMQuantile:
    """Harness wrapper around the SHIPPED model class.

    Deliberately thin: the model itself lives in
    backend.models.quantile_model so that what the harness measures is
    literally what production runs. A second copy here would drift.
    """

    name = "lgbm_quantile"

    def __init__(self, phase_key: str, params: dict | None = None,
                 ta_target_encoding: bool = True, log_target: bool = True,
                 conformal: bool = True, calib_frac: float = 0.2,
                 coverage: float = 0.80, calib_strategy: str = "recent",
                 transform: str = "log1p"):
        self.phase_key = phase_key
        self.model = ConformalQuantileModel(
            phase_key,
            transform=transform if log_target else "none",
            params=params, ta_target_encoding=ta_target_encoding,
            conformal=conformal, calib_frac=calib_frac,
            coverage=coverage, calib_strategy=calib_strategy,
        )

    def fit(self, train: pd.DataFrame, target: str = "duration_days"):
        self.model.fit(train, target)
        return self

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        return self.model.predict(build_features(test, self.phase_key))

    def predict_interval(self, test: pd.DataFrame):
        return self.model.predict_interval(build_features(test, self.phase_key))


class LGBMPoint(LGBMQuantile):
    """Median-only variant — isolates how much of the gain is the interval setup."""

    name = "lgbm_point"

    def predict_interval(self, test: pd.DataFrame):
        raise NotImplementedError


class ShippedArtifact:
    """The actual pickled model currently in production.

    Scored only for the record — its training set overlaps any temporal test
    fold, so its numbers here are optimistic and must not be compared to
    candidates. It exists so the ledger holds the real deployed behaviour.
    """

    name = "v1_shipped"

    def __init__(self, phase_key: str, artifacts_dir: str | None = None):
        self.phase_key = phase_key
        self.artifacts_dir = artifacts_dir

    def fit(self, train: pd.DataFrame, target: str = "duration_days"):
        import json
        from pathlib import Path

        import joblib

        from backend.config import settings

        base = Path(self.artifacts_dir) if self.artifacts_dir else settings.models_dir
        self.pipe = joblib.load(base / self.phase_key / "model.pkl")
        self.rmse = json.loads(
            (base / self.phase_key / "metadata.json").read_text()
        )["rmse"]
        return self

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        return self.pipe.predict(build_features(test, self.phase_key))

    def predict_interval(self, test: pd.DataFrame):
        X = build_features(test, self.phase_key)
        rf = self.pipe.named_steps["model"]
        Xt = self.pipe.named_steps["pre"].transform(X)
        tree_std = np.stack([t.predict(Xt) for t in rf.estimators_]).std(axis=0)
        std_used = np.maximum(tree_std, self.rmse * 0.5)
        point = self.pipe.predict(X)
        return (np.maximum(1.0, point - _Z_80 * std_used), point + _Z_80 * std_used)


# ── Survival models (v3.1) ────────────────────────────────────────────────────

class SurvivalModel:
    """Right-censored duration models.

    The point of these is not a fancier learner — it is being allowed to LOOK at
    the trials v2 had to throw away. The corpus is 59-66% censored once ongoing
    trials are admitted, and those trials have already run LONGER than the
    completed ones took to finish (P3: 25.2 months elapsed and counting, against
    an observed median of 21.0). Dropping them is not a neutral filter; it
    removes the slow half of recent history.

    `kind`:
      weibull_aft — lifelines, log-linear parametric. The 2025 duration survey
                    puts this class at C-index 0.754.
      rsf         — random survival forest (0.762 in the same survey).
      gbsa        — gradient-boosted survival analysis.
    """

    name = "survival"

    def __init__(self, phase_key: str, kind: str = "weibull_aft",
                 ta_target_encoding: bool = True, params: dict | None = None):
        self.phase_key = phase_key
        self.kind = kind
        self.ta_target_encoding = ta_target_encoding
        self.params = dict(params or {})

    def _prep(self, train: pd.DataFrame):
        X = build_features(train, self.phase_key)
        self.pre_ = make_preprocessor(ta_target_encoding=self.ta_target_encoding)
        y_time = train["duration_days"].to_numpy(dtype=float)
        # The encoder needs a target; give it the observed/censored time. It is
        # fitted train-fold only and out-of-fold within it, so this does not leak.
        Xt = self.pre_.fit_transform(X, y_time)
        return np.nan_to_num(np.asarray(Xt, dtype=float))

    def fit(self, train: pd.DataFrame, target: str = "duration_days"):
        if "event_observed" not in train.columns:
            raise ValueError(
                "SurvivalModel needs an `event_observed` column — load the frame "
                "with experiments.dataset.load_clean_censored")

        Xt = self._prep(train)
        time = train["duration_days"].to_numpy(dtype=float)
        event = train["event_observed"].to_numpy(dtype=int)

        if self.kind == "weibull_aft":
            from lifelines import WeibullAFTFitter

            df = pd.DataFrame(Xt, columns=[f"f{i}" for i in range(Xt.shape[1])])
            # Drop zero-variance columns; a constant column makes the AFT design
            # matrix singular and the fit throws rather than degrading.
            keep = df.std(axis=0) > 1e-9
            self.cols_ = list(df.columns[keep])
            df = df[self.cols_]
            df["_T"] = np.maximum(time, 1.0)
            df["_E"] = event
            self.model_ = WeibullAFTFitter(penalizer=self.params.get("penalizer", 0.1))
            self.model_.fit(df, duration_col="_T", event_col="_E")
        else:
            from sksurv.ensemble import (GradientBoostingSurvivalAnalysis,
                                         RandomSurvivalForest)

            y = np.array([(bool(e), t) for e, t in zip(event, time)],
                         dtype=[("event", "?"), ("time", "<f8")])
            if self.kind == "rsf":
                self.model_ = RandomSurvivalForest(
                    n_estimators=self.params.get("n_estimators", 300),
                    min_samples_leaf=self.params.get("min_samples_leaf", 15),
                    max_features="sqrt", n_jobs=-1, random_state=42)
            elif self.kind == "gbsa":
                self.model_ = GradientBoostingSurvivalAnalysis(
                    n_estimators=self.params.get("n_estimators", 300),
                    learning_rate=self.params.get("learning_rate", 0.05),
                    max_depth=self.params.get("max_depth", 3), random_state=42)
            else:
                raise ValueError(f"Unknown survival kind {self.kind!r}")
            self.model_.fit(Xt, y)
        return self

    def _transform(self, test: pd.DataFrame) -> np.ndarray:
        Xt = self.pre_.transform(build_features(test, self.phase_key))
        return np.nan_to_num(np.asarray(Xt, dtype=float))

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        """Median predicted duration in days."""
        Xt = self._transform(test)

        if self.kind == "weibull_aft":
            df = pd.DataFrame(Xt, columns=[f"f{i}" for i in range(Xt.shape[1])])
            return np.maximum(1.0, self.model_.predict_median(df[self.cols_]).to_numpy())

        # sksurv gives survival curves; read the median off each one.
        surv = self.model_.predict_survival_function(Xt, return_array=True)
        times = self.model_.unique_times_
        out = np.empty(len(Xt))
        for i, curve in enumerate(surv):
            below = np.where(curve <= 0.5)[0]
            out[i] = times[below[0]] if len(below) else times[-1]
        return np.maximum(1.0, out)

    def risk(self, test: pd.DataFrame) -> np.ndarray:
        """Higher = shorter predicted duration, for C-index ranking."""
        return -self.predict(test)

    def predict_interval(self, test: pd.DataFrame):
        if self.kind != "weibull_aft":
            raise NotImplementedError
        Xt = self._transform(test)
        df = pd.DataFrame(Xt, columns=[f"f{i}" for i in range(Xt.shape[1])])[self.cols_]
        lo = self.model_.predict_percentile(df, p=0.1).to_numpy()
        hi = self.model_.predict_percentile(df, p=0.9).to_numpy()
        lo = np.nan_to_num(lo, nan=1.0, posinf=1.0)
        hi = np.nan_to_num(hi, nan=np.nanmax(lo) * 3 if len(lo) else 1000.0,
                           posinf=np.nanmax(lo) * 3 if len(lo) else 1000.0)
        return np.minimum(lo, hi), np.maximum(lo, hi)


class IPCWLGBMQuantile(LGBMQuantile):
    """LightGBM trained on completed trials, reweighted to undo censoring bias.

    Why this shape. The naive v3.1 arms showed that admitting censored trials
    halves the systematic bias (P2 −4.22 → −2.32 months on a near-unbiased test
    window) but costs more in scatter than it wins, because survival learners
    are weaker point predictors than gradient boosting. So keep the learner and
    move the correction into the sample weights.

    Inverse-probability-of-censoring weighting: estimate G(t), the probability a
    trial is still uncensored at t, with a reverse Kaplan-Meier over the
    censoring distribution, then weight each completed trial by 1/G(T_i). Long
    trials are the ones censoring removes, so they are exactly the ones that get
    upweighted — the training set is pushed back toward the duration mix that
    would exist if every trial had been allowed to finish.
    """

    name = "ipcw_lgbm"

    def __init__(self, phase_key: str, censored_frame: pd.DataFrame | None = None,
                 weight_cap: float = 10.0, **kwargs):
        super().__init__(phase_key, **kwargs)
        self.censored_frame = censored_frame
        self.weight_cap = weight_cap

    def _censoring_survival(self, frame: pd.DataFrame):
        """Reverse Kaplan-Meier: survival of the CENSORING process."""
        from lifelines import KaplanMeierFitter

        kmf = KaplanMeierFitter()
        # Flip the indicator: an "event" here is being censored.
        kmf.fit(frame["duration_days"].to_numpy(dtype=float),
                event_observed=1 - frame["event_observed"].to_numpy(dtype=int))
        return kmf

    def _weights(self, train: pd.DataFrame) -> np.ndarray:
        frame = self.censored_frame
        if frame is None or "event_observed" not in frame.columns:
            log.warning("No censored frame supplied — IPCW falls back to uniform weights")
            return np.ones(len(train))

        kmf = self._censoring_survival(frame)
        t = train["duration_days"].to_numpy(dtype=float)
        g = kmf.survival_function_at_times(t).to_numpy()
        g = np.clip(g, 1.0 / self.weight_cap, 1.0)
        w = 1.0 / g
        w = np.clip(w, 1.0, self.weight_cap)
        w = w / w.mean()  # keep the effective sample size stable
        log.info("%s IPCW weights: min %.2f max %.2f mean %.2f",
                 self.phase_key, w.min(), w.max(), w.mean())
        return w

    def fit(self, train: pd.DataFrame, target: str = "duration_days"):
        from backend.models.quantile_model import make_preprocessor as _mp  # noqa: F401
        import lightgbm as lgb
        from sklearn.pipeline import Pipeline as _P

        m = self.model
        train = train[train[target].notna()].reset_index(drop=True)
        X = build_features(train, self.phase_key)
        y = m._fwd(train[target].to_numpy(dtype=float))
        w = self._weights(train)

        def fit_quantiles(Xf, yf, wf):
            out = {}
            for alpha in (0.1, 0.5, 0.9):
                pipe = _P([
                    ("pre", make_preprocessor(ta_target_encoding=m.ta_target_encoding)),
                    ("model", lgb.LGBMRegressor(objective="quantile", alpha=alpha,
                                                **m.params)),
                ])
                pipe.fit(Xf, yf, model__sample_weight=wf)
                out[alpha] = pipe
            return out

        m.qhat_ = 0.0
        if not m.conformal or len(y) < 100:
            m.models = fit_quantiles(X, y, w)
            return self

        n_cal = max(30, int(len(y) * m.calib_frac))
        order = np.argsort(pd.to_datetime(train["Start Date"]).to_numpy())
        cal_idx, tr_idx = order[-n_cal:], order[:-n_cal]
        m.models = fit_quantiles(X.iloc[tr_idx], y[tr_idx], w[tr_idx])

        Xc = X.iloc[cal_idx]
        lo = m.models[0.1].predict(Xc)
        hi = m.models[0.9].predict(Xc)
        scores = np.maximum(lo - y[cal_idx], y[cal_idx] - hi)
        level = min(1.0, np.ceil((len(scores) + 1) * m.coverage) / len(scores))
        m.qhat_ = float(np.quantile(scores, level, method="higher"))
        return self


# V3.3 two-stage duration now lives in backend.models.quantile_model so the
# harness measures exactly what ships.
from backend.models.quantile_model import TwoStageDuration  # noqa: E402,F401


class StratifiedTwoStage:
    """One model per (phase x therapeutic area) instead of one per phase.

    The architecture question: does giving each indication its own model let it
    learn structure a pooled model has to compromise on, or does splitting the
    data cost more in variance than it wins in specificity?

    Pooling already gets the area signal through TATargetEncoder and the area
    one-hots. Stratifying goes further: separate trees, separate splits, separate
    calibration per cell. It also cuts each model's training set by roughly an
    order of magnitude, which is the whole risk.

    A trial can map to several areas, so each is assigned its RAREST qualifying
    area as its primary. Rare beats common because it is the more specific claim:
    a trial tagged both Oncology and Other is an oncology trial.

    Cells below `min_cell` fall back to the pooled model rather than fitting a
    model on 40 rows.
    """

    name = "stratified_two_stage"

    def __init__(self, phase_key: str, min_cell: int = 150,
                 params: dict | None = None):
        self.phase_key = phase_key
        self.min_cell = min_cell
        self.params = params

    @staticmethod
    def _primary_area(df: pd.DataFrame, eligible: set) -> pd.Series:
        from experiments.metrics import ta_masks

        masks = ta_masks(df)
        sizes = {a: int(m.sum()) for a, m in masks.items()}
        out = []
        for i in range(len(df)):
            areas = [a for a, m in masks.items() if m.iloc[i] and a in eligible]
            areas = [a for a in areas if a != "Other"] or areas
            out.append(min(areas, key=lambda a: sizes[a]) if areas else None)
        return pd.Series(out, index=df.index)

    def fit(self, train: pd.DataFrame, target: str = "duration_days"):
        from backend.models.quantile_model import TwoStageDuration
        from experiments.metrics import ta_masks

        train = train.reset_index(drop=True)
        masks = ta_masks(train)
        self.eligible_ = {a for a, m in masks.items() if m.sum() >= self.min_cell}

        # The pooled model is both the fallback and the honest control: if no cell
        # beats it, stratifying bought nothing.
        self.pooled_ = TwoStageDuration(self.phase_key, params=self.params)
        self.pooled_.fit(train, target)

        self.models_ = {}
        primary = self._primary_area(train, self.eligible_)
        for area in sorted(self.eligible_):
            sub = train[primary == area]
            if len(sub) < self.min_cell:
                continue
            try:
                m = TwoStageDuration(self.phase_key, params=self.params)
                m.fit(sub.reset_index(drop=True), target)
                self.models_[area] = m
            except Exception as exc:
                log.warning("%s/%s cell failed (%d rows): %s",
                            self.phase_key, area, len(sub), exc)
        log.info("%s: %d cell models fitted (min_cell=%d), pooled fallback ready",
                 self.phase_key, len(self.models_), self.min_cell)
        return self

    def _route(self, test: pd.DataFrame) -> pd.Series:
        return self._primary_area(test.reset_index(drop=True), set(self.models_))

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        test = test.reset_index(drop=True)
        out = self.pooled_.predict(test)
        route = self._route(test)
        for area, m in self.models_.items():
            sel = (route == area).to_numpy()
            if sel.any():
                out[sel] = m.predict(test[sel].reset_index(drop=True))
        return out

    def predict_interval(self, test: pd.DataFrame):
        test = test.reset_index(drop=True)
        lo, hi = self.pooled_.predict_interval(test)
        route = self._route(test)
        for area, m in self.models_.items():
            sel = (route == area).to_numpy()
            if sel.any():
                l, h = m.predict_interval(test[sel].reset_index(drop=True))
                lo[sel], hi[sel] = l, h
        return lo, hi

    def routing_report(self, test: pd.DataFrame) -> pd.Series:
        return self._route(test).fillna("(pooled)").value_counts()
