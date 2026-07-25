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
