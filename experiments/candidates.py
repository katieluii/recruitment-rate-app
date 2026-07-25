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

from backend.models.trainer import _RF_PARAMS
from backend.preprocessing.pipeline import build_features, make_preprocessor

log = logging.getLogger(__name__)

_Z_80 = 1.28  # matches backend/models/inference.py


class V1Recipe:
    """The shipped recipe, refit on whatever fold it is given."""

    name = "v1_recipe"

    def __init__(self, phase_key: str, params: dict | None = None):
        self.phase_key = phase_key
        self.params = dict(params or _RF_PARAMS)

    def _X(self, df: pd.DataFrame) -> pd.DataFrame:
        return build_features(df, self.phase_key)

    def fit(self, train: pd.DataFrame, target: str = "duration_days"):
        X = self._X(train)
        y = train[target].to_numpy(dtype=float)
        self.pipe = Pipeline([
            ("pre", make_preprocessor()),
            ("model", RandomForestRegressor(**self.params)),
        ])
        self.pipe.fit(X, y)

        # v1 derives its interval half-width from in-sample-split RMSE; reproduce
        # that by scoring on the training fold's own held-out portion.
        from sklearn.model_selection import train_test_split
        Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.2, random_state=42)
        probe = Pipeline([
            ("pre", make_preprocessor()),
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
