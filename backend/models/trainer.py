"""Train phase-specific RandomForest models and save artifacts."""
from __future__ import annotations
import asyncio
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from backend.config import settings
from backend.constants import PHASES
from backend.data.data_layer import get_raw_dataframe
from backend.preprocessing.cleaner import clean
from backend.preprocessing.pipeline import build_features, make_preprocessor

log = logging.getLogger(__name__)

_RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 15,
    "min_samples_leaf": 10,
    "max_features": 0.5,
    "random_state": 42,
    "n_jobs": -1,
}


def _artifact_paths(phase_key: str) -> tuple[Path, Path, Path]:
    base = settings.models_dir / phase_key
    return base / "model.pkl", base / "metadata.json", base / "analytics.json"


def _save(phase_key: str, pipeline: Pipeline, rmse: float, n_train: int,
          analytics: dict, defaults: dict, ranges: dict) -> None:
    model_path, meta_path, analytics_path = _artifact_paths(phase_key)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    meta_path.write_text(json.dumps({
        "rmse": rmse,
        "n_train": n_train,
        "feature_defaults": defaults,
        "feature_ranges": ranges,
    }, indent=2))
    analytics_path.write_text(json.dumps(analytics))
    log.info("Saved %s model (RMSE=%.1f days, n=%d)", phase_key, rmse, n_train)


def _feature_defaults(X: pd.DataFrame) -> dict:
    """Median for numerics, mode for categoricals, taken from the training set.

    Inference used to hardcode these (Allocation="RANDOMIZED", Masking="DOUBLE",
    site_count from a guessed table). Deriving them from the fitted data means
    an unspecified field falls back to what that phase actually looks like.
    """
    out: dict[str, object] = {}
    for col in X.columns:
        s = X[col]
        if pd.api.types.is_numeric_dtype(s):
            val = s.median()
            out[col] = None if pd.isna(val) else float(val)
        else:
            mode = s.mode()
            out[col] = str(mode.iloc[0]) if len(mode) else "UNKNOWN"
    return out


def _feature_ranges(X: pd.DataFrame) -> dict:
    """1st/99th percentile per numeric feature, for the inference-time
    extrapolation guard.

    This is the check that would have caught the original bug: v1 trained
    site_count on 1..20 (it was really a country count) and served predictions
    at site_count=40, outside the trained range, where the forest is flat and
    every therapeutic area collapses to the same answer.
    """
    out: dict[str, dict] = {}
    for col in X.columns:
        s = X[col]
        if not pd.api.types.is_numeric_dtype(s) or s.dropna().empty:
            continue
        out[col] = {
            "p01": float(s.quantile(0.01)),
            "p99": float(s.quantile(0.99)),
            "min": float(s.min()),
            "max": float(s.max()),
        }
    return out


def _build_analytics(X: pd.DataFrame, y: pd.Series) -> dict:
    """Pre-compute box-plot stats per therapeutic area for the analytics endpoint."""
    from backend.constants import THERAPEUTIC_AREAS
    stats = {}
    for ta in THERAPEUTIC_AREAS:
        if ta not in X.columns:
            continue
        mask = X[ta] == 1
        vals = y[mask].dropna()
        if len(vals) < 5:
            continue
        q = vals.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).tolist()
        stats[ta] = {
            "q10": round(q[0], 1),
            "q25": round(q[1], 1),
            "median": round(q[2], 1),
            "q75": round(q[3], 1),
            "q90": round(q[4], 1),
            "mean": round(float(vals.mean()), 1),
            "n": int(len(vals)),
        }
    return stats


async def train_phase(phase_key: str) -> None:
    log.info("Training %s ...", phase_key)
    raw = await get_raw_dataframe(phase_key)
    df = clean(raw, phase_key)

    # Filter HV / non-HV
    hv_flag = PHASES[phase_key]["hv"]
    if "is_hv" in df.columns:
        df = df[df["is_hv"] == int(hv_flag)].reset_index(drop=True)

    X = build_features(df, phase_key)
    y = df["duration_days"]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = make_preprocessor()
    model = RandomForestRegressor(**_RF_PARAMS)
    pipe = Pipeline([("pre", preprocessor), ("model", model)])
    pipe.fit(X_tr, y_tr)

    y_pred = pipe.predict(X_te)
    rmse = float(np.sqrt(mean_squared_error(y_te, y_pred)))
    log.info("%s test RMSE: %.1f days", phase_key, rmse)

    analytics = _build_analytics(X_te, y_te)
    _save(phase_key, pipe, rmse, len(X_tr), analytics,
          _feature_defaults(X_tr), _feature_ranges(X_tr))


async def train_all() -> None:
    for phase_key in PHASES:
        try:
            await train_phase(phase_key)
        except Exception as exc:
            log.error("Failed to train %s: %s", phase_key, exc)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(train_all())
