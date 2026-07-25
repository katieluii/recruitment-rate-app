from __future__ import annotations
"""Train the two prediction heads per phase and save artifacts.

HEAD A — duration       : start → primary completion, in days.
HEAD B — recruitment rate: patients per site per month.

They are separate models because they are separate processes. Duration fuses
recruitment with follow-up, and follow-up is what makes Oncology Phase 3 run a
median 34 months while Dermatology runs 15. One model over the blended target
could not express that and regressed everything to the phase mean.

Evaluation does NOT happen here — `experiments/` owns that, on a temporal
holdout against a per-therapeutic-area median baseline. This module fits the
final models on all available data and records what inference needs.
"""
import asyncio
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from backend.config import settings
from backend.constants import PHASES
from backend.data.data_layer import get_raw_dataframe
from backend.models.quantile_model import QUANTILES, ConformalQuantileModel
from backend.preprocessing.cleaner import clean
from backend.preprocessing.pipeline import build_features

log = logging.getLogger(__name__)

#: head name → (target column, transform)
HEADS = {
    "duration": ("duration_days", "log1p"),
    "rate": ("recruitment_rate", "log"),
}


def _phase_dir(phase_key: str) -> Path:
    return settings.models_dir / phase_key


def _build_analytics(df: pd.DataFrame) -> dict:
    """Per-therapeutic-area duration box-plot stats for the analytics endpoint."""
    from experiments.metrics import ta_masks  # shared TA masking convention

    stats = {}
    y = df["duration_days"]
    for area, mask in ta_masks(df).items():
        vals = y[mask.to_numpy()].dropna()
        if len(vals) < 5:
            continue
        q = vals.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).tolist()
        stats[area] = {
            "q10": round(q[0], 1), "q25": round(q[1], 1), "median": round(q[2], 1),
            "q75": round(q[3], 1), "q90": round(q[4], 1),
            "mean": round(float(vals.mean()), 1), "n": int(len(vals)),
        }
    return stats


def _feature_defaults(X: pd.DataFrame) -> dict:
    """Median for numerics, mode for categoricals, from the fitted data.

    Inference used to hardcode these (Allocation="RANDOMIZED", Masking="DOUBLE",
    primary_completion_year=2024), so the only thing that varied between two
    requests for different therapeutic areas was 22 sparse binary columns.
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
    """Trained range per numeric feature, for the extrapolation guard.

    This is the check that would have caught the original bug: v1 trained
    site_count on 1..20 (it was really a country count) and then served
    predictions at site_count=40, outside the trained range, where a forest is
    flat and every therapeutic area collapses onto the same answer.
    """
    out: dict[str, dict] = {}
    for col in X.columns:
        s = X[col]
        if not pd.api.types.is_numeric_dtype(s) or s.dropna().empty:
            continue
        out[col] = {"p01": float(s.quantile(0.01)), "p99": float(s.quantile(0.99)),
                    "min": float(s.min()), "max": float(s.max())}
    return out


async def train_phase(phase_key: str) -> None:
    log.info("Training %s ...", phase_key)
    raw = await get_raw_dataframe(phase_key)
    df = clean(raw, phase_key)

    hv_flag = PHASES[phase_key]["hv"]
    if "is_hv" in df.columns:
        df = df[df["is_hv"] == int(hv_flag)].reset_index(drop=True)

    base = _phase_dir(phase_key)
    base.mkdir(parents=True, exist_ok=True)

    meta: dict = {"n_train": int(len(df)), "heads": {}}

    for head, (target, transform) in HEADS.items():
        sub = df[df[target].notna()].reset_index(drop=True)
        if len(sub) < 50:
            log.warning("%s/%s: only %d usable rows — skipping head",
                        phase_key, head, len(sub))
            continue

        model = ConformalQuantileModel(phase_key, transform=transform)
        model.fit(sub, target)

        for alpha, pipe in model.models.items():
            joblib.dump(pipe, base / f"{head}_q{int(alpha * 100)}.pkl")

        meta["heads"][head] = {
            "target": target,
            "transform": transform,
            "qhat": model.qhat_,
            "quantiles": list(QUANTILES),
            "n_fit": int(len(sub)),
            "coverage_nominal": model.coverage,
        }
        log.info("Saved %s/%s (n=%d, qhat=%.4f)", phase_key, head, len(sub), model.qhat_)

    X = build_features(df, phase_key)
    meta["feature_defaults"] = _feature_defaults(X)
    meta["feature_ranges"] = _feature_ranges(X)

    # Kept for backwards compatibility with the v1 metadata contract; the
    # interval no longer derives from it.
    meta["rmse"] = float(np.sqrt(np.mean(
        (df["duration_days"] - df["duration_days"].median()) ** 2)))

    (base / "metadata.json").write_text(json.dumps(meta, indent=2))
    (base / "analytics.json").write_text(json.dumps(_build_analytics(df)))

    # Site-level priors are derived from the same cleaned frame, so they are
    # built here rather than recomputed per request.
    from backend.analytics.site_rates import build_priors
    (base / "site_priors.json").write_text(json.dumps(build_priors(df)))

    log.info("Saved %s metadata (%d rows)", phase_key, len(df))


async def train_all() -> None:
    for phase_key in PHASES:
        try:
            await train_phase(phase_key)
        except Exception as exc:
            log.error("Failed to train %s: %s", phase_key, exc, exc_info=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(train_all())
