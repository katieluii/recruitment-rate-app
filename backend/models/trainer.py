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
from backend.models.quantile_model import (POINT_KEY, QUANTILES,
                                           ConformalQuantileModel,
                                           TwoStageDuration, slot_name)
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


#: Fields whose typical value differs sharply by therapeutic area. An oncology
#: Phase 3 runs a median 89 sites and 502 patients; a dermatology Phase 3 runs 33
#: sites and 380. Falling back to one phase-wide median for every area makes two
#: very different trials look alike before the model has said anything, which is
#: part of why predictions clustered even after the model itself was fixed.
_TA_CONDITIONAL_FIELDS = (
    "Enrollment", "site_count", "country_count", "followup_months",
    "number_of_arms", "total_primary_outcomes", "total_secondary_outcomes",
    "n_inclusion_criteria", "n_exclusion_criteria", "criteria_chars",
    "enrollment_per_site",
)


def _feature_defaults_by_ta(df: pd.DataFrame, X: pd.DataFrame) -> dict:
    """Per-therapeutic-area medians for the fields that vary most by area.

    Only areas with enough trials get an entry; everything else falls back to
    the phase-wide default.
    """
    from experiments.metrics import ta_masks

    out: dict[str, dict] = {}
    for area, mask in ta_masks(df).items():
        m = mask.to_numpy()
        if m.sum() < 15:
            continue
        sub = X[m]
        vals = {}
        for col in _TA_CONDITIONAL_FIELDS:
            if col not in sub.columns:
                continue
            v = pd.to_numeric(sub[col], errors="coerce").median()
            if not pd.isna(v):
                vals[col] = float(v)
        if vals:
            out[area] = {"n": int(m.sum()), **vals}
    log.info("TA-conditional defaults for %d areas", len(out))
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


async def _censoring_frame(phase_key: str,
                           loader=None) -> pd.DataFrame | None:
    """Completed + still-running trials, for the IPCW censoring correction.

    Only the shape of the censoring distribution is needed, not the ongoing
    trials' features, so this is deliberately cheap. If the fetch fails the
    duration head simply trains unweighted rather than the whole run dying —
    losing a bias correction is worse than nothing, but not worse than no model.
    """
    from backend.data.ct_api_client import (ONGOING_STATUSES, fetch_studies,
                                            flatten_study)

    try:
        if loader is not None:
            completed, ongoing = loader.completed(phase_key), loader.ongoing(phase_key)
        else:
            completed = await get_raw_dataframe(phase_key)
            raw = await fetch_studies(PHASES[phase_key]["api_phases"],
                                      statuses=ONGOING_STATUSES)
            ongoing = pd.DataFrame(
                [r for s in raw if (r := flatten_study(s)) is not None])
        if ongoing.empty:
            return None
        for frame, flag in ((completed, 0), (ongoing, 1)):
            if "is_ongoing" not in frame.columns:
                frame["is_ongoing"] = flag
        combined = pd.concat([completed, ongoing], ignore_index=True, sort=False)
        frame = clean(combined, phase_key, censored=True)
        log.info("%s censoring frame: %d rows, %.0f%% censored", phase_key,
                 len(frame), 100 * (1 - frame["event_observed"].mean()))
        return frame
    except Exception as exc:
        log.warning("%s: could not build censoring frame (%s) — the duration head "
                    "will train without the IPCW correction", phase_key, exc)
        return None


async def train_phase(phase_key: str, loader=None) -> None:
    """Fit and save one phase's artifacts.

    `loader` supplies the raw frames instead of the CT.gov API. Passing the
    local parquet cache does two things: it avoids the twelve back-to-back
    fetches that retraining four phases would otherwise issue, and it makes the
    model train on EXACTLY the corpus the experiment harness measured, which is
    the point of keeping the fitted class in backend/ in the first place.
    """
    log.info("Training %s ...", phase_key)
    raw = loader.completed(phase_key) if loader is not None else await get_raw_dataframe(phase_key)
    df = clean(raw, phase_key)

    hv_flag = PHASES[phase_key]["hv"]
    if "is_hv" in df.columns:
        df = df[df["is_hv"] == int(hv_flag)].reset_index(drop=True)

    censoring = await _censoring_frame(phase_key, loader=loader)

    base = _phase_dir(phase_key)
    base.mkdir(parents=True, exist_ok=True)

    meta: dict = {"n_train": int(len(df)), "heads": {}}

    for head, (target, transform) in HEADS.items():
        sub = df[df[target].notna()].reset_index(drop=True)
        if len(sub) < 50:
            log.warning("%s/%s: only %d usable rows — skipping head",
                        phase_key, head, len(sub))
            continue

        if head == "duration":
            # Two stages, because they are two processes: the components are
            # near-uncorrelated (r = +0.03) and follow-up is where the
            # therapeutic-area signal lives. A P3 survival endpoint follows up
            # for a median 26.0 months against 5.5 for a biomarker endpoint,
            # while their recruiting windows are 11.1 and 13.5 — oncology is not
            # slow to recruit, it is slow to finish. Splitting them cut Phase 3
            # MAE from 7.29 to 6.90 months and Phase 2 interval width by 18%.
            model = TwoStageDuration(phase_key, censoring_frame=censoring)
            model.fit(sub, target)
            for stage, sub_model in (("enrolment", model.enrol), ("followup", model.fu)):
                for alpha, pipe in sub_model.models.items():
                    joblib.dump(pipe, base / f"{stage}_{slot_name(alpha)}.pkl")
            meta["heads"][head] = {
                "target": target,
                "kind": "two_stage",
                "stages": ["enrolment", "followup"],
                "transform": transform,
                "band_scale": model.scale_,
                "quantiles": list(QUANTILES),
                "n_fit": int(len(sub)),
                "coverage_nominal": model.coverage,
                "ipcw_applied": bool(getattr(model.enrol, "ipcw_applied_", False)),
            }
            log.info("Saved %s/duration two-stage (n=%d, band scale %.2f)",
                     phase_key, len(sub), model.scale_)
            continue

        # The rate head's censored rows carry no usable rate — their Enrollment
        # field is the target, not what has been recruited — so there is nothing
        # to reweight toward and IPCW would be noise.
        model = ConformalQuantileModel(phase_key, transform=transform)
        model.fit(sub, target)

        for alpha, pipe in model.models.items():
            joblib.dump(pipe, base / f"{head}_{slot_name(alpha)}.pkl")

        meta["heads"][head] = {
            "target": target,
            "kind": "single",
            "transform": transform,
            "qhat": model.qhat_,
            "quantiles": list(QUANTILES),
            "n_fit": int(len(sub)),
            "coverage_nominal": model.coverage,
            "ipcw_applied": bool(getattr(model, "ipcw_applied_", False)),
        }
        log.info("Saved %s/%s (n=%d, qhat=%.4f)", phase_key, head, len(sub), model.qhat_)

    X = build_features(df, phase_key)
    meta["feature_defaults"] = _feature_defaults(X)
    meta["feature_ranges"] = _feature_ranges(X)
    meta["feature_defaults_by_ta"] = _feature_defaults_by_ta(df, X)

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

    # Endpoint combinations, same reasoning: derived from the corpus, and the
    # deployed app has no corpus.
    from backend.analytics.endpoint_profiles import build_profiles
    (base / "endpoint_profiles.json").write_text(json.dumps(build_profiles(df)))

    log.info("Saved %s metadata (%d rows)", phase_key, len(df))


async def train_all(loader=None) -> None:
    for phase_key in PHASES:
        try:
            await train_phase(phase_key, loader=loader)
        except Exception as exc:
            log.error("Failed to train %s: %s", phase_key, exc, exc_info=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(train_all())
