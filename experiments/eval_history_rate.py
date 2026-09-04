from __future__ import annotations
"""Evaluate a direct recruitment-rate model on registry-history targets."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backend.models.quantile_model import ConformalQuantileModel
from backend.preprocessing.cleaner import clean
from experiments.baselines import TAMedianBaseline
from experiments.dataset import load_raw
from experiments.metrics import evaluate, skill_score
from experiments.splits import check_split_viability, temporal_split


def _frame(phase: str, targets: pd.DataFrame) -> pd.DataFrame:
    raw = load_raw(phase)
    if phase == "P1" and "is_hv" in raw:
        raw = raw[raw["is_hv"] == 0]
    frame = clean(raw.copy(), phase)
    phase_targets = targets[
        (targets["phase"] == phase) & targets["usable"].astype(bool)
    ][["nct_id", "recruitment_rate", "quality_tier"]].copy()
    phase_targets["history_recruitment_rate"] = pd.to_numeric(
        phase_targets["recruitment_rate"], errors="coerce")
    frame = frame.merge(
        phase_targets[["nct_id", "history_recruitment_rate", "quality_tier"]],
        on="nct_id", how="inner", validate="one_to_one",
    )
    frame["recruitment_rate"] = frame["history_recruitment_rate"]
    frame = frame.replace([np.inf, -np.inf], np.nan)
    return frame.dropna(subset=["recruitment_rate"]).reset_index(drop=True)


def _multiplicative(y: np.ndarray, pred: np.ndarray) -> dict:
    ratio = np.maximum(y, pred) / np.maximum(np.minimum(y, pred), 1e-9)
    return {
        "log_mae": round(float(np.mean(np.abs(np.log(pred) - np.log(y)))), 4),
        "within_1_5x": round(float(np.mean(ratio <= 1.5)), 3),
        "within_2x": round(float(np.mean(ratio <= 2.0)), 3),
        "median_factor_error": round(float(np.median(ratio)), 3),
    }


def run_phase(phase: str, targets: pd.DataFrame,
              cutoff: str, test_end: str) -> dict:
    frame = _frame(phase, targets)
    train, test = temporal_split(frame, cutoff=cutoff, test_end=test_end)
    warning = check_split_viability(train, test, min_rows=100)
    if warning:
        return {"phase": phase, "error": warning, "n_total": len(frame)}

    model = ConformalQuantileModel(
        phase, transform="log", coverage=0.80,
        country_mix=True, criteria_text=True,
    ).fit(train, "recruitment_rate")
    pred, lower, upper = model.predict_df(test)
    baseline = TAMedianBaseline().fit(train, "recruitment_rate")
    base_pred = baseline.predict(test)

    y = test["recruitment_rate"].to_numpy(dtype=float)
    metrics = evaluate(test, y, pred, lower, upper, unit="raw", nominal=0.80)
    model_mae = float(np.mean(np.abs(pred - y)))
    baseline_mae = float(np.mean(np.abs(base_pred - y)))
    metrics.update(_multiplicative(y, pred))
    metrics.update({
        "phase": phase,
        "n_total": int(len(frame)),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "baseline_mae_raw": round(baseline_mae, 4),
        "skill_vs_ta_median": skill_score(model_mae, baseline_mae),
        "target_tier": sorted(frame["quality_tier"].dropna().unique().tolist()),
        "test_rate_median": round(float(np.median(y)), 4),
        "test_rate_p90": round(float(np.quantile(y, 0.9)), 4),
    })
    metrics.pop("_per_ta", None)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", default="experiments/reports/recruitment-history-pilot.csv")
    parser.add_argument("--phases", default="P1,P2,P3")
    parser.add_argument("--cutoff", default="2021-01-01")
    parser.add_argument("--test-end", default="2023-01-01")
    args = parser.parse_args()
    targets = pd.read_csv(args.targets)
    results = [run_phase(phase.strip(), targets, args.cutoff, args.test_end)
               for phase in args.phases.split(",")]
    out = {"target_definition": "actual enrollment / (initiated sites x recorded recruiting months)",
           "cutoff": args.cutoff, "test_end": args.test_end, "phases": results}
    path = Path("experiments/reports/history-rate-evaluation.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
