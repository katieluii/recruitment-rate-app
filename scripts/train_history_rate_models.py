from __future__ import annotations
"""Fit the production rate heads on quality-gated record-history targets.

This intentionally updates only ``rate_*`` artifacts and the rate block in
metadata.  WSi v5 duration artifacts are byte-for-byte untouched.
"""

import argparse
import json
from datetime import date
from pathlib import Path

import joblib
import pandas as pd

from backend.config import settings
from backend.models.quantile_model import (POINT_KEY, QUANTILES,
                                           ConformalQuantileModel, slot_name)
from experiments.eval_history_rate import _frame


def train_phase(phase: str, targets: pd.DataFrame,
                evaluation: dict) -> dict:
    frame = _frame(phase, targets)
    model = ConformalQuantileModel(
        phase, transform="log", coverage=0.80,
        country_mix=True, criteria_text=True,
    ).fit(frame, "recruitment_rate")

    base = settings.models_dir / phase
    for key, fitted in model.models.items():
        joblib.dump(fitted, base / f"rate_{slot_name(key)}.pkl")

    meta_path = base / "metadata.json"
    meta = json.loads(meta_path.read_text())
    phase_eval = next(p for p in evaluation["phases"] if p["phase"] == phase)
    meta.setdefault("heads", {})["rate"] = {
        "target": "recruitment_rate",
        "target_definition": (
            "actual enrollment / (initiated sites x recorded recruiting months)"
        ),
        "target_quality_tier": "B",
        "target_source": "ClinicalTrials.gov record history",
        "kind": "single",
        "transform": "log",
        "qhat": model.qhat_,
        "quantiles": list(QUANTILES),
        "point_model": f"rate_{slot_name(POINT_KEY)}.pkl",
        "n_fit": int(len(frame)),
        "coverage_nominal": model.coverage,
        "ipcw_applied": False,
        "built": date.today().isoformat(),
        "validation": {
            key: phase_eval.get(key) for key in (
                "n_test", "mae_raw", "medae_raw", "log_mae",
                "within_1_5x", "within_2x", "median_factor_error",
                "interval_coverage", "skill_vs_ta_median",
            )
        },
        "validation_window": {
            "train_before": evaluation["cutoff"],
            "test_start": evaluation["cutoff"],
            "test_end": evaluation["test_end"],
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return {"phase": phase, "n_fit": len(frame),
            "qhat": round(model.qhat_, 4),
            "validation": meta["heads"]["rate"]["validation"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", default="experiments/reports/recruitment-history-pilot.csv")
    parser.add_argument("--evaluation", default="experiments/reports/history-rate-evaluation.json")
    parser.add_argument("--phases", default="P1,P2,P3")
    args = parser.parse_args()
    targets = pd.read_csv(args.targets)
    evaluation = json.loads(Path(args.evaluation).read_text())
    results = [train_phase(p.strip(), targets, evaluation)
               for p in args.phases.split(",")]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
