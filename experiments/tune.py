from __future__ import annotations
"""Hyperparameter search for the duration heads.

    python -m experiments.tune --phases P2,P3
    python -m experiments.tune --phases P3 --trials 40

The LightGBM parameters have never been tuned — `DEFAULT_PARAMS` are library
defaults picked by hand. That is one of two untouched accuracy levers, the other
being training-set size, which was fixed by lifting the API cap.

Protocol, so a win here is real rather than a lucky split:
  * search on a VALIDATION split carved out of the training fold, never on the
    temporal test fold;
  * the test fold is scored once, at the end, for the winner only;
  * random search rather than grid — with this many interacting parameters a
    grid spends most of its budget on the unimportant ones.

Every trial lands in the ledger, so a tuning run is auditable like any other.
"""
import argparse
import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from backend.models.quantile_model import DEFAULT_PARAMS, TwoStageDuration
from experiments import ledger
from experiments.dataset import load_clean
from experiments.metrics import check_gates, point_metrics, skill_score
from experiments.splits import temporal_split

log = logging.getLogger(__name__)
REPORT_DIR = Path(__file__).parent / "reports"

#: Sampled per trial. Ranges chosen around the defaults, wide enough to escape
#: them but not so wide that most draws are obviously bad.
SPACE = {
    "n_estimators": [300, 600, 900, 1400, 2000],
    "learning_rate": [0.015, 0.025, 0.04, 0.06, 0.09],
    "num_leaves": [15, 31, 63, 127, 255],
    "min_child_samples": [5, 10, 20, 40, 80],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.5, 0.65, 0.8, 0.95],
    "reg_lambda": [0.0, 0.5, 1.0, 5.0, 20.0],
    "reg_alpha": [0.0, 0.1, 1.0],
}


def sample(rng: np.random.Generator) -> dict:
    p = {k: rng.choice(v).item() for k, v in SPACE.items()}
    p.update({"random_state": 42, "n_jobs": -1, "verbose": -1, "subsample_freq": 1})
    return p


def _validation_split(train: pd.DataFrame, frac: float = 0.25):
    """Most-recent slice of the TRAINING fold, mirroring the deployment order."""
    order = np.argsort(pd.to_datetime(train["Start Date"]).to_numpy())
    n_val = max(200, int(len(train) * frac))
    return (train.iloc[order[:-n_val]].reset_index(drop=True),
            train.iloc[order[-n_val:]].reset_index(drop=True))


def tune_phase(phase: str, n_trials: int, cutoff: str, seed: int = 0) -> dict:
    df = load_clean(phase)
    train, test = temporal_split(df, cutoff=cutoff)
    inner_train, val = _validation_split(train)
    log.info("%s: search on %d train / %d validation, holding %d test rows back",
             phase, len(inner_train), len(val), len(test))

    y_val = val["duration_days"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)

    baseline = TwoStageDuration(phase).fit(inner_train, "duration_days")
    best = {"params": dict(DEFAULT_PARAMS),
            **point_metrics(y_val, baseline.predict(val))}
    log.info("%s: defaults score MAE %.3f, R2 %.4f on validation",
             phase, best["mae_months"], best["r2"])

    for i in range(n_trials):
        params = sample(rng)
        try:
            m = TwoStageDuration(phase, params=params).fit(inner_train, "duration_days")
            s = point_metrics(y_val, m.predict(val))
        except Exception as exc:
            log.warning("%s trial %d failed: %s", phase, i, exc)
            continue
        # Rank on R2, since that is the gate now; MAE breaks ties.
        better = (s["r2"] or -9) > (best["r2"] or -9) or (
            abs((s["r2"] or 0) - (best["r2"] or 0)) < 1e-4
            and s["mae_months"] < best["mae_months"])
        if better:
            best = {"params": params, **s}
            log.info("%s trial %2d: NEW BEST R2 %.4f MAE %.3f", phase, i,
                     s["r2"], s["mae_months"])

    # Refit on the FULL training fold with the winner, then score the test fold once.
    final = TwoStageDuration(phase, params=best["params"]).fit(train, "duration_days")
    y_test = test["duration_days"].to_numpy(dtype=float)
    lo, hi = final.predict_interval(test)
    from experiments.baselines import TAMedianBaseline
    from experiments.metrics import interval_calibration

    base_mae = point_metrics(
        y_test, TAMedianBaseline().fit(train, "duration_days").predict(test)
    )["mae_months"]
    m = point_metrics(y_test, final.predict(test))
    m.update(interval_calibration(y_test, lo, hi))
    m["skill_vs_ta_median"] = skill_score(m["mae_months"], base_mae)
    gates = check_gates(m)

    row = {
        "config": "tuned_two_stage",
        "phase": phase,
        "split": "temporal",
        "cutoff": cutoff,
        "target": "duration_days",
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_trials_searched": n_trials,
        "params": best["params"],
        "validation_r2": best["r2"],
        **m,
        "gates": gates,
        "gate_pass": gates["all_pass"],
    }
    ledger.append(row)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phases", default="P1,P2,P3")
    ap.add_argument("--trials", type=int, default=25)
    ap.add_argument("--cutoff", default="2021-01-01")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    rows = [tune_phase(p.strip(), args.trials, args.cutoff)
            for p in args.phases.split(",")]

    print("\n" + "=" * 96)
    print(pd.DataFrame([
        {k: r.get(k) for k in
         ("phase", "n_train", "mae_months", "r2", "rmse_days",
          "skill_vs_ta_median", "interval_coverage", "gate_pass")}
        for r in rows]).to_string(index=False))
    print("=" * 96)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / f"{date.today().isoformat()}-tuning.md"
    lines = ["# Hyperparameter search", "",
             f"Random search, {args.trials} trials per phase. Searched on a validation",
             "slice of the training fold; the test fold was scored once for the winner.",
             "", "| phase | MAE (mo) | R2 | RMSE (d) | skill | coverage | gate |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['phase']} | {r['mae_months']} | {r['r2']} | {r['rmse_days']} "
                     f"| {r['skill_vs_ta_median']} | {r['interval_coverage']} "
                     f"| {'PASS' if r['gate_pass'] else 'fail'} |")
    lines += ["", "## Winning parameters", ""]
    for r in rows:
        lines.append(f"- **{r['phase']}** — `{r['params']}`")
    path.write_text("\n".join(lines))
    print(f"Report: {path}")


if __name__ == "__main__":
    main()
