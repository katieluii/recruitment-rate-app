from __future__ import annotations
"""The decisive test of v3.1's premise, by simulating censoring where we already
know the truth.

    python -m experiments.censoring_backtest

THE PROBLEM WITH THE OBVIOUS EXPERIMENT
Censoring only exists in recent data. Unbiased ground truth only exists in old
data. Since the test fold must come after the training fold, any straightforward
temporal split gives you one or the other, never both: either the training set
has no censoring to correct, or the test set is itself biased toward fast trials
and quietly rewards a model for sharing that bias.

THE FIX
Stand at a past vantage date and pretend. Take trials that started before an
`as_of` date, and hide anything that had not finished by then — exactly what a
model trained at that moment would have seen. Then score against what actually
happened, which we know today.

  arm A `completed_only`   what v2 does: train on trials that LOOKED finished
                           as of the vantage date. Slow trials are invisible.
  arm B `ipcw`             same rows, reweighted by inverse probability of
                           censoring to undo the missing slow trials.
  arm C `survival`         trains on the hidden trials too, as censored rows.

All three are scored on the same held-out trials using their REAL durations.
"""
import argparse
import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from experiments import ledger
from experiments.baselines import TAMedianBaseline
from experiments.candidates import IPCWLGBMQuantile, LGBMQuantile, SurvivalModel
from experiments.dataset import load_clean_censored
from experiments.metrics import DAYS_PER_MONTH, concordance, point_metrics, skill_score

log = logging.getLogger(__name__)
REPORT_DIR = Path(__file__).parent / "reports"


def apply_retrospective_censoring(df: pd.DataFrame, as_of: str) -> pd.DataFrame:
    """Rewrite the frame as it would have looked on `as_of`.

    A trial that truly finished after the vantage date becomes censored at
    (as_of − start): back then you knew only that it was still running.
    """
    out = df.copy()
    cut = pd.Timestamp(as_of)
    start = pd.to_datetime(out["Start Date"])
    true_end = start + pd.to_timedelta(out["duration_days"], unit="D")

    finished_by_then = (out["event_observed"] == 1) & (true_end <= cut)
    out["event_observed_asof"] = finished_by_then.astype(int)
    out["duration_days_asof"] = np.where(
        finished_by_then, out["duration_days"], (cut - start).dt.days)

    out = out[out["duration_days_asof"] > 0].reset_index(drop=True)
    return out


def run_phase(phase_key: str, as_of: str, train_before: str,
              test_start: str, test_end: str) -> list[dict]:
    df = load_clean_censored(phase_key)
    start = pd.to_datetime(df["Start Date"])

    train_pool = df[start < pd.Timestamp(train_before)].reset_index(drop=True)
    test = df[(start >= pd.Timestamp(test_start))
              & (start < pd.Timestamp(test_end))
              & (df["event_observed"] == 1)].reset_index(drop=True)

    if len(train_pool) < 150 or len(test) < 50:
        log.warning("%s: too few rows (train pool %d, test %d)",
                    phase_key, len(train_pool), len(test))
        return []

    asof = apply_retrospective_censoring(train_pool, as_of)
    # What a model standing at `as_of` would have been able to train on.
    visible = asof[asof["event_observed_asof"] == 1].copy()
    visible["duration_days"] = visible["duration_days_asof"]
    visible["event_observed"] = 1

    hidden_frame = asof.copy()
    hidden_frame["duration_days"] = hidden_frame["duration_days_asof"]
    hidden_frame["event_observed"] = hidden_frame["event_observed_asof"]

    pct_hidden = 100 * (1 - asof["event_observed_asof"].mean())
    true_med = train_pool.loc[train_pool.event_observed == 1, "duration_days"].median()
    vis_med = visible["duration_days"].median()
    log.info("%s @ %s: %d in pool, %.1f%% hidden. Median duration LOOKED "
             "%.1f mo but truly was %.1f mo.",
             phase_key, as_of, len(asof), pct_hidden,
             vis_med / DAYS_PER_MONTH, true_med / DAYS_PER_MONTH)

    y_true = test["duration_days"].to_numpy(dtype=float)
    base = TAMedianBaseline().fit(visible, "duration_days")
    base_mae = point_metrics(y_true, base.predict(test))["mae_months"]

    arms = {
        "completed_only": lambda: LGBMQuantile(
            phase_key, calib_strategy="recent").fit(visible, "duration_days"),
        "ipcw": lambda: IPCWLGBMQuantile(
            phase_key, censored_frame=hidden_frame).fit(visible, "duration_days"),
        "survival_gbsa": lambda: SurvivalModel(
            phase_key, kind="gbsa").fit(hidden_frame, "duration_days"),
    }

    rows: list[dict] = []
    for name, build in arms.items():
        try:
            model = build()
            pred = model.predict(test)
            m = point_metrics(y_true, pred)
            rows.append({
                "config": f"backtest_{name}",
                "phase": phase_key,
                "as_of": as_of,
                "train_before": train_before,
                "test_window": f"{test_start}..{test_end}",
                "pct_hidden_at_asof": round(pct_hidden, 1),
                "n_train_visible": int(len(visible)),
                "n_train_total": int(len(hidden_frame)),
                "n_test": int(len(test)),
                **m,
                "baseline_mae": base_mae,
                "skill_vs_ta_median": skill_score(m["mae_months"], base_mae),
                "c_index": concordance(y_true, pred, np.ones(len(y_true), dtype=int)),
                "median_pred_months": round(float(np.median(pred)) / DAYS_PER_MONTH, 2),
                "median_actual_months": round(float(np.median(y_true)) / DAYS_PER_MONTH, 2),
            })
        except Exception as exc:
            log.error("%s/%s failed: %s", name, phase_key, exc, exc_info=True)
            rows.append({"config": f"backtest_{name}", "phase": phase_key,
                         "error": str(exc)})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", default="2018-01-01",
                    help="vantage date — trials unfinished by then are hidden")
    ap.add_argument("--train-before", default="2016-01-01")
    ap.add_argument("--test-start", default="2016-01-01")
    ap.add_argument("--test-end", default="2019-01-01")
    ap.add_argument("--phases", default="P2,P3")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    rows: list[dict] = []
    for phase in [p.strip() for p in args.phases.split(",")]:
        rows.extend(run_phase(phase, args.as_of, args.train_before,
                              args.test_start, args.test_end))
    for r in rows:
        ledger.append(r)

    ok = [r for r in rows if not r.get("error")]
    if ok:
        print("\n" + "=" * 112)
        print(pd.DataFrame([
            {k: r.get(k) for k in
             ("config", "phase", "pct_hidden_at_asof", "n_train_visible",
              "n_test", "mae_months", "bias_months", "skill_vs_ta_median",
              "median_pred_months", "median_actual_months")}
            for r in ok]).to_string(index=False))
        print("=" * 112)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / f"{date.today().isoformat()}-censoring-backtest.md"
    lines = ["# Censoring backtest — does admitting unfinished trials help?", "",
             f"Vantage date {args.as_of}. Trials that had not finished by then are",
             "hidden from the `completed_only` arm, exactly as they would have been.",
             "All arms scored on real durations we know today.", "",
             "| arm | phase | % hidden | MAE (mo) | bias (mo) | skill |",
             "|---|---|---|---|---|---|"]
    for r in ok:
        lines.append(f"| {r['config']} | {r['phase']} | {r['pct_hidden_at_asof']} "
                     f"| {r['mae_months']} | {r['bias_months']} "
                     f"| {r.get('skill_vs_ta_median')} |")
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report: {path}")


if __name__ == "__main__":
    main()
