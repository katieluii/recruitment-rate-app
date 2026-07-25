from __future__ import annotations
"""v3.1 — does admitting censored trials actually help?

    python -m experiments.run_survival

The comparison has to be apples-to-apples or it proves nothing, so:

  * BOTH arms are tested on the SAME trials — held-out studies that started
    after the cutoff and have since genuinely completed, so a real duration
    exists to score against.
  * The v2 arm trains only on completed trials, exactly as it ships today.
  * The survival arm trains on completed AND ongoing trials, with the ongoing
    ones right-censored at time-elapsed-so-far.

The only difference between the arms is what they were allowed to learn from.

C-index is additionally reported over the full test fold including still-running
trials, since ranking is defined under censoring even where MAE is not.
"""
import argparse
import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from backend.constants import PHASES
from experiments import ledger
from experiments.baselines import TAMedianBaseline
from experiments.candidates import LGBMQuantile, SurvivalModel
from experiments.dataset import load_clean, load_clean_censored
from experiments.metrics import DAYS_PER_MONTH, concordance, point_metrics, skill_score
from experiments.splits import temporal_split

log = logging.getLogger(__name__)

REPORT_DIR = Path(__file__).parent / "reports"

ARMS = {
    "v2_lgbm_completed_only": "LightGBM quantile, completed trials only (ships today)",
    "surv_weibull_aft": "Weibull AFT, completed + censored",
    "surv_rsf": "Random survival forest, completed + censored",
    "surv_gbsa": "Gradient-boosted survival, completed + censored",
}


def run_phase(phase_key: str, cutoff: str) -> list[dict]:
    censored = load_clean_censored(phase_key)
    completed = load_clean(phase_key)

    cens_train, cens_test = temporal_split(censored, cutoff=cutoff)
    comp_train, _ = temporal_split(completed, cutoff=cutoff)

    # Scoreable test set: post-cutoff trials with a REAL observed endpoint.
    scoreable = cens_test[cens_test["event_observed"] == 1].reset_index(drop=True)
    if len(comp_train) < 100 or len(scoreable) < 30:
        log.warning("%s: insufficient rows (train %d, scoreable test %d)",
                    phase_key, len(comp_train), len(scoreable))
        return []

    log.info("%s: train completed=%d, train censored-corpus=%d (%.0f%% censored), "
             "scoreable test=%d, full test=%d",
             phase_key, len(comp_train), len(cens_train),
             100 * (1 - cens_train["event_observed"].mean()),
             len(scoreable), len(cens_test))

    y_true = scoreable["duration_days"].to_numpy(dtype=float)
    full_time = cens_test["duration_days"].to_numpy(dtype=float)
    full_event = cens_test["event_observed"].to_numpy(dtype=int)

    # Baseline, fitted on completed trials only.
    base = TAMedianBaseline().fit(comp_train, "duration_days")
    base_mae = point_metrics(y_true, base.predict(scoreable))["mae_months"]

    rows: list[dict] = []
    for arm in ARMS:
        try:
            if arm == "v2_lgbm_completed_only":
                model = LGBMQuantile(phase_key, calib_strategy="recent")
                model.fit(comp_train, "duration_days")
                pred = model.predict(scoreable)
                full_pred = model.predict(cens_test)
            else:
                kind = arm.replace("surv_", "")
                model = SurvivalModel(phase_key, kind=kind)
                model.fit(cens_train, "duration_days")
                pred = model.predict(scoreable)
                full_pred = model.predict(cens_test)

            m = point_metrics(y_true, pred)
            row = {
                "config": arm,
                "phase": phase_key,
                "split": "temporal",
                "cutoff": cutoff,
                "target": "duration_days",
                "trains_on_censored": arm != "v2_lgbm_completed_only",
                "n_train": int(len(comp_train) if arm == "v2_lgbm_completed_only"
                               else len(cens_train)),
                "n_test_scoreable": int(len(scoreable)),
                **m,
                "baseline_mae": base_mae,
                "skill_vs_ta_median": skill_score(m["mae_months"], base_mae),
                "beats_baseline": bool(m["mae_months"] < base_mae),
                # Ranking over EVERY test trial, censored ones included.
                "c_index_full_test": concordance(full_time, full_pred, full_event),
                "c_index_scoreable": concordance(
                    y_true, pred, np.ones(len(y_true), dtype=int)),
                # Direction of the miss. A model de-biased against censoring
                # SHOULD predict longer, and will look worse on a test set that
                # is itself survivorship-biased toward fast trials.
                "median_pred_months": round(float(np.median(pred)) / DAYS_PER_MONTH, 2),
                "median_actual_months": round(float(np.median(y_true)) / DAYS_PER_MONTH, 2),
            }
            rows.append(row)
        except Exception as exc:
            log.error("%s/%s failed: %s", arm, phase_key, exc, exc_info=True)
            rows.append({"config": arm, "phase": phase_key, "error": str(exc)})
    return rows


def write_report(rows: list[dict], cutoff: str) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / f"{date.today().isoformat()}-v31-survival.md"
    lines = [
        "# v3.1 — right-censored duration models",
        "",
        f"Temporal split at {cutoff}. Both arms scored on the SAME held-out trials:",
        "post-cutoff studies that have since genuinely completed. The only difference",
        "is that the survival arms were also allowed to learn from trials that are",
        "still running, entered as right-censored observations.",
        "",
        "| arm | phase | n_train | MAE (mo) | skill vs TA-median | C-index (full test) |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        if r.get("error"):
            lines.append(f"| {r['config']} | {r['phase']} | _failed: {r['error'][:60]}_ | | | |")
            continue
        lines.append(
            f"| {r['config']} | {r['phase']} | {r['n_train']} | {r['mae_months']} "
            f"| {r.get('skill_vs_ta_median')} | {r.get('c_index_full_test')} |")
    lines += ["", "Arms:", ""] + [f"- `{k}` — {v}" for k, v in ARMS.items()]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", default="2021-01-01")
    ap.add_argument("--phases", default="P1,P2,P3")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    rows: list[dict] = []
    for phase in [p.strip() for p in args.phases.split(",")]:
        rows.extend(run_phase(phase, args.cutoff))

    for r in rows:
        ledger.append(r)

    ok = [r for r in rows if not r.get("error")]
    if ok:
        print("\n" + "=" * 108)
        print(pd.DataFrame([
            {k: r.get(k) for k in
             ("config", "phase", "n_train", "n_test_scoreable", "mae_months",
              "skill_vs_ta_median", "beats_baseline", "c_index_full_test")}
            for r in ok]).to_string(index=False))
        print("=" * 108)
    print(f"Report: {write_report(rows, args.cutoff)}")


if __name__ == "__main__":
    main()
