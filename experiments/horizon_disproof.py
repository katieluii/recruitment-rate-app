from __future__ import annotations
"""Is the horizon-matching gain real, or is it fitting the truncation?

    python -m experiments.horizon_disproof --phase P1

Horizon matching (dropping long trials from training) raises R2 by 0.048 on the
2021+ fold. That fold structurally cannot contain long trials, so "the model got
better" and "the model stopped predicting values the fold cannot hold" produce
the SAME number. They are told apart by scoring on a fold that is not truncated.

    train:            starts before 2018
    truncated test:   starts 2021+          (median horizon ~3.6 yr, capped)
    UNtruncated test: starts 2018 to 2020   (horizon 5.4-8.6 yr, effectively uncapped
                                             against a corpus whose p95 duration is 5.9)

If matching wins on the truncated fold and loses on the untruncated one, the gain
is an artifact of the metric and must not be shipped. If it wins on both, the long
trials really are hurting training and it is a real lever.
"""
import argparse
import logging

import numpy as np
import pandas as pd

from backend.models.quantile_model import TwoStageDuration
from backend.preprocessing.pipeline import set_top_sponsors
from backend.preprocessing.text_features import top_sponsors
from experiments.dataset import load_clean
from experiments.metrics import evaluate

log = logging.getLogger(__name__)


def _score(model, test: pd.DataFrame) -> dict:
    y = test["duration_days"].to_numpy(dtype=float)
    pred = model.predict(test)
    lo, hi = model.predict_interval(test)
    m = evaluate(test, y, pred, lo, hi, unit="days")
    m.pop("_per_ta", None)
    return m


def run(phase_key: str, max_years: float = 5.0) -> pd.DataFrame:
    df = load_clean(phase_key)
    df = df[df["duration_days"].notna()].reset_index(drop=True)
    start = pd.to_datetime(df["Start Date"])

    train = df[start < pd.Timestamp("2018-01-01")].reset_index(drop=True)
    untrunc = df[(start >= pd.Timestamp("2018-01-01"))
                 & (start < pd.Timestamp("2021-01-01"))].reset_index(drop=True)
    trunc = df[start >= pd.Timestamp("2021-01-01")].reset_index(drop=True)

    vintage = start.max()
    for nm, f in (("train <2018", train), ("test 2018-2020", untrunc),
                  ("test 2021+", trunc)):
        h = (vintage - pd.to_datetime(f["Start Date"])).dt.days / 365.25
        log.warning("%-16s n=%4d  horizon min %.1f yr  median %.1f yr",
                    nm, len(f), h.min(), h.median())

    set_top_sponsors(top_sponsors(train))

    keep = train["duration_days"].to_numpy(dtype=float) / 365.25 <= max_years
    matched = train[keep].reset_index(drop=True)
    log.warning("horizon match at %.1f yr keeps %d/%d training rows (%.1f%%)",
                max_years, len(matched), len(train), 100 * len(matched) / len(train))

    rows = []
    for label, tr in (("full", train), (f"matched<={max_years:g}y", matched)):
        model = TwoStageDuration(phase_key).fit(tr, "duration_days")
        for fold, te in (("UNtruncated 2018-2020", untrunc),
                         ("truncated 2021+", trunc)):
            m = _score(model, te)
            rows.append({"train": label, "test_fold": fold, "n_train": len(tr),
                         "n_test": len(te), "r2": m["r2"],
                         "mae_months": m["mae_months"],
                         "bias_months": m["bias_months"],
                         "coverage": m.get("interval_coverage")})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="P1")
    ap.add_argument("--max-years", type=float, default=5.0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    out = run(args.phase, args.max_years)
    print(f"\n{args.phase} — horizon matching, scored on both folds\n")
    print(out.to_string(index=False))

    piv = out.pivot(index="test_fold", columns="train", values="r2")
    delta = piv.iloc[:, 1] - piv.iloc[:, 0]
    print("\nR2 change from matching:")
    for fold, d in delta.items():
        print(f"  {fold:<24} {d:+.4f}")
    print("\nA gain on the truncated fold ONLY means the metric moved, not the model.")


if __name__ == "__main__":
    main()
