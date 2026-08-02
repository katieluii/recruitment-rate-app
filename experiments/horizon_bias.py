from __future__ import annotations
"""Is the temporal test fold truncated by observation horizon, and does it show?

    python -m experiments.horizon_bias --phase P1

docs/OPEN_LEVERS.md §3. The corpus holds completed trials only, so a trial that
started in 2025 can only appear if it finished within about a year. The test fold
is therefore not a sample of trials — it is a sample of trials SHORT ENOUGH to
have finished by the data vintage, and the cut gets tighter the later the start.

This reports the model's bias per test-fold start year against the horizon each
year actually had. If the bias grows as the horizon shrinks, the model is being
scored against a truncation artifact rather than against trial duration.
"""
import argparse
import logging

import numpy as np
import pandas as pd

from backend.models.quantile_model import TwoStageDuration
from backend.preprocessing.pipeline import set_top_sponsors
from backend.preprocessing.text_features import top_sponsors
from experiments.dataset import load_clean
from experiments.splits import get_split

log = logging.getLogger(__name__)


def run(phase_key: str, cutoff: str = "2021-01-01") -> pd.DataFrame:
    df = load_clean(phase_key)
    df = df[df["duration_days"].notna()].reset_index(drop=True)
    train, test = get_split(df, "temporal", cutoff=cutoff)
    set_top_sponsors(top_sponsors(train))

    model = TwoStageDuration(phase_key).fit(train, "duration_days")
    pred = model.predict(test)

    start = pd.to_datetime(test["Start Date"])
    vintage = pd.to_datetime(df["Start Date"]).max()
    out = pd.DataFrame({
        "yr": start.dt.year,
        "horizon_yr": (vintage - start).dt.days / 365.25,
        "true_mo": test["duration_days"].to_numpy(dtype=float) / 30.44,
        "pred_mo": pred / 30.44,
    })
    out["err_mo"] = out["pred_mo"] - out["true_mo"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="P1")
    ap.add_argument("--cutoff", default="2021-01-01")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    out = run(args.phase, args.cutoff)

    g = out.groupby("yr").agg(
        n=("err_mo", "size"),
        horizon_yr=("horizon_yr", "median"),
        true_median_mo=("true_mo", "median"),
        pred_median_mo=("pred_mo", "median"),
        bias_mo=("err_mo", "mean"),
        mae_mo=("err_mo", lambda s: s.abs().mean()),
    ).round(2)

    print(f"\n{args.phase} — error by test-fold start year "
          f"(temporal cutoff {args.cutoff})\n")
    print(g.to_string())

    # The claim to falsify: bias is unrelated to how much horizon the year had.
    r = np.corrcoef(out["horizon_yr"], out["err_mo"])[0, 1]
    print(f"\ncorr(horizon, signed error) = {r:+.3f} over {len(out)} rows")
    print("Negative = the model over-predicts most where the horizon is "
          "shortest, i.e. it is scored against trials the horizon truncated.")


if __name__ == "__main__":
    main()
