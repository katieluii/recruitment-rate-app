from __future__ import annotations
"""How much of the enrolment target is the floor constant rather than data?

    python -m experiments.clip_audit --phases P1

Reports, per phase, the share of rows whose recruiting window is SET BY
`MIN_ENROL_FRACTION` (docs/OPEN_LEVERS.md §1), and how that share moves with the
floor. Also splits the share by therapeutic area, because the claim that matters
is not the headline 1-in-6 — it is that the clip fires hardest on the
long-follow-up trials the model already handles worst.
"""
import argparse

import numpy as np
import pandas as pd

from backend.constants import PHASES
from backend.preprocessing.cleaner import (MIN_ENROL_FRACTION, _followup_months,
                                           clipped_by_floor)
from experiments.dataset import load_clean

SWEEP = (0.0, 0.1, 0.25, 0.4)


def audit(phase_key: str) -> dict:
    df = load_clean(phase_key)
    df = df[df["duration_days"].notna()].reset_index(drop=True)

    total = df["duration_days"] / 30.44
    fu = _followup_months(df)
    raw = total - fu

    out = {"phase": phase_key, "n": len(df),
           "share_on_floor": float(clipped_by_floor(df).mean())}
    for frac in SWEEP:
        out[f"share_at_{frac}"] = float(clipped_by_floor(df, frac).mean())

    clipped = clipped_by_floor(df).to_numpy(dtype=bool)
    out["median_total_months_clipped"] = round(float(total[clipped].median()), 1)
    out["median_total_months_clean"] = round(float(total[~clipped].median()), 1)
    out["median_fu_months_clipped"] = round(float(fu[clipped].median()), 1)
    out["median_fu_months_clean"] = round(float(fu[~clipped].median()), 1)
    # How far below the floor the honest value sits, for the rows it rewrites.
    out["median_raw_window_clipped"] = round(float(raw[clipped].median()), 1)

    if "therapeutic_area" in df.columns:
        by_ta = (pd.DataFrame({"ta": df["therapeutic_area"], "c": clipped})
                 .groupby("ta")["c"].agg(["mean", "size"])
                 .sort_values("mean", ascending=False))
        out["_by_ta"] = by_ta[by_ta["size"] >= 30]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phases", default=",".join(PHASES))
    args = ap.parse_args()

    rows = [audit(p.strip()) for p in args.phases.split(",")]

    print(f"\nFloor constant MIN_ENROL_FRACTION = {MIN_ENROL_FRACTION}\n")
    cols = ["phase", "n", "share_on_floor"] + [f"share_at_{f}" for f in SWEEP]
    print(pd.DataFrame([{k: r[k] for k in cols} for r in rows]).to_string(index=False))

    print("\nWhat the clipped rows look like (months):")
    cols2 = ["phase", "median_total_months_clipped", "median_total_months_clean",
             "median_fu_months_clipped", "median_fu_months_clean",
             "median_raw_window_clipped"]
    print(pd.DataFrame([{k: r[k] for k in cols2} for r in rows]).to_string(index=False))

    for r in rows:
        if "_by_ta" not in r:
            continue
        print(f"\n{r['phase']} — clipped share by therapeutic area (n >= 30):")
        t = r["_by_ta"].copy()
        t["mean"] = (100 * t["mean"]).round(1)
        print(t.rename(columns={"mean": "clipped_%", "size": "n"}).to_string())


if __name__ == "__main__":
    main()
