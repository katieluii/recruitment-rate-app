from __future__ import annotations
"""Evaluation harness for the endpoint archetype classifier.

The classifier is a model too, so it gets measured like one rather than
assumed correct.

    python -m experiments.eval_endpoints --sample     # draw a labelling sheet
    python -m experiments.eval_endpoints --score      # score rules against it

`--sample` writes a stratified draw to experiments/endpoint_eval_set.csv with
the `true_archetype` column blank. Filling that column is a human step — the
scoring half deliberately refuses to run against an unlabelled sheet rather
than inventing agreement with itself.
"""
import argparse
import logging
from pathlib import Path

import pandas as pd

from backend.preprocessing.endpoints import ARCHETYPES, add_endpoint_features

log = logging.getLogger(__name__)

EVAL_PATH = Path(__file__).parent / "endpoint_eval_set.csv"


def draw_sample(n: int = 200, seed: int = 42) -> Path:
    """Stratified sample across phases and predicted archetypes.

    Oversamples UNKNOWN — the abstention bucket is where the classifier's real
    error lives, and a proportional draw would barely touch the rare archetypes.
    """
    from experiments.dataset import load_clean

    frames = []
    for phase in ("P1", "P2", "P3"):
        df = add_endpoint_features(load_clean(phase))
        df["phase"] = phase
        frames.append(df[["nct_id", "phase", "primary_outcome_measures",
                          "endpoint_archetype"]])
    allrows = pd.concat(frames, ignore_index=True)
    allrows = allrows[allrows["primary_outcome_measures"].fillna("").str.strip() != ""]

    per_group = max(3, n // (len(ARCHETYPES) * 3))
    sample = (
        allrows.groupby(["phase", "endpoint_archetype"], group_keys=False)
        .apply(lambda g: g.sample(min(len(g), per_group), random_state=seed))
        .reset_index(drop=True)
    )
    if len(sample) > n:
        sample = sample.sample(n, random_state=seed).reset_index(drop=True)

    sample["primary_outcome_measures"] = (
        sample["primary_outcome_measures"].str.split("|").str[0].str.slice(0, 300)
    )
    sample = sample.rename(columns={"endpoint_archetype": "rule_archetype"})
    sample["true_archetype"] = ""  # ← fill this in by hand
    sample.to_csv(EVAL_PATH, index=False)
    log.info("Wrote %d rows to %s — fill the true_archetype column",
             len(sample), EVAL_PATH)
    return EVAL_PATH


def score() -> dict:
    if not EVAL_PATH.exists():
        raise SystemExit(f"No eval set at {EVAL_PATH}. Run with --sample first.")
    df = pd.read_csv(EVAL_PATH).fillna("")
    labelled = df[df["true_archetype"].str.strip() != ""]
    if labelled.empty:
        raise SystemExit(
            f"{EVAL_PATH} has no filled `true_archetype` values. "
            "The classifier cannot be scored against an unlabelled sheet."
        )

    bad = set(labelled["true_archetype"].str.strip().str.upper()) - set(ARCHETYPES)
    if bad:
        raise SystemExit(f"Out-of-vocabulary labels in the eval set: {sorted(bad)}")

    truth = labelled["true_archetype"].str.strip().str.upper()
    pred = labelled["rule_archetype"].str.strip().str.upper()
    correct = (truth == pred)

    non_abstain = pred != "UNKNOWN"
    result = {
        "n_labelled": int(len(labelled)),
        "accuracy": round(float(correct.mean()), 3),
        "accuracy_when_not_abstaining": round(
            float(correct[non_abstain].mean()), 3) if non_abstain.any() else None,
        "abstention_rate": round(float((~non_abstain).mean()), 3),
    }
    per_class = (
        pd.DataFrame({"truth": truth, "correct": correct})
        .groupby("truth")["correct"].agg(["mean", "count"]).round(3)
    )
    print(pd.Series(result).to_string())
    print("\nPer-archetype recall:")
    print(per_class.to_string())
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true", help="draw a labelling sheet")
    ap.add_argument("--score", action="store_true", help="score rules vs labels")
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.sample:
        print(f"Labelling sheet: {draw_sample(args.n)}")
        print(f"Fill `true_archetype` with one of: {', '.join(ARCHETYPES)}")
    elif args.score:
        score()
    else:
        ap.error("pass --sample or --score")


if __name__ == "__main__":
    main()
