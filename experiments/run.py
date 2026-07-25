from __future__ import annotations
"""Experiment runner.

    python -m experiments.run --config v1_recipe
    python -m experiments.run --config all --split temporal --cutoff 2021-01-01
    python -m experiments.run --config v1_shipped --split random   # v1's own protocol

Every run appends one row per (config, phase) to experiments/ledger.jsonl and
writes a readable report to experiments/reports/.
"""
import argparse
import logging
from datetime import date
from pathlib import Path

import pandas as pd

from backend.constants import PHASES
from experiments import ledger
from experiments.baselines import ALL_BASELINES, PRIMARY_BASELINE
from experiments.candidates import ShippedArtifact, V1Recipe
from experiments.dataset import load_clean
from experiments.metrics import evaluate, skill_score
from experiments.splits import check_split_viability, get_split

log = logging.getLogger(__name__)

REPORT_DIR = Path(__file__).parent / "reports"

#: name → (factory taking phase_key, is_baseline)
CONFIGS = {
    "median":               (lambda p: ALL_BASELINES[0](), True),
    "ta_median":            (lambda p: ALL_BASELINES[1](), True),
    "ta_enrollment_median": (lambda p: ALL_BASELINES[2](), True),
    "v1_recipe":            (lambda p: V1Recipe(p), False),
    # Isolates the therapeutic-area target encoder's contribution.
    "rf_ta_encoded":        (lambda p: V1Recipe(p, ta_target_encoding=True), False),
    "v1_shipped":           (lambda p: ShippedArtifact(
        p, artifacts_dir="models/artifacts_v1_baseline"), False),
}

BASELINE_CONFIGS = [n for n, (_, is_b) in CONFIGS.items() if is_b]


def run_one(config: str, phase_key: str, split: str, cutoff: str,
            target: str = "duration_days") -> dict | None:
    df = load_clean(phase_key)
    train, test = get_split(df, split, **({"cutoff": cutoff} if split == "temporal" else {}))

    warning = check_split_viability(train, test)
    if warning:
        log.warning("%s/%s skipped: %s", config, phase_key, warning)
        return {"config": config, "phase": phase_key, "split": split,
                "cutoff": cutoff if split == "temporal" else None,
                "target": target, "skipped": warning}

    factory, _ = CONFIGS[config]
    model = factory(phase_key).fit(train, target)

    y_true = test[target].to_numpy(dtype=float)
    y_pred = model.predict(test)
    try:
        lower, upper = model.predict_interval(test)
    except (NotImplementedError, AttributeError):
        lower = upper = None

    metrics = evaluate(test, y_true, y_pred, lower, upper)
    per_ta = metrics.pop("_per_ta")

    return {
        "config": config,
        "phase": phase_key,
        "split": split,
        "cutoff": cutoff if split == "temporal" else None,
        "target": target,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        **metrics,
        "per_ta": per_ta,
    }


def add_skill_scores(rows: list[dict]) -> list[dict]:
    """Attach each row's MAE skill against the primary baseline for its phase."""
    bar = {
        r["phase"]: r.get("mae_months")
        for r in rows
        if r.get("config") == PRIMARY_BASELINE.name and not r.get("skipped")
    }
    for r in rows:
        if r.get("skipped") or r.get("mae_months") is None:
            continue
        base = bar.get(r["phase"])
        r["baseline_mae_months"] = base
        r["skill_vs_ta_median"] = (
            skill_score(r["mae_months"], base) if base else None
        )
        r["beats_baseline"] = (
            None if base is None else bool(r["mae_months"] < base)
        )
    return rows


def write_report(rows: list[dict], name: str, split: str, cutoff: str) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / f"{date.today().isoformat()}-{name}.md"

    lines = [
        f"# Experiment report — {name}",
        "",
        f"Split: **{split}**" + (f" (cutoff {cutoff})" if split == "temporal" else ""),
        f"Target: duration_days · generated {date.today().isoformat()}",
        "",
        "## Summary",
        "",
        "| config | phase | n_train | n_test | MAE (mo) | RMSE (d) | skill vs TA-median | beats bar | TA spread ratio | TA rank corr | distinct TAs | 80% coverage |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if r.get("skipped"):
            lines.append(
                f"| {r['config']} | {r['phase']} | — | — | _skipped: {r['skipped']}_ "
                "| | | | | | | |"
            )
            continue
        beats = r.get("beats_baseline")
        beats_s = "—" if beats is None else ("yes" if beats else "**NO**")
        lines.append(
            f"| {r['config']} | {r['phase']} | {r['n_train']} | {r['n_test']} "
            f"| {r['mae_months']} | {r['rmse_days']} "
            f"| {r.get('skill_vs_ta_median', '—')} | {beats_s} "
            f"| {r.get('ta_spread_ratio', '—')} | {r.get('ta_rank_corr', '—')} "
            f"| {r.get('ta_n_distinct', '—')}/{r.get('ta_n_areas', '—')} "
            f"| {r.get('interval_coverage', '—')} |"
        )

    lines += ["", "## Per-therapeutic-area detail", ""]
    for r in rows:
        if r.get("skipped") or not r.get("per_ta"):
            continue
        lines += [f"### {r['config']} — {r['phase']}", "",
                  "| therapeutic area | n | true median (mo) | predicted median (mo) | MAE (mo) |",
                  "|---|---|---|---|---|"]
        for t in r["per_ta"]:
            lines.append(
                f"| {t['therapeutic_area']} | {t['n']} | {t['true_median_months']} "
                f"| {t['pred_median_months']} | {t['mae_months']} |"
            )
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="all",
                    help=f"one of {list(CONFIGS)} or 'all' or 'baselines'")
    ap.add_argument("--split", default="temporal", choices=["temporal", "random"])
    ap.add_argument("--cutoff", default="2021-01-01")
    ap.add_argument("--phases", default=",".join(PHASES))
    ap.add_argument("--name", default=None, help="report filename stem")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")

    if args.config == "all":
        configs = list(CONFIGS)
    elif args.config == "baselines":
        configs = BASELINE_CONFIGS
    else:
        configs = [c.strip() for c in args.config.split(",")]
    unknown = [c for c in configs if c not in CONFIGS]
    if unknown:
        raise SystemExit(f"Unknown config(s): {unknown}. Available: {list(CONFIGS)}")

    # The primary baseline must run so skill scores can be computed.
    if PRIMARY_BASELINE.name not in configs:
        configs = [PRIMARY_BASELINE.name] + configs

    phases = [p.strip() for p in args.phases.split(",")]

    rows: list[dict] = []
    for config in configs:
        for phase_key in phases:
            try:
                row = run_one(config, phase_key, args.split, args.cutoff)
            except Exception as exc:
                log.error("%s/%s failed: %s", config, phase_key, exc, exc_info=True)
                row = {"config": config, "phase": phase_key, "split": args.split,
                       "error": str(exc)}
            if row:
                rows.append(row)

    rows = add_skill_scores(rows)
    for row in rows:
        ledger.append({k: v for k, v in row.items() if k != "per_ta"})

    name = args.name or f"{args.config}-{args.split}"
    path = write_report(rows, name, args.split, args.cutoff)

    print("\n" + "=" * 100)
    summary = pd.DataFrame([
        {k: r.get(k) for k in
         ("config", "phase", "n_test", "mae_months", "skill_vs_ta_median",
          "beats_baseline", "ta_spread_ratio", "ta_rank_corr",
          "ta_n_distinct", "interval_coverage")}
        for r in rows if not r.get("skipped") and not r.get("error")
    ])
    if not summary.empty:
        print(summary.to_string(index=False))
    print("=" * 100)
    print(f"Report: {path}")
    print(f"Ledger: {ledger.LEDGER_PATH}")


if __name__ == "__main__":
    main()
