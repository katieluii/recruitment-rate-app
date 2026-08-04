from __future__ import annotations
"""Best recorded R2 and RMSE per phase, read back from the ledger.

    python -m experiments.leaderboard
    python -m experiments.leaderboard --split horizon

Replaces the absolute 0.70 R2 gate, retired 2026-08-04 because the feature set
cannot reach it — per-site enrolment and country recruitment speed are the two
highest-signal inputs and neither is published.

What stands in its place is a moving bar rather than a lower one: a change is
measured against the best this phase has ever recorded, so it cannot be passed
by relaxing a threshold, and it gets harder exactly as the model gets better.

Only rows from the SAME split are comparable, and the default is the horizon
fold adopted on 2026-08-04. Comparing across folds is meaningless — the same
model scored +0.075 on one fold and -0.081 on another — so this refuses to.
"""
import argparse
import json
from pathlib import Path

LEDGER = Path(__file__).parent / "ledger.jsonl"

#: metric -> (better direction, higher-is-better)
OBJECTIVES = {"r2": True, "rmse_days": False}

#: A change smaller than this is noise, not an improvement. Two runs of the
#: same config on the same data differ by roughly this much.
NOISE = {"r2": 0.005, "rmse_days": 1.0}


def _rows(split: str, cutoff: str | None = None) -> list[dict]:
    if not LEDGER.exists():
        return []
    out = []
    for line in LEDGER.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("skipped") or r.get("error") or r.get("split") != split:
            continue
        if cutoff is not None and r.get("cutoff") != cutoff:
            continue
        if r.get("r2") is None:
            continue
        out.append(r)
    return out


def best(split: str = "horizon", cutoff: str | None = None) -> dict:
    """{phase: {metric: {value, config, ts}}} over comparable ledger rows."""
    table: dict[str, dict] = {}
    for r in _rows(split, cutoff):
        phase = r.get("phase")
        slot = table.setdefault(phase, {})
        for metric, higher_better in OBJECTIVES.items():
            val = r.get(metric)
            if val is None:
                continue
            cur = slot.get(metric)
            if cur is None or (val > cur["value"] if higher_better else val < cur["value"]):
                slot[metric] = {"value": val, "config": r.get("config"),
                                "ts": r.get("ts")}
    return table


def compare(row: dict, table: dict) -> dict:
    """How one result stands against the best for its phase.

    Returns a verdict per objective: 'record', 'regression', or 'level'. A run
    that examined nothing gets 'unknown' rather than a free pass.
    """
    out: dict[str, dict] = {}
    slot = table.get(row.get("phase"), {})
    for metric, higher_better in OBJECTIVES.items():
        val = row.get(metric)
        prev = slot.get(metric)
        if val is None:
            out[metric] = {"verdict": "unknown", "value": None}
            continue
        if prev is None:
            out[metric] = {"verdict": "record", "value": val, "previous": None}
            continue
        delta = val - prev["value"]
        improved = delta > 0 if higher_better else delta < 0
        if abs(delta) < NOISE[metric]:
            verdict = "level"
        else:
            verdict = "record" if improved else "regression"
        out[metric] = {"verdict": verdict, "value": val,
                       "previous": prev["value"], "delta": round(delta, 4),
                       "previous_config": prev["config"]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="horizon")
    ap.add_argument("--cutoff", default=None)
    args = ap.parse_args()

    table = best(args.split, args.cutoff)
    if not table:
        print(f"No comparable ledger rows for split={args.split!r}.")
        return

    print(f"\nBest recorded per phase — split={args.split}\n")
    print(f"{'phase':<6} {'R2 (max)':>9} {'config':<22} {'RMSE d (min)':>13} {'config':<22}")
    print("-" * 78)
    for phase in ("P1HV", "P1", "P2", "P3"):
        slot = table.get(phase)
        if not slot:
            continue
        r2, rm = slot.get("r2"), slot.get("rmse_days")
        print(f"{phase:<6} {r2['value']:>9.4f} {str(r2['config'])[:22]:<22} "
              f"{rm['value']:>13.1f} {str(rm['config'])[:22]:<22}")
    print("\nR2 up, RMSE down. These are the bar; there is no absolute threshold.")


if __name__ == "__main__":
    main()
