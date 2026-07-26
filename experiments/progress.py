from __future__ import annotations
"""Progress log — what each change bought, straight from the ledger.

    python -m experiments.progress            # console
    python -m experiments.progress --md       # RESULTS.md for the project card

Reads `ledger.jsonl` rather than any hand-kept notes, so a claim on the project
card can always be traced back to the run that produced it. Where a config was
run more than once the LATEST row wins, because later runs reflect later fixes.
"""
import argparse
import json
from datetime import date
from pathlib import Path

LEDGER = Path(__file__).parent / "ledger.jsonl"
OUT = Path(__file__).parent.parent / "RESULTS.md"

PHASES = ["P1HV", "P1", "P2", "P3"]

#: The story in order. Each entry is one step of the rebuild, the ledger config
#: that measures it, and what the change actually was.
MILESTONES = [
    ("ta_median", "Baseline — per-therapeutic-area median lookup",
     "The bar every model must clear. A learned model that loses to this is a "
     "lookup table with worse latency."),
    ("v1_recipe@first", "v1 as it actually shipped",
     "The RandomForest recipe with its original feature set, refit on a temporal "
     "fold. v1 was never compared to a baseline, so nobody knew it lost to one."),
    ("v1_recipe", "+ data-layer fixes (real site count, no leaked year)",
     "Same RandomForest, repaired inputs. `primary_completion_year` leaked the "
     "label's endpoint and `site_count` counted countries."),
    ("rf_ta_encoded", "+ therapeutic-area target encoding",
     "Replaces 22 sparse binaries with one smoothed continuous signal the trees "
     "will actually split on."),
    ("lgbm_point", "+ LightGBM on a log target",
     "Gradient boosting and a log target for the right-skewed duration."),
    ("lgbm_conformal_recent", "+ conformal intervals (v2 shipped)",
     "Real quantile intervals widened on the most recent training slice, "
     "replacing an interval pinned at rmse*0.5 for every input."),
    ("two_stage", "+ enrolment / follow-up split (v3.3)",
     "Duration modelled as two near-independent processes rather than one "
     "blended number."),
    ("two_stage_geo", "+ country site-mix effect (v3.2) — NOT SHIPPED",
     "Adds the geography lever the tool was missing, but costs accuracy on 3 of "
     "4 phases. Recorded, not merged; see the note below."),
]

RATE_MILESTONES = [
    ("ta_median", "Baseline — per-area median rate"),
    ("lgbm_conformal_recent", "LightGBM, log1p target"),
    ("lgbm_rate", "LightGBM, plain log target"),
]


def load(target: str = "duration_days", pick: str = "latest") -> dict:
    """One row per (config, phase). `pick="first"` keeps the EARLIEST run.

    Latest-wins is right for most configs, since later runs reflect later fixes.
    It is wrong for `v1_recipe`: that config was re-run after the data layer was
    repaired, so its latest row no longer shows what v1 actually did. The run
    that matters there is the first one, before anything was fixed.
    """
    latest: dict[tuple[str, str], dict] = {}
    for line in LEDGER.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("target") != target or r.get("error") or r.get("skipped"):
            continue
        if r.get("split") != "temporal":
            continue
        key = (r.get("config"), r.get("phase"))
        if pick == "first" and key in latest:
            continue
        latest[key] = r
    return latest


def _mae(r: dict):
    return r.get("mae_months", r.get("mae_raw"))


def table(rows: dict, milestones, metric: str = "mae"):
    out = []
    for cfg, label, *rest in milestones:
        cells = {}
        for ph in PHASES:
            r = rows.get((cfg, ph))
            if not r:
                continue
            cells[ph] = {
                "mae": _mae(r),
                "skill": r.get("skill_vs_ta_median"),
                "cov": r.get("interval_coverage"),
                "distinct": r.get("ta_n_distinct"),
                "rank": r.get("ta_rank_corr"),
                "r2": r.get("r2"),
                "rmse": r.get("rmse_days"),
            }
        if cells:
            out.append((cfg, label, rest[0] if rest else "", cells))
    return out


def _merged(target: str) -> dict:
    """Latest rows, plus the first-run rows exposed under a `@first` alias."""
    rows = load(target, "latest")
    for (cfg, ph), r in load(target, "first").items():
        rows[(f"{cfg}@first", ph)] = r
    return rows


def render_md() -> str:
    dur = table(_merged("duration_days"), MILESTONES)
    rate = table(_merged("recruitment_rate"), RATE_MILESTONES)

    L = [
        "# Results log",
        "",
        f"Generated {date.today().isoformat()} from `experiments/ledger.jsonl` "
        f"— every figure traces to a recorded run.",
        "",
        "Protocol: train on trials starting before 2021-01-01, test on those "
        "starting after. `skill` is the fraction of the per-therapeutic-area "
        "median baseline's error removed; **negative means worse than a lookup "
        "table**.",
        "",
        "## Duration — mean absolute error, months",
        "",
        "| step | P1HV | P1 | P2 | P3 |",
        "|---|---|---|---|---|",
    ]
    for _cfg, label, _note, cells in dur:
        row = [f"**{label}**"]
        for ph in PHASES:
            c = cells.get(ph)
            row.append(f"{c['mae']:.2f}" if c and c["mae"] is not None else "—")
        L.append("| " + " | ".join(row) + " |")

    L += ["", "## Duration — skill against the baseline", "",
          "| step | P1HV | P1 | P2 | P3 |", "|---|---|---|---|---|"]
    for _cfg, label, _note, cells in dur:
        row = [f"**{label}**"]
        for ph in PHASES:
            c = cells.get(ph)
            s = c["skill"] if c else None
            row.append("—" if s is None else (f"**{s:+.3f}**" if s < 0 else f"{s:+.3f}"))
        L.append("| " + " | ".join(row) + " |")

    L += ["", "## Interval calibration — share of actuals inside the 80% band", "",
          "| step | P1HV | P1 | P2 | P3 |", "|---|---|---|---|---|"]
    for _cfg, label, _note, cells in dur:
        if not any((cells.get(p) or {}).get("cov") for p in PHASES):
            continue
        row = [f"**{label}**"]
        for ph in PHASES:
            c = cells.get(ph)
            row.append(f"{c['cov']:.3f}" if c and c.get("cov") is not None else "—")
        L.append("| " + " | ".join(row) + " |")

    L += ["", "## Therapeutic-area differentiation",
          "", "Distinct predicted medians out of the areas with enough test "
          "trials — the metric that caught the original failure, where 17 of 22 "
          "Phase 1 areas returned the identical 10.9 months.", "",
          "| step | P1HV | P1 | P2 | P3 |", "|---|---|---|---|---|"]
    for _cfg, label, _note, cells in dur:
        row = [f"**{label}**"]
        for ph in PHASES:
            c = cells.get(ph)
            row.append(str(c["distinct"]) if c and c.get("distinct") else "—")
        L.append("| " + " | ".join(row) + " |")

    if rate:
        L += ["", "## Recruitment rate — MAE, patients per site per month", "",
              "| step | P1HV | P1 | P2 | P3 |", "|---|---|---|---|---|"]
        for _cfg, label, _note, cells in rate:
            row = [f"**{label}**"]
            for ph in PHASES:
                c = cells.get(ph)
                row.append(f"{c['mae']:.3f}" if c and c["mae"] is not None else "—")
            L.append("| " + " | ".join(row) + " |")

    L += ["", "## R-squared and RMSE",
          "",
          "Reported for continuity with the original project. Neither is the gate.",
          "",
          "R-squared scores against predicting the MEAN, which is a weak reference for",
          "a right-skewed target: the per-therapeutic-area median lookup posts a NEGATIVE",
          "R-squared (-0.12 on P2, -0.14 on P3) while being the harder bar on MAE. A model",
          "can therefore look respectable on R-squared while losing to a lookup table,",
          "which is exactly what v1 did. `skill_vs_ta_median` is the same fraction-of-error-",
          "removed idea measured against that harder reference, and it is what decides",
          "whether a change ships.",
          "",
          "RMSE squares the error, so a handful of eight-year trials dominate it. MAE is",
          "the headline because the median quantile model minimises absolute error by",
          "construction, and a metric that disagrees with the loss will reward the wrong",
          "model.",
          "",
          "| step | P2 R2 | P3 R2 | P2 RMSE (d) | P3 RMSE (d) |",
          "|---|---|---|---|---|"]
    for _cfg, label, _note, cells in dur:
        p2, p3 = cells.get("P2") or {}, cells.get("P3") or {}
        if p2.get("r2") is None and p3.get("r2") is None:
            continue
        def _f(v, nd=3):
            return "—" if v is None else f"{v:.{nd}f}"
        L.append(f"| **{label}** | {_f(p2.get('r2'))} | {_f(p3.get('r2'))} "
                 f"| {_f(p2.get('rmse'), 0)} | {_f(p3.get('rmse'), 0)} |")

    L += ["", "## What each step was", ""]
    for _cfg, label, note, _cells in dur:
        if note:
            L.append(f"- **{label}** — {note}")

    L += ["",
          "## Findings that changed the work",
          "",
          "- **`primary_completion_year` leaked the label's own endpoint.** "
          "Removing it took Phase 2 MAE from 25.41 to 8.88 months.",
          "- **`site_count` counted countries, not sites.** Training values sat "
          "in 1–20 while inference passed real site counts of 40+, outside the "
          "trained range where a forest returns a constant.",
          "- **Completed-trials-only data is survivorship-biased.** At a 2018 "
          "vantage, Phase 3 duration looked 20.9 months when it was truly 24.6. "
          "Corrected by inverse-probability-of-censoring weighting.",
          "- **Survival models lost.** Weibull AFT, random survival forest and "
          "gradient-boosted survival all cut the bias but lost more on scatter. "
          "Recorded rather than quietly dropped.",
          "- **Duration is two processes.** Enrolment window and follow-up are "
          "near-uncorrelated (r = +0.03). A Phase 3 survival endpoint follows up "
          "for 26.0 months against 5.5 for a biomarker endpoint.",
          "",
          "## Open",
          "",
          "- The enrolment head and `N / (sites × rate)` disagree for some areas "
          "(P3 infectious disease 21.1 vs 13.0 months). Medians do not compose "
          "and the heads are fitted independently. V3.2 dissolves this by "
          "deriving the window from per-site rates.",
          ]
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", action="store_true", help="write RESULTS.md")
    args = ap.parse_args()

    md = render_md()
    if args.md:
        OUT.write_text(md, encoding="utf-8")
        print(f"Wrote {OUT}")
    else:
        print(md)


if __name__ == "__main__":
    main()
