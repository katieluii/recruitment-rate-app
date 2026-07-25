from __future__ import annotations
"""Compare v1 and v2 predictions against REAL completed trials.

    python -m experiments.real_trials --html experiments/reports/real-trials.html

Honesty constraint that shapes this whole module: the models in
`models/artifacts/` are fitted on every completed trial in the corpus, so
scoring them against any real trial would be scoring them on their own training
data. Instead both recipes are refitted here on trials that STARTED BEFORE the
cutoff, and compared on real trials that started after it. Every row is a real
NCT id with a real, known outcome that neither model was shown.
"""
import argparse
import html
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.candidates import LGBMQuantile, V1Recipe
from experiments.dataset import load_clean
from experiments.metrics import DAYS_PER_MONTH, ta_masks
from experiments.splits import temporal_split

log = logging.getLogger(__name__)

PHASE_LABELS = {"P1HV": "Phase 1 (healthy vol.)", "P1": "Phase 1",
                "P2": "Phase 2", "P3": "Phase 3"}


def _primary_ta(df: pd.DataFrame) -> pd.Series:
    """One display label per trial: the rarest area it maps to, so a trial
    tagged both Oncology and Other reads as Oncology."""
    masks = ta_masks(df)
    sizes = {a: int(m.sum()) for a, m in masks.items()}
    out = []
    for i in range(len(df)):
        areas = [a for a, m in masks.items() if m.iloc[i]]
        areas = [a for a in areas if a != "Other"] or areas
        out.append(min(areas, key=lambda a: sizes[a]) if areas else "Other")
    return pd.Series(out, index=df.index)


def build_comparison(cutoff: str = "2021-01-01") -> pd.DataFrame:
    frames = []
    for phase in ("P1HV", "P1", "P2", "P3"):
        df = load_clean(phase)
        train, test = temporal_split(df, cutoff=cutoff)
        if len(train) < 100 or len(test) < 20:
            log.warning("%s: split too small, skipping", phase)
            continue

        log.info("%s: fitting v1 and v2 on %d pre-%s trials", phase, len(train), cutoff)
        v1 = V1Recipe(phase).fit(train, "duration_days")
        v2 = LGBMQuantile(phase, calib_strategy="recent").fit(train, "duration_days")

        lo, hi = v2.predict_interval(test)
        actual = test["duration_days"].to_numpy(dtype=float)
        p1 = v1.predict(test)
        p2 = v2.predict(test)

        frames.append(pd.DataFrame({
            "nct_id": test.get("nct_id", pd.Series([""] * len(test))),
            "phase": phase,
            "therapeutic_area": _primary_ta(test).values,
            "conditions": test["conditions"].fillna("").str.split("|").str[0].str.slice(0, 60),
            "start": pd.to_datetime(test["Start Date"]).dt.strftime("%Y-%m"),
            "enrollment": pd.to_numeric(test["Enrollment"], errors="coerce"),
            "sites": pd.to_numeric(test["site_count"], errors="coerce"),
            "actual_months": actual / DAYS_PER_MONTH,
            "v1_months": p1 / DAYS_PER_MONTH,
            "v2_months": p2 / DAYS_PER_MONTH,
            "v2_lower": lo / DAYS_PER_MONTH,
            "v2_upper": hi / DAYS_PER_MONTH,
        }))

    out = pd.concat(frames, ignore_index=True)
    out["v1_error"] = (out["v1_months"] - out["actual_months"]).abs()
    out["v2_error"] = (out["v2_months"] - out["actual_months"]).abs()
    out["v2_closer"] = out["v2_error"] < out["v1_error"]
    out["in_interval"] = ((out["actual_months"] >= out["v2_lower"])
                          & (out["actual_months"] <= out["v2_upper"]))
    return out


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for phase, g in df.groupby("phase", sort=False):
        rows.append({
            "phase": phase,
            "n_trials": len(g),
            "actual_median_mo": round(g["actual_months"].median(), 1),
            "v1_MAE_mo": round(g["v1_error"].mean(), 2),
            "v2_MAE_mo": round(g["v2_error"].mean(), 2),
            "v2_closer_pct": round(100 * g["v2_closer"].mean(), 1),
            "v2_interval_hit_pct": round(100 * g["in_interval"].mean(), 1),
        })
    rows.append({
        "phase": "ALL",
        "n_trials": len(df),
        "actual_median_mo": round(df["actual_months"].median(), 1),
        "v1_MAE_mo": round(df["v1_error"].mean(), 2),
        "v2_MAE_mo": round(df["v2_error"].mean(), 2),
        "v2_closer_pct": round(100 * df["v2_closer"].mean(), 1),
        "v2_interval_hit_pct": round(100 * df["in_interval"].mean(), 1),
    })
    return pd.DataFrame(rows)


def by_area(df: pd.DataFrame, min_n: int = 15) -> pd.DataFrame:
    g = df.groupby("therapeutic_area").agg(
        n=("actual_months", "size"),
        actual_median=("actual_months", "median"),
        v1_median=("v1_months", "median"),
        v2_median=("v2_months", "median"),
        v1_MAE=("v1_error", "mean"),
        v2_MAE=("v2_error", "mean"),
    )
    g = g[g["n"] >= min_n].round(1).sort_values("actual_median", ascending=False)
    return g.reset_index()


# ── HTML report ───────────────────────────────────────────────────────────────

_CSS = """
:root{--bg:#fbfaf7;--fg:#22201d;--muted:#6d675f;--line:#e2ddd4;--accent:#7a5c3e;
--good:#2f6b4f;--bad:#a33a2c;--card:#fff}
@media(prefers-color-scheme:dark){:root{--bg:#171614;--fg:#eae6df;--muted:#9c958a;
--line:#332f2a;--accent:#c9a77c;--good:#6fbf95;--bad:#e08472;--card:#1f1d1a}}
*{box-sizing:border-box}body{margin:0;padding:2.5rem 1.5rem;background:var(--bg);
color:var(--fg);font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
.wrap{max-width:1180px;margin:0 auto}
h1{font-family:Georgia,serif;font-size:1.9rem;margin:0 0 .3rem}
h2{font-family:Georgia,serif;font-size:1.25rem;margin:2.5rem 0 .5rem;
padding-bottom:.4rem;border-bottom:1px solid var(--line)}
.sub{color:var(--muted);margin:0 0 1.5rem}
.note{background:var(--card);border-left:3px solid var(--accent);padding:.9rem 1.1rem;
border-radius:5px;margin:1.2rem 0;font-size:.9rem}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
table{border-collapse:collapse;width:100%;font-size:.86rem;background:var(--card);
border-radius:6px;overflow:hidden}
th,td{padding:.5rem .7rem;text-align:right;border-bottom:1px solid var(--line);
white-space:nowrap}
th{background:rgba(122,92,62,.08);font-weight:600;text-align:right;position:sticky;top:0}
td:first-child,th:first-child,td.l,th.l{text-align:left}
tr:last-child td{border-bottom:none}
tfoot td{font-weight:700;background:rgba(122,92,62,.06)}
.good{color:var(--good);font-weight:600}.bad{color:var(--bad);font-weight:600}
.kpis{display:flex;flex-wrap:wrap;gap:.9rem;margin:1.4rem 0}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:7px;
padding:.8rem 1.1rem;min-width:150px}
.kpi .v{font-size:1.5rem;font-weight:700;font-family:Georgia,serif}
.kpi .l{font-size:.74rem;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
code{background:rgba(122,92,62,.1);padding:.1rem .3rem;border-radius:3px;font-size:.85em}
"""


def _fmt_row(r) -> str:
    cls = "good" if r["v2_closer"] else "bad"
    hit = "✓" if r["in_interval"] else "✗"
    return (
        f"<tr><td class='l'><code>{html.escape(str(r['nct_id']))}</code></td>"
        f"<td class='l'>{html.escape(str(r['conditions']))}</td>"
        f"<td class='l'>{html.escape(str(r['therapeutic_area']))}</td>"
        f"<td>{PHASE_LABELS.get(r['phase'], r['phase'])}</td>"
        f"<td>{r['start']}</td>"
        f"<td>{r['enrollment']:.0f}</td><td>{r['sites']:.0f}</td>"
        f"<td><b>{r['actual_months']:.1f}</b></td>"
        f"<td>{r['v1_months']:.1f}</td>"
        f"<td class='{cls}'>{r['v2_months']:.1f}</td>"
        f"<td>{r['v2_lower']:.0f}–{r['v2_upper']:.0f} {hit}</td>"
        f"<td>{r['v1_error']:.1f}</td><td class='{cls}'>{r['v2_error']:.1f}</td></tr>"
    )


def write_html(df: pd.DataFrame, summary: pd.DataFrame, areas: pd.DataFrame,
               path: Path, cutoff: str, n_examples: int = 60) -> Path:
    all_row = summary[summary["phase"] == "ALL"].iloc[0]

    kpis = "".join(
        f"<div class='kpi'><div class='v'>{v}</div><div class='l'>{l}</div></div>"
        for v, l in [
            (f"{int(all_row['n_trials']):,}", "real held-out trials"),
            (f"{all_row['v1_MAE_mo']:.1f} mo", "v1 mean error"),
            (f"{all_row['v2_MAE_mo']:.1f} mo", "v2 mean error"),
            (f"{all_row['v2_closer_pct']:.0f}%", "trials v2 called closer"),
            (f"{all_row['v2_interval_hit_pct']:.0f}%", "inside 80% interval"),
        ])

    sum_rows = "".join(
        "<tr>" + f"<td class='l'>{PHASE_LABELS.get(r['phase'], r['phase'])}</td>"
        + f"<td>{int(r['n_trials']):,}</td><td>{r['actual_median_mo']}</td>"
        + f"<td>{r['v1_MAE_mo']}</td>"
        + f"<td class='{'good' if r['v2_MAE_mo'] < r['v1_MAE_mo'] else 'bad'}'>{r['v2_MAE_mo']}</td>"
        + f"<td>{r['v2_closer_pct']}%</td><td>{r['v2_interval_hit_pct']}%</td></tr>"
        for _, r in summary.iterrows())

    area_rows = "".join(
        "<tr>" + f"<td class='l'>{html.escape(r['therapeutic_area'])}</td>"
        + f"<td>{int(r['n'])}</td><td><b>{r['actual_median']}</b></td>"
        + f"<td>{r['v1_median']}</td><td>{r['v2_median']}</td>"
        + f"<td>{r['v1_MAE']}</td>"
        + f"<td class='{'good' if r['v2_MAE'] < r['v1_MAE'] else 'bad'}'>{r['v2_MAE']}</td></tr>"
        for _, r in areas.iterrows())

    sample = df.sample(min(n_examples, len(df)), random_state=7).sort_values(
        "actual_months", ascending=False)
    ex_rows = "".join(_fmt_row(r) for _, r in sample.iterrows())

    body = f"""<div class="wrap">
<h1>v2 vs v1 on real trials</h1>
<p class="sub">Every row is a real, completed ClinicalTrials.gov study with a known
outcome. Both models were refitted on trials starting before {cutoff} and are
compared here on trials that started after it — neither has seen any of them.</p>

<div class="note"><b>Why refit rather than use the shipped models?</b> The models in
<code>models/artifacts/</code> are trained on every completed trial in the corpus,
so testing them against any real trial would be testing them on their own training
data. That would flatter both versions and tell you nothing.</div>

<div class="kpis">{kpis}</div>

<h2>By phase</h2>
<div class="scroll"><table><thead><tr>
<th class="l">Phase</th><th>Trials</th><th>Actual median</th>
<th>v1 error</th><th>v2 error</th><th>v2 closer</th><th>In 80% interval</th>
</tr></thead><tbody>{sum_rows}</tbody></table></div>

<h2>By therapeutic area — does it track reality?</h2>
<p class="sub">The original complaint: too many areas shared the same predicted length.
Compare the v1 and v2 median columns against the actual median.</p>
<div class="scroll"><table><thead><tr>
<th class="l">Therapeutic area</th><th>n</th><th>Actual median</th>
<th>v1 median</th><th>v2 median</th><th>v1 error</th><th>v2 error</th>
</tr></thead><tbody>{area_rows}</tbody></table></div>

<h2>Individual trials <span style="font-weight:400;font-size:.8rem;color:var(--muted)">
({len(sample)} sampled at random, longest first)</span></h2>
<div class="scroll"><table><thead><tr>
<th class="l">NCT</th><th class="l">Condition</th><th class="l">Area</th><th>Phase</th>
<th>Start</th><th>Enrol</th><th>Sites</th><th>Actual</th><th>v1</th><th>v2</th>
<th>v2 80% interval</th><th>v1 err</th><th>v2 err</th>
</tr></thead><tbody>{ex_rows}</tbody></table></div>

<p class="sub" style="margin-top:2rem">All durations in months, start to primary
completion. Green means v2 was closer than v1 on that trial; red means v1 was.</p>
</div>"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"<!doctype html><html><head><meta charset='utf-8'>"
        f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>v2 vs v1 on real trials</title><style>{_CSS}</style></head>"
        f"<body>{body}</body></html>", encoding="utf-8")
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", default="2021-01-01")
    ap.add_argument("--html", default="experiments/reports/real-trials.html")
    ap.add_argument("--csv", default="experiments/reports/real-trials.csv")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    df = build_comparison(args.cutoff)
    summary = summarise(df)
    areas = by_area(df)

    print("\n=== BY PHASE (real held-out trials) ===")
    print(summary.to_string(index=False))
    print("\n=== BY THERAPEUTIC AREA ===")
    print(areas.to_string(index=False))

    df.to_csv(args.csv, index=False)
    path = write_html(df, summary, areas, Path(args.html), args.cutoff)
    print(f"\nHTML : {path}\nCSV  : {args.csv}")


if __name__ == "__main__":
    main()
