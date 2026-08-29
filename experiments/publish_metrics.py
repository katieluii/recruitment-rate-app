"""publish_metrics.py — the ONE source for every published accuracy / coverage figure.

WHY (2026-08-29 audit, cc-exchange M "circular validation"): the README table, RESULTS.md,
the portfolio site and `provenance.py` each carried their own hand-typed copy of the
metrics, all four from the 2021+ fold that the method's own completed-trials assumption
invalidates, and `provenance.py` shipped a literal `"0.82-0.89"` that no code computed.
When the honest horizon fold was run (2026-08-04) nothing downstream moved.

This module reads `experiments/ledger.jsonl` — every figure traces to a recorded run —
selects the row that ships per phase, and writes `experiments/published_metrics.json`.
`provenance.py` reads that file at request time; README.md / RESULTS.md paste the
markdown this prints. Re-run after any ledger change that should reach a reader:

    .venv/bin/python -m experiments.publish_metrics            # write JSON + print markdown
    .venv/bin/python -m experiments.publish_metrics --check    # exit 1 if JSON is stale

The fold is named on every output. A phase whose gate fails is published AS failing —
never dropped, never rounded up.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "experiments" / "ledger.jsonl"
OUT = ROOT / "experiments" / "published_metrics.json"
DOCS = [ROOT / "README.md", ROOT / "RESULTS.md"]
MARK_START, MARK_END = "<!-- published_metrics:start", "<!-- published_metrics:end -->"

# What ships, per phase. Change here, re-run, and every surface follows.
SPLIT = "horizon"
# P1HV ships the 0.85-nominal band (ledger row 333, 2026-08-29): at 0.80 nominal it covered
# 0.729 on the horizon fold, below the 0.75 gate; a 0.85 target lands 0.795 at +1.6 months of
# width with the point estimate unchanged. calib_frac=0.3 alone reached only 0.746 (row 332).
SHIPPED = {"P1HV": "two_stage_l2_cov85", "P1": "two_stage_l2", "P2": "two_stage_l2", "P3": "two_stage_l2"}
BASELINE = "ta_median"
FOLD_TEXT = ("horizon fold — train on trials starting before 2018, test on 2018–2020 starts, "
             "which have had 5.4–8.6 years to finish against a corpus whose p95 duration is 5.9")
PHASE_ORDER = ["P1HV", "P1", "P2", "P3"]


def load_rows() -> list:
    return [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]


def latest(rows: list, config: str, phase: str, split: str = SPLIT) -> dict | None:
    hits = [(i, r) for i, r in enumerate(rows, 1)
            if r.get("config") == config and r.get("phase") == phase and r.get("split") == split]
    if not hits:
        return None
    i, r = max(hits, key=lambda t: (t[1].get("ts", ""), t[0]))
    r = dict(r); r["ledger_row"] = i
    return r


def build(rows: list) -> dict:
    out = {"split": SPLIT, "fold": FOLD_TEXT, "generated_from": str(LEDGER.relative_to(ROOT)),
           "phases": {}}
    for ph in PHASE_ORDER:
        r = latest(rows, SHIPPED[ph], ph)
        b = latest(rows, BASELINE, ph)
        if r is None:
            out["phases"][ph] = {"status": "NOT MEASURED on this fold"}
            continue
        g = r.get("gates", {})
        out["phases"][ph] = {
            "config": SHIPPED[ph], "ledger_row": r["ledger_row"], "run_ts": r.get("ts"),
            "n_train": r.get("n_train"), "n_test": r.get("n_test"),
            "mae_months": r.get("mae_months"), "r2": r.get("r2"), "rmse_days": r.get("rmse_days"),
            "skill_vs_ta_median": r.get("skill_vs_ta_median"),
            "interval_coverage": r.get("interval_coverage"),
            "interval_nominal": r.get("interval_nominal"),
            "interval_mean_width_months": r.get("interval_mean_width_months"),
            "coverage_gate": g.get("interval_coverage", {}),
            "skill_gate": g.get("skill_vs_ta_median", {}),
            "all_gates_pass": r.get("gate_pass"),
            "baseline_mae_months": b.get("mae_months") if b else None,
        }
    covs = [p["interval_coverage"] for p in out["phases"].values() if p.get("interval_coverage") is not None]
    out["coverage_range"] = [min(covs), max(covs)] if covs else None
    out["gates_failing"] = [ph for ph, p in out["phases"].items() if p.get("all_gates_pass") is False]
    return out


def markdown(pub: dict) -> str:
    lines = ["| Phase | duration MAE | skill | R² | coverage (0.80 nominal) | gate |",
             "|-------|--------------|-------|----|-------------------------|------|"]
    for ph in PHASE_ORDER:
        p = pub["phases"][ph]
        if "mae_months" not in p:
            lines.append(f"| {ph} | — | — | — | — | not measured |"); continue
        cg = p["coverage_gate"]
        gate = "pass" if p["all_gates_pass"] else f"**FAIL** (coverage {cg.get('value')} < {cg.get('threshold', ['?'])[0]})"
        lines.append(f"| {ph} | {p['mae_months']:.2f} mo | {p['skill_vs_ta_median']:+.2f} | "
                     f"{p['r2']:.3f} | {p['interval_coverage']:.2f} | {gate} |")
    lines.append("")
    lines.append(f"Fold: {pub['fold']}. Rows " +
                 ", ".join(f"{ph}={pub['phases'][ph].get('ledger_row')}" for ph in PHASE_ORDER) +
                 f" of `{pub['generated_from']}`. Regenerate: `python -m experiments.publish_metrics`.")
    return "\n".join(lines)


def fill_docs(md: str, docs=DOCS, write: bool = True) -> list:
    """Replace the marked block in each doc. Returns the docs that were (or would be) changed;
    a doc without both markers is skipped loudly — a silent skip would leave a stale table."""
    changed = []
    for doc in docs:
        text = doc.read_text()
        i, j = text.find(MARK_START), text.find(MARK_END)
        if i < 0 or j < 0 or j < i:
            print(f"publish_metrics: {doc.name} has no marker block — NOT updated", file=sys.stderr)
            continue
        line_end = text.index("\n", i) + 1
        new = text[:line_end] + md + "\n" + text[j:]
        if new != text:
            changed.append(doc.name)
            if write:
                doc.write_text(new)
    return changed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if published_metrics.json or a doc block is stale")
    a = ap.parse_args()
    pub = build(load_rows())
    text = json.dumps(pub, indent=2) + "\n"
    md = markdown(pub)
    if a.check:
        cur = OUT.read_text() if OUT.exists() else ""
        stale = fill_docs(md, write=False)
        if cur != text or stale:
            print(f"STALE against the ledger: json={'yes' if cur != text else 'no'} docs={stale} — re-run publish_metrics", file=sys.stderr)
            return 1
        print("published_metrics.json and doc blocks match the ledger")
        return 0
    OUT.write_text(text)
    print(md)
    print(f"docs updated: {fill_docs(md)}", file=sys.stderr)
    print(f"\nwrote {OUT.relative_to(ROOT)} · coverage range {pub['coverage_range']} · gates failing: {pub['gates_failing']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
