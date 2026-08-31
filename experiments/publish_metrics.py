"""publish_metrics.py: the ONE source for every published accuracy / coverage figure.

WHY (2026-08-29 audit, cc-exchange M "circular validation"): the README table, RESULTS.md,
the portfolio site and `provenance.py` each carried their own hand-typed copy of the
metrics, all four from the 2021+ fold that the method's own completed-trials assumption
invalidates, and `provenance.py` shipped a literal `"0.82-0.89"` that no code computed.
When the honest horizon fold was run (2026-08-04) nothing downstream moved.

This module reads `experiments/ledger.jsonl` — every figure traces to a recorded run —
selects the row that ships per phase, and writes `experiments/published_metrics.json`.
`provenance.py` reads that file at request time; README.md / RESULTS.md carry the markdown
this prints inside marker blocks. Re-run after any ledger change that should reach a reader:

    .venv/bin/python -m experiments.publish_metrics            # write JSON + fill docs
    .venv/bin/python -m experiments.publish_metrics --check    # exit 1 if JSON/docs are stale

The fold is named on every output. A phase whose gate fails is published AS failing —
never dropped, never rounded up.

PARITY GATE (2026-08-30): the shipped duration head trains with IPCW censoring weights
(`trainer.train_phase` passes a censoring frame). Until this date every eval config passed
none, so every published number measured an unweighted model that was not the one serving.
A duration row now publishes only if its `ipcw_applied` is True — anything else (False,
missing, an older config) exits 2 with the row named. "Not measured" stays allowed; "measured
the wrong model" does not.
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

# What ships, per phase. Change here, re-run, and every surface follows.
SPLIT = "horizon"
DURATION, RATE = "duration_days", "recruitment_rate"
# Both configs construct TwoStageDuration exactly as trainer.train_phase does — same class,
# same censoring-frame builder (trainer.build_censoring_frame). P1HV ships the 0.85-nominal
# band (trainer.COVERAGE_TARGET): at 0.80 nominal it covered 0.729 on this fold, below the
# 0.75 gate; a 0.85 target landed 0.795 at +1.6 months of width (ledger row 333, 2026-08-29).
# `_total` (2026-08-30): one IPCW weight per trial from TOTAL duration, applied to both stages
# (trainer.IPCW_SCOPE). The plain `_ipcw` configs are the earlier scope — the enrolment stage
# alone, looked up at the enrolment window — kept for the record, not shipped.
# v5 (2026-08-31): the v1 random forest's point estimate inside the two-stage model's
# calibrated band, forest refit on all training rows after calibration, split rescaled to
# the forest total (experiments.candidates.HybridForestPoint; trainer.DURATION_MODEL).
SHIPPED = {ph: "hybrid_rf_refit_fband_ipcw_total" for ph in ("P1HV", "P1", "P2", "P3")}
# The rate the API SERVES is derived from the duration head's enrolment window (inference.py,
# Task 13 `6a20fd5`), band inverted from the duration band — so the served rate inherits the
# duration configs, censoring frame and coverage target included. `DerivedRate` in
# experiments/candidates.py mirrors that derivation line for line.
RATE_SHIPPED = {ph: "derived_rate_hybrid_refit_fband_ipcw_total" for ph in ("P1HV", "P1", "P2", "P3")}
# The standalone rate head reaches a response only as `recruitment_rate_crosscheck` — a point,
# never its band. Published as a labelled cross-check, not as the rate figure. It trains
# unweighted by design (a censored row's Enrollment is the target, not what was recruited).
# P1HV's head aims at 0.85 (trainer.RATE_COVERAGE_TARGET): 0.744 at 0.80, 0.800 at 0.85.
RATE_HEAD_SHIPPED = {"P1HV": "lgbm_rate_cov85", "P1": "lgbm_rate", "P2": "lgbm_rate", "P3": "lgbm_rate"}
BASELINE = "ta_median"
# Version ladder on the SAME mature fold (2026-08-31): every version's recipe refit on today's
# corpus and scored where v4 is scored, so one table holds them all. v1 = the shipped v1 recipe
# with its target leak removed (V1Recipe); v2 = LightGBM quantile + conformal (what v2 shipped);
# v3 = the two-stage split with a quantile point; v4 = SHIPPED. Refitting on today's corpus
# means v4's data-cap fix benefits every column — the ladder isolates the MODEL changes.
# docs/VERSION_HISTORY.md maps the labels (v4 was called v3.1 until 2026-08-31).
VERSION_LADDER = {"v1": "v1_recipe", "v2": "lgbm_conformal_recent", "v3": "two_stage",
                  "v4": {"P1HV": "two_stage_l2_cov85_ipcw_total", "P1": "two_stage_l2_ipcw_total",
                         "P2": "two_stage_l2_ipcw_total", "P3": "two_stage_l2_ipcw_total"}}
LIVE_VERSION = "v5"
FOLD_TEXT = ("horizon fold: train on trials starting before 2018, test on 2018–2020 starts, "
             "which have had 5.4–8.6 years to finish against a corpus whose p95 duration is 5.9")
PHASE_ORDER = ["P1HV", "P1", "P2", "P3"]
RATE_UNIT = "patients per site per month"


class ParityError(RuntimeError):
    """A shipped-config row that does not measure the shipped model."""


def load_rows(path: Path = LEDGER) -> list:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def latest(rows: list, config: str, phase: str, target: str = DURATION,
           split: str = SPLIT) -> dict | None:
    """Newest ledger row for (config, phase, target, split), 1-indexed row attached.

    `target` is part of the key: the baseline runs under both targets, and without
    it a newer rate-head baseline row would silently stand in for the duration one."""
    hits = [(i, r) for i, r in enumerate(rows, 1)
            if r.get("config") == config and r.get("phase") == phase
            and r.get("split") == split and r.get("target", DURATION) == target
            and not r.get("skipped") and not r.get("error")]
    if not hits:
        return None
    i, r = max(hits, key=lambda t: (t[1].get("ts", ""), t[0]))
    r = dict(r); r["ledger_row"] = i
    return r


def _gates(r: dict) -> dict:
    g = r.get("gates", {})
    return {"coverage_gate": g.get("interval_coverage", {}),
            "skill_gate": g.get("skill_vs_ta_median", {}),
            "all_gates_pass": r.get("gate_pass")}


def _parity(ph: str, r: dict, config: str, what: str) -> None:
    if r.get("ipcw_applied") is not True:
        raise ParityError(
            f"{ph} {what}: ledger row {r['ledger_row']} ({config}, {r.get('ts')}) has "
            f"ipcw_applied={r.get('ipcw_applied')!r}. The shipped duration head trains with "
            f"IPCW weights (backend/models/trainer.train_phase), so this row measures a model "
            f"nobody serves. Refusing to publish it. Re-run: python -m experiments.run "
            f"--config {config} --phases {ph} --split {SPLIT}"
            + (f" --target {RATE}" if what != "duration" else ""))


def _rate_entry(rows: list, ph: str, config: str) -> dict:
    r = latest(rows, config, ph, target=RATE)
    b = latest(rows, BASELINE, ph, target=RATE)
    if r is None:
        return {"status": "NOT MEASURED on this fold"}
    return {
        "config": config, "ledger_row": r["ledger_row"], "run_ts": r.get("ts"),
        "n_train": r.get("n_train"), "n_test": r.get("n_test"),
        "ipcw_applied": r.get("ipcw_applied"),
        "mae": r.get("mae_raw"), "rmse": r.get("rmse_raw", r.get("rmse_days")),
        "skill_vs_ta_median": r.get("skill_vs_ta_median"),
        "interval_coverage": r.get("interval_coverage"),
        "interval_nominal": r.get("interval_nominal"),
        **_gates(r),
        "baseline_mae": b.get("mae_raw") if b else None,
    }


def build(rows: list) -> dict:
    out = {"split": SPLIT, "fold": FOLD_TEXT, "generated_from": str(LEDGER.relative_to(ROOT)),
           "phases": {},
           "rate": {"target": RATE, "unit": RATE_UNIT,
                    "what": "the rate the API serves, derived from the duration head's "
                            "enrolment window with the band inverted from the duration band",
                    "phases": {}},
           "rate_head": {"target": RATE, "unit": RATE_UNIT,
                         "what": "the standalone rate head, served only as "
                                 "recruitment_rate_crosscheck (a point, no band)",
                         "phases": {}}}
    for ph in PHASE_ORDER:
        r = latest(rows, SHIPPED[ph], ph)
        b = latest(rows, BASELINE, ph)
        if r is None:
            out["phases"][ph] = {"status": "NOT MEASURED on this fold"}
            continue
        _parity(ph, r, SHIPPED[ph], "duration")
        out["phases"][ph] = {
            "config": SHIPPED[ph], "ledger_row": r["ledger_row"], "run_ts": r.get("ts"),
            "n_train": r.get("n_train"), "n_test": r.get("n_test"),
            "ipcw_applied": True,
            "mae_months": r.get("mae_months"), "r2": r.get("r2"), "rmse_days": r.get("rmse_days"),
            "skill_vs_ta_median": r.get("skill_vs_ta_median"),
            "interval_coverage": r.get("interval_coverage"),
            "interval_nominal": r.get("interval_nominal"),
            "interval_mean_width_months": r.get("interval_mean_width_months"),
            **_gates(r),
            "baseline_mae_months": b.get("mae_months") if b else None,
        }
    for ph in PHASE_ORDER:
        e = _rate_entry(rows, ph, RATE_SHIPPED[ph])
        if "mae" in e:
            # The served rate is the duration head's window inverted, so it carries the
            # duration head's parity obligation.
            _parity(ph, e, RATE_SHIPPED[ph], "served rate")
        out["rate"]["phases"][ph] = e
        out["rate_head"]["phases"][ph] = _rate_entry(rows, ph, RATE_HEAD_SHIPPED[ph])
    # Version ladder: same fold, every version, plus the baseline row. Not parity-gated —
    # v1-v3 never trained with a censoring frame, and that is part of what they were.
    out["versions"] = {"fold": FOLD_TEXT, "rows": {}}
    ladder = {"baseline": BASELINE, **VERSION_LADDER}
    for label, cfg in ladder.items():
        entry = {"config": cfg, "phases": {}}
        for ph in PHASE_ORDER:
            r = latest(rows, cfg[ph] if isinstance(cfg, dict) else cfg, ph)
            entry["phases"][ph] = ({"status": "NOT MEASURED on this fold"} if r is None else
                                   {"ledger_row": r["ledger_row"], "mae_months": r.get("mae_months"),
                                    "r2": r.get("r2"), "rmse_days": r.get("rmse_days")})
        out["versions"]["rows"][label] = entry
    out["versions"]["rows"][LIVE_VERSION] = {
        "config": "shipped (see phases)",
        "phases": {ph: ({"ledger_row": p["ledger_row"], "mae_months": p["mae_months"],
                         "r2": p["r2"], "rmse_days": p["rmse_days"]} if "mae_months" in p else p)
                   for ph, p in out["phases"].items()}}
    covs = [p["interval_coverage"] for p in out["phases"].values()
            if p.get("interval_coverage") is not None]
    out["coverage_range"] = [min(covs), max(covs)] if covs else None
    out["gates_failing"] = (
        [ph for ph, p in out["phases"].items() if p.get("all_gates_pass") is False]
        + [f"{ph} rate" for ph, p in out["rate"]["phases"].items()
           if p.get("all_gates_pass") is False]
        + [f"{ph} rate_head" for ph, p in out["rate_head"]["phases"].items()
           if p.get("all_gates_pass") is False])
    return out


def _gate_text(p: dict) -> str:
    if p["all_gates_pass"]:
        return "pass"
    cg, sg = p["coverage_gate"], p["skill_gate"]
    why = []
    if cg and cg.get("pass") is False:
        lo, hi = (cg.get("threshold") or ["?", "?"])[:2]
        why.append(f"coverage {cg.get('value')} outside {lo}–{hi}")
    if sg and sg.get("pass") is False:
        why.append(f"skill {sg.get('value')} ≤ {sg.get('threshold')}")
    return "**FAIL** (" + "; ".join(why or ["gate not evaluated"]) + ")"


def _cov(p: dict) -> str:
    nom = p.get("interval_nominal")
    return f"{p['interval_coverage']:.2f} ({nom:.2f})" if nom is not None else f"{p['interval_coverage']:.2f}"


def _rows_line(pub: dict, phases: dict) -> str:
    return (f"Fold: {pub['fold']}. Rows " +
            ", ".join(f"{ph}={phases[ph].get('ledger_row')}" for ph in PHASE_ORDER) +
            f" of `{pub['generated_from']}`. Regenerate: `python -m experiments.publish_metrics`.")


def markdown(pub: dict) -> str:
    lines = ["| Phase | duration MAE | skill | R² | coverage (nominal) | gate |",
             "|-------|--------------|-------|----|--------------------|------|"]
    for ph in PHASE_ORDER:
        p = pub["phases"][ph]
        if "mae_months" not in p:
            lines.append(f"| {ph} | n/a | n/a | n/a | n/a | not measured |"); continue
        lines.append(f"| {ph} | {p['mae_months']:.2f} mo | {p['skill_vs_ta_median']:+.2f} | "
                     f"{p['r2']:.3f} | {_cov(p)} | {_gate_text(p)} |")
    lines += ["", _rows_line(pub, pub["phases"]) +
              " Every row measures the shipped configuration, IPCW censoring weights applied "
              "as in `trainer.train_phase`."]
    return "\n".join(lines)


def _rate_table(phases: dict, label: str) -> list:
    lines = [f"| Phase | {label} MAE ({RATE_UNIT}) | baseline MAE | skill | coverage (nominal) | gate |",
             "|-------|------------------------------------|--------------|-------|--------------------|------|"]
    for ph in PHASE_ORDER:
        p = phases[ph]
        if "mae" not in p:
            lines.append(f"| {ph} | n/a | n/a | n/a | n/a | not measured |"); continue
        base = f"{p['baseline_mae']:.2f}" if p.get("baseline_mae") is not None else "n/a"
        lines.append(f"| {ph} | {p['mae']:.2f} | {base} | {p['skill_vs_ta_median']:+.2f} | "
                     f"{_cov(p)} | {_gate_text(p)} |")
    return lines


def markdown_rate(pub: dict) -> str:
    lines = ["**Served rate**: " + pub["rate"]["what"] + ".", ""]
    lines += _rate_table(pub["rate"]["phases"], "served rate")
    lines += ["", _rows_line(pub, pub["rate"]["phases"]), "",
              "**Cross-check**: " + pub["rate_head"]["what"] + ".", ""]
    lines += _rate_table(pub["rate_head"]["phases"], "rate-head")
    lines += ["", _rows_line(pub, pub["rate_head"]["phases"])]
    return "\n".join(lines)


#: marker name → renderer. Each doc carries `<!-- <name>:start … -->` … `<!-- <name>:end -->`.
BLOCKS = {"published_metrics": markdown, "published_metrics_rate": markdown_rate}


def fill_docs(pub: dict, docs=DOCS, write: bool = True) -> list:
    """Replace every marker block in each doc. Returns the docs that were (or would be)
    changed; a doc missing a block is skipped loudly — a silent skip would leave a stale table."""
    changed = []
    for doc in docs:
        text = doc.read_text()
        new = text
        for name, render in BLOCKS.items():
            start, end = f"<!-- {name}:start", f"<!-- {name}:end -->"
            i, j = new.find(start), new.find(end)
            if i < 0 or j < 0 or j < i:
                print(f"publish_metrics: {doc.name} has no {name} block; NOT updated",
                      file=sys.stderr)
                continue
            line_end = new.index("\n", i) + 1
            new = new[:line_end] + render(pub) + "\n" + new[j:]
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
    try:
        pub = build(load_rows())
    except ParityError as exc:
        print(f"PARITY: {exc}", file=sys.stderr)
        return 2
    text = json.dumps(pub, indent=2) + "\n"
    if a.check:
        cur = OUT.read_text() if OUT.exists() else ""
        stale = fill_docs(pub, write=False)
        if cur != text or stale:
            print(f"STALE against the ledger: json={'yes' if cur != text else 'no'} docs={stale} "
                  f"; re-run publish_metrics", file=sys.stderr)
            return 1
        print("published_metrics.json and doc blocks match the ledger")
        return 0
    OUT.write_text(text)
    print(markdown(pub)); print(); print(markdown_rate(pub))
    print(f"docs updated: {fill_docs(pub)}", file=sys.stderr)
    print(f"\nwrote {OUT.relative_to(ROOT)}, coverage range {pub['coverage_range']}, "
          f"gates failing: {pub['gates_failing']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
