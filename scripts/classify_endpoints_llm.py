from __future__ import annotations
"""Batched endpoint classification for the trials the regex rules abstain on.

    python -m scripts.classify_endpoints_llm --phase P2 --mode validate
    python -m scripts.classify_endpoints_llm --phase P2 --mode resolve

Two modes, and VALIDATE MUST RUN FIRST.

  validate  Sample trials the deterministic rules already classified, hide the
            rule's answer, and ask the model. Reports agreement per archetype.
            This is the only evidence that the model's labels on the ABSTENTIONS
            are worth anything — without it, resolving abstentions produces
            confident labels with no way to know if they are right.

  resolve   Classify only rows where the rules abstained. Never overwrites a
            deterministic label; the rules stay the source of truth where they
            fired.

Design constraints, each of which exists for a reason:

* Closed vocabulary. Every returned archetype goes through `endpoints.validate`,
  which rejects anything outside ARCHETYPES to UNKNOWN. A free-form label would
  silently create a category the model never saw at training time.
* Multi-label. The whole point of the profiles work is that a trial carries a
  COMBINATION; asking for one label would rebuild the defect being fixed.
* Cached by NCT id in its own file, so a rerun costs nothing and a crash mid-run
  loses nothing. The verdict file is append-merged, never blind-overwritten.
* Batched, ~25 trials per call, grouped by phase and indication so each call sees
  one clinical context rather than a scramble.
* Runs on the Claude Max subscription via `claude -p`, cheap tier by default.
"""
import argparse
import json
import logging
import re
import subprocess
from pathlib import Path

import pandas as pd

from backend.preprocessing.cleaner import clean
from backend.preprocessing.endpoints import (ARCHETYPES, add_endpoint_features,
                                             validate)
from experiments.dataset import load_raw
from experiments.metrics import ta_masks

log = logging.getLogger(__name__)

CACHE = Path("data/cache/endpoint_llm_multilabel.json")
BATCH = 25
MODEL = "sonnet"

_VOCAB = [a for a in ARCHETYPES if a != "UNKNOWN"]

PROMPT_HEAD = f"""You are classifying clinical trial PRIMARY endpoints into a closed vocabulary.

Vocabulary (use these exact strings, nothing else):
{chr(10).join('  ' + a for a in _VOCAB)}

Definitions:
  SURVIVAL        OS, PFS, DFS, EFS - event-driven time-to-event endpoints
  EVENT_COMPOSITE MACE, hospitalisation or mortality composites
  EVENT_RATE      annualised rates - bleeding rate, seizure frequency, attack rate
  RESPONSE        ORR, pCR, DCR, clinical remission, treatment success
  IMMUNOGENICITY  seroconversion, seroprotection, antibody titre
  CLINICAL_SCORE  ACR20, PASI75, ADAS-Cog, EDSS, 6MWD and similar scales
  BIOMARKER       HbA1c, LDL, viral load, eGFR and similar lab measures
  PRO             patient-reported outcomes, quality of life
  SAFETY          AE, TEAE, DLT, MTD, tolerability
  PK_PD           Cmax, AUC, half-life, clearance, receptor occupancy

Rules:
- A trial may have SEVERAL archetypes across its primary outcomes. Return every
  one that genuinely appears. Most trials have one or two.
- Return an EMPTY list if the text does not support any archetype. An empty list
  is a correct and useful answer; guessing is not.
- Judge ONLY the text given. Do not infer from the indication or the trial name.

Return STRICT JSON only, no prose, no markdown fence:
{{"NCT01234567": ["SAFETY", "PK_PD"], "NCT07654321": []}}

Trials:
"""


def _load_cache() -> dict:
    if CACHE.exists():
        return json.loads(CACHE.read_text())
    return {}


def _save_cache(cache: dict) -> None:
    """Re-read and merge before writing: a concurrent phase run must not lose
    the other's verdicts to a stale in-memory copy."""
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    merged = _load_cache()
    merged.update(cache)
    CACHE.write_text(json.dumps(merged, indent=2, sort_keys=True,
                                ensure_ascii=False) + "\n", encoding="utf-8")


def _call(prompt: str) -> str:
    out = subprocess.run(
        ["claude", "-p", prompt, "--model", MODEL],
        capture_output=True, text=True, timeout=600,
    )
    if out.returncode != 0:
        raise RuntimeError(f"claude -p failed: {out.stderr[:300]}")
    return out.stdout.strip()


def _parse(raw: str) -> dict[str, list[str]]:
    """Strict JSON out, closed vocabulary in. Anything unparseable is dropped
    rather than guessed at, and says so."""
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        log.warning("No JSON object in response: %s", raw[:200])
        return {}
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        log.warning("Unparseable JSON (%s): %s", exc, raw[:200])
        return {}

    out = {}
    for nct, labels in data.items():
        if not isinstance(labels, list):
            log.warning("%s: expected a list, got %r", nct, labels)
            continue
        clean_labels = []
        for lab in labels:
            v = validate(str(lab).strip().upper())
            if v != "UNKNOWN" and v not in clean_labels:
                clean_labels.append(v)
        out[str(nct).strip()] = clean_labels
    return out


def _batches(frame: pd.DataFrame):
    for i in range(0, len(frame), BATCH):
        yield frame.iloc[i:i + BATCH]


def _classify(frame: pd.DataFrame, cache: dict, label: str) -> dict:
    """Classify a frame batch by batch, skipping anything already cached."""
    todo = frame[~frame["nct_id"].astype(str).isin(cache)]
    if not len(todo):
        log.info("%s: all %d already cached", label, len(frame))
        return {}

    got = {}
    for n, batch in enumerate(_batches(todo), 1):
        lines = []
        for _, r in batch.iterrows():
            measures = " ; ".join(
                m.strip() for m in str(r["primary_outcome_measures"]).split("|")
                if m.strip())[:1200]
            lines.append(f'{r["nct_id"]}: {measures}')
        try:
            parsed = _parse(_call(PROMPT_HEAD + "\n".join(lines)))
        except Exception as exc:
            log.error("%s batch %d failed: %s", label, n, exc)
            continue
        # Only accept ids we actually asked about.
        asked = set(batch["nct_id"].astype(str))
        parsed = {k: v for k, v in parsed.items() if k in asked}
        missing = len(asked) - len(parsed)
        if missing:
            log.warning("%s batch %d: %d of %d trials came back unanswered",
                        label, n, missing, len(asked))
        got.update(parsed)
        log.info("%s batch %d: %d/%d classified (%d cumulative)",
                 label, n, len(parsed), len(asked), len(got))
    return got


def _frame(phase_key: str) -> pd.DataFrame:
    df = clean(load_raw(phase_key), phase_key)
    hv = int(__import__("backend.constants", fromlist=["PHASES"]).PHASES[phase_key]["hv"])
    if "is_hv" in df.columns:
        df = df[df["is_hv"] == hv]
    df = df[df["duration_days"].notna()].reset_index(drop=True)
    return add_endpoint_features(df)


def _by_indication(df: pd.DataFrame):
    """Yield (indication, frame) so each call sees one clinical context."""
    masks = ta_masks(df)
    claimed = pd.Series(False, index=df.index)
    for area, mask in sorted(masks.items(), key=lambda kv: -int(kv[1].sum())):
        sel = mask.to_numpy() & ~claimed.to_numpy()
        if not sel.any():
            continue
        claimed |= pd.Series(sel, index=df.index)
        yield area, df[sel]


def run_validate(phase_key: str, sample: int) -> None:
    """Blind agreement against the rules, on rows the rules DID classify."""
    df = _frame(phase_key)
    classified = df[df["endpoint_archetype"] != "UNKNOWN"]
    # Stratify so a rare archetype is not represented by one trial.
    per = max(5, sample // max(1, classified["endpoint_archetype"].nunique()))
    strat = (classified.groupby("endpoint_archetype", group_keys=False)
             .apply(lambda g: g.head(per)))
    log.info("%s validate: %d trials across %d archetypes",
             phase_key, len(strat), strat["endpoint_archetype"].nunique())

    verdicts = _classify(strat, {}, f"{phase_key}/validate")

    rows = []
    for _, r in strat.iterrows():
        got = verdicts.get(str(r["nct_id"]))
        if got is None:
            continue
        rule = r["endpoint_archetype"]
        rows.append({"rule": rule, "llm": " + ".join(got) or "(none)",
                     "contains_rule_label": rule in got})
    if not rows:
        print("No verdicts returned — cannot report agreement.")
        return

    res = pd.DataFrame(rows)
    agree = res["contains_rule_label"].mean()
    print(f"\n{phase_key} — does the model's set CONTAIN the rule's label?")
    print(f"  overall {agree:.1%} on n={len(res)}\n")
    per_arch = (res.groupby("rule")["contains_rule_label"]
                .agg(["mean", "size"]).sort_values("mean"))
    per_arch["mean"] = (100 * per_arch["mean"]).round(1)
    print(per_arch.rename(columns={"mean": "agreement_%", "size": "n"}).to_string())
    print("\nLow agreement on an archetype means the abstention labels for that "
          "kind of endpoint should not be trusted.")


def run_resolve(phase_key: str) -> None:
    """Classify only the rows the rules abstained on."""
    df = _frame(phase_key)
    todo = df[df["endpoint_archetype"] == "UNKNOWN"]
    log.info("%s: %d abstentions of %d rows (%.1f%%)",
             phase_key, len(todo), len(df), 100 * len(todo) / max(1, len(df)))
    if not len(todo):
        return

    cache = _load_cache()
    for area, frame in _by_indication(todo):
        got = _classify(frame, cache, f"{phase_key}/{area[:24]}")
        if got:
            cache.update(got)
            _save_cache(got)

    resolved = sum(1 for k in df[df["endpoint_archetype"] == "UNKNOWN"]["nct_id"]
                   .astype(str) if cache.get(k))
    non_empty = sum(1 for k in df[df["endpoint_archetype"] == "UNKNOWN"]["nct_id"]
                    .astype(str) if cache.get(k))
    print(f"\n{phase_key}: {resolved} of {len(todo)} abstentions now have a verdict "
          f"({non_empty} non-empty). Cache: {CACHE}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True)
    ap.add_argument("--mode", choices=["validate", "resolve"], required=True)
    ap.add_argument("--sample", type=int, default=120)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.mode == "validate":
        run_validate(args.phase, args.sample)
    else:
        run_resolve(args.phase)


if __name__ == "__main__":
    main()
