from __future__ import annotations
"""Feasibility audit for an actual patients/centre/month training target.

Examples:
    python -m experiments.recruitment_history_pilot --sample-per-phase 10
    python -m experiments.recruitment_history_pilot --sample-per-phase 100 --concurrency 3

Each trial result is cached independently.  Interrupted runs resume without
re-querying completed IDs, and the report records every exclusion reason rather
than silently shortening the sample.
"""

import argparse
import asyncio
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import httpx

from backend.data.trial_history import RecruitmentTarget, TrialHistoryClient
from experiments.dataset import load_raw


log = logging.getLogger(__name__)
CACHE_DIR = Path("data/cache/history_targets")
REPORT_DIR = Path("experiments/reports")


def _candidates(phase: str, sample: int, seed: int,
                min_start_year: int | None = None) -> list[str]:
    frame = load_raw(phase)
    if phase == "P1" and "is_hv" in frame:
        frame = frame[frame["is_hv"] == 0]
    frame = frame.dropna(subset=["nct_id", "Start Date"])
    if min_start_year is not None:
        starts = pd.to_datetime(frame["Start Date"], errors="coerce")
        frame = frame[starts.dt.year >= min_start_year]
    # Prefer completed records with a listed site; the history extractor still
    # verifies final ACTUAL enrollment and start-date type independently.
    if "site_count" in frame:
        frame = frame[pd.to_numeric(frame["site_count"], errors="coerce") > 0]
    frame = frame.drop_duplicates("nct_id")
    if len(frame) <= sample:
        return frame["nct_id"].astype(str).tolist()
    return frame.sample(sample, random_state=seed)["nct_id"].astype(str).tolist()


def _cache_path(nct_id: str, mode: str) -> Path:
    suffix = "summary" if mode == "summary" else "detailed"
    return CACHE_DIR / suffix / f"{nct_id}.json"


def _read_cached(nct_id: str, mode: str) -> dict[str, Any] | None:
    path = _cache_path(nct_id, mode)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _write_cached(result: RecruitmentTarget, mode: str) -> dict[str, Any]:
    payload = result.to_dict()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(result.nct_id, mode)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return payload


async def _fetch_one(nct_id: str, semaphore: asyncio.Semaphore,
                     history_client: TrialHistoryClient, mode: str,
                     http_client: httpx.AsyncClient | None = None) -> dict[str, Any]:
    cached = _read_cached(nct_id, mode)
    if cached is not None:
        return cached
    async with semaphore:
        try:
            result = (await history_client.fetch_summary_target(nct_id, http_client)
                      if mode == "summary" else await history_client.fetch_target(nct_id))
            log.info("%s: %s", nct_id,
                     result.quality_tier if result.usable else result.exclusion_reason)
            return _write_cached(result, mode)
        except Exception as exc:
            # Network errors are written to the report, but not cached: a future
            # resume should retry them instead of treating a transient failure as
            # a data exclusion.
            log.warning("%s: fetch error: %s", nct_id, exc)
            return {"nct_id": nct_id, "usable": False,
                    "exclusion_reason": "fetch_error", "error": str(exc)}


async def run(phases: list[str], sample_per_phase: int,
              concurrency: int, seed: int,
              min_start_year: int | None = None,
              mode: str = "detailed") -> pd.DataFrame:
    history_client = TrialHistoryClient()
    semaphore = asyncio.Semaphore(concurrency)
    phase_ids = {phase: _candidates(phase, sample_per_phase, seed, min_start_year)
                 for phase in phases}
    unique_ids = sorted({nct for ids in phase_ids.values() for nct in ids})
    if mode == "summary":
        async with httpx.AsyncClient(
            timeout=history_client.timeout,
            limits=httpx.Limits(max_connections=concurrency,
                                max_keepalive_connections=concurrency),
            headers={"User-Agent": "WSi research audit"},
        ) as http_client:
            payloads = await asyncio.gather(*[
                _fetch_one(nct_id, semaphore, history_client, mode, http_client)
                for nct_id in unique_ids
            ])
    else:
        payloads = await asyncio.gather(*[
            _fetch_one(nct_id, semaphore, history_client, mode)
            for nct_id in unique_ids
        ])
    by_id = {row["nct_id"]: row for row in payloads}
    rows = []
    for phase, ids in phase_ids.items():
        for nct_id in ids:
            rows.append({"phase": phase, **by_id[nct_id]})
    return pd.DataFrame(rows)


def _summary(frame: pd.DataFrame) -> str:
    lines = [
        "# Recruitment history feasibility pilot",
        "",
        "Historical target: final ACTUAL enrollment divided by recruiting centre-months.",
        "Tier A integrates dated active-site snapshots. Tier B divides final actual",
        "enrolment by initiated sites and the recorded recruiting interval; it trains",
        "the released planning-rate model and is disclosed as an assumption, not",
        "observed performance for each centre. Tier A is the validation reference.",
        "",
        "| Phase | Sample | Tier A | Tier B | Excluded/error | Tier-A median PPCM |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for phase, group in frame.groupby("phase", sort=False):
        tiers = group.get("quality_tier", pd.Series(index=group.index, dtype=object))
        a = group[tiers == "A"]
        median = pd.to_numeric(a.get("recruitment_rate"), errors="coerce").median()
        lines.append(
            f"| {phase} | {len(group)} | {(tiers == 'A').sum()} | "
            f"{(tiers == 'B').sum()} | {(~tiers.isin(['A', 'B'])).sum()} | "
            f"{median:.3f} |" if pd.notna(median) else
            f"| {phase} | {len(group)} | {(tiers == 'A').sum()} | "
            f"{(tiers == 'B').sum()} | {(~tiers.isin(['A', 'B'])).sum()} | n/a |"
        )
    lines.extend(["", "## Exclusion reasons", ""])
    counts = Counter(frame.loc[frame["quality_tier"].isna(),
                               "exclusion_reason"].fillna("unknown"))
    for reason, count in counts.most_common():
        lines.append(f"- {reason}: {count}")
    lines.extend([
        "",
        "## Interpretation gate",
        "",
        "Proceed to model training only if Tier A coverage can support at least 1,000",
        "eligible trials in each of P1, P2 and P3, and a 20-trial manual recalculation",
        "agrees with the extractor for at least 18 trials within 10%.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phases", default="P1,P2,P3")
    parser.add_argument("--sample-per-phase", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2105)
    parser.add_argument("--min-start-year", type=int)
    parser.add_argument("--mode", choices=("detailed", "summary"),
                        default="detailed")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    phases = [p.strip() for p in args.phases.split(",") if p.strip()]
    frame = asyncio.run(run(phases, args.sample_per_phase,
                            args.concurrency, args.seed, args.min_start_year,
                            args.mode))
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = REPORT_DIR / "recruitment-history-pilot.csv"
    md_path = REPORT_DIR / "recruitment-history-pilot.md"
    frame.to_csv(csv_path, index=False)
    md_path.write_text(_summary(frame))
    print(md_path.read_text())
    print(f"Detailed rows: {csv_path}")


if __name__ == "__main__":
    main()
