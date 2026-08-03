from __future__ import annotations
"""Write endpoint_profiles.json into existing artifacts, without retraining.

    python -m scripts.build_endpoint_profiles --phases P1
    python -m scripts.build_endpoint_profiles            # all four

The fitted models are not touched and not reloaded — this only adds a derived
lookup beside them, so it cannot move a prediction. Retraining to obtain a JSON
file would risk changing the served numbers for no reason, and each phase's
retrain needs its own CT.gov fetch.

Needs the phase's parquet cache (see experiments/dataset.py). One phase per run
if the caches are cold: fetching all four back to back has rate-limited CT.gov.
"""
import argparse
import json
import logging
from pathlib import Path

from backend.analytics.endpoint_profiles import build_profiles
from backend.config import settings
from backend.constants import PHASES
from experiments.dataset import load_clean

log = logging.getLogger(__name__)


def build(phase_key: str) -> Path:
    base = settings.models_dir / phase_key
    if not (base / "metadata.json").exists():
        raise SystemExit(f"No trained artifacts for {phase_key} at {base}")

    df = load_clean(phase_key)
    df = df[df["duration_days"].notna()].reset_index(drop=True)
    profiles = build_profiles(df)

    path = base / "endpoint_profiles.json"
    path.write_text(json.dumps(profiles, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")

    areas = [k for k in profiles if k != "_phase"]
    log.info("%s: %d rows -> %d areas with their own profile (+ phase fallback)",
             phase_key, len(df), len(areas))
    top = profiles.get("_phase", [])
    if top:
        log.info("  phase-wide top profile: %s (n=%d, median %.1f mo)",
                 top[0]["label"], top[0]["n"], top[0]["median_months"])
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phases", default=",".join(PHASES))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    for phase_key in [p.strip() for p in args.phases.split(",")]:
        if phase_key not in PHASES:
            raise SystemExit(f"Unknown phase '{phase_key}'")
        print(f"wrote {build(phase_key)}")


if __name__ == "__main__":
    main()
