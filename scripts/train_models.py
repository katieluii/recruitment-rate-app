"""
One-shot training script.
Run from the project root:

    python -m scripts.train_models [--phase P1HV] [--all]

Fetches data from ClinicalTrials.gov API (or Postgres if configured),
trains a LightGBM model per phase, and saves artifacts to models/artifacts/.
"""
import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.models.trainer import train_phase, train_all
from backend.constants import PHASES


class CacheLoader:
    """Serve the trainer from the local parquet cache instead of the API.

    Opt-in via --use-cache. Two reasons it exists: retraining four phases
    otherwise issues twelve back-to-back CT.gov fetches, which has rate-limited
    this project before; and it guarantees the fitted model sees exactly the
    corpus experiments/ measured, rather than a fetch taken minutes later.
    """

    def completed(self, phase_key):
        from experiments.dataset import load_raw
        return load_raw(phase_key)

    def ongoing(self, phase_key):
        """Ongoing cohort from cache, or a LOUD failure.

        Returning an empty frame here is silently expensive: the trainer's
        censoring-frame builder catches everything, logs one warning, and trains
        the duration head with no IPCW correction at all — which cost P3 roughly
        3.7 months of optimistic bias the one time it happened unnoticed. A
        missing cache must stop the run, not quietly downgrade the model.
        """
        from experiments.dataset import cache_path, load_ongoing

        path = cache_path(phase_key, "ongoing")
        if not path.exists():
            raise SystemExit(
                f"No ongoing-cohort cache for {phase_key} at {path}.\n"
                f"Training from cache without it would skip the IPCW censoring "
                f"correction silently. Fetch it first:\n"
                f"  python -c \"from experiments.dataset import load_ongoing; "
                f"load_ongoing('{phase_key}')\""
            )
        return load_ongoing(phase_key)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


async def main():
    parser = argparse.ArgumentParser(description="Train clinical trial duration models.")
    parser.add_argument("--phase", choices=list(PHASES), help="Train a single phase.")
    parser.add_argument("--all", action="store_true", help="Train all phases (default).")
    parser.add_argument("--use-cache", action="store_true",
                        help="Train from the local parquet cache instead of "
                             "fetching from ClinicalTrials.gov.")
    parser.add_argument("--heads", default=None,
                        help="Comma-separated subset of heads (duration,rate) to retrain "
                             "for --phase; the other head's artifacts and metadata are "
                             "carried forward untouched.")
    args = parser.parse_args()

    loader = CacheLoader() if args.use_cache else None
    heads = [h.strip() for h in args.heads.split(",")] if args.heads else None
    if args.phase:
        await train_phase(args.phase, loader=loader, heads=heads)
    else:
        if heads:
            raise SystemExit("--heads needs --phase; a partial retrain is one phase at a time")
        await train_all(loader=loader)


if __name__ == "__main__":
    asyncio.run(main())
