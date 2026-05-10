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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


async def main():
    parser = argparse.ArgumentParser(description="Train clinical trial duration models.")
    parser.add_argument("--phase", choices=list(PHASES), help="Train a single phase.")
    parser.add_argument("--all", action="store_true", help="Train all phases (default).")
    args = parser.parse_args()

    if args.phase:
        await train_phase(args.phase)
    else:
        await train_all()


if __name__ == "__main__":
    asyncio.run(main())
