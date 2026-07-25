from __future__ import annotations
"""Train/test splitting.

The default is a TEMPORAL split, not a random one. The model's job in
production is to predict a trial that has not run yet, so the honest test is
"train on trials that started before date X, predict trials that started after".
A random split lets the model see 2023 trials while predicting 2019 ones and
flatters it — v1 was scored that way (trainer.py used train_test_split with
random_state=42).
"""
import logging

import pandas as pd

log = logging.getLogger(__name__)

DEFAULT_CUTOFF = "2021-01-01"


def temporal_split(
    df: pd.DataFrame,
    cutoff: str = DEFAULT_CUTOFF,
    date_col: str = "Start Date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split on trial START date. Returns (train, test)."""
    dates = pd.to_datetime(df[date_col], errors="coerce")
    cut = pd.Timestamp(cutoff)
    train = df[dates < cut].reset_index(drop=True)
    test = df[dates >= cut].reset_index(drop=True)
    log.info(
        "Temporal split @ %s: train=%d test=%d", cutoff, len(train), len(test)
    )
    return train, test


def random_split(
    df: pd.DataFrame, test_size: float = 0.2, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The v1 protocol, kept only so we can quantify how much it flatters."""
    from sklearn.model_selection import train_test_split

    train, test = train_test_split(df, test_size=test_size, random_state=seed)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def get_split(df: pd.DataFrame, kind: str, **kwargs):
    if kind == "temporal":
        return temporal_split(df, **kwargs)
    if kind == "random":
        return random_split(df, **kwargs)
    raise ValueError(f"Unknown split kind '{kind}' (expected temporal|random)")


def check_split_viability(
    train: pd.DataFrame, test: pd.DataFrame, min_rows: int = 30
) -> str | None:
    """Return a warning string if the split leaves too little to learn or score."""
    if len(train) < min_rows:
        return f"train fold has only {len(train)} rows (< {min_rows})"
    if len(test) < min_rows:
        return f"test fold has only {len(test)} rows (< {min_rows})"
    return None
