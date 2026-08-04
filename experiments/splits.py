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

#: The gate fold, chosen 2026-08-04. Trials that started in 2021 or later have
#: had between 0 and 5.6 years to finish, against a corpus whose 95th-percentile
#: duration is 5.9 years, so the 2021+ fold cannot contain a long trial and
#: scores a model higher for predicting short. Measured: the longest trial in
#: each 2021+ start-year sits within ~0.2 years of that year's available
#: horizon, and model bias ran -1.27 months where the horizon was longest to
#: +2.03 where it was shortest.
#:
#: Trials starting 2018-2020 have had 5.4 to 8.6 years, so a long trial CAN
#: appear and the target is effectively uncapped. The cost is real and worth
#: stating: 2021+ trials become training data only and are never scored.
HORIZON_CUTOFF = "2018-01-01"
HORIZON_TEST_END = "2021-01-01"


def temporal_split(
    df: pd.DataFrame,
    cutoff: str = DEFAULT_CUTOFF,
    date_col: str = "Start Date",
    test_end: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split on trial START date. Returns (train, test).

    `test_end` closes the test window on the right. Without it the test fold
    runs to the data vintage and its later years are truncated by observation
    horizon — see HORIZON_CUTOFF above for why that flatters short predictions.
    """
    dates = pd.to_datetime(df[date_col], format="ISO8601", errors="coerce")
    cut = pd.Timestamp(cutoff)
    train = df[dates < cut].reset_index(drop=True)
    mask = dates >= cut
    if test_end is not None:
        mask &= dates < pd.Timestamp(test_end)
    test = df[mask].reset_index(drop=True)
    log.info("Temporal split @ %s%s: train=%d test=%d", cutoff,
             f" (test < {test_end})" if test_end else "", len(train), len(test))
    return train, test


def random_split(
    df: pd.DataFrame, test_size: float = 0.2, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The v1 protocol, kept only so we can quantify how much it flatters."""
    from sklearn.model_selection import train_test_split

    train, test = train_test_split(df, test_size=test_size, random_state=seed)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def get_split(df: pd.DataFrame, kind: str, **kwargs):
    if kind == "horizon":
        kwargs.setdefault("cutoff", HORIZON_CUTOFF)
        kwargs.setdefault("test_end", HORIZON_TEST_END)
        return temporal_split(df, **kwargs)
    if kind == "temporal":
        return temporal_split(df, **kwargs)
    if kind == "random":
        return random_split(df, **kwargs)
    raise ValueError(
        f"Unknown split kind '{kind}' (expected horizon|temporal|random)")


def check_split_viability(
    train: pd.DataFrame, test: pd.DataFrame, min_rows: int = 30
) -> str | None:
    """Return a warning string if the split leaves too little to learn or score."""
    if len(train) < min_rows:
        return f"train fold has only {len(train)} rows (< {min_rows})"
    if len(test) < min_rows:
        return f"test fold has only {len(test)} rows (< {min_rows})"
    return None
