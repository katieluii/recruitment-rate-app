from __future__ import annotations
"""Reference predictors.

These exist to answer the question v1 never asked: is the model actually adding
anything over a lookup table? `TAMedianBaseline` is the bar. A learned model
that does not beat the median duration for its therapeutic area is a lookup
table with extra steps and worse latency.

Each baseline also emits an empirical 10th/90th percentile interval, so model
intervals have something honest to be compared against.
"""
import logging

import numpy as np
import pandas as pd

from experiments.metrics import ta_masks

log = logging.getLogger(__name__)


class _Baseline:
    name = "baseline"

    def fit(self, train: pd.DataFrame, target: str) -> "_Baseline":
        raise NotImplementedError

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def predict_interval(self, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class MedianBaseline(_Baseline):
    """Predict the training-fold median for every trial.

    Because models are trained per phase, this is the per-phase median.
    """

    name = "median"

    def fit(self, train: pd.DataFrame, target: str = "duration_days"):
        vals = train[target].dropna()
        self._median = float(vals.median())
        self._q10 = float(vals.quantile(0.10))
        self._q90 = float(vals.quantile(0.90))
        return self

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        return np.full(len(test), self._median)

    def predict_interval(self, test: pd.DataFrame):
        return (np.full(len(test), self._q10), np.full(len(test), self._q90))


class TAMedianBaseline(_Baseline):
    """Per-therapeutic-area median from the training fold. THE BAR.

    Trials mapping to several areas get the mean of those areas' medians.
    Areas with too few training rows fall back to the global median rather than
    fitting noise.
    """

    name = "ta_median"

    def __init__(self, min_rows: int = 5):
        self.min_rows = min_rows

    def fit(self, train: pd.DataFrame, target: str = "duration_days"):
        y = train[target].to_numpy(dtype=float)
        self._global = float(np.median(y))
        self._global_q10 = float(np.quantile(y, 0.10))
        self._global_q90 = float(np.quantile(y, 0.90))

        self._med: dict[str, float] = {}
        self._q10: dict[str, float] = {}
        self._q90: dict[str, float] = {}
        for area, mask in ta_masks(train).items():
            m = mask.to_numpy()
            if m.sum() < self.min_rows:
                continue
            vals = y[m]
            self._med[area] = float(np.median(vals))
            self._q10[area] = float(np.quantile(vals, 0.10))
            self._q90[area] = float(np.quantile(vals, 0.90))
        log.info("TAMedianBaseline: %d areas above n>=%d",
                 len(self._med), self.min_rows)
        return self

    def _lookup(self, test: pd.DataFrame, table: dict[str, float],
                fallback: float) -> np.ndarray:
        masks = ta_masks(test)
        acc = np.zeros(len(test))
        hits = np.zeros(len(test))
        for area, mask in masks.items():
            if area not in table:
                continue
            m = mask.to_numpy()
            acc[m] += table[area]
            hits[m] += 1
        out = np.where(hits > 0, acc / np.maximum(hits, 1), fallback)
        return out

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        return self._lookup(test, self._med, self._global)

    def predict_interval(self, test: pd.DataFrame):
        return (
            self._lookup(test, self._q10, self._global_q10),
            self._lookup(test, self._q90, self._global_q90),
        )


class TAEnrollmentMedianBaseline(TAMedianBaseline):
    """Per-TA median, further split by enrollment tercile.

    A harder bar than TA alone, and a cheap check on whether the model is
    adding anything beyond "big trial, slow trial".
    """

    name = "ta_enrollment_median"

    def fit(self, train: pd.DataFrame, target: str = "duration_days"):
        super().fit(train, target)
        enrol = pd.to_numeric(train["Enrollment"], errors="coerce").fillna(0)
        self._edges = [float(enrol.quantile(1 / 3)), float(enrol.quantile(2 / 3))]
        y = train[target].to_numpy(dtype=float)
        self._bucket_med: dict[tuple[str, int], float] = {}
        buckets = np.digitize(enrol.to_numpy(), self._edges)
        for area, mask in ta_masks(train).items():
            m = mask.to_numpy()
            for b in (0, 1, 2):
                sel = m & (buckets == b)
                if sel.sum() < self.min_rows:
                    continue
                self._bucket_med[(area, b)] = float(np.median(y[sel]))
        return self

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        enrol = pd.to_numeric(test["Enrollment"], errors="coerce").fillna(0)
        buckets = np.digitize(enrol.to_numpy(), self._edges)
        masks = ta_masks(test)
        acc = np.zeros(len(test))
        hits = np.zeros(len(test))
        for area, mask in masks.items():
            m = mask.to_numpy()
            for b in (0, 1, 2):
                key = (area, b)
                if key not in self._bucket_med:
                    continue
                sel = m & (buckets == b)
                acc[sel] += self._bucket_med[key]
                hits[sel] += 1
        fallback = super().predict(test)
        return np.where(hits > 0, acc / np.maximum(hits, 1), fallback)


ALL_BASELINES = [MedianBaseline, TAMedianBaseline, TAEnrollmentMedianBaseline]

#: The baseline every candidate model must beat before it can ship.
PRIMARY_BASELINE = TAMedianBaseline
