from __future__ import annotations
"""Therapeutic-area target encoding.

This is the direct fix for the failure that started the rework: 22 sparse binary
therapeutic-area columns could not compete against six scaled continuous
features inside a Random Forest with max_features=0.5, so rare areas almost
never won a split and tree averaging pulled them all to the phase mean. 17 of
22 Phase 1 areas returned the identical 10.9 months.

Replacing those 22 weak binaries with one strong continuous column — the
smoothed median duration of the area — gives the model a therapeutic-area
signal it will actually split on.

Leakage control:
  * The encoder is fitted inside the sklearn pipeline, so it only ever sees the
    training fold. Test-fold rows are encoded from statistics they did not
    contribute to.
  * Within the training fold, `fit_transform` uses OUT-OF-FOLD encoding. Encoding
    a training row with a statistic computed from that same row is how target
    encoding quietly overfits; K-fold prevents the model from learning to trust
    the column more than it should.
  * Areas with few trials are shrunk toward the global median rather than
    fitted to noise.
"""
import logging

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold

log = logging.getLogger(__name__)


class TATargetEncoder(BaseEstimator, TransformerMixin):
    """Encode a multi-label one-hot block as smoothed target statistics.

    Input  : (n_samples, n_areas) binary matrix — a trial may belong to several.
    Output : (n_samples, 2) — [smoothed target statistic, log1p(support)].

    The support column lets the model discount the encoding where it rests on
    few trials, instead of treating a 6-trial area's median as being as solid
    as a 500-trial area's.
    """

    def __init__(self, smoothing: float = 10.0, n_splits: int = 5,
                 random_state: int = 42):
        self.smoothing = smoothing
        self.n_splits = n_splits
        self.random_state = random_state

    # ── fitting ──────────────────────────────────────────────────────────────

    def _fit_tables(self, X: np.ndarray, y: np.ndarray):
        """Return (per-area smoothed statistic, per-area support, global stat)."""
        global_stat = float(np.median(y))
        n_areas = X.shape[1]
        stats = np.full(n_areas, global_stat, dtype=float)
        support = np.zeros(n_areas, dtype=float)
        for j in range(n_areas):
            mask = X[:, j] == 1
            n = int(mask.sum())
            if n == 0:
                continue
            raw = float(np.median(y[mask]))
            # Shrink toward the global median in proportion to how thin the
            # evidence is: n=5 with smoothing=10 lands a third of the way.
            stats[j] = (n * raw + self.smoothing * global_stat) / (n + self.smoothing)
            support[j] = n
        return stats, support, global_stat

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("TATargetEncoder requires y")
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.stats_, self.support_, self.global_ = self._fit_tables(X, y)
        self.n_features_in_ = X.shape[1]
        return self

    def _encode(self, X: np.ndarray, stats: np.ndarray, support: np.ndarray,
                global_stat: float) -> np.ndarray:
        hits = X.sum(axis=1)
        # Mean of the areas the trial belongs to; global median if it belongs to none.
        summed = X @ stats
        enc = np.where(hits > 0, summed / np.maximum(hits, 1), global_stat)
        sup = np.where(hits > 0, (X @ support) / np.maximum(hits, 1), 0.0)
        return np.column_stack([enc, np.log1p(sup)])

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return self._encode(X, self.stats_, self.support_, self.global_)

    def fit_transform(self, X, y=None, **fit_params):
        """Out-of-fold encoding for the training rows.

        Falls back to plain fit().transform() when there are too few rows to
        fold, which only happens on tiny phase subsets.
        """
        if y is None:
            raise ValueError("TATargetEncoder requires y")
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.fit(X, y)

        if len(y) < self.n_splits * 2:
            log.warning("Only %d rows — skipping out-of-fold encoding", len(y))
            return self.transform(X)

        out = np.zeros((len(y), 2), dtype=float)
        kf = KFold(n_splits=self.n_splits, shuffle=True,
                   random_state=self.random_state)
        for tr_idx, te_idx in kf.split(X):
            stats, support, global_stat = self._fit_tables(X[tr_idx], y[tr_idx])
            out[te_idx] = self._encode(X[te_idx], stats, support, global_stat)
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array(["ta_target_stat", "ta_log_support"])
