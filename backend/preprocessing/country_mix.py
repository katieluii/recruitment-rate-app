from __future__ import annotations
"""Site mix as a model input — the thing that makes geography change the answer.

In v2 the site layer was a lookup that hung off the side of the model: moving 20
sites from the US to Poland changed the simulator's output and left the duration
prediction untouched. This encoder closes that, by turning a trial's per-country
site distribution into a feature the enrolment model actually consumes.

WHY THE ENROLMENT WINDOW AND NOT THE PER-SITE RATE
The obvious target — patients per site per month — turned out to be mostly
arithmetic. Regressing log(rate) on log(site_count) gives a slope near -1 because
sites sit in the denominator, and sponsors add sites precisely because sites are
slow, so the two are chosen together. Ranking countries on it correlates only
0.29 with a size-adjusted ranking. The enrolment WINDOW correlates with site
count at just +0.20 to +0.31, so it is the quantity that carries real signal
rather than a restatement of the sponsor's site count.

Per-country site counts are recoverable for 92.9% of trials from the retained
locations field, and half of trials run in more than one country, so the mix is
real data rather than an assumption.
"""
import logging
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold

log = logging.getLogger(__name__)

#: Countries appearing in at least 60 trials across P1-P3. Anything rarer is
#: pooled into the residual rather than fitted, so a country seen twice cannot
#: acquire an effect.
MODELLED_COUNTRIES: list[str] = [
    "United States", "Japan", "Germany", "China", "Spain", "France", "Canada",
    "Poland", "United Kingdom", "Italy", "Russia", "Australia", "South Korea",
    "Hungary", "Czechia", "Brazil", "Belgium", "Ukraine", "Argentina",
    "Bulgaria", "Taiwan", "Netherlands", "Israel", "India", "Turkey (Türkiye)",
    "Mexico", "Romania", "South Africa", "Sweden", "Austria", "Greece",
    "Slovakia", "Denmark", "Thailand", "Portugal", "Chile", "Finland",
    "New Zealand", "Serbia", "Colombia", "Switzerland", "Malaysia",
    "Philippines", "Georgia", "Peru", "Norway", "Puerto Rico", "Latvia",
    "Lithuania", "Croatia", "Ireland", "Singapore", "Hong Kong", "Estonia",
    "Belarus", "Vietnam", "Slovenia", "Moldova",
]

_COUNTRY_INDEX = {c: i for i, c in enumerate(MODELLED_COUNTRIES)}
SHARE_PREFIX = "csite_"
SHARE_COLUMNS = [f"{SHARE_PREFIX}{c}" for c in MODELLED_COUNTRIES]

#: Trial-size columns handed to the encoder alongside the shares, so it can
#: strip the size effect BEFORE attributing anything to geography. Without this
#: the encoder reads raw target means and inherits the confound: measured
#: like-for-like, country effects on the P3 enrolment window span about 30%,
#: but an unconditioned encoder produced an 86% counterfactual spread.
SIZE_COLUMNS = ["site_count", "Enrollment"]
ENCODER_COLUMNS = SHARE_COLUMNS + SIZE_COLUMNS


def country_site_counts(locations: str | None) -> dict[str, int]:
    """Sites per country, parsed from the encoded locations field."""
    from backend.analytics.site_rates import _parse_locations

    if not locations or pd.isna(locations):
        return {}
    return dict(Counter(c for _, _, c in _parse_locations(locations) if c))


def site_share_frame(df: pd.DataFrame) -> pd.DataFrame:
    """One column per modelled country holding that country's SHARE of sites.

    Shares rather than counts, so the mix is separated from the scale — total
    site count is already its own feature, and leaving counts here would make
    the encoder relearn it.
    """
    n = len(df)
    mat = np.zeros((n, len(MODELLED_COUNTRIES)), dtype=float)

    if "country_sites" in df.columns:
        source = df["country_sites"]
        parse = _decode_counts
    else:
        source = df.get("locations", pd.Series([""] * n, index=df.index))
        parse = country_site_counts

    for row, enc in enumerate(source):
        counts = parse(enc)
        total = sum(counts.values())
        if not total:
            continue
        for country, k in counts.items():
            j = _COUNTRY_INDEX.get(country)
            if j is not None:
                mat[row, j] = k / total

    return pd.DataFrame(mat, columns=SHARE_COLUMNS, index=df.index)


def encode_counts(counts: dict[str, int]) -> str:
    return "|".join(f"{c}:{n}" for c, n in sorted(counts.items()))


def _decode_counts(encoded: str | None) -> dict[str, int]:
    if not encoded or pd.isna(encoded):
        return {}
    out: dict[str, int] = {}
    for part in str(encoded).split("|"):
        if ":" not in part:
            continue
        country, _, n = part.rpartition(":")
        try:
            out[country] = int(n)
        except ValueError:
            continue
    return out


class CountryMixEncoder(BaseEstimator, TransformerMixin):
    """Share-weighted country effect on the target.

    Input  : (n_samples, n_countries) matrix of site shares, rows summing to
             1 where any country was resolved.
    Output : (n_samples, 2) — [share-weighted effect, log1p(evidence)].

    The effect is the country's mean target deviation, shrunk toward the global
    mean by how few trials support it, then blended by the trial's own site
    shares. A trial running 80% of its sites in Poland picks up 80% of Poland's
    effect, so changing the mix changes the prediction — which is the entire
    point and the thing v2 could not do.

    Leakage is controlled the same way as the therapeutic-area encoder: fitted
    inside the pipeline so it only sees the training fold, and out-of-fold within
    that fold so a row is never encoded using its own target.
    """

    def __init__(self, smoothing: float = 12.0, n_splits: int = 5,
                 random_state: int = 42):
        self.smoothing = smoothing
        self.n_splits = n_splits
        self.random_state = random_state

    @staticmethod
    def _split(X: np.ndarray):
        """Separate the share block from the trial-size columns."""
        n_share = len(SHARE_COLUMNS)
        return X[:, :n_share], X[:, n_share:]

    def _residualise(self, size: np.ndarray, y: np.ndarray, fit: bool):
        """Remove the trial-size component of the target.

        Country effects must be what is left AFTER accounting for how big the
        trial is and how many sites it runs, otherwise a country that mostly
        appears in large global studies is scored as slow for that reason alone.
        """
        if size.size == 0:
            return y
        logs = np.log(np.clip(np.nan_to_num(size, nan=1.0), 1.0, None))
        A = np.hstack([logs, np.ones((len(y), 1))])
        if fit:
            self.size_beta_, *_ = np.linalg.lstsq(A, y, rcond=None)
        beta = getattr(self, "size_beta_", None)
        if beta is None:
            return y
        return y - A @ beta

    def _fit_tables(self, X: np.ndarray, y: np.ndarray):
        global_mean = float(np.mean(y))
        n_c = X.shape[1]
        effects = np.zeros(n_c, dtype=float)
        support = np.zeros(n_c, dtype=float)
        for j in range(n_c):
            w = X[:, j]
            mass = float(w.sum())
            if mass <= 0:
                continue
            # Weighted mean of the target across trials with sites in country j,
            # each trial counting in proportion to how much of it sat there.
            wm = float((w * y).sum() / mass)
            effects[j] = ((mass * wm + self.smoothing * global_mean)
                          / (mass + self.smoothing)) - global_mean
            support[j] = mass
        return effects, support, global_mean

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("CountryMixEncoder requires y")
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        shares, size = self._split(X)
        resid = self._residualise(size, y, fit=True)
        self.effects_, self.support_, self.global_ = self._fit_tables(shares, resid)
        self.n_features_in_ = X.shape[1]
        return self

    def _encode(self, X, effects, support):
        total = X.sum(axis=1)
        safe = np.maximum(total, 1e-9)
        eff = (X @ effects) / safe
        sup = (X @ support) / safe
        # Trials whose countries are all outside the modelled set get no effect
        # rather than a spurious zero-weighted one.
        eff = np.where(total > 0, eff, 0.0)
        sup = np.where(total > 0, sup, 0.0)
        return np.column_stack([eff, np.log1p(sup)])

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        shares, _ = self._split(X)
        return self._encode(shares, self.effects_, self.support_)

    def fit_transform(self, X, y=None, **fit_params):
        if y is None:
            raise ValueError("CountryMixEncoder requires y")
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.fit(X, y)
        if len(y) < self.n_splits * 2:
            return self.transform(X)

        shares, size = self._split(X)
        out = np.zeros((len(y), 2), dtype=float)
        kf = KFold(n_splits=self.n_splits, shuffle=True,
                   random_state=self.random_state)
        for tr, te in kf.split(X):
            resid = self._residualise(size[tr], y[tr], fit=True)
            eff, sup, _ = self._fit_tables(shares[tr], resid)
            out[te] = self._encode(shares[te], eff, sup)
        # Restore the full-data tables; the folds above were only for encoding.
        self.fit(X, y)
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array(["country_mix_effect", "country_mix_support"])

    def country_table(self) -> pd.DataFrame:
        """Fitted per-country effects, for inspection and for the UI."""
        return pd.DataFrame({
            "country": MODELLED_COUNTRIES,
            "effect": np.round(self.effects_, 4),
            "site_mass": np.round(self.support_, 1),
        }).sort_values("effect")
