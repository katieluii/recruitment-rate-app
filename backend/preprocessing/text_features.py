from __future__ import annotations
"""Eligibility-criteria text and sponsor identity as features (v3.4).

Two additions, both cheap and both suggested by the literature rather than by
guesswork:

* **Criteria wording.** We already count inclusion and exclusion bullets, but
  TrialEnroll (2024, 31k trials) shows the WORDING carries signal a count cannot
  see — "prior systemic therapy", "washout", "treatment-naive" and "ECOG 0-1"
  narrow an eligible population in ways that ten bullets and two bullets do not
  distinguish.
* **Sponsor identity.** The 2025 duration-prediction survey ranks sponsor as its
  second strongest driver, behind indication. We already collect it and have
  never used it.

WHY HASHED N-GRAMS AND NOT SENTENCE EMBEDDINGS
A transformer would need a model download, would put a network dependency in the
training path, and would make runs non-reproducible across environments. Hashed
character and word n-grams reduced by truncated SVD are deterministic, run
offline, and are testable in the existing harness. If the harness says the text
carries signal and this representation is the bottleneck, that is the moment to
reach for embeddings — not before.
"""
import logging
import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

log = logging.getLogger(__name__)

N_TEXT_COMPONENTS = 12

#: Restrictiveness markers worth counting explicitly rather than leaving to the
#: SVD to rediscover. Each narrows the eligible pool in a way clinical teams
#: would recognise, so a hit is interpretable when it shows up in provenance.
CRITERIA_MARKERS: dict[str, str] = {
    "prior_therapy": r"prior (?:systemic |anti-?cancer |chemo)?(?:therapy|treatment|line)",
    "treatment_naive": r"treatment[- ]na(?:i|ï)ve|previously untreated|no prior",
    "washout": r"washout|wash[- ]out",
    "biomarker_required": r"(?:positive|mutation|amplification|expression)\s*(?:status)?\b|"
                          r"\bher2\b|\begfr\b|\bpd-?l1\b|\bbrca\b|\balk\b",
    "performance_status": r"\becog\b|karnofsky|performance status",
    "organ_function": r"(?:adequate|normal)\s+(?:organ|renal|hepatic|bone marrow|cardiac)",
    "contraception": r"contracept|childbearing potential",
    "pregnancy_excluded": r"pregnan|lactat|breast[- ]?feed",
    "comorbidity_excluded": r"uncontrolled|clinically significant|history of",
    "hospitalised": r"hospitali[sz]|inpatient",
}


def marker_frame(text: pd.Series) -> pd.DataFrame:
    """Binary hit per restrictiveness marker."""
    t = text.fillna("").astype(str).str.lower()
    return pd.DataFrame(
        {f"crit_{name}": t.str.contains(pat, regex=True, na=False).astype(int)
         for name, pat in CRITERIA_MARKERS.items()},
        index=text.index,
    )


def sponsor_tier(sponsor: pd.Series, top: set[str] | None = None) -> pd.Series:
    """Large-cap sponsor, other named industry sponsor, or unknown.

    Deliberately coarse. Fitting an effect per sponsor across 610 distinct names
    would give most of them a handful of trials each — the same thin-cell trap
    that made the country effects unidentifiable.
    """
    s = sponsor.fillna("").astype(str)
    if top is None:
        return pd.Series(np.where(s.str.strip() == "", "UNKNOWN", "OTHER"),
                         index=sponsor.index)
    return pd.Series(
        np.where(s.isin(top), "LARGE_CAP",
                 np.where(s.str.strip() == "", "UNKNOWN", "OTHER")),
        index=sponsor.index)


def top_sponsors(df: pd.DataFrame, min_trials: int = 25) -> set[str]:
    if "lead_sponsor" not in df.columns:
        return set()
    counts = df["lead_sponsor"].fillna("").value_counts()
    return set(counts[counts >= min_trials].index) - {""}


class CriteriaTextEncoder(BaseEstimator, TransformerMixin):
    """TF-IDF over eligibility text, reduced to a handful of dense components.

    Fitted inside the sklearn pipeline so the vocabulary comes from the training
    fold only — a vocabulary built on all data would let test-fold wording leak
    into training.
    """

    def __init__(self, n_components: int = N_TEXT_COMPONENTS,
                 max_features: int = 20000, random_state: int = 42):
        self.n_components = n_components
        self.max_features = max_features
        self.random_state = random_state

    def _texts(self, X) -> list[str]:
        if isinstance(X, pd.DataFrame):
            col = X.columns[0]
            return X[col].fillna("").astype(str).tolist()
        return pd.Series(np.asarray(X).ravel()).fillna("").astype(str).tolist()

    def fit(self, X, y=None):
        texts = self._texts(X)
        non_empty = sum(1 for t in texts if t.strip())
        if non_empty < 50:
            log.warning("Only %d non-empty criteria texts — text encoder disabled",
                        non_empty)
            self.pipe_ = None
            return self
        n_comp = min(self.n_components, max(2, non_empty - 1))
        self.pipe_ = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=self.max_features, ngram_range=(1, 2),
                min_df=5, max_df=0.85, strip_accents="unicode",
                lowercase=True, stop_words="english")),
            ("svd", TruncatedSVD(n_components=n_comp,
                                 random_state=self.random_state)),
        ])
        self.pipe_.fit(texts)
        self.n_out_ = n_comp
        return self

    def transform(self, X):
        texts = self._texts(X)
        if self.pipe_ is None:
            return np.zeros((len(texts), 1), dtype=float)
        return self.pipe_.transform(texts)

    def get_feature_names_out(self, input_features=None):
        n = getattr(self, "n_out_", 1)
        return np.array([f"crit_svd_{i}" for i in range(n)])


def top_terms(encoder: CriteriaTextEncoder, component: int = 0, k: int = 8):
    """Highest-weighted terms in a component — used by the provenance surface so
    a text feature is inspectable rather than a black box."""
    if encoder.pipe_ is None:
        return []
    tfidf = encoder.pipe_.named_steps["tfidf"]
    svd = encoder.pipe_.named_steps["svd"]
    names = np.array(tfidf.get_feature_names_out())
    weights = svd.components_[component]
    idx = np.argsort(-np.abs(weights))[:k]
    return [(str(names[i]), round(float(weights[i]), 4)) for i in idx]
