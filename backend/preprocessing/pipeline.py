from __future__ import annotations
"""Build and apply the sklearn feature pipeline."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from backend.constants import THERAPEUTIC_AREAS, REGIONS
from backend.preprocessing.endpoints import ARCHETYPES, add_endpoint_features
from backend.preprocessing.target_encoding import TATargetEncoder
from backend.preprocessing.features import (
    assign_therapeutic_area,
    assign_region,
    classify_sad_mad,
    one_hot_pipe_col,
)

_CAT_COLS = ["Drug_Type", "Allocation", "Intervention_Model", "Masking",
             "Primary_Purpose", "Sex", "sad_mad", "endpoint_archetype"]

#: Presence flags per endpoint archetype, e.g. endpoint_has_SURVIVAL.
_ENDPOINT_FLAGS = [f"endpoint_has_{a}" for a in ARCHETYPES if a != "UNKNOWN"]

# NOTE ON CALENDAR YEAR — deliberately absent.
# v1 used `primary_completion_year`, which is the label's own endpoint. On a
# temporal holdout that produced a +25 month bias on Phase 2 (MAE 25.4 vs 8.9
# without it): every test trial has a completion year above the trained range,
# so the forest, which cannot extrapolate, predicted uniformly long.
# `start_year` has the same defect in deployment — a trial being quoted today
# starts later than anything in the training set. Era effects belong in the
# training window or a recency weight, not in a tree split on a year.
_NUM_COLS = [
    "Enrollment",
    "site_count",             # real site count as of Phase 2, not country count
    "country_count",
    "total_primary_outcomes",
    "total_secondary_outcomes",
    "number_of_arms",
    "followup_months",        # parsed from the primary outcome time frame
    "min_age_years",
    "age_span_years",
    "criteria_chars",
    "n_inclusion_criteria",
    "n_exclusion_criteria",
    "n_collaborators",
]

# Derived ratios — the tree has to spend splits to discover these otherwise.
_RATIO_COLS = ["enrollment_per_site", "outcomes_total"]

_BIN_TA = THERAPEUTIC_AREAS
_BIN_RE = REGIONS


def build_features(df: pd.DataFrame, phase_key: str) -> pd.DataFrame:
    """Apply all feature engineering steps and return a model-ready DataFrame."""
    df = df.copy()

    # Therapeutic area + region (pipe-separated → one-hot)
    # At inference time, conditions_str IS already a canonical TA label — bypass keyword mapping.
    df["Therapeutic_Area"] = df["conditions"].apply(
        lambda c: c if c in THERAPEUTIC_AREAS else assign_therapeutic_area(c)
    )
    df["Region"] = df["countries"].apply(assign_region)

    ta_ohe = one_hot_pipe_col(df, "Therapeutic_Area", THERAPEUTIC_AREAS)
    re_ohe = one_hot_pipe_col(df, "Region", REGIONS)

    # Endpoint archetype + per-archetype flags. On Phase 3 this spans 29 months
    # of median duration on its own (immunogenicity 10.6 → survival 39.6), which
    # is the follow-up half of the target that therapeutic area alone cannot see.
    df = add_endpoint_features(df)

    # SAD/MAD (P1 only — set to "None" for other phases). v1 computed this and
    # then never added it to X; it is a real categorical now.
    if phase_key in ("P1", "P1HV"):
        df["sad_mad"] = df["brief_summary"].apply(classify_sad_mad)
    else:
        df["sad_mad"] = "None"

    # `is_hv` is deliberately NOT a feature: the trainer filters each phase on it,
    # so it is constant within every model and carries zero information.
    # df.get returns a bare scalar when the column is absent, so build the
    # Series explicitly rather than chaining .fillna off the default.
    if "has_collaborators" not in df.columns:
        df["has_collaborators"] = 0
    df["has_collaborators"] = (
        pd.to_numeric(df["has_collaborators"], errors="coerce").fillna(0).astype(int)
    )

    # Tidy categoricals
    for col in _CAT_COLS:
        if col not in df.columns:
            df[col] = "UNKNOWN"
        df[col] = df[col].fillna("UNKNOWN").astype(str)

    # Numeric defaults. NaN is left in place for the imputer to handle rather
    # than being flattened to 0 — "no maximum age stated" is not "max age 0".
    for col in _NUM_COLS:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["enrollment_per_site"] = df["Enrollment"] / df["site_count"].replace(0, np.nan)
    df["outcomes_total"] = (
        df["total_primary_outcomes"].fillna(0) + df["total_secondary_outcomes"].fillna(0)
    )

    for col in _ENDPOINT_FLAGS:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    X = pd.concat([
        df[_CAT_COLS + _NUM_COLS + _RATIO_COLS + _ENDPOINT_FLAGS
           + ["has_collaborators"]].reset_index(drop=True),
        ta_ohe.reset_index(drop=True),
        re_ohe.reset_index(drop=True),
    ], axis=1)

    return X


def make_preprocessor(ta_target_encoding: bool = True) -> ColumnTransformer:
    """Return an unfitted sklearn preprocessor matching build_features output.

    The therapeutic-area block is fed to BOTH the target encoder and the
    passthrough. The encoder supplies the strong continuous signal the trees
    will actually split on; the raw binaries stay so the model can still pick
    out an area whose behaviour is not captured by its median alone.
    """
    ohe_cols = _CAT_COLS
    num_cols = _NUM_COLS + _RATIO_COLS
    passthrough_cols = (
        ["has_collaborators"] + _ENDPOINT_FLAGS + _BIN_TA + _BIN_RE
    )

    transformers = [
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(sparse_output=False, handle_unknown="ignore")),
        ]), ohe_cols),
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), num_cols),
        ("bin", "passthrough", passthrough_cols),
    ]
    if ta_target_encoding:
        transformers.append(("ta_target", TATargetEncoder(), _BIN_TA))

    return ColumnTransformer(transformers=transformers, remainder="drop")
