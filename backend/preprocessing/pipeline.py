"""Build and apply the sklearn feature pipeline."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from backend.constants import THERAPEUTIC_AREAS, REGIONS
from backend.preprocessing.features import (
    assign_therapeutic_area,
    assign_region,
    classify_sad_mad,
    normalise_phase,
    one_hot_pipe_col,
)

_CAT_COLS = ["Drug_Type", "Allocation", "Intervention_Model", "Masking", "Primary_Purpose"]
_NUM_COLS = ["Enrollment", "site_count", "primary_completion_year",
             "total_primary_outcomes", "total_secondary_outcomes", "number_of_arms"]
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

    # SAD/MAD (P1 only — set to "None" for other phases)
    if phase_key in ("P1", "P1HV"):
        df["sad_mad"] = df["brief_summary"].apply(classify_sad_mad)
    else:
        df["sad_mad"] = "None"

    # Healthy volunteer flag (already in 'is_hv' from data layer)
    df["is_hv"] = df.get("is_hv", 0).fillna(0).astype(int)
    df["has_collaborators"] = df.get("has_collaborators", 0).fillna(0).astype(int)

    # Phase numeric (needed by some models)
    df["phase_num"] = df["Phases"].apply(normalise_phase)

    # Tidy categoricals
    for col in _CAT_COLS:
        if col not in df.columns:
            df[col] = "UNKNOWN"
        df[col] = df[col].fillna("UNKNOWN").astype(str)

    # Numeric defaults
    df["site_count"] = df.get("site_count", 1).fillna(1)
    for col in _NUM_COLS:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    X = pd.concat([
        df[_CAT_COLS + _NUM_COLS + ["is_hv", "has_collaborators"]].reset_index(drop=True),
        ta_ohe.reset_index(drop=True),
        re_ohe.reset_index(drop=True),
    ], axis=1)

    return X


def make_preprocessor() -> ColumnTransformer:
    """Return an unfitted sklearn preprocessor matching build_features output."""
    ohe_cols = _CAT_COLS
    num_cols = _NUM_COLS
    passthrough_cols = ["is_hv", "has_collaborators"] + _BIN_TA + _BIN_RE

    return ColumnTransformer(
        transformers=[
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(sparse_output=False, handle_unknown="ignore")),
            ]), ohe_cols),
            ("num", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), num_cols),
            ("bin", "passthrough", passthrough_cols),
        ],
        remainder="drop",
    )
