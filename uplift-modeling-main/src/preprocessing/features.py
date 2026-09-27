import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

FEATURES = ["recency", "history_segment", "history", "mens", "womens", "zip_code", "newbie", "channel"]
CATEGORICAL = ["history_segment", "zip_code", "channel"]
NUMERICAL = [c for c in FEATURES if c not in CATEGORICAL]

def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ("num", "passthrough", NUMERICAL),
    ], remainder="drop")

def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df[FEATURES].copy()
