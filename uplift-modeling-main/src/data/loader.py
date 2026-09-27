from pathlib import Path
import pandas as pd

EXPECTED_COLUMNS = ["recency", "history_segment", "history", "mens", "womens", "zip_code", "newbie", "channel", "segment", "visit", "conversion", "spend"]

def load_hillstrom(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    validate_schema(df)
    return df

def validate_schema(df: pd.DataFrame) -> None:
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df["conversion"].isna().any() or df["visit"].isna().any():
        raise ValueError("Critical outcomes contain missing values")
    if not set(df["segment"].dropna().unique()).issubset({"No E-Mail", "Mens E-Mail", "Womens E-Mail"}):
        raise ValueError("Unexpected treatment labels found")

def add_binary_treatment(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["treatment"] = (out["segment"] != "No E-Mail").astype(int)
    return out
