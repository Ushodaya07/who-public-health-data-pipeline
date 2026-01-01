import pandas as pd
import numpy as np

# ----------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------

def _safe_str(x):
    try:
        return str(x)
    except Exception:
        return ""


def _coerce_num(s):
    """Convert WHO value like '78.1 [78.1-78.2]' -> 78.1"""
    if pd.isna(s):
        return np.nan
    if isinstance(s, (int, float, np.number)):
        return float(s)

    import re
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(s))
    return float(m.group(0)) if m else np.nan


def _to_int(x):
    try:
        return int(x)
    except Exception:
        return np.nan


# ----------------------------------------------------------------------
# CLEAN WHO DATA — FIXED
# ----------------------------------------------------------------------

def clean_who_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Clean WHO historical and future data to a consistent schema.
    """
    print(f"🧹 Cleaning WHO data: initial shape {df_raw.shape}")

    df = df_raw.copy()
    df.columns = [c.strip() for c in df.columns]

    # ----------------------------------------------------
    # FIX 1 — Avoid duplicate IndicatorCode
    # ----------------------------------------------------
    # If API created both "IndicatorCode" AND "indicator_code"
    if "IndicatorCode" in df.columns and "indicator_code" in df.columns:
        df.drop(columns=["IndicatorCode"], inplace=True)

    # If only "IndicatorCode" is present, rename it
    if "IndicatorCode" in df.columns:
        df.rename(columns={"IndicatorCode": "indicator_code"}, inplace=True)

    # Ensure no duplicate columns remain
    df = df.loc[:, ~df.columns.duplicated()]

    # ----------------------------------------------------
    # Keep relevant columns only
    # ----------------------------------------------------
    important = [
        "id", "IndicatorCode", "indicator_code",
        "SpatialDim", "SpatialDimType",
        "ParentLocationCode", "ParentLocation",
        "TimeDim", "TimeDimType",
        "Dim1", "Dim1Type",
        "Value", "NumericValue", "Low", "High",
        "Date", "TimeDimensionValue"
    ]
    existing = [c for c in important if c in df.columns]
    df = df[existing].copy()

    # ----------------------------------------------------
    # Rename to clean names
    # ----------------------------------------------------
    df.rename(columns={
        "SpatialDim": "country_iso3",
        "ParentLocationCode": "region_code",
        "ParentLocation": "region",
        "TimeDim": "year",
        "Dim1": "sex",
        "Value": "value_text",
        "NumericValue": "value_numeric",
        "Low": "low",
        "High": "high",
        "Date": "date_reported",
    }, inplace=True)

    # Ensure indicator_code exists always
    if "indicator_code" not in df.columns:
        df["indicator_code"] = ""

    # ----------------------------------------------------
    # Clean string columns
    # ----------------------------------------------------
    for col in ["indicator_code", "country_iso3", "region_code", "region", "sex"]:
        if col in df.columns:
            df[col] = df[col].map(_safe_str).str.strip()
        else:
            df[col] = ""

    # ----------------------------------------------------
    # Convert year
    # ----------------------------------------------------
    if "year" in df.columns:
        df["year"] = df["year"].apply(_to_int)

    # ----------------------------------------------------
    # Numeric value cleanup
    # ----------------------------------------------------
    if "value_numeric" in df.columns and df["value_numeric"].notna().any():
        df["value"] = pd.to_numeric(df["value_numeric"], errors="coerce")
    else:
        df["value"] = df["value_text"].apply(_coerce_num)

    df["low"] = pd.to_numeric(df.get("low", np.nan), errors="coerce")
    df["high"] = pd.to_numeric(df.get("high", np.nan), errors="coerce")

    # ----------------------------------------------------
    # Date
    # ----------------------------------------------------
    df["date_reported"] = pd.to_datetime(df.get("date_reported", pd.NaT), errors="coerce")

    # ----------------------------------------------------
    # Filter invalid ISO3
    # ----------------------------------------------------
    df = df[df["country_iso3"].str.len() == 3]

    # Remove missing values
    df = df[df["indicator_code"] != ""]
    df = df[df["value"].notna()]

    # ----------------------------------------------------
    # Final schema
    # ----------------------------------------------------
    expected_cols = [
        "id", "indicator_code", "region_code", "region",
        "country_iso3", "year", "sex",
        "value", "low", "high", "date_reported"
    ]
    df = df.reindex(columns=expected_cols)

    df = df.drop_duplicates()

    print(f"✅ Cleaned WHO data: {df.shape}")
    return df


# ----------------------------------------------------------------------
# SPLIT FUNCTION (REQUIRED BY PIPELINE)
# ----------------------------------------------------------------------

def split_by_year(df: pd.DataFrame):
    """
    Train: <= 2017  
    Test: 2018–2022
    """
    print("📆 Splitting data...")

    df["year"] = df["year"].astype(int)

    train_df = df[df["year"] <= 2017].copy()
    test_df = df[(df["year"] >= 2018) & (df["year"] <= 2022)].copy()

    print(f"📘 Train: {train_df.shape}")
    print(f"📙 Test: {test_df.shape}")

    return train_df, test_df
