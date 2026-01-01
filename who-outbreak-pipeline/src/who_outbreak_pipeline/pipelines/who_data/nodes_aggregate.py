import pandas as pd

def aggregate_who_data(who_clean_data: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate WHO cleaned data by Country, Region, Year and IndicatorCode.
    Computes average numeric values for each indicator per country/year.
    """
    df = who_clean_data.copy()
    print(f"📊 Aggregating WHO data: {df.shape}")

    # Ensure numeric type for value_numeric
    df["value_numeric"] = pd.to_numeric(df["value_numeric"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    # Define grouping columns dynamically
    group_cols = [col for col in ["region", "country", "indicator_code", "year"] if col in df.columns]

    agg_df = (
        df.groupby(group_cols, dropna=False)["value_numeric"]
        .mean()
        .reset_index()
    )

    print(f"✅ Aggregated WHO data: final shape {agg_df.shape}")
    return agg_df
