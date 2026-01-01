import pandas as pd
import numpy as np

CONTINENT_MAP = {
    "AFR": "Africa",
    "AMR": "Americas",
    "EMR": "Eastern Mediterranean",
    "EUR": "Europe",
    "SEAR": "South-East Asia",
    "WPR": "Western Pacific",
    # Sometimes WHO uses these short codes too:
    "AFRO": "Africa",
    "AMRO": "Americas",
    "EMRO": "Eastern Mediterranean",
    "EURO": "Europe",
    "SEARO": "South-East Asia",
    "WPRO": "Western Pacific",
}

def engineer_features(df_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Add date, rolling stats, z-scores per indicator, continent, etc.
    Works purely from WHO life-expectancy style indicators.
    """
    print(f"🧪 Feature engineering on: {df_clean.shape}")
    df = df_clean.copy()

    # date: prefer Jan-01 of year (WHO often reports annual)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["date"] = pd.to_datetime(df["year"].astype("Int64").astype(str) + "-01-01", errors="coerce")

    # continent/region label normalization
    df["continent"] = df["region_code"].map(CONTINENT_MAP).fillna(df["region"])

    # group keys
    grp = ["indicator_code", "country_iso3"]

    # sort for rolling
    df = df.sort_values(grp + ["year"])

    # rolling mean (3-year) per country+indicator
    df["value_roll3"] = (
        df.groupby(grp)["value"]
          .transform(lambda s: s.rolling(window=3, min_periods=1).mean())
    )

    # z-score per indicator (global)
    df["value_z_global"] = (
        df.groupby("indicator_code")["value"]
          .transform(lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-9))
    )

    # z-score per indicator & year (cross-section)
    df["value_z_year"] = (
        df.groupby(["indicator_code", "year"])["value"]
          .transform(lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-9))
    )

    # country growth rate vs previous year (pct)
    df["value_prev"] = df.groupby(grp)["value"].shift(1)
    df["value_pct_change"] = ((df["value"] - df["value_prev"]) / (df["value_prev"].replace(0, np.nan))) * 100.0

    # cleanliness flags
    df["has_ci"] = (~df["low"].isna()) & (~df["high"].isna())
    df["ci_width"] = (df["high"] - df["low"]).where(df["has_ci"])

    # quality proxy: inverse of ci width / value magnitude
    df["quality_score"] = (1.0 / (df["ci_width"].abs() + 1e-9)).clip(upper=1e6)
    df.loc[df["ci_width"].isna(), "quality_score"] = np.nan

    # keep a clean feature set
    cols = [
        "id", "indicator_code", "country_iso3", "continent",
        "region_code", "region", "year", "sex",
        "value", "low", "high", "date",
        "value_roll3", "value_z_global", "value_z_year",
        "value_pct_change", "ci_width", "quality_score"
    ]
    df = df[cols].sort_values(["indicator_code", "country_iso3", "year"])
    print(f"✅ Engineered features: {df.shape}")
    return df
