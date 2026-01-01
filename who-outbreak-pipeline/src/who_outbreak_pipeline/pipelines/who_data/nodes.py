import requests
import pandas as pd
from pathlib import Path

# ========================================================================
# 1) FIXED — Fetch FULL historical data
# ========================================================================

def fetch_who_data(indicator_codes: list, output_path: str) -> pd.DataFrame:
    base_url = "https://ghoapi.azureedge.net/api"
    dfs = []

    for code in indicator_codes:
        url = f"{base_url}/{code}"
        print(f"Fetching data from {url} ...")

        response = requests.get(url)
        response.raise_for_status()
        data = response.json().get("value", [])

        df = pd.DataFrame(data)
        df["indicator_code"] = code      # FIXED
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined[combined["SpatialDimType"] == "COUNTRY"]

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_file, index=False)

    print(f"Data saved to {output_file}")
    return combined


# ========================================================================
# 2) FIXED — Fetch FUTURE data (2023–2025)
# ========================================================================

def fetch_future_who_data(indicator_codes: list, start_year: int = 2023) -> pd.DataFrame:
    base_url = "https://ghoapi.azureedge.net/api"
    dfs = []

    print(f"🔮 Fetching future WHO data for years >= {start_year} ...")

    for code in indicator_codes:
        url = f"{base_url}/{code}?$filter=TimeDim ge {start_year}"
        print(f"Fetching future data from {url} ...")

        response = requests.get(url)
        response.raise_for_status()
        data = response.json().get("value", [])

        df = pd.DataFrame(data)
        df["indicator_code"] = code      # FIXED
        dfs.append(df)

    future_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    print(f"📦 Future WHO Data fetched: {future_df.shape}")
    return future_df
