import json
import pandas as pd
import plotly.express as px
from pathlib import Path

def _ensure_parent(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def aggregate_who_data(df_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Country-year median per indicator (stable for reporting).
    """
    print(f"📊 Aggregating WHO data: {df_clean.shape}")
    key = ["indicator_code", "country_iso3", "year"]
    out = (
        df_clean.groupby(key, as_index=False)
                .agg(value_median=("value", "median"))
                .sort_values(key)
    )
    print(f"✅ Aggregated WHO data: final shape {out.shape}")
    return out


def summarize_who_trends(
    df_agg: pd.DataFrame,
    model_info: dict,
    preds_df: pd.DataFrame,
    out_html: str,
) -> pd.DataFrame:
    """
    Create interactive HTML + CSV summaries:
     - Time series by indicator (global median)
     - Choropleth by country (latest year per indicator)
     - Save region & country summaries for Streamlit

    Return a summary table (for catalog).
    """
    print(f"📈 Visualizing WHO data: {df_agg.shape}")

    out_html_path = Path(out_html)
    _ensure_parent(out_html_path)

    # ---------- Global median per indicator/year -------------
    ts = (
        df_agg.groupby(["indicator_code", "year"], as_index=False)["value_median"]
             .median()
    )

    fig_ts = px.line(
        ts,
        x="year",
        y="value_median",
        color="indicator_code",
        title="WHO Indicators – Global median over time",
    )

    # ---------- Latest year choropleth per indicator ---------
    latest_per_indicator = (
        df_agg.sort_values(["indicator_code", "year"])
              .groupby("indicator_code", as_index=False)
              .tail(1)
              .copy()
    )
    # attach iso3 for plotly
    latest_per_indicator["iso_alpha"] = latest_per_indicator["country_iso3"]

    fig_map = px.choropleth(
        latest_per_indicator,
        locations="iso_alpha",
        color="value_median",
        hover_name="country_iso3",
        color_continuous_scale="Viridis",
        title="Latest year – country median by indicator",
    )

    # ---------- Model metrics / feature importances ----------
    metrics = model_info.get("metrics", {})
    feature_rows = model_info.get("top_features", [])
    metrics_text = f"""
    <h3>Model metrics</h3>
    <pre>{json.dumps(metrics, indent=2)}</pre>
    <h3>Top features</h3>
    <pre>{json.dumps(feature_rows, indent=2)[:4000]}</pre>
    """

    html = f"""
    <html>
      <head><meta charset="utf-8"></head>
      <body>
        <h2>WHO Indicator Summary</h2>
        <div>{fig_ts.to_html(full_html=False, include_plotlyjs='cdn')}</div>
        <hr/>
        <div>{fig_map.to_html(full_html=False, include_plotlyjs=False)}</div>
        <hr/>
        {metrics_text}
      </body>
    </html>
    """

    out_html_path.write_text(html, encoding="utf-8")
    print(f"✅ Visualization generated: {out_html_path}")

    # Region & country aggregation for Streamlit dashboards
    # (You can use these directly in streamlit)
    # NOTE: If you want region here, join df_clean before aggregating.
    # For now, summarise by country and indicator across years.
    country_summary = (
        df_agg.groupby(["indicator_code", "country_iso3"], as_index=False)
              .agg(median_value=("value_median", "median"),
                   last_year=("year", "max"))
    )

    region_summary = (
        preds_df.groupby(["indicator_code", "continent"], as_index=False)
                .agg(mean_predicted=("predicted_value", "mean"),
                     mean_actual=("value", "mean"))
    )

    # Return one summary (others saved via catalog)
    return country_summary, region_summary
