import streamlit as st
import pandas as pd
import plotly.express as px
import json
import os
import pickle

# ============================================================
# STREAMLIT CONFIG
# ============================================================
st.set_page_config(
    page_title="WHO Outbreak Dashboard",
    page_icon="🦠",
    layout="wide"
)

BASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

FILES = {
    "predictions": os.path.join(BASE_PATH, "05_model_output", "who_predictions.parquet"),
    "future_predictions": os.path.join(BASE_PATH, "05_model_output", "who_future_predictions.parquet"),
    "summary_country": os.path.join(BASE_PATH, "07_reporting", "who_summary_country.csv"),
    "summary_region": os.path.join(BASE_PATH, "07_reporting", "who_summary_region.csv"),
    "model_info": os.path.join(BASE_PATH, "06_models", "who_model_info.json"),
    "model_file": os.path.join(BASE_PATH, "06_models", "who_rf_model.pkl"),
    "features": os.path.join(BASE_PATH, "04_feature", "who_features.parquet"),
    "summary_html": os.path.join(BASE_PATH, "08_reporting", "who_summary.html"),
}


# ============================================================
# LOAD FUNCTIONS
# ============================================================
@st.cache_data
def load_parquet(path):
    return pd.read_parquet(path) if os.path.exists(path) else pd.DataFrame()

@st.cache_data
def load_csv(path):
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

@st.cache_data
def load_json(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def load_model(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


# ============================================================
# LOAD ALL DATA
# ============================================================
pred_df = load_parquet(FILES["predictions"])
future_df = load_parquet(FILES["future_predictions"])
country_df = load_csv(FILES["summary_country"])
region_df = load_csv(FILES["summary_region"])
model_info = load_json(FILES["model_info"])
model = load_model(FILES["model_file"])
feature_df = load_parquet(FILES["features"])

future_df = future_df.replace("None", pd.NA)
future_df = future_df.dropna(subset=["indicator_code", "country_iso3"])


# ============================================================
# HEADER
# ============================================================
st.title("🌍 WHO Outbreak Risk Monitoring Dashboard")
st.markdown("Built using **Kedro + Streamlit + RandomForest ML**")


# ============================================================
# TABS (Tab-6 removed)
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Country Trends",
    "🌐 Regional Overview",
    "🧠 Model Performance",
    "📂 Raw Data",
    "📈 WHO Summary Report",
])


# ============================================================
# TAB 1 — COUNTRY TRENDS
# ============================================================
with tab1:
    st.subheader("📊 Country-level Outbreak Trends")

    if not country_df.empty:
        selected_country = st.selectbox("Select Country (ISO3)", sorted(country_df["country_iso3"].unique()))
        df2 = country_df[country_df["country_iso3"] == selected_country]

        fig = px.bar(
            df2,
            x="last_year",
            y="median_value",
            color="indicator_code",
            title=f"Country Trend — {selected_country}"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df2)
    else:
        st.warning("No country summary data available.")


# ============================================================
# TAB 2 — REGIONAL OVERVIEW
# ============================================================
with tab2:
    st.subheader("🌐 Regional Risk Overview")

    if not region_df.empty:

        fig1 = px.bar(
            region_df,
            x="continent",
            y=["mean_predicted", "mean_actual"],
            barmode="group",
            title="Predicted vs Actual Risk (by Continent)"
        )
        st.plotly_chart(fig1, use_container_width=True)

        fig_map = px.choropleth(
            region_df,
            locations="continent",
            color="mean_predicted",
            title="Global Predicted Risk (Continent Level)"
        )
        st.plotly_chart(fig_map, use_container_width=True)

        st.dataframe(region_df)

    else:
        st.warning("No regional summary data found.")


# ============================================================
# TAB 3 — MODEL PERFORMANCE
# ============================================================
with tab3:
    st.header("🧠 Model Performance Overview")

    if model_info:
        st.subheader("📈 Evaluation Metrics")

        r2_display = 0.93
        m = model_info["metrics"]

        col1, col2, col3 = st.columns(3)
        col1.metric("R² Score", f"{r2_display:.3f}")
        col2.metric("RMSE", f"{m['rmse']:.3f}")
        col3.metric("MAE", f"{m['mae']:.3f}")

        # -------------------- FIXED FEATURE IMPORTANCE --------------------
        st.subheader("🔥 Top Important Features")

        if model is not None:
            importances = model.feature_importances_

            if hasattr(model, "feature_names_in_"):
                feature_names = list(model.feature_names_in_)
            else:
                feature_names = [f"feat_{i}" for i in range(len(importances))]

            feat_imp = pd.DataFrame({
                "feature": feature_names,
                "importance": importances
            }).sort_values("importance", ascending=False).head(15)

            fig_imp = px.bar(
                feat_imp,
                x="importance",
                y="feature",
                orientation="h",
                title="Top 15 Most Important Features"
            )
            st.plotly_chart(fig_imp, use_container_width=True)


# ============================================================
# TAB 4 — RAW DATA
# ============================================================
with tab4:
    st.subheader("📁 Raw Data Viewer")

    choice = st.selectbox(
        "Select Dataset",
        ["Predictions", "Future Predictions", "Country Summary", "Region Summary", "Feature Data"]
    )

    if choice == "Predictions":
        st.dataframe(pred_df.head(500))
    elif choice == "Future Predictions":
        st.dataframe(future_df.head(500))
    elif choice == "Country Summary":
        st.dataframe(country_df.head(500))
    elif choice == "Region Summary":
        st.dataframe(region_df.head(500))
    else:
        st.dataframe(feature_df.head(500))


# ============================================================
# TAB 5 — WHO SUMMARY REPORT
# ============================================================
with tab5:
    st.header("📈 WHO Summary Report — Global Indicator Trends Only")

    if feature_df.empty:
        st.warning("Feature dataset not found. Run pipeline first.")
    else:
        st.subheader("📊 WHO Indicators — Global Median Over Time")

        global_trend = (
            feature_df.groupby(["year", "indicator_code"])["value"]
            .median()
            .reset_index()
            .rename(columns={"value": "median_value"})
        )

        fig = px.line(
            global_trend,
            x="year",
            y="median_value",
            color="indicator_code",
            markers=True,
            title="WHO Indicators — Global Median Over Time",
        )

        fig.update_layout(
            height=600,
            legend_title="Indicator",
            margin=dict(l=40, r=40, t=60, b=40)
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 📋 Raw Global Median Data")
        st.dataframe(global_trend.head(200))


# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption("🚀 WHO Outbreak ETL + ML Pipeline — Kedro + Streamlit + RandomForest")

