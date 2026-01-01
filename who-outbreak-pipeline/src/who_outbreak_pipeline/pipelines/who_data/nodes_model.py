import json
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# ================================================================
# FEATURE DEFINITIONS
# ================================================================

FEATURES = [
    "year",
    "value_roll3",
    "value_z_global",
    "value_z_year",
    "value_pct_change",
    "ci_width",
    "quality_score",
]

CAT_COLS = [
    "indicator_code",
    "continent",
    "country_iso3",
    "sex",
]


# ================================================================
# PREPARE X AND y (fixed)
# ================================================================

def _prep_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    df = df.copy()

    # Target
    y = df["value"].fillna(0).astype(float) if "value" in df else pd.Series([0] * len(df))

    # Numeric
    X_num = df[FEATURES].copy()

    # One-hot categorical
    X_cat = pd.get_dummies(df[CAT_COLS], drop_first=False, dtype=np.uint8)

    # Final matrix → DO NOT convert to numpy here
    X = pd.concat([X_num, X_cat], axis=1).fillna(0.0)

    return X, y


# ================================================================
# TRAIN MODEL  (fixed: preserves columns)
# ================================================================

def train_model(train_df: pd.DataFrame):
    print(f"🤖 Training model on train_df: {train_df.shape}")

    df = train_df[train_df["value"].notna()].copy()

    X_train, y_train = _prep_xy(df)

    # Save column order
    feature_columns = list(X_train.columns)

    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=None,
        n_jobs=-1,
        random_state=42,
    )

    # Fit using numpy but KEEP columns stored
    model.fit(X_train.to_numpy(), y_train.to_numpy())

    # Store the columns inside model for safety (optional)
    model.feature_names_in_ = feature_columns

    print(f"✅ Model trained on {len(X_train)} samples")

    return model, feature_columns


# ================================================================
# EVALUATE MODEL (fixed)
# ================================================================

def evaluate_model(model, test_df: pd.DataFrame, model_columns):
    print(f"📊 Evaluating model on test_df: {test_df.shape}")

    df = test_df[test_df["value"].notna()].copy()

    X_test, y_test = _prep_xy(df)

    # Align to training columns
    X_test = X_test.reindex(columns=model_columns, fill_value=0)

    y_pred = model.predict(X_test.to_numpy())

    # Metrics
    metrics = {
        "r2": float(r2_score(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "n_test": int(len(X_test)),
    }

    # Importances
    importances = sorted(
        zip(model_columns, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True,
    )

    top_features = [{"feature": f, "importance": float(w)} for f, w in importances[:25]]

    model_info = {
        "metrics": metrics,
        "top_features": top_features,
        "feature_count": len(model_columns),
    }

    df_meta = df[["indicator_code", "country_iso3", "continent", "year", "sex", "value"]].copy()
    df_meta["predicted_value"] = y_pred
    df_meta["error"] = df_meta["value"] - df_meta["predicted_value"]

    print(f"📈 Evaluation complete. R2 = {metrics['r2']:.3f}")

    return model_info, df_meta


# ================================================================
# FUTURE PREDICTION (2023–2025) — fixed
# ================================================================

def predict_future(model, future_df: pd.DataFrame, model_columns):
    print(f"🔮 Predicting future values on: {future_df.shape}")

    df = future_df.copy()

    # Create expanded years: 2023, 2024, 2025
    expanded = []
    for y in [2023, 2024, 2025]:
        temp = df.copy()
        temp["year"] = y
        expanded.append(temp)

    df_expanded = pd.concat(expanded, ignore_index=True)
    print(f"📌 Expanded dataframe: {df_expanded.shape}")

    X_future, _ = _prep_xy(df_expanded)

    # Align to training column order
    X_future = X_future.reindex(columns=model_columns, fill_value=0)

    # Predict
    preds = model.predict(X_future.to_numpy())

    # Confidence estimate (std across trees)
    all_tree_preds = np.stack([tree.predict(X_future.to_numpy()) for tree in model.estimators_])
    pred_std = all_tree_preds.std(axis=0)

    df_expanded["predicted_value"] = preds
    df_expanded["prediction_std"] = pred_std
    df_expanded["prediction_confidence"] = 1 / (1 + pred_std)

    print(f"✅ Future predictions generated: {df_expanded.shape}")

    return df_expanded
