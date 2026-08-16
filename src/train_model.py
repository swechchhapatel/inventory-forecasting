"""
Retail Demand Forecasting - Model Training Script
===================================================
Run this in Kaggle Notebook or Google Colab where the dataset is available:
https://www.kaggle.com/datasets/atomicd/retail-store-inventory-and-demand-forecasting

WHAT THIS SCRIPT DOES:
1. Loads the raw CSV
2. Engineers time-series features (lags, rolling stats, calendar features)
3. Encodes categoricals
4. Trains an XGBoost regressor to predict `Demand`
5. Evaluates the model (MAE, RMSE, MAPE, R2)
6. Saves the trained model + encoders + feature list + processed data
   -> these artifacts get downloaded and used later by the Streamlit app

HOW TO RUN IN KAGGLE:
- Add the dataset to your notebook (Add Data -> search "Retail Store Inventory and
  Demand Forecasting")
- Update DATA_PATH below to the path Kaggle mounts it at, e.g.
  "/kaggle/input/retail-store-inventory-and-demand-forecasting/retail_store_inventory.csv"
- Run all cells
- Download the /kaggle/working/artifacts folder (zip it) and bring it to your
  local machine / wherever you run the Streamlit app
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

# -----------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------
DATA_PATH = "data\\sales_data.csv"

ARTIFACT_DIR = "artifacts"   # change to "artifacts" if running locally
os.makedirs(ARTIFACT_DIR, exist_ok=True)

TARGET = "Demand"
DATE_COL = "Date"
GROUP_COLS = ["Store ID", "Product ID"]

RANDOM_STATE = 42
TEST_SIZE_DAYS = 30   # last 30 days per store-item held out as test set (time-based split)


# -----------------------------------------------------------------------
# 1. LOAD DATA
# -----------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(GROUP_COLS + [DATE_COL]).reset_index(drop=True)
    print(f"Loaded {len(df):,} rows | {df[DATE_COL].min().date()} to {df[DATE_COL].max().date()}")
    print(f"Stores: {df['Store ID'].nunique()} | Products: {df['Product ID'].nunique()}")
    return df


# -----------------------------------------------------------------------
# 2. FEATURE ENGINEERING
# -----------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ---- Calendar features ----
    df["day_of_week"] = df[DATE_COL].dt.dayofweek
    df["day_of_month"] = df[DATE_COL].dt.day
    df["month"] = df[DATE_COL].dt.month
    df["week_of_year"] = df[DATE_COL].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["quarter"] = df[DATE_COL].dt.quarter

    # ---- Price-derived features ----
    df["price_gap_vs_competitor"] = df["Price"] - df["Competitor Pricing"]
    df["price_gap_pct"] = df["price_gap_vs_competitor"] / df["Price"].replace(0, np.nan)
    df["discounted_price"] = df["Price"] * (1 - df["Discount"] / 100.0)

    # ---- Inventory-derived features ----
    df["stockout_flag"] = (df["Inventory Level"] <= 0).astype(int)
    df["inventory_to_sales_ratio"] = df["Inventory Level"] / df["Units Sold"].replace(0, np.nan)

    # ---- Group-wise lag & rolling features (per store-item time series) ----
    g = df.groupby(GROUP_COLS, group_keys=False)

    for lag in [1, 7, 14]:
        df[f"demand_lag_{lag}"] = g[TARGET].shift(lag)
        df[f"units_sold_lag_{lag}"] = g["Units Sold"].shift(lag)

    for window in [7, 14, 30]:
        df[f"demand_rollmean_{window}"] = (
            g[TARGET].shift(1).rolling(window, min_periods=1).mean().reset_index(drop=True)
        )
        df[f"demand_rollstd_{window}"] = (
            g[TARGET].shift(1).rolling(window, min_periods=1).std().reset_index(drop=True)
        )

    df["units_sold_rollmean_7"] = (
        g["Units Sold"].shift(1).rolling(7, min_periods=1).mean().reset_index(drop=True)
    )

    # Trend: difference between short and long rolling mean (momentum indicator)
    df["demand_trend"] = df["demand_rollmean_7"] - df["demand_rollmean_30"]

    # ---- Fill NaNs created by lag/rolling at series start ----
    lag_roll_cols = [c for c in df.columns if "lag" in c or "roll" in c or c == "demand_trend"]
    df[lag_roll_cols] = df[lag_roll_cols].fillna(0)
    df["inventory_to_sales_ratio"] = df["inventory_to_sales_ratio"].fillna(0)
    df["price_gap_pct"] = df["price_gap_pct"].fillna(0)

    return df


# -----------------------------------------------------------------------
# 3. ENCODE CATEGORICALS
# -----------------------------------------------------------------------
def encode_categoricals(df: pd.DataFrame, cat_cols):
    encoders = {}
    df = df.copy()
    for col in cat_cols:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return df, encoders


# -----------------------------------------------------------------------
# 4. TIME-BASED TRAIN/TEST SPLIT
# -----------------------------------------------------------------------
def time_based_split(df: pd.DataFrame, date_col: str, test_days: int):
    cutoff = df[date_col].max() - pd.Timedelta(days=test_days)
    train = df[df[date_col] <= cutoff]
    test = df[df[date_col] > cutoff]
    print(f"Train: {len(train):,} rows (up to {cutoff.date()})")
    print(f"Test:  {len(test):,} rows (after {cutoff.date()})")
    return train, test


# -----------------------------------------------------------------------
# 5. TRAIN MODEL
# -----------------------------------------------------------------------
def train_xgboost(X_train, y_train, X_val, y_val):
    model = xgb.XGBRegressor(
        n_estimators=800,
        max_depth=7,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        early_stopping_rounds=50,
        eval_metric="mae",
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )
    return model


# -----------------------------------------------------------------------
# 6. EVALUATE
# -----------------------------------------------------------------------
def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    preds = np.clip(preds, 0, None)  # demand can't be negative

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    # MAPE, avoiding div-by-zero
    mask = y_test != 0
    mape = np.mean(np.abs((y_test[mask] - preds[mask]) / y_test[mask])) * 100

    print("\n" + "=" * 40)
    print("MODEL EVALUATION (held-out test set)")
    print("=" * 40)
    print(f"MAE:  {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"R2:   {r2:.4f}")
    print("=" * 40)

    return {"mae": mae, "rmse": rmse, "mape": mape, "r2": r2}, preds


# -----------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------
def main():
    df = load_data(DATA_PATH)
    df = engineer_features(df)

    cat_cols = ["Store ID", "Product ID", "Category", "Region",
                "Weather Condition", "Seasonality"]
    df, encoders = encode_categoricals(df, cat_cols)

    feature_cols = [
        "Inventory Level", "Units Sold", "Units Ordered", "Price", "Discount",
        "Promotion", "Competitor Pricing", "Epidemic",
        "day_of_week", "day_of_month", "month", "week_of_year", "is_weekend", "quarter",
        "price_gap_vs_competitor", "price_gap_pct", "discounted_price",
        "stockout_flag", "inventory_to_sales_ratio",
        "demand_lag_1", "demand_lag_7", "demand_lag_14",
        "units_sold_lag_1", "units_sold_lag_7", "units_sold_lag_14",
        "demand_rollmean_7", "demand_rollmean_14", "demand_rollmean_30",
        "demand_rollstd_7", "demand_rollstd_14", "demand_rollstd_30",
        "units_sold_rollmean_7", "demand_trend",
    ] + [c + "_enc" for c in cat_cols]

    train_df, test_df = time_based_split(df, DATE_COL, TEST_SIZE_DAYS)

    # further split train into train/val for early stopping
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=RANDOM_STATE)

    X_train, y_train = train_df[feature_cols], train_df[TARGET]
    X_val, y_val = val_df[feature_cols], val_df[TARGET]
    X_test, y_test = test_df[feature_cols], test_df[TARGET]

    model = train_xgboost(X_train, y_train, X_val, y_val)
    metrics, preds = evaluate(model, X_test, y_test)

    # ---- Feature importance ----
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    print("\nTop 10 features:")
    print(importance.head(10).to_string(index=False))

    # -----------------------------------------------------------------
    # SAVE ARTIFACTS
    # -----------------------------------------------------------------
    model.save_model(f"{ARTIFACT_DIR}/xgb_demand_model.json")
    joblib.dump(encoders, f"{ARTIFACT_DIR}/label_encoders.pkl")

    with open(f"{ARTIFACT_DIR}/feature_cols.json", "w") as f:
        json.dump(feature_cols, f)

    with open(f"{ARTIFACT_DIR}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    importance.to_csv(f"{ARTIFACT_DIR}/feature_importance.csv", index=False)

    # Save the FULL engineered dataset (with predictions on test period attached)
    # This is what the Streamlit app will use to bootstrap history + lag features
    # for any new uploaded CSV that doesn't have enough history of its own.
    df.to_parquet(f"{ARTIFACT_DIR}/historical_engineered.parquet", index=False)

    test_df = test_df.copy()
    test_df["Predicted_Demand"] = preds
    test_df.to_csv(f"{ARTIFACT_DIR}/test_predictions_sample.csv", index=False)

    print(f"\nAll artifacts saved to: {ARTIFACT_DIR}")
    print("Files: xgb_demand_model.json, label_encoders.pkl, feature_cols.json,")
    print("       metrics.json, feature_importance.csv, historical_engineered.parquet,")
    print("       test_predictions_sample.csv")


if __name__ == "__main__":
    main()
