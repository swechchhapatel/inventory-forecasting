"""
Inference utilities shared by the Streamlit app.
Mirrors the feature engineering done in train_model.py so that predictions
on new uploaded CSVs are consistent with how the model was trained.
"""

import pandas as pd
import numpy as np

DATE_COL = "Date"
GROUP_COLS = ["Store ID", "Product ID"]
TARGET = "Demand"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Same logic as train_model.engineer_features — kept in sync manually.
    If you change one, change the other."""
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(GROUP_COLS + [DATE_COL]).reset_index(drop=True)

    df["day_of_week"] = df[DATE_COL].dt.dayofweek
    df["day_of_month"] = df[DATE_COL].dt.day
    df["month"] = df[DATE_COL].dt.month
    df["week_of_year"] = df[DATE_COL].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["quarter"] = df[DATE_COL].dt.quarter

    df["price_gap_vs_competitor"] = df["Price"] - df["Competitor Pricing"]
    df["price_gap_pct"] = (df["price_gap_vs_competitor"] / df["Price"].replace(0, np.nan)).fillna(0)
    df["discounted_price"] = df["Price"] * (1 - df["Discount"] / 100.0)

    df["stockout_flag"] = (df["Inventory Level"] <= 0).astype(int)
    df["inventory_to_sales_ratio"] = (
        df["Inventory Level"] / df["Units Sold"].replace(0, np.nan)
    ).fillna(0)

    g = df.groupby(GROUP_COLS, group_keys=False)

    if TARGET not in df.columns:
        df[TARGET] = np.nan  # allow inference-only data without ground truth

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
    df["demand_trend"] = df["demand_rollmean_7"] - df["demand_rollmean_30"]

    lag_roll_cols = [c for c in df.columns if "lag" in c or "roll" in c or c == "demand_trend"]
    df[lag_roll_cols] = df[lag_roll_cols].fillna(0)

    return df


def encode_with_saved_encoders(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    """Applies previously-fitted LabelEncoders. Unseen categories are mapped
        to a fallback code (-1 handled via a safe transform)."""
    df = df.copy()

    for col, le in encoders.items():
        mapping = {
            str(cls): idx
            for idx, cls in enumerate(le.classes_)
        }

        df[col + "_enc"] = (
            df[col]
            .astype(str)
            .map(mapping)
            .fillna(-1)
            .astype(np.int32)
        )

    return df


def bootstrap_history(new_df: pd.DataFrame, historical_df: pd.DataFrame) -> pd.DataFrame:
    """If the uploaded CSV has limited history per store-item (so lag/rolling
    features would be mostly zero), prepend matching historical rows so lag
    features are meaningful. historical_df is the engineered training data
    artifact (historical_engineered.parquet)."""
    """Add matching historical rows needed for lag/rolling features."""

    raw_cols = [
        DATE_COL, "Store ID", "Product ID", "Category", "Region",
        "Inventory Level", "Units Sold", "Units Ordered", "Price", "Discount",
        "Weather Condition", "Promotion", "Competitor Pricing", "Seasonality",
        "Epidemic", TARGET
    ]

    # Only keep the columns actually needed BEFORE merging.
    available_cols = [c for c in raw_cols if c in historical_df.columns]
    historical_raw = historical_df[available_cols].copy()

    # Get only the unique store/product combinations from the upload.
    keys = new_df[GROUP_COLS].drop_duplicates()

    # Filter historical data to only matching store/product combinations.
    hist_subset = historical_raw.merge(
        keys,
        on=GROUP_COLS,
        how="inner"
    )

    # Only keep raw columns from the uploaded file.
    new_raw = new_df[
        [c for c in raw_cols if c in new_df.columns]
    ].copy()

    combined = pd.concat(
        [hist_subset, new_raw],
        ignore_index=True
    )

    combined = combined.drop_duplicates(
        subset=GROUP_COLS + [DATE_COL],
        keep="last"
    )

    combined = combined.sort_values(
        GROUP_COLS + [DATE_COL]
    ).reset_index(drop=True)

    return combined


def predict_demand(model, df_encoded: pd.DataFrame, feature_cols: list) -> np.ndarray:
    missing = [c for c in feature_cols if c not in df_encoded.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    preds = model.predict(df_encoded[feature_cols])
    return np.clip(preds, 0, None)


# -----------------------------------------------------------------------
# STOCK ALERT LOGIC
# -----------------------------------------------------------------------
def generate_alerts(
    df: pd.DataFrame,
    lead_time_days: int = 7,
    safety_stock_days: int = 3,
    critical_days_threshold: float = 3,
    warning_days_threshold: float = 7,
) -> pd.DataFrame:
    """
    Given a dataframe with one row per store-item (latest snapshot) containing:
        - Inventory Level (current stock)
        - Predicted_Demand (forecasted daily demand, e.g. avg of next N days)
    computes:
        - days_of_stock_left
        - reorder_point (lead_time + safety_stock, in units)
        - recommended_order_qty
        - alert_level (Critical / Warning / OK / Overstock)
    """
    df = df.copy()
    daily_demand = df["Predicted_Demand"].replace(0, np.nan)

    df["days_of_stock_left"] = (df["Inventory Level"] / daily_demand).fillna(np.inf)
    df["reorder_point_units"] = daily_demand.fillna(0) * (lead_time_days + safety_stock_days)

    # Recommended order = bring stock up to cover lead time + safety stock,
    # topped up beyond what's already on hand
    target_stock = daily_demand.fillna(0) * (lead_time_days + safety_stock_days)
    df["recommended_order_qty"] = np.maximum(target_stock - df["Inventory Level"], 0).round().astype(int)

    def classify(row):
        days_left = row["days_of_stock_left"]
        if row["Predicted_Demand"] <= 0:
            return "OK"
        if days_left <= critical_days_threshold:
            return "Critical"
        if days_left <= warning_days_threshold:
            return "Warning"
        if days_left > warning_days_threshold * 4:
            return "Overstock"
        return "OK"

    df["alert_level"] = df.apply(classify, axis=1)
    return df
