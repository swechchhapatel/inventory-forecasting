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


def encode_with_saved_encoders(df: pd.DataFrame,encoders: dict) -> pd.DataFrame:
    """Apply previously fitted LabelEncoders using vectorized pandas mapping.
    This is considerably more memory/CPU efficient than calling
    le.transform() once for every individual cell.
    """

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
    """
    Combine the required historical rows with the uploaded rows.

    Historical rows are used ONLY to calculate lag/rolling features.
    The __is_uploaded marker lets run_pipeline() later discard the
    historical rows before encoding and prediction.
    """

    raw_cols = [
        DATE_COL,
        "Store ID",
        "Product ID",
        "Category",
        "Region",
        "Inventory Level",
        "Units Sold",
        "Units Ordered",
        "Price",
        "Discount",
        "Weather Condition",
        "Promotion",
        "Competitor Pricing",
        "Seasonality",
        "Epidemic",
        TARGET,
    ]

    # Only select the raw columns needed for feature engineering.
    # This avoids carrying all 47 columns from historical_engineered.parquet
    # through the merge.
    available_hist_cols = [
        c for c in raw_cols
        if c in historical_df.columns
    ]

    historical_raw = historical_df.loc[:, available_hist_cols].copy()

    # Only historical records belonging to store/product combinations
    # present in the uploaded file are needed.
    keys = new_df[GROUP_COLS].drop_duplicates()

    hist_subset = historical_raw.merge(
        keys,
        on=GROUP_COLS,
        how="inner",
        sort=False,
    )

    # Mark historical rows.
    hist_subset["__is_uploaded"] = False

    # Keep only the raw columns needed from the uploaded data.
    available_new_cols = [
        c for c in raw_cols
        if c in new_df.columns
    ]

    new_raw = new_df.loc[:, available_new_cols].copy()

    # Mark uploaded rows.
    new_raw["__is_uploaded"] = True

    # Historical first, uploaded second.
    # Therefore if the exact same Store/Product/Date occurs in both,
    # the uploaded row wins because drop_duplicates(... keep="last")
    # is used below.
    combined = pd.concat(
        [hist_subset, new_raw],
        ignore_index=True,
        copy=False,
    )

    del historical_raw
    del hist_subset
    del new_raw

    combined = combined.drop_duplicates(
        subset=GROUP_COLS + [DATE_COL],
        keep="last",
    )

    combined = combined.sort_values(
        GROUP_COLS + [DATE_COL]
    ).reset_index(drop=True)

    return combined


def predict_demand(model, df_encoded: pd.DataFrame, feature_cols: list) -> np.ndarray:
    """Predict demand using only the required model features."""

    missing = [
        c for c in feature_cols
        if c not in df_encoded.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required feature columns: {missing}"
        )

    # Only pass the model's required columns to XGBoost.
    X = df_encoded.loc[:, feature_cols]

    preds = model.predict(X)

    del X

    return np.clip(
        preds,
        0,
        None,
    )


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
    Calculate stock health and reorder recommendations.

    Uses vectorized pandas/numpy operations to reduce memory and CPU
    compared with row-by-row DataFrame.apply().
    """

    df = df.copy()

    daily_demand = df["Predicted_Demand"].replace(
        0,
        np.nan,
    )

    # Days of stock remaining.
    df["days_of_stock_left"] = (
        df["Inventory Level"] / daily_demand
    ).fillna(np.inf)

    # Lead time + safety stock target.
    target_days = (
        lead_time_days +
        safety_stock_days
    )

    df["reorder_point_units"] = (
        daily_demand.fillna(0) * target_days
    )

    # Recommended order quantity.
    target_stock = (
        daily_demand.fillna(0) * target_days
    )

    df["recommended_order_qty"] = (
        np.maximum(
            target_stock - df["Inventory Level"],
            0,
        )
        .round()
        .astype(int)
    )

    # ---------------------------------------------------------------
    # Vectorized alert classification
    # ---------------------------------------------------------------
    df["alert_level"] = "OK"

    positive_demand = df["Predicted_Demand"] > 0

    critical_mask = (
        positive_demand &
        (df["days_of_stock_left"] <= critical_days_threshold)
    )

    warning_mask = (
        positive_demand &
        (df["days_of_stock_left"] > critical_days_threshold) &
        (df["days_of_stock_left"] <= warning_days_threshold)
    )

    overstock_mask = (
        positive_demand &
        (df["days_of_stock_left"] > warning_days_threshold * 4)
    )

    df.loc[critical_mask, "alert_level"] = "Critical"
    df.loc[warning_mask, "alert_level"] = "Warning"
    df.loc[overstock_mask, "alert_level"] = "Overstock"

    return df