# Retail Demand Forecasting & Inventory Alert System

A machine-learning powered inventory forecasting system that predicts daily demand for each **Store × Product** combination and converts predictions into actionable stock alerts.

The project uses **XGBoost** for demand forecasting and provides a **Streamlit dashboard** for uploading sales data, viewing forecasts, monitoring inventory, and generating reorder recommendations.

## Features

* Daily demand forecasting for Store × Product combinations
* Time-series feature engineering using:

  * Lag features
  * Rolling averages and standard deviations
  * Calendar features
  * Price-related features
* Inventory alerts:

  * 🔴 **Critical**
  * 🟠 **Warning**
  * 🟢 **OK**
  * 🔵 **Overstock**
* Recommended reorder quantity
* Interactive Streamlit dashboard
* Forecast vs. actual demand visualization
* Inventory trend analysis
* Model performance and feature-importance insights
* Configurable lead time, safety stock, and alert thresholds

## Dataset

The model is trained using the Kaggle **Retail Store Inventory and Demand Forecasting** dataset.

**Dataset:** [Retail Store Inventory and Demand Forecasting](https://www.kaggle.com/datasets/atomicd/retail-store-inventory-and-demand-forecasting)

For production use, the model should be retrained using the company's actual historical sales and inventory data.

## Project Structure

```text
project/
├── src/
│   ├── train_model.py
│   └── inference_utils.py
├── app.py
├── requirements.txt
├── artifacts/
│   ├── xgb_demand_model.pkl
│   ├── label_encoders.pkl
│   ├── feature_cols.json
│   ├── metrics.json
│   ├── feature_importance.csv
│   ├── historical_engineered.parquet
│   └── test_predictions_sample.csv
└── README.md
```

## Model Training

Training is performed in Kaggle or Google Colab.

`train_model.py`:

1. Loads the retail dataset
2. Creates time-series and calendar features
3. Trains an XGBoost regression model
4. Evaluates the model using a time-based test set
5. Saves the trained model and required artifacts

The evaluation includes:

* MAE
* RMSE
* MAPE
* R²

After training, place the generated `artifacts/` folder in the project root.

## Inventory Alert Logic

The dashboard uses the predicted daily demand to estimate inventory coverage.

### Days of Stock

```text
Days of Stock = Inventory Level / Forecasted Daily Demand
```

### Reorder Point

```text
Reorder Point = Forecasted Daily Demand × (Lead Time + Safety Stock Days)
```

### Recommended Order Quantity

```text
Recommended Order = max(Reorder Point - Inventory Level, 0)
```

### Alert Levels

| Alert        | Condition               |
| ------------ | ----------------------- |
| 🔴 Critical  | Days of stock ≤ 3       |
| 🟠 Warning   | Days of stock ≤ 7       |
| 🟢 OK        | Healthy inventory level |
| 🔵 Overstock | Days of stock > 28      |

The thresholds can be adjusted from the Streamlit dashboard.

## Streamlit Dashboard

The application provides three main areas:

### Overview & Alerts

View all Store × Product combinations with:

* Forecasted daily demand
* Current inventory
* Days of stock remaining
* Reorder point
* Recommended order quantity
* Alert level
* Demand trend

### Store-Item Analysis

Select a store and product to inspect:

* Historical demand
* Predicted demand
* Inventory trends
* Forecast performance

### Model Insights

View:

* Model evaluation metrics
* Feature importance
* Forecasting insights

## Important Notes

* The included model is trained on the Kaggle dataset for demonstration purposes.
* For real business use, retrain the model on historical company data.
* Keep the `artifacts/` directory synchronized with the model used by the application.
* Large datasets should not be uploaded to the Git repository; use Git LFS or external storage if necessary.
* The current application is intended as a forecasting and inventory decision-support tool, not an autonomous purchasing system.

## Future Improvements

* Automated model retraining
* Live database integration
* Scheduled demand forecasts
* Model drift monitoring
* Per-category and regional forecasting models
* Automated inventory notifications
* Authentication and role-based access
* Integration with ERP/inventory management systems
