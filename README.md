# Retail Demand Forecasting & Inventory Alert System

Predicts daily demand for every Store × Product combination and turns those
forecasts into actionable stock alerts (Critical / Warning / OK / Overstock),
with a recommended reorder quantity for each item.

Dataset: [Retail Store Inventory and Demand Forecasting](https://www.kaggle.com/datasets/atomicd/retail-store-inventory-and-demand-forecasting/data) (Kaggle)

---

## 1. Project structure

```
project/
├── src/
│   ├── train_model.py       # Run in Kaggle/Colab to train the model
│   └── inference_utils.py   # Shared feature engineering + alert logic (used by both app.py and the API)
├── api/
│   ├── main.py               # FastAPI service — the production integration point
│   └── schemas.py            # Request/response models
├── app.py                   # Streamlit dashboard (internal demo/testing tool)
├── requirements.txt
├── artifacts/                # <- created after training, holds model + encoders (you bring this in)
└── README.md
```

## 2. Train the model (Kaggle or Colab)

1. Open a new Kaggle Notebook.
2. Add the dataset: **Add Input → search "Retail Store Inventory and Demand Forecasting"**.
3. Copy `src/train_model.py` into a notebook cell (or upload the file and `%run` it).
4. Check `DATA_PATH` in the script matches the path Kaggle mounts the CSV at
   (shown in the notebook's Data pane, typically
   `/kaggle/input/retail-store-inventory-and-demand-forecasting/retail_store_inventory.csv`
   — adjust if the actual filename differs).
5. Run all cells. This will:
   - Engineer time-series features (lags, rolling means/stds, calendar features, price gaps)
   - Train an XGBoost regressor to predict `Demand`
   - Print MAE / RMSE / MAPE / R² on a held-out time-based test split (last 30 days)
   - Save all artifacts to `/kaggle/working/artifacts/`
6. Zip and download that folder:
   ```python
   import shutil
   shutil.make_archive("artifacts", "zip", "/kaggle/working/artifacts")
   ```
   Then download `artifacts.zip` from the Kaggle output pane.
7. Unzip it locally so you have:
   ```
   project/artifacts/
     ├── xgb_demand_model.pkl
     ├── label_encoders.pkl
     ├── feature_cols.json
     ├── metrics.json
     ├── feature_importance.csv
     ├── historical_engineered.parquet
     └── test_predictions_sample.csv
   ```

## 3. Run the API (for website integration — the production path)

### Set up API keys first (required)

The API refuses to start unless `api/api_keys.json` exists with at least one valid
entry — this makes it impossible to accidentally run it with no authentication.
Each client (the website, an internal tool, etc.) gets its **own** key, so you can
revoke one client's access later without affecting anyone else.

1. Copy the template:
   ```bash
   cp api/api_keys.json.example api/api_keys.json
   ```
2. Edit `api/api_keys.json` and replace the placeholder values with real random
   secrets — one entry per client:
   ```json
   {
     "website": "a-long-random-secret-for-the-website",
     "internal-testing": "a-different-long-random-secret"
   }
   ```
3. `api/api_keys.json` is gitignored — never commit real secrets to source control.

Every request to `/forecast`, `/forecast/records`, and `/model/info` must include a
valid key from this file in an `X-API-Key` header, or it's rejected with
`401 Unauthorized`. `/health` and `/schema` are intentionally left open (no
sensitive data, useful for uptime monitors).

**To revoke a client's access**: delete its entry from `api/api_keys.json` and
restart the server. Every other client's key keeps working unaffected.

**To add a new client**: add a new entry with a freshly generated secret and
restart the server.

**Every successful call is logged with the calling client's name** (e.g.
`/forecast called by client 'website'`), so you can see who's using the API and
debug issues per-client.

### Then start the server

```bash
cd project
pip install -r requirements.txt
uvicorn api.main:app --reload --port 8000
```

Interactive docs (auto-generated, explore and test every endpoint from the browser):
- Swagger UI: `http://localhost:8000/docs` — click the **"Authorize"** button (top
  right) and paste one client's key from `api/api_keys.json`; it'll then be sent
  automatically on every "Try it out" request for the rest of your session.
- ReDoc: `http://localhost:8000/redoc`

### Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Liveness check — confirms the API is up and the model loaded |
| `/schema` | GET | Returns the required CSV columns + an example row, so a client app can validate uploads before sending |
| `/model/info` | GET | Accuracy metrics (MAE/RMSE/MAPE/R²) and top feature importances |
| `/forecast` | POST | Upload a CSV, get back a forecast + restock alert for every store-item in it. **Primary endpoint for the website.** |
| `/forecast/records` | POST | JSON alternative to `/forecast` — send records directly as JSON if the website already holds the data in its own database |

`/forecast` and `/forecast/records` both accept optional parameters to override the default alert thresholds: `lead_time_days`, `safety_stock_days`, `critical_days_threshold`, `warning_days_threshold`.

Example request (CSV upload):
```bash
curl -X POST "http://localhost:8000/forecast?lead_time_days=7&safety_stock_days=3" \
  -H "X-API-Key: a-long-random-secret-for-the-website" \
  -F "file=@store_sales_export.csv"
```

Example request (JSON):
```bash
curl -X POST "http://localhost:8000/forecast/records" \
  -H "X-API-Key: a-long-random-secret-for-the-website" \
  -H "Content-Type: application/json" \
  -d '{
    "records": [ {"Date": "2024-06-15", "Store ID": "S001", "Product ID": "P0042", "...": "..."} ],
    "settings": {"lead_time_days": 7, "safety_stock_days": 3}
  }'
```

Both return the same shape:
```json
{
  "n_items": 2, "n_critical": 1, "n_warning": 1, "n_ok": 0, "n_overstock": 0,
  "items": [
    {
      "store_id": "S001", "product_id": "P0001", "category": "Electronics", "region": "North",
      "inventory_level": 34.3, "forecasted_daily_demand": 44.86,
      "days_of_stock_left": 0.76, "reorder_point_units": 448.61,
      "recommended_order_qty": 414, "alert_level": "Critical", "trend": "down"
    }
  ]
}
```

**CORS**: currently allows requests from any origin (`allow_origins=["*"]`) for easy testing. Narrow `allow_origins` in `api/main.py` to the actual website domain(s) before going live.

**Note**: the model bundled here is trained on the Kaggle dataset for demonstration. Retrain on the company's actual sales data (Section 2) and swap the `artifacts/` folder before this API goes live on the real website.

## 4. Run the Streamlit dashboard (internal demo / quick testing)

```bash
cd project
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL Streamlit prints (usually `http://localhost:8501`). Kept as an internal tool for quickly sanity-checking the model or demoing to non-technical stakeholders — the actual website integration should go through the API above, not this dashboard.

## 5. Using the dashboard

1. **Upload a CSV** in the sidebar — same columns as the training data. It can be:
   - A full historical export (best case), or
   - A short recent slice (e.g. last few days) — the app automatically pulls in
     matching historical rows from `historical_engineered.parquet` for the same
     Store-Product pairs so lag/rolling features stay meaningful.
2. **Overview & Alerts tab**: sortable, filterable table of every store-item combo,
   ranked by urgency (days of stock left), with a recommended order quantity.
   Download the alert report as CSV.
3. **Store-Item Drill-down tab**: pick any Store + Product to see actual vs.
   predicted demand over time, plus the inventory trend.
4. **Model Insights tab**: model accuracy metrics and feature importance, so you
   can explain to your supervisor *why* the model predicts what it does.
5. Adjust **lead time**, **safety stock buffer**, and **alert thresholds** in the
   sidebar — these drive the reorder point and alert classification live, no retraining needed.

## 6. Alert logic (explainable, no black box)

For each store-item, using the model's forecasted daily demand and current inventory:

| Metric | Formula |
|---|---|
| Days of stock left | `Inventory Level ÷ Forecasted Daily Demand` |
| Reorder point | `Forecasted Daily Demand × (Lead Time + Safety Stock days)` |
| Recommended order qty | `max(Reorder Point − Inventory Level, 0)` |

Alert levels:
- 🔴 **Critical** — days of stock left ≤ critical threshold (default 3)
- 🟠 **Warning** — days of stock left ≤ warning threshold (default 7)
- 🟢 **OK** — healthy stock level
- 🔵 **Overstock** — days of stock left > 4× warning threshold (capital tied up, consider promotions)

## 7. Next steps for production deployment

- Retrain on the company's actual sales data (not the Kaggle dataset) and swap `artifacts/`.
- Point the website at `/forecast` or `/forecast/records` instead of manual CSV upload —
  ideally the website's backend calls the API on a schedule (e.g. nightly) rather than
  the browser calling it directly on every page load.
- Retrain periodically (e.g. weekly) as new sales data accumulates — the time-based
  split in `train_model.py` makes this straightforward to automate.
- Per-client API key authentication is already in place (`api/api_keys.json`,
  `X-API-Key` header). For production, consider upgrading further to short-lived
  tokens (OAuth/JWT) if clients need automatic key rotation, or move the keys
  file into a proper secrets manager (e.g. AWS Secrets Manager, Azure Key Vault)
  instead of a local JSON file.
- Narrow CORS (`allow_origins` in `api/main.py`) to the real website domain(s).
- Add model monitoring (track live MAE/MAPE against actuals) to catch drift.
- Consider per-category or per-region models if error rates vary significantly
  across segments (see `/model/info` after evaluating on real data).
- Run the API behind a process manager (e.g. `gunicorn` with `uvicorn` workers,
  or a container orchestrator) rather than the single `--reload` dev server shown above.
