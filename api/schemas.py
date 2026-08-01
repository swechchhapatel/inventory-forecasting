"""
Pydantic models for the Store Stock Assistant API.
Defines the request/response contracts used by all endpoints.
"""

from pydantic import BaseModel, Field
from typing import Optional, List


class SalesRecord(BaseModel):
    """One row of raw sales/inventory data -- matches the required CSV schema."""
    Date: str = Field(..., description="Date, e.g. '2024-06-15'")
    Store_ID: str = Field(..., alias="Store ID")
    Product_ID: str = Field(..., alias="Product ID")
    Category: str
    Region: str
    Inventory_Level: float = Field(..., alias="Inventory Level")
    Units_Sold: float = Field(..., alias="Units Sold")
    Units_Ordered: float = Field(..., alias="Units Ordered")
    Price: float
    Discount: float
    Weather_Condition: str = Field(..., alias="Weather Condition")
    Promotion: int = Field(..., ge=0, le=1)
    Competitor_Pricing: float = Field(..., alias="Competitor Pricing")
    Seasonality: str
    Epidemic: int = Field(..., ge=0, le=1)

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "Date": "2024-06-15",
                "Store ID": "S001",
                "Product ID": "P0042",
                "Category": "Electronics",
                "Region": "North",
                "Inventory Level": 120,
                "Units Sold": 35,
                "Units Ordered": 0,
                "Price": 49.99,
                "Discount": 10,
                "Weather Condition": "Sunny",
                "Promotion": 1,
                "Competitor Pricing": 47.50,
                "Seasonality": "Summer",
                "Epidemic": 0,
            }
        }


class ForecastSettings(BaseModel):
    """Business-policy parameters controlling alert thresholds. All optional --
    defaults match what the Streamlit app used."""
    lead_time_days: int = Field(7, ge=1, le=60, description="Supplier lead time in days")
    safety_stock_days: int = Field(3, ge=0, le=30, description="Safety stock buffer in days")
    critical_days_threshold: float = Field(3, ge=0, description="Days-of-stock-left cutoff for Critical alert")
    warning_days_threshold: float = Field(7, ge=0, description="Days-of-stock-left cutoff for Warning alert")


class ItemForecast(BaseModel):
    """Forecast + alert result for a single store-item combination."""
    store_id: str
    product_id: str
    category: Optional[str] = None
    region: Optional[str] = None
    inventory_level: float
    forecasted_daily_demand: float
    days_of_stock_left: float
    reorder_point_units: float
    recommended_order_qty: int
    alert_level: str
    trend: Optional[str] = None  # "up" / "down" / "flat"


class ForecastResponse(BaseModel):
    """Response for a batch CSV forecast request."""
    n_items: int
    n_critical: int
    n_warning: int
    n_ok: int
    n_overstock: int
    items: List[ItemForecast]


class ModelInfo(BaseModel):
    """Model metadata for admin/monitoring dashboards."""
    mae: Optional[float] = None
    rmse: Optional[float] = None
    mape: Optional[float] = None
    r2: Optional[float] = None
    top_features: Optional[List[dict]] = None


class ForecastRecordsRequest(BaseModel):
    """Request body for POST /forecast/records."""
    records: List[SalesRecord]
    settings: ForecastSettings = ForecastSettings()


class SchemaResponse(BaseModel):
    """The required CSV columns -- lets client apps validate uploads before sending."""
    required_columns: List[str]
    example_row: dict


class ErrorResponse(BaseModel):
    detail: str
