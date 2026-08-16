"""
Store Stock Assistant
=======================================================
Run with: streamlit run app.py

Expects trained model artifacts in ./artifacts/ (produced by src/train_model.py):
  - xgb_demand_model.pkl
  - label_encoders.pkl
  - feature_cols.json
  - metrics.json
  - historical_engineered.parquet
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import math
import plotly.graph_objects as go
import plotly.express as px

from src.inference_utils import (
    engineer_features, encode_with_saved_encoders, bootstrap_history,
    predict_demand, generate_alerts, GROUP_COLS, DATE_COL, TARGET
)

# -----------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------
st.set_page_config(
    page_title="Store Stock Assistant",
    page_icon="🧺",
    layout="wide",
    initial_sidebar_state="expanded",
)

ARTIFACT_DIR = "artifacts"

INK = "#1E2A38"
MUTED_INK = "#5B6672"
CARD_BG = "#FFFFFF"
CARD_BORDER = "#E7E3DA"
PAGE_BG = "#FAF9F6"
ACCENT = "#2F6F5E"

ALERT_COLORS = {
    "Critical": "#C1443A",
    "Warning": "#D79A3B",
    "OK": "#2F8F5B",
    "Overstock": "#3E6FA8",
}
ALERT_ICON = {
    "Critical": "🔴",
    "Warning": "🟠",
    "OK": "🟢",
    "Overstock": "🔵",
}
ALERT_PLAIN = {
    "Critical": "Order now",
    "Warning": "Order soon",
    "OK": "Well stocked",
    "Overstock": "Too much stock",
}
TREND_ARROW = {"up": "↑", "down": "↓", "flat": "→"}
TREND_COLOR = {"up": "#C1443A", "down": "#2F8F5B", "flat": "#8B93A0"}

CUSTOM_CSS = f"""
<style>
.stock-card {{
    background: {CARD_BG};
    border: 1px solid {CARD_BORDER};
    border-radius: 14px;
    padding: 22px 26px;
    box-shadow: 0 1px 3px rgba(30,42,56,0.06);
}}
.stock-headline {{
    font-size: 1.35rem;
    font-weight: 700;
    color: {INK};
    margin-bottom: 2px;
}}
.stock-subline {{
    font-size: 0.95rem;
    color: {MUTED_INK};
    margin-bottom: 14px;
}}
.pill {{
    display: inline-block;
    padding: 5px 14px;
    border-radius: 999px;
    font-size: 13px;
    font-weight: 700;
    color: white;
    letter-spacing: 0.2px;
}}
.pill-soft {{
    display: inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 600;
    background: #F0EEE8;
    color: {MUTED_INK};
}}
.stat-row {{
    display: flex;
    gap: 28px;
    margin-top: 16px;
    flex-wrap: wrap;
}}
.stat-block .stat-num {{
    font-size: 1.5rem;
    font-weight: 700;
    color: {INK};
}}
.stat-block .stat-label {{
    font-size: 0.82rem;
    color: {MUTED_INK};
}}
.action-item {{
    background: {CARD_BG};
    border: 1px solid {CARD_BORDER};
    border-left: 5px solid var(--accent);
    border-radius: 10px;
    padding: 14px 18px 6px 18px;
    margin-bottom: 10px;
}}
.action-item.done {{
    opacity: 0.5;
}}
.action-title {{
    font-weight: 700;
    color: {INK};
    font-size: 1.0rem;
}}
.action-sub {{
    color: {MUTED_INK};
    font-size: 0.88rem;
}}
.why-line {{
    color: {MUTED_INK};
    font-size: 0.85rem;
    font-style: italic;
    margin-top: 4px;
}}
section[data-testid="stSidebar"] {{
    border-right: 1px solid {CARD_BORDER};
}}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------
# CACHED LOADERS
# -----------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(ARTIFACT_DIR, "xgb_demand_model.pkl"))
    encoders = joblib.load(os.path.join(ARTIFACT_DIR, "label_encoders.pkl"))
    with open(os.path.join(ARTIFACT_DIR, "feature_cols.json")) as f:
        feature_cols = json.load(f)
    metrics = {}
    metrics_path = os.path.join(ARTIFACT_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
    return model, encoders, feature_cols, metrics


@st.cache_data
def load_historical():
    path = os.path.join(ARTIFACT_DIR, "historical_engineered.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    return None


@st.cache_data(show_spinner=False)
def run_pipeline(raw_df: pd.DataFrame, _encoders, feature_cols, _model, historical_df):
    """Full pipeline: bootstrap history -> engineer features -> encode -> predict."""
    
    if historical_df is not None:
        combined_raw = bootstrap_history(raw_df, historical_df)
    else:
        combined_raw = raw_df.copy()

    engineered = engineer_features(combined_raw)
    encoded = encode_with_saved_encoders(engineered, _encoders)
    preds = predict_demand(_model, encoded, feature_cols)
    encoded["Predicted_Demand"] = preds

    upload_keys = raw_df[GROUP_COLS + [DATE_COL]].copy()
    upload_keys[DATE_COL] = pd.to_datetime(upload_keys[DATE_COL])
    encoded[DATE_COL] = pd.to_datetime(encoded[DATE_COL])
    result = encoded.merge(upload_keys, on=GROUP_COLS + [DATE_COL], how="inner")

    # how many historical rows (pre-upload) each store-item had available --
    # used later for the confidence heuristic
    hist_counts = (
        combined_raw.groupby(GROUP_COLS).size().rename("history_days").reset_index()
    )
    result = result.merge(hist_counts, on=GROUP_COLS, how="left")
    return result


def stock_health_gauge(days_left: float, alert_level: str, max_days: float = 21):
    """A simple ring gauge showing days-of-stock-left at a glance."""
    capped = min(days_left, max_days)
    pct = capped / max_days
    color = ALERT_COLORS.get(alert_level, "#999")

    fig = go.Figure(go.Pie(
        values=[pct, 1 - pct],
        hole=0.72,
        marker=dict(colors=[color, "#EFECE3"]),
        textinfo="none",
        sort=False,
        direction="clockwise",
        rotation=0,
    ))
    label = "999+" if days_left > 900 else f"{days_left:.0f}"
    fig.update_layout(
        showlegend=False,
        margin=dict(t=0, b=0, l=0, r=0),
        height=180, width=180,
        annotations=[dict(
            text=f"<b>{label}</b><br><span style='font-size:11px;color:{MUTED_INK}'>days left</span>",
            x=0.5, y=0.5, showarrow=False, font=dict(size=22, color=INK),
        )],
    )
    return fig


def get_trend_direction(demand_trend: float, threshold: float = 1.0) -> str:
    """demand_trend = rollmean_7 - rollmean_30. Small threshold avoids noise
    being labeled as a trend."""
    if pd.isna(demand_trend):
        return "flat"
    if demand_trend > threshold:
        return "up"
    if demand_trend < -threshold:
        return "down"
    return "flat"


def format_order_line(row) -> str:
    """Plain-language instruction for a store manager."""
    qty = int(row["recommended_order_qty"])
    days_left = row["days_of_stock_left"]
    if row["alert_level"] == "Critical":
        when = "today" if days_left <= 1 else f"within {math.ceil(days_left)} day(s)"
        return f"Order **{qty} units** {when} - you'll run out soon."
    if row["alert_level"] == "Warning":
        return f"Order **{qty} units** in the next few days to stay ahead of demand."
    if row["alert_level"] == "Overstock":
        return "No need to order - consider a promotion to move extra stock."
    return "No action needed right now."


def explain_alert(row) -> str:
    """'Why this alert' one-liner built from features we already compute."""
    trend_dir = get_trend_direction(row.get("demand_trend", 0))
    volatility = row.get("demand_rollstd_7", 0) or 0
    mean_demand = row.get("demand_rollmean_7", 0) or 0
    is_volatile = mean_demand > 0 and (volatility / mean_demand) > 0.4
    promo = row.get("Promotion", 0) == 1

    reasons = []
    if trend_dir == "up":
        reasons.append("sales have been trending up over the last week")
    elif trend_dir == "down":
        reasons.append("sales have been slowing down recently")

    if promo:
        reasons.append("a promotion is currently boosting demand")

    if is_volatile:
        reasons.append("this item's daily sales swing a lot, so treat the forecast as a rough guide")

    if row["alert_level"] == "Overstock":
        reasons.append("stock on hand is far more than recent demand needs")

    if not reasons:
        if row["alert_level"] in ("Critical", "Warning"):
            reasons.append("stock is running down at its normal, steady pace")
        else:
            reasons.append("sales are steady and stock levels are on track")

    return "Because " + " and ".join(reasons) + "."

# -----------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------
st.sidebar.markdown("## 🧺 Store Stock Assistant")
st.sidebar.caption("Upload your sales data to see what needs restocking.")

uploaded_file = st.sidebar.file_uploader("Upload sales data (CSV)", type=["csv"])

with st.sidebar.expander("⚙️ Restocking preferences", expanded=False):
    st.caption("These control how early you get warned before running out.")
    lead_time = st.slider("How many days does a supplier order take to arrive?", 1, 21, 7)
    safety_stock = st.slider("Extra buffer stock to keep on hand (days)", 0, 14, 3)
    critical_days = st.slider("Warn urgently when days left is under", 1, 10, 3)
    warning_days = st.slider("Give a heads-up when days left is under", 2, 21, 7)

if not os.path.exists(os.path.join(ARTIFACT_DIR, "xgb_demand_model.pkl")):
    st.error(
        "No trained model found in `./artifacts/`. Run `src/train_model.py` in "
        "Kaggle/Colab first, then place the downloaded `artifacts/` folder next to "
        "this app before launching Streamlit."
    )
    st.stop()

model, encoders, feature_cols, metrics = load_artifacts()
historical_df = load_historical()

if uploaded_file is None:
    st.title("🧺 Store Stock Assistant")
    st.markdown(
        "##### See what to restock, before you run out.\n"
        "Upload your store's sales data in the sidebar and this tool will tell you, "
        "in plain terms, what's running low, what's overstocked, and how much to order."
    )
    st.markdown(" ")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**📤 1. Upload your data**")
        st.caption("A CSV export of daily sales, inventory, and pricing per store and item.")
    with c2:
        st.markdown("**🔮 2. We forecast demand**")
        st.caption("Each item gets a demand forecast based on its own sales history.")
    with c3:
        st.markdown("**✅ 3. You get a to-do list**")
        st.caption("A ranked list of what to reorder - and how much - before it's a problem.")

    with st.expander("How accurate is this forecast?"):
        if metrics:
            st.write(
                f"On past data, this tool's forecasts were off by about "
                f"**{metrics.get('mape', 0):.0f}%** on average - solid enough to plan "
                f"reorders with confidence, though always use judgement for unusual events "
                f"(promotions, holidays, local disruptions)."
            )
        else:
            st.caption("No accuracy report found. Retrain the model to generate one.")
    st.stop()


# -----------------------------------------------------------------------
# LOAD + VALIDATE UPLOADED CSV
# -----------------------------------------------------------------------
try:
    raw_df = pd.read_csv(uploaded_file)
    raw_df.columns = [c.strip() for c in raw_df.columns]
except Exception as e:
    st.error(f"Could not read that file: {e}")
    st.stop()

required_cols = [
    DATE_COL, "Store ID", "Product ID", "Category", "Region",
    "Inventory Level", "Units Sold", "Units Ordered", "Price", "Discount",
    "Weather Condition", "Promotion", "Competitor Pricing", "Seasonality", "Epidemic"
]
missing_cols = [c for c in required_cols if c not in raw_df.columns]
if missing_cols:
    st.error(
        f"This file is missing some columns we need: {', '.join(missing_cols)}. "
        "Please check your export matches the expected format."
    )
    st.stop()

with st.spinner("Reading your sales history and forecasting demand..."):
    result = run_pipeline(raw_df, encoders, feature_cols, model, historical_df)

latest = (
    result.sort_values(DATE_COL)
    .groupby(GROUP_COLS, as_index=False)
    .last()
)
alerts = generate_alerts(
    latest,
    lead_time_days=lead_time,
    safety_stock_days=safety_stock,
    critical_days_threshold=critical_days,
    warning_days_threshold=warning_days,
)

# ---- session state for "mark as ordered" checkboxes (resets per upload) ----
file_key = f"{uploaded_file.name}_{uploaded_file.size}"
if "ordered_items" not in st.session_state or st.session_state.get("_file_key") != file_key:
    st.session_state.ordered_items = set()
    st.session_state._file_key = file_key

alerts["item_key"] = alerts["Store ID"].astype(str) + "__" + alerts["Product ID"].astype(str)


# -----------------------------------------------------------------------
# HEADER
# -----------------------------------------------------------------------
st.title("🧺 Store Stock Assistant")

n_critical = int((alerts["alert_level"] == "Critical").sum())
n_warning = int((alerts["alert_level"] == "Warning").sum())
n_ok = int((alerts["alert_level"] == "OK").sum())
n_over = int((alerts["alert_level"] == "Overstock").sum())

if n_critical > 0:
    st.markdown(
        f"### 🔴 {n_critical} item{'s' if n_critical != 1 else ''} need"
        f"{'s' if n_critical == 1 else ''} ordering **today**"
    )
else:
    st.markdown("### 🟢 Nothing urgent today")

k1, k2, k3, k4 = st.columns(4)
k1.metric("🔴 Order now", n_critical)
k2.metric("🟠 Order soon", n_warning)
k3.metric("🟢 Well stocked", n_ok)
k4.metric("🔵 Overstocked", n_over)

st.markdown("---")

tab_overview, tab_drilldown, tab_summary, tab_about = st.tabs(
    ["✅ What to restock", "🔍 Look up an item", "📤 Order summary", "ℹ️ About this forecast"]
)

# =========================================================================
# TAB 1: OVERVIEW -- action-first, now with why/trend/confidence/sparkline/checkbox
# =========================================================================
with tab_overview:
    st.markdown("#### Today's restock list")
    st.caption("Most urgent items first. Use the filters below to narrow down to what you manage.")

    colf1, colf2, colf3, colf4 = st.columns([2, 1.3, 1.3, 1.4])
    with colf1:
        alert_filter = st.multiselect(
            "Show",
            options=["Critical", "Warning", "OK", "Overstock"],
            default=["Critical", "Warning"],
            format_func=lambda x: f"{ALERT_ICON[x]} {ALERT_PLAIN[x]}",
        )
    with colf2:
        region_filter = st.multiselect(
            "Region",
            options=sorted(alerts["Region"].dropna().unique().tolist()),
            default=[],
        )
    with colf3:
        category_filter = st.multiselect(
            "Category",
            options=sorted(alerts["Category"].dropna().unique().tolist()),
            default=[],
        )
    with colf4:
        hide_ordered = st.checkbox("Hide items already ordered", value=False)

    filtered = alerts.copy()
    if alert_filter:
        filtered = filtered[filtered["alert_level"].isin(alert_filter)]
    if region_filter:
        filtered = filtered[filtered["Region"].isin(region_filter)]
    if category_filter:
        filtered = filtered[filtered["Category"].isin(category_filter)]
    if hide_ordered:
        filtered = filtered[~filtered["item_key"].isin(st.session_state.ordered_items)]

    filtered = filtered.sort_values("days_of_stock_left")

    st.markdown(" ")

    if len(filtered) == 0:
        st.info("Nothing matches these filters. Try widening your selection above.")
    else:
        top_n = 12
        show_cards = filtered.head(top_n)
        for _, row in show_cards.iterrows():
            color = ALERT_COLORS.get(row["alert_level"], "#999")
            days_left = row["days_of_stock_left"]
            days_txt = "999+" if days_left > 900 else f"{days_left:.0f} days"
            is_ordered = row["item_key"] in st.session_state.ordered_items
            trend_dir = get_trend_direction(row.get("demand_trend", 0))

            item_hist_row = result[
                (result["Store ID"] == row["Store ID"]) & (result["Product ID"] == row["Product ID"])
            ].sort_values(DATE_COL)

            card_class = "action-item done" if is_ordered else "action-item"
            col_check, col_body = st.columns([0.35, 6.5])

            with col_check:
                st.write("")
                checked = st.checkbox(
                    "Ordered", key=f"chk_{row['item_key']}", value=is_ordered,
                    label_visibility="collapsed",
                )
                if checked and row["item_key"] not in st.session_state.ordered_items:
                    st.session_state.ordered_items.add(row["item_key"])
                    st.rerun()
                elif not checked and row["item_key"] in st.session_state.ordered_items:
                    st.session_state.ordered_items.discard(row["item_key"])
                    st.rerun()

            with col_body:
                st.markdown(
                    f"""
                    <div class="{card_class}" style="--accent:{color}; border-left-color:{color};">
                        <span class="pill" style="background-color:{color}">
                            {ALERT_ICON[row['alert_level']]} {ALERT_PLAIN[row['alert_level']]}
                        </span>
                        <span class="pill-soft" style="margin-left:6px;">
                            <span style="color:{TREND_COLOR[trend_dir]};font-weight:800;">
                                {TREND_ARROW[trend_dir]}
                            </span> trend
                        </span>
                        &nbsp;
                        <br>
                        <span class="action-title">{row['Product ID']} · Store {row['Store ID']}</span>
                        <br>
                        <span class="action-sub">
                            {row['Category']} · {row['Region']} &nbsp;|&nbsp;
                            {row['Inventory Level']:.0f} units on hand &nbsp;|&nbsp;
                            {days_txt} of stock left &nbsp;|&nbsp;
                            {format_order_line(row)}
                        </span>
                        <div class="why-line">{explain_alert(row)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        if len(filtered) > top_n:
            st.caption(
                f"Showing the {top_n} most urgent of {len(filtered)} matching items. "
                "Download the full list below."
            )

    st.markdown(" ")
    display_cols = [
        "Store ID", "Product ID", "Category", "Region", "Inventory Level",
        "Predicted_Demand", "days_of_stock_left", "recommended_order_qty", "alert_level"
    ]
    show_df = filtered[display_cols].rename(columns={
        "Predicted_Demand": "Forecasted Daily Demand",
        "days_of_stock_left": "Days of Stock Left",
        "recommended_order_qty": "Suggested Order Qty",
        "alert_level": "Status",
    })
    show_df["Days of Stock Left"] = show_df["Days of Stock Left"].replace(np.inf, 999).round(1)
    show_df["Forecasted Daily Demand"] = show_df["Forecasted Daily Demand"].round(1)
    show_df["Status"] = show_df["Status"].map(lambda s: f"{ALERT_ICON.get(s, '')} {ALERT_PLAIN.get(s, s)}")

    with st.expander(f"View full list as a table ({len(show_df)} items)"):
        st.dataframe(show_df, use_container_width=True, height=380)

    st.download_button(
        "⬇ Download full restock list (CSV)",
        data=show_df.to_csv(index=False).encode("utf-8"),
        file_name="restock_list.csv",
        mime="text/csv",
    )

    st.markdown("#### Stock status across the store")
    col_dist, col_cat = st.columns(2)

    with col_dist:
        dist = alerts["alert_level"].value_counts().reindex(
            ["Critical", "Warning", "OK", "Overstock"]
        ).fillna(0)
        fig = go.Figure(go.Bar(
            x=[f"{ALERT_ICON[k]} {ALERT_PLAIN[k]}" for k in dist.index],
            y=dist.values,
            marker_color=[ALERT_COLORS[k] for k in dist.index],
            text=dist.values, textposition="outside",
        ))
        fig.update_layout(
            height=320, margin=dict(t=30, b=10), yaxis_title="Number of items",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color=INK), title="By status",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_cat:
        # Category breakdown -- which categories are driving Critical/Warning alerts
        cat_urgent = alerts[alerts["alert_level"].isin(["Critical", "Warning"])]
        cat_counts = cat_urgent.groupby("Category").size().sort_values(ascending=False)
        fig_cat = go.Figure(go.Bar(
            x=cat_counts.values, y=cat_counts.index,
            orientation="h", marker_color=ACCENT,
            text=cat_counts.values, textposition="outside",
        ))
        fig_cat.update_layout(
            height=320, margin=dict(t=30, b=10), xaxis_title="Items needing attention",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color=INK), title="By category (order now + order soon)",
        )
        st.plotly_chart(fig_cat, use_container_width=True)

# =========================================================================
# TAB 2: DRILL-DOWN -- plain-English summary + gauge + why + trend
# =========================================================================
with tab_drilldown:
    st.markdown("#### Look up a specific item")

    stores = sorted(result["Store ID"].unique().tolist())
    c1, c2 = st.columns(2)
    with c1:
        sel_store = st.selectbox("Store", stores)
    with c2:
        products_for_store = sorted(
            result[result["Store ID"] == sel_store]["Product ID"].unique().tolist()
        )
        sel_product = st.selectbox("Item", products_for_store)

    item_hist = result[
        (result["Store ID"] == sel_store) & (result["Product ID"] == sel_product)
    ].sort_values(DATE_COL)

    item_alert = alerts[
        (alerts["Store ID"] == sel_store) & (alerts["Product ID"] == sel_product)
    ]

    st.markdown(" ")

    if len(item_alert) > 0:
        row = item_alert.iloc[0]
        color = ALERT_COLORS.get(row["alert_level"], "#888")
        days_left = row["days_of_stock_left"]
        trend_dir = get_trend_direction(row.get("demand_trend", 0))

        col_gauge, col_text = st.columns([1, 2.2])
        with col_gauge:
            st.plotly_chart(
                stock_health_gauge(days_left, row["alert_level"]),
                use_container_width=False,
                config={"displayModeBar": False},
            )
        with col_text:
            st.markdown(
                f"""
                <div class="stock-card">
                    <span class="pill" style="background-color:{color}">
                        {ALERT_ICON[row['alert_level']]} {ALERT_PLAIN[row['alert_level']]}
                    </span>
                    <span class="pill-soft" style="margin-left:6px;">
                        <span style="color:{TREND_COLOR[trend_dir]};font-weight:800;">
                            {TREND_ARROW[trend_dir]}
                        </span> {trend_dir} trend
                    </span>
                    <div class="stock-headline" style="margin-top:10px;">
                        {sel_product} at Store {sel_store}
                    </div>
                    <div class="stock-subline">{format_order_line(row)}</div>
                    <div class="why-line" style="margin-bottom:10px;">{explain_alert(row)}</div>
                    <div class="stat-row">
                        <div class="stat-block">
                            <div class="stat-num">{row['Inventory Level']:.0f}</div>
                            <div class="stat-label">units on hand</div>
                        </div>
                        <div class="stat-block">
                            <div class="stat-num">{row['Predicted_Demand']:.1f}</div>
                            <div class="stat-label">forecasted daily sales</div>
                        </div>
                        <div class="stat-block">
                            <div class="stat-num">{row['recommended_order_qty']:.0f}</div>
                            <div class="stat-label">units to order</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(" ")
    st.markdown("##### Sales trend & forecast")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=item_hist[DATE_COL], y=item_hist["Units Sold"],
        name="Actual sales", mode="lines", line=dict(color="#9AA3AD", width=1.5),
    ))
    fig2.add_trace(go.Scatter(
        x=item_hist[DATE_COL], y=item_hist["Predicted_Demand"],
        name="Forecasted demand", mode="lines+markers",
        line=dict(color=ACCENT, width=2.5),
    ))
    fig2.add_trace(go.Scatter(
        x=item_hist[DATE_COL], y=item_hist["Inventory Level"],
        name="Stock on hand", mode="lines", yaxis="y2",
        line=dict(color="#3E6FA8", width=1.5, dash="dash"),
    ))
    fig2.update_layout(
        height=420,
        yaxis=dict(title="Units sold / forecasted"),
        yaxis2=dict(title="Stock on hand", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=40),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK),
    )
    st.plotly_chart(fig2, use_container_width=True)

    with st.expander("View the underlying data for this item"):
        st.dataframe(item_hist, use_container_width=True)

# =========================================================================
# TAB 3: ORDER SUMMARY -- shareable, supplier-ready summary
# =========================================================================
with tab_summary:
    st.markdown("#### Order summary")
    st.caption("A ready-to-send summary of what needs ordering, grouped by category and store.")

    to_order = alerts[alerts["alert_level"].isin(["Critical", "Warning"])].copy()
    to_order = to_order.sort_values(["Category", "Store ID", "days_of_stock_left"])

    if len(to_order) == 0:
        st.info("Nothing needs ordering right now - nothing to summarize.")
    else:
        group_by = st.radio("Group summary by", ["Category", "Store ID"], horizontal=True)

        lines = [f"RESTOCK ORDER SUMMARY", f"Generated from current stock data", ""]
        total_units = 0
        for group_val, group_df in to_order.groupby(group_by):
            lines.append(f"== {group_by}: {group_val} ==")
            for _, row in group_df.iterrows():
                qty = int(row["recommended_order_qty"])
                total_units += qty
                urgency = "URGENT" if row["alert_level"] == "Critical" else "soon"
                lines.append(
                    f"  - {row['Product ID']} (Store {row['Store ID']}): "
                    f"order {qty} units [{urgency}]"
                )
            lines.append("")
        lines.append(f"TOTAL UNITS TO ORDER: {total_units}")
        summary_text = "\n".join(lines)

        st.text_area("Summary (copy or download below)", summary_text, height=380)

        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                "⬇ Download summary (.txt)",
                data=summary_text.encode("utf-8"),
                file_name="order_summary.txt",
                mime="text/plain",
            )
        with col_dl2:
            csv_export = to_order[[
                "Store ID", "Product ID", "Category", "Region",
                "Inventory Level", "recommended_order_qty", "alert_level"
            ]].rename(columns={"recommended_order_qty": "Order Qty"})
            st.download_button(
                "⬇ Download summary (.csv)",
                data=csv_export.to_csv(index=False).encode("utf-8"),
                file_name="order_summary.csv",
                mime="text/csv",
            )

# =========================================================================
# TAB 4: ABOUT -- technical detail tucked away, plain language up top
# =========================================================================
with tab_about:
    st.markdown("#### How this works")
    st.markdown(
        """
        This tool looks at each item's recent sales, price, promotions, and seasonal
        patterns to forecast how much will sell tomorrow - then compares that to what's
        currently on the shelf to tell you what needs restocking.

        - **Order now** - you'll likely run out within a few days
        - **Order soon** - worth reordering in the next week
        - **Well stocked** - no action needed
        - **Too much stock** - you have far more than you need; consider a promotion
        """
    )

    st.markdown("#### About the trend arrows and confidence label")
    st.markdown(
        f"""
        - **Trend arrow** compares this week's average daily sales to the last month's
          average. ↑ means sales are picking up, ↓ means they're slowing, → means steady.
        """
    )

    if metrics:
        st.markdown("#### Forecast accuracy")
        st.write(
            f"Tested against real historical sales, this tool's forecasts were within "
            f"about **{metrics.get('mape', 0):.0f}%** of actual demand on average."
        )
        with st.expander("Technical details"):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("MAE", f"{metrics.get('mae', 0):.2f} units")
            c2.metric("RMSE", f"{metrics.get('rmse', 0):.2f} units")
            c3.metric("MAPE", f"{metrics.get('mape', 0):.1f}%")
            c4.metric("R²", f"{metrics.get('r2', 0):.3f}")

            fi_path = os.path.join(ARTIFACT_DIR, "feature_importance.csv")
            if os.path.exists(fi_path):
                st.markdown("**What the forecast pays attention to:**")
                fi = pd.read_csv(fi_path).head(12)
                fig3 = px.bar(
                    fi.sort_values("importance"), x="importance", y="feature",
                    orientation="h", color_discrete_sequence=[ACCENT],
                )
                fig3.update_layout(
                    height=380, margin=dict(t=10),
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color=INK),
                )
                st.plotly_chart(fig3, use_container_width=True)

    st.markdown("#### How the order quantities are calculated")
    with st.expander("Show the formulas"):
        st.markdown(
            """
            - **Days of stock left** = units on hand ÷ forecasted daily demand
            - **Reorder point** = forecasted daily demand × (supplier lead time + safety buffer)
            - **Suggested order quantity** = reorder point − units currently on hand
              (never less than zero)

            You can adjust lead time and buffer days in the sidebar under
            **Restocking preferences** - the whole list updates instantly.
            """
        )
