import streamlit as st
import pandas as pd
import warnings
import plotly.express as px
import datetime
import calendar
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import re
import io

# Filter warnings for a clean output
warnings.filterwarnings("ignore")

# --- Page Configuration ---
st.set_page_config(page_title="YOY Dashboard", page_icon=":chart_with_upwards_trend:", layout="wide")

# --- Title and Logo ---
col1, col2 = st.columns([3, 1])
with col1:
    st.title("YOY Dashboard 📊")
with col2:
    st.image("logo.png", width=300)

# Initialize session state keys for data if not already set
if "sales_data" not in st.session_state:
    st.session_state["sales_data"] = None
   
# --- Sidebar ---
st.sidebar.header("YOY Dashboard")

# =============================================================================
# Helper Functions for Sales Data
# =============================================================================
def compute_custom_week(date):
    custom_dow = (date.weekday() + 2) % 7  # Saturday=0, Sunday=1, ..., Friday=6
    week_start = date - datetime.timedelta(days=custom_dow)
    week_end = week_start + datetime.timedelta(days=6)
    custom_year = week_end.year
    first_day = datetime.date(custom_year, 1, 1)
    first_day_custom_dow = (first_day.weekday() + 2) % 7
    first_week_start = first_day - datetime.timedelta(days=first_day_custom_dow)
    custom_week = ((week_start - first_week_start).days // 7) + 1
    return custom_week, custom_year, week_start, week_end

def get_custom_week_date_range(week_year, week_number):
    first_day = datetime.date(week_year, 1, 1)
    first_day_custom_dow = (first_day.weekday() + 2) % 7
    first_week_start = first_day - datetime.timedelta(days=first_day_custom_dow)
    week_start = first_week_start + datetime.timedelta(weeks=int(week_number) - 1)
    week_end = week_start + datetime.timedelta(days=6)
    return week_start, week_end

def get_quarter(week):
    if 1 <= week <= 13:
        return "Q1"
    elif 14 <= week <= 26:
        return "Q2"
    elif 27 <= week <= 39:
        return "Q3"
    elif 40 <= week <= 52:
        return "Q4"
    else:
        return None

def format_currency(value):
    return f"£{value:,.2f}"

def format_currency_int(value):
    return f"£{int(round(value)):,}"

@st.cache_data(show_spinner=False)
def load_data(file):
    if file.name.endswith(".csv"):
        data = pd.read_csv(file)
    else:
        data = pd.read_excel(file)
    return data

@st.cache_data(show_spinner=False)
def preprocess_data(data):
    required_cols = {"Week", "Year", "Sales Value (£)", "Date"}
    if not required_cols.issubset(set(data.columns)):
        st.error(f"Dataset is missing required columns: {required_cols}")
        st.stop()
    data["Date"] = pd.to_datetime(data["Date"], errors='coerce')
    data["Year_from_date"] = data["Date"].dt.year
    data["Custom_Week"], data["Custom_Week_Year"], data["Custom_Week_Start"], data["Custom_Week_End"] = zip(
        *data["Date"].apply(lambda d: compute_custom_week(d.date()))
    )
    data["Week"] = data["Custom_Week"]
    data["Quarter"] = data["Week"].apply(get_quarter)
    return data

def create_yoy_trends_chart(data, selected_years, selected_quarters,
                            selected_channels=None, selected_listings=None,
                            selected_products=None, time_grouping="Week"):
    filtered = data.copy()
    if selected_years:
        filtered = filtered[filtered["Custom_Week_Year"].isin(selected_years)]
    if selected_quarters:
        filtered = filtered[filtered["Quarter"].isin(selected_quarters)]
    if selected_channels and len(selected_channels) > 0:
        filtered = filtered[filtered["Sales Channel"].isin(selected_channels)]
    if selected_listings and len(selected_listings) > 0:
        filtered = filtered[filtered["Listing"].isin(selected_listings)]
    if selected_products and len(selected_products) > 0:
        filtered = filtered[filtered["Product"].isin(selected_products)]
    
    if time_grouping == "Week":
        grouped = filtered.groupby(["Custom_Week_Year", "Week"])["Sales Value (£)"].sum().reset_index()
        x_col = "Week"
        x_axis_label = "Week"
        grouped = grouped.sort_values(by=["Custom_Week_Year", "Week"])
        title = "Weekly Revenue Trends by Custom Week Year"
    else:
        grouped = filtered.groupby(["Custom_Week_Year", "Quarter"])["Sales Value (£)"].sum().reset_index()
        x_col = "Quarter"
        x_axis_label = "Quarter"
        quarter_order = ["Q1", "Q2", "Q3", "Q4"]
        grouped["Quarter"] = pd.Categorical(grouped["Quarter"], categories=quarter_order, ordered=True)
        grouped = grouped.sort_values(by=["Custom_Week_Year", "Quarter"])
        title = "Quarterly Revenue Trends by Custom Week Year"
    
    grouped["RevenueK"] = grouped["Sales Value (£)"] / 1000
    fig = px.line(grouped, x=x_col, y="Sales Value (£)", color="Custom_Week_Year", markers=True,
                  title=title,
                  labels={"Sales Value (£)": "Revenue (£)", x_col: x_axis_label},
                  custom_data=["RevenueK"])
    fig.update_traces(hovertemplate=f"{x_axis_label}: %{{x}}<br>Revenue: %{{customdata[0]:.1f}}K")
    
    if time_grouping == "Week":
        min_week = 0.8
        max_week = grouped["Week"].max() if not grouped["Week"].empty else 52
        fig.update_xaxes(range=[min_week, max_week])
    
    fig.update_yaxes(rangemode="tozero")
    fig.update_layout(margin=dict(t=50, b=50))
    return fig

def create_pivot_table(data, selected_years, selected_quarters, selected_channels,
                           selected_listings, selected_products, grouping_key="Listing"):
    filtered = data.copy()
    if selected_years:
        filtered = filtered[filtered["Custom_Week_Year"].isin(selected_years)]
    if selected_quarters:
        filtered = filtered[filtered["Quarter"].isin(selected_quarters)]
    if selected_channels and len(selected_channels) > 0:
        filtered = filtered[filtered["Sales Channel"].isin(selected_channels)]
    if selected_listings and len(selected_listings) > 0:
        filtered = filtered[filtered["Listing"].isin(selected_listings)]
    if grouping_key == "Product" and selected_products and len(selected_products) > 0:
        filtered = filtered[filtered["Product"].isin(selected_products)]
    
    pivot = pd.pivot_table(filtered, values="Sales Value (£)", index=grouping_key,
                           columns="Week", aggfunc="sum", fill_value=0)
    pivot["Total Revenue"] = pivot.sum(axis=1)
    pivot = pivot.round(0)
    new_columns = {}
    for col in pivot.columns:
        if isinstance(col, (int, float)):
            new_columns[col] = f"Week {int(col)}"
    pivot = pivot.rename(columns=new_columns)
    return pivot

def create_sku_line_chart(data, sku_text, selected_years, selected_channels=None, week_range=None):
    if "Product SKU" not in data.columns:
        st.error("The dataset does not contain a 'Product SKU' column.")
        st.stop()
    
    filtered = data.copy()
    filtered = filtered[filtered["Product SKU"].str.contains(sku_text, case=False, na=False)]
    if selected_years:
        filtered = filtered[filtered["Custom_Week_Year"].isin(selected_years)]
    if selected_channels and len(selected_channels) > 0:
        filtered = filtered[filtered["Sales Channel"].isin(selected_channels)]
    
    if week_range:
        filtered = filtered[(filtered["Custom_Week"] >= week_range[0]) & (filtered["Custom_Week"] <= week_range[1])]
    
    if filtered.empty:
        st.warning("No data available for the entered SKU and filters.")
        return None
    
    weekly_sku = filtered.groupby(["Custom_Week_Year", "Week"]).agg({
        "Sales Value (£)": "sum",
        "Order Quantity": "sum"
    }).reset_index().sort_values(by=["Custom_Week_Year", "Week"])
    
    weekly_sku["RevenueK"] = weekly_sku["Sales Value (£)"] / 1000
    if week_range:
        min_week, max_week = week_range
    else:
        min_week, max_week = 1, 52
    
    fig = px.line(weekly_sku, x="Week", y="Sales Value (£)", color="Custom_Week_Year", markers=True,
                  title=f"Weekly Revenue Trends for SKU matching: '{sku_text}'",
                  labels={"Sales Value (£)": "Revenue (£)"},
                  custom_data=["RevenueK", "Order Quantity"])
    fig.update_traces(hovertemplate="Week: %{x}<br>Revenue: %{customdata[0]:.1f}K<br>Units Sold: %{customdata[1]}")
    fig.update_layout(xaxis=dict(tickmode="linear", range=[min_week, max_week]),
                      margin=dict(t=50, b=50))
    return fig

def create_daily_price_chart(data, listing, selected_years, selected_quarters, selected_channels, week_range=None):
    if "Date" not in data.columns:
        st.error("The dataset does not contain a 'Date' column required for daily price analysis.")
        return None
    
    df_listing = data[(data["Listing"] == listing) & (data["Year"].isin(selected_years))].copy()
    if selected_quarters:
        df_listing = df_listing[df_listing["Quarter"].isin(selected_quarters)]
    if selected_channels and len(selected_channels) > 0:
        df_listing = df_listing[df_listing["Sales Channel"].isin(selected_channels)]
    # Filter by week range if provided
    if week_range:
        start_week, end_week = week_range
        df_listing = df_listing[(df_listing["Custom_Week"] >= start_week) & (df_listing["Custom_Week"] <= end_week)]
    
    if df_listing.empty:
        st.warning(f"No data available for {listing} for the selected filters.")
        return None
    if selected_channels and len(selected_channels) > 0:
        unique_currencies = df_listing["Original Currency"].unique()
        display_currency = unique_currencies[0] if len(unique_currencies) > 0 else "GBP"
    else:
        display_currency = "GBP"
    df_listing["Date"] = pd.to_datetime(df_listing["Date"])
    grouped = df_listing.groupby([df_listing["Date"].dt.date, "Year"]).agg({
        "Sales Value in Transaction Currency": "sum",
        "Order Quantity": "sum"
    }).reset_index()
    grouped.rename(columns={"Date": "Date"}, inplace=True)
    grouped["Average Price"] = grouped["Sales Value in Transaction Currency"] / grouped["Order Quantity"]
    grouped["Date"] = pd.to_datetime(grouped["Date"])
    dfs = []
    for yr in selected_years:
        df_year = grouped[grouped["Year"] == yr].copy()
        if df_year.empty:
            continue
        df_year["Day"] = df_year["Date"].dt.dayofyear
        start_day = int(df_year["Day"].min())
        end_day = int(df_year["Day"].max())
        df_year = df_year.set_index("Day").reindex(range(start_day, end_day + 1))
        df_year.index.name = "Day"
        df_year["Average Price"] = df_year["Average Price"].ffill()
        prices = df_year["Average Price"].values.copy()
        for i in range(1, len(prices)):
            if prices[i] < 0.75 * prices[i-1]:
                prices[i] = prices[i-1]
            if prices[i] > 1.25 * prices[i-1]:
                prices[i] = prices[i-1]
        df_year["Average Price"] = prices
        df_year["Year"] = yr
        df_year = df_year.reset_index()
        dfs.append(df_year)
    if not dfs:
        st.warning("No data available after processing for the selected filters.")
        return None
    combined = pd.concat(dfs, ignore_index=True)
    fig = px.line(
        combined, 
        x="Day", 
        y="Average Price", 
        color="Year",
        title=f"Daily Average Price for {listing}",
        labels={"Day": "Day of Year", "Average Price": f"Average Price ({display_currency})"},
        color_discrete_sequence=px.colors.qualitative.Set1
    )
    fig.update_yaxes(rangemode="tozero")
    fig.update_layout(margin=dict(t=50, b=50))
    return fig

# =============================================================================
# Main Dashboard Code
# =============================================================================

# --- Sales Data File Uploader ---
uploaded_file = st.sidebar.file_uploader("Upload Sales Data (CSV or Excel)", type=["csv", "xlsx"], key="sales_file")
if uploaded_file is not None:
    st.session_state["sales_data"] = load_data(uploaded_file)
if st.session_state["sales_data"] is None:
    st.info("Please use the file uploader in the sidebar to upload your Sales Data and view the dashboard.")
    st.stop()

df = st.session_state["sales_data"]
df = preprocess_data(df)

available_custom_years = sorted(df["Custom_Week_Year"].dropna().unique())
if not available_custom_years:
    st.error("No custom week year data available.")
    st.stop()
current_custom_year = available_custom_years[-1]
if len(available_custom_years) >= 2:
    prev_custom_year = available_custom_years[-2]
    yoy_default_years = [prev_custom_year, current_custom_year]
else:
    selected_daily_quarters = []
    yoy_default_years = [current_custom_year]
default_current_year = [current_custom_year]

tabs = st.tabs([
    "KPIs", 
    "YOY Trends", 
    "Daily Prices", 
    "SKU Trends", 
    "Pivot Table", 
    "Unrecognised Sales"
])

# -------------------------------
# Tab 1: KPIs
# -------------------------------
with tabs[0]:
    st.markdown("### Key Performance Indicators")
    with st.expander("KPI Filters", expanded=False):
        today = datetime.date.today()
        available_weeks = sorted(df[df["Custom_Week_Year"] == current_custom_year]["Custom_Week"].dropna().unique())
        full_weeks = [wk for wk in available_weeks if get_custom_week_date_range(current_custom_year, wk)[1] <= today]
        default_week = full_weeks[-1] if full_weeks else (available_weeks[-1] if available_weeks else 1)
        selected_week = st.selectbox(
            "Select Week for KPI Calculation",
            options=available_weeks,
            index=available_weeks.index(default_week) if default_week in available_weeks else 0,
            key="kpi_week",
            help="Select the week to calculate KPIs for. (Defaults to the last full week)"
        )
        week_start_custom, week_end_custom = get_custom_week_date_range(current_custom_year, selected_week)
        st.info(f"Week {selected_week}: {week_start_custom.strftime('%d %b')} - {week_end_custom.strftime('%d %b, %Y')}")
    kpi_data = df[df["Custom_Week"] == selected_week]
    revenue_summary = kpi_data.groupby("Custom_Week_Year")["Sales Value (£)"].sum()
    if "Order Quantity" in kpi_data.columns:
        units_summary = kpi_data.groupby("Custom_Week_Year")["Order Quantity"].sum()
    else:
        units_summary = None
    all_custom_years = sorted(df["Custom_Week_Year"].dropna().unique())
    kpi_cols = st.columns(len(all_custom_years))
    for idx, year in enumerate(all_custom_years):
        with kpi_cols[idx]:
            revenue = revenue_summary.get(year, 0)
            if idx > 0:
                prev_rev = revenue_summary.get(all_custom_years[idx - 1], 0)
                delta_rev = revenue - prev_rev
                delta_rev_str = f"{int(round(delta_rev)):,}"
            else:
                delta_rev_str = ""
            if revenue == 0:
                st.metric(label=f"Revenue {year} (Week {selected_week})", value="N/A")
            else:
                st.metric(
                    label=f"Revenue {year} (Week {selected_week})",
                    value=format_currency_int(revenue),
                    delta=delta_rev_str
                )
            if units_summary is not None:
                total_units = units_summary.get(year, 0)
                if idx > 0:
                    prev_units = units_summary.get(all_custom_years[idx - 1], 0)
                    if prev_units != 0:
                        delta_units_percent = ((total_units - prev_units) / prev_units) * 100
                    else:
                        delta_units_percent = 0
                    delta_units_str = f"{delta_units_percent:.1f}%"
                else:
                    delta_units_str = ""
                st.metric(
                    label=f"Total Units Sold {year} (Week {selected_week})",
                    value=f"{total_units:,}",
                    delta=delta_units_str
                )
                aov = revenue / total_units if total_units != 0 else 0
                if idx > 0:
                    prev_total_units = units_summary.get(all_custom_years[idx - 1], 0)
                    prev_rev = revenue_summary.get(all_custom_years[idx - 1], 0)
                    prev_aov = prev_rev / prev_total_units if prev_total_units != 0 else 0
                    if prev_aov != 0:
                        delta_aov_percent = ((aov - prev_aov) / prev_aov) * 100
                    else:
                        delta_aov_percent = 0
                    delta_aov_str = f"{delta_aov_percent:.1f}%"
                else:
                    delta_aov_str = ""
                st.metric(
                    label=f"AOV {year} (Week {selected_week})",
                    value=f"£{aov:,.2f}",
                    delta=delta_aov_str
                )

# -----------------------------------------
# Tab 2: YOY Trends
# -----------------------------------------
with tabs[1]:
    st.markdown("### YOY Weekly Revenue Trends")
    with st.expander("Chart Filters", expanded=False):
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            yoy_years = st.multiselect("Year(s)", options=available_custom_years, default=yoy_default_years, key="yoy_years")
        with col2:
            quarter_options = ["All Quarters", "Q1", "Q2", "Q3", "Q4", "Custom..."]
            quarter_selection = st.selectbox("Quarter(s)", options=quarter_options, index=0, key="quarter_dropdown_yoy")
            if quarter_selection == "All Quarters":
                selected_quarters = ["Q1", "Q2", "Q3", "Q4"]
            elif quarter_selection == "Custom...":
                selected_quarters = st.multiselect("Select quarters", options=["Q1", "Q2", "Q3", "Q4"], default=["Q1", "Q2", "Q3", "Q4"], key="custom_quarters_yoy")
            else:
                selected_quarters = [quarter_selection]
        with col3:
            selected_channels = st.multiselect("Channel(s)", options=sorted(df["Sales Channel"].dropna().unique()), default=[], key="yoy_channels")
        with col4:
            selected_listings = st.multiselect("Listing(s)", options=sorted(df["Listing"].dropna().unique()), default=[], key="yoy_listings")
        with col5:
            if selected_listings:
                product_options = sorted(df[df["Listing"].isin(selected_listings)]["Product"].dropna().unique())
            else:
                product_options = sorted(df["Product"].dropna().unique())
            selected_products = st.multiselect("Product(s)", options=product_options, default=[], key="yoy_products")
        time_grouping = "Week"
    fig_yoy = create_yoy_trends_chart(df, yoy_years, selected_quarters, selected_channels, selected_listings, selected_products, time_grouping=time_grouping)
    st.plotly_chart(fig_yoy, use_container_width=True)
    st.markdown("### Revenue Summary")
    st.markdown("")
    filtered_df = df.copy()
    if yoy_years:
        filtered_df = filtered_df[filtered_df["Custom_Week_Year"].isin(yoy_years)]
    if selected_quarters:
        filtered_df = filtered_df[filtered_df["Quarter"].isin(selected_quarters)]
    if selected_channels:
        filtered_df = filtered_df[filtered_df["Sales Channel"].isin(selected_channels)]
    if selected_listings:
        filtered_df = filtered_df[filtered_df["Listing"].isin(selected_listings)]
    if selected_products:
        filtered_df = filtered_df[filtered_df["Product"].isin(selected_products)]
    df_revenue = filtered_df.copy()
    if df_revenue.empty:
        st.info("No data available for the selected filters to build the revenue summary table.")
    else:
        df_revenue["Custom_Week_Year"] = df_revenue["Custom_Week_Year"].astype(int)
        df_revenue["Week"] = df_revenue["Week"].astype(int)
        filtered_current_year = df_revenue["Custom_Week_Year"].max()
        df_revenue_current = df_revenue[df_revenue["Custom_Week_Year"] == filtered_current_year].copy()
        df_revenue_current["Week_Start"] = df_revenue_current.apply(lambda row: row["Custom_Week_Start"], axis=1)
        df_revenue_current["Week_End"] = df_revenue_current.apply(lambda row: row["Custom_Week_End"], axis=1)
        today = datetime.date.today()
        current_week_info = compute_custom_week(today)
        last_complete_week_end = current_week_info[3] if today > current_week_info[3] else current_week_info[3] - datetime.timedelta(weeks=1)
        df_full_weeks_current = df_revenue_current[df_revenue_current["Week_End"] <= last_complete_week_end].copy()
        unique_weeks_current = (df_full_weeks_current.groupby(["Custom_Week_Year", "Week"])
                                .first().reset_index()[["Custom_Week_Year", "Week", "Week_End"]]
                                .sort_values("Week_End"))
        if unique_weeks_current.empty:
            st.info("Not enough complete week data in the filtered current year to build the revenue summary table.")
        else:
            last_complete_week_row_current = unique_weeks_current.iloc[-1]
            last_week_tuple_current = (last_complete_week_row_current["Custom_Week_Year"], last_complete_week_row_current["Week"])
            last_week_number = last_week_tuple_current[1]
            last_4_weeks_current = unique_weeks_current.tail(4)
            last_4_week_numbers = last_4_weeks_current["Week"].tolist()
            grouping_key = "Product" if (selected_listings and len(selected_listings) == 1) else "Listing"
            rev_last_4_current = (df_full_weeks_current[df_full_weeks_current["Week"].isin(last_4_week_numbers)]
                                 .groupby(grouping_key)["Sales Value (£)"].sum()
                                 .rename("Last 4 Weeks Revenue (Current Year)").round(0).astype(int))
            rev_last_1_current = (df_full_weeks_current[df_full_weeks_current.apply(lambda row: (row["Custom_Week_Year"], row["Week"]) == last_week_tuple_current, axis=1)]
                                 .groupby(grouping_key)["Sales Value (£)"].sum()
                                 .rename("Last Week Revenue (Current Year)").round(0).astype(int))
            if len(filtered_df["Custom_Week_Year"].unique()) >= 2:
                filtered_years = sorted(filtered_df["Custom_Week_Year"].unique())
                last_year = filtered_years[-2]
                df_revenue_last_year = df_revenue[df_revenue["Custom_Week_Year"] == last_year].copy()
                rev_last_1_last_year = (df_revenue_last_year[df_revenue_last_year["Week"] == last_week_number]
                                        .groupby(grouping_key)["Sales Value (£)"].sum()
                                        .rename("Last Week Revenue (Last Year)").round(0).astype(int))
                rev_last_4_last_year = (df_revenue_last_year[df_revenue_last_year["Week"].isin(last_4_week_numbers)]
                                        .groupby(grouping_key)["Sales Value (£)"].sum()
                                        .rename("Last 4 Weeks Revenue (Last Year)").round(0).astype(int))
            else:
                rev_last_4_last_year = pd.Series(dtype=float, name="Last 4 Weeks Revenue (Last Year)")
                rev_last_1_last_year = pd.Series(dtype=float, name="Last Week Revenue (Last Year)")
            all_keys_current = pd.Series(sorted(df_revenue_current[grouping_key].unique()), name=grouping_key)
            revenue_summary = pd.DataFrame(all_keys_current).set_index(grouping_key)
            revenue_summary = revenue_summary.join(rev_last_4_current, how="left")\
                                             .join(rev_last_1_current, how="left")\
                                             .join(rev_last_4_last_year, how="left")\
                                             .join(rev_last_1_last_year, how="left")
            revenue_summary = revenue_summary.fillna(0).reset_index()
            revenue_summary["Last 4 Weeks Diff"] = revenue_summary["Last 4 Weeks Revenue (Current Year)"] - revenue_summary["Last 4 Weeks Revenue (Last Year)"]
            revenue_summary["Last Week Diff"] = revenue_summary["Last Week Revenue (Current Year)"] - revenue_summary["Last Week Revenue (Last Year)"]
            revenue_summary["Last 4 Weeks % Change"] = revenue_summary.apply(lambda row: (row["Last 4 Weeks Diff"] / row["Last 4 Weeks Revenue (Last Year)"] * 100) if row["Last 4 Weeks Revenue (Last Year)"] != 0 else 0, axis=1)
            revenue_summary["Last Week % Change"] = revenue_summary.apply(lambda row: (row["Last Week Diff"] / row["Last Week Revenue (Last Year)"] * 100) if row["Last Week Revenue (Last Year)"] != 0 else 0, axis=1)
            desired_order = [grouping_key,
                             "Last 4 Weeks Revenue (Current Year)",
                             "Last 4 Weeks Revenue (Last Year)",
                             "Last 4 Weeks Diff",
                             "Last 4 Weeks % Change",
                             "Last Week Revenue (Current Year)",
                             "Last Week Revenue (Last Year)",
                             "Last Week Diff",
                             "Last Week % Change"]
            revenue_summary = revenue_summary[desired_order]
            
            # ---- Total Summary Table ----
            summary_row = {
                grouping_key: "Total",
                "Last 4 Weeks Revenue (Current Year)": revenue_summary["Last 4 Weeks Revenue (Current Year)"].sum(),
                "Last 4 Weeks Revenue (Last Year)": revenue_summary["Last 4 Weeks Revenue (Last Year)"].sum(),
                "Last Week Revenue (Current Year)": revenue_summary["Last Week Revenue (Current Year)"].sum(),
                "Last Week Revenue (Last Year)": revenue_summary["Last Week Revenue (Last Year)"].sum(),
                "Last 4 Weeks Diff": revenue_summary["Last 4 Weeks Diff"].sum(),
                "Last Week Diff": revenue_summary["Last Week Diff"].sum()
            }
            total_last4_last_year = summary_row["Last 4 Weeks Revenue (Last Year)"]
            total_last_week_last_year = summary_row["Last Week Revenue (Last Year)"]
            summary_row["Last 4 Weeks % Change"] = (summary_row["Last 4 Weeks Diff"] / total_last4_last_year * 100) if total_last4_last_year != 0 else 0
            summary_row["Last Week % Change"] = (summary_row["Last Week Diff"] / total_last_week_last_year * 100) if total_last_week_last_year != 0 else 0
            total_df = pd.DataFrame([summary_row])[desired_order]
            def color_diff(val):
                try:
                    if val < 0:
                        return 'color: red'
                    elif val > 0:
                        return 'color: green'
                    else:
                        return ''
                except Exception:
                    return ''
            styled_total = total_df.style.format({
                "Last 4 Weeks Revenue (Current Year)": "{:,}",
                "Last 4 Weeks Revenue (Last Year)": "{:,}",
                "Last Week Revenue (Current Year)": "{:,}",
                "Last Week Revenue (Last Year)": "{:,}",
                "Last 4 Weeks Diff": "{:,.0f}",
                "Last Week Diff": "{:,.0f}",
                "Last 4 Weeks % Change": "{:.1f}%",
                "Last Week % Change": "{:.1f}%"
            }).applymap(color_diff, subset=["Last 4 Weeks Diff", "Last Week Diff", "Last 4 Weeks % Change", "Last Week % Change"])\
              .set_properties(**{'font-weight': 'bold'})
            st.markdown("##### Total Summary")
            st.dataframe(styled_total, use_container_width=True)
            
            # ---- Detailed Summary Table ----
            styled_main = revenue_summary.style.format({
                "Last 4 Weeks Revenue (Current Year)": "{:,}",
                "Last 4 Weeks Revenue (Last Year)": "{:,}",
                "Last Week Revenue (Current Year)": "{:,}",
                "Last Week Revenue (Last Year)": "{:,}",
                "Last 4 Weeks Diff": "{:,.0f}",
                "Last Week Diff": "{:,.0f}",
                "Last 4 Weeks % Change": "{:.1f}%",
                "Last Week % Change": "{:.1f}%"
            }).applymap(color_diff, subset=["Last 4 Weeks Diff", "Last Week Diff", "Last 4 Weeks % Change", "Last Week % Change"])
            st.markdown("##### Detailed Summary")
            st.dataframe(styled_main, use_container_width=True)
        
# -------------------------------
# Tab 3: Daily Prices
# -------------------------------
with tabs[2]:
    st.markdown("### Daily Prices for Top Listings")
    with st.expander("Daily Price Filters", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            default_daily_years = [year for year in available_custom_years if year in (2024, 2025)]
            if not default_daily_years:
                default_daily_years = [current_custom_year]
            selected_daily_years = st.multiselect("Select Year(s)", options=available_custom_years, default=default_daily_years, key="daily_years", help="Default shows 2024 and 2025 if available.")
        with col2:
            quarter_options = ["All Quarters", "Q1", "Q2", "Q3", "Q4", "Custom..."]
            quarter_selection = st.selectbox("Quarter(s)", options=quarter_options, index=0, key="quarter_dropdown_daily")
            if quarter_selection == "Custom...":
                selected_daily_quarters = st.multiselect("Select Quarter(s)", options=["Q1", "Q2", "Q3", "Q4"], default=[], key="daily_quarters_custom", help="Select one or more quarters to filter.")
            elif quarter_selection == "All Quarters":
                selected_daily_quarters = ["Q1", "Q2", "Q3", "Q4"]
            else:
                selected_daily_quarters = [quarter_selection]
        with col3:
            selected_daily_channels = st.multiselect("Select Sales Channel(s)", options=sorted(df["Sales Channel"].dropna().unique()), default=[], key="daily_channels", help="Select one or more sales channels to filter the daily price data.")
        with col4:
            daily_week_range = st.slider("Select Week Range", min_value=1, max_value=52, value=(1, 52), step=1, key="daily_week_range", help="Select the range of weeks to display in the Daily Prices section.")
    main_listings = ["Pattern Pants", "Pattern Shorts", "Solid Pants", "Solid Shorts", "Patterned Polos"]
    for listing in main_listings:
        st.subheader(listing)
        fig_daily = create_daily_price_chart(df, listing, selected_daily_years, selected_daily_quarters, selected_daily_channels, week_range=daily_week_range)
        if fig_daily:
            st.plotly_chart(fig_daily, use_container_width=True)
    st.markdown("### Daily Prices Comparison")
    with st.expander("Comparison Chart Filters", expanded=False):
        comp_col1, comp_col2, comp_col3 = st.columns(3)
        with comp_col1:
            comp_years = st.multiselect("Select Year(s)", options=available_custom_years, default=default_daily_years, key="comp_years", help="Select the year(s) for the comparison chart.")
        with comp_col2:
            comp_quarter_options = ["All Quarters", "Q1", "Q2", "Q3", "Q4", "Custom..."]
            comp_quarter_selection = st.selectbox("Quarter(s)", options=comp_quarter_options, index=0, key="quarter_dropdown_comp")
            if comp_quarter_selection == "Custom...":
                comp_quarters = st.multiselect("Select Quarter(s)", options=["Q1", "Q2", "Q3", "Q4"], default=[], key="comp_quarters_custom", help="Select one or more quarters for comparison.")
            elif comp_quarter_selection == "All Quarters":
                comp_quarters = ["Q1", "Q2", "Q3", "Q4"]
            else:
                comp_quarters = [comp_quarter_selection]
        with comp_col3:
            comp_channels = st.multiselect("Select Sales Channel(s)", options=sorted(df["Sales Channel"].dropna().unique()), default=[], key="comp_channels", help="Select the sales channel(s) for the comparison chart.")
            comp_listing = st.selectbox("Select Listing", options=sorted(df["Listing"].dropna().unique()), key="comp_listing", help="Select a listing for daily prices comparison.")
        fig_comp = create_daily_price_chart(df, comp_listing, comp_years, comp_quarters, comp_channels)
        if fig_comp:
            st.plotly_chart(fig_comp, use_container_width=True)

# -------------------------------
# Tab 4: SKU Trends 
# -------------------------------
with tabs[3]:
    st.markdown("### SKU Trends")
    if "Product SKU" not in df.columns:
        st.error("The dataset does not contain a 'Product SKU' column.")
    else:
        with st.expander("Chart Filters", expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                sku_text = st.text_input("Enter Product SKU", value="", key="sku_input", help="Enter a SKU (or part of it) to display its weekly revenue trends.")
            with col2:
                sku_years = st.multiselect("Select Year(s)", options=available_custom_years, default=default_current_year, key="sku_years", help="Default is the current custom week year.")
            with col3:
                sku_channels = st.multiselect("Select Sales Channel(s)", options=sorted(df["Sales Channel"].dropna().unique()), default=[], key="sku_channels", help="Select one or more sales channels to filter SKU trends. If empty, all channels are shown.")
            with col4:
                week_range = st.slider("Select Week Range", min_value=1, max_value=52, value=(1, 52), step=1, key="sku_week_range", help="Select the range of weeks to display.")
        
        if sku_text.strip() == "":
            st.info("Please enter a Product SKU to view its trends.")
        else:
            fig_sku = create_sku_line_chart(df, sku_text, sku_years, selected_channels=sku_channels, week_range=week_range)
            if fig_sku is not None:
                st.plotly_chart(fig_sku, use_container_width=True)
            
            filtered_sku_data = df[df["Product SKU"].str.contains(sku_text, case=False, na=False)]
            if sku_years:
                filtered_sku_data = filtered_sku_data[filtered_sku_data["Custom_Week_Year"].isin(sku_years)]
            if sku_channels and len(sku_channels) > 0:
                filtered_sku_data = filtered_sku_data[filtered_sku_data["Sales Channel"].isin(sku_channels)]
            if week_range:
                filtered_sku_data = filtered_sku_data[(filtered_sku_data["Custom_Week"] >= week_range[0]) & (filtered_sku_data["Custom_Week"] <= week_range[1])]
            if "Order Quantity" in filtered_sku_data.columns:
                total_units = filtered_sku_data.groupby("Custom_Week_Year")["Order Quantity"].sum().reset_index()
                total_units_summary = total_units.set_index("Custom_Week_Year").T
                total_units_summary.index = ["Total Units Sold"]
                st.markdown("##### Total Units Sold Summary")
                st.dataframe(total_units_summary, use_container_width=True)
                sku_units = filtered_sku_data.groupby(["Product SKU", "Custom_Week_Year"])["Order Quantity"].sum().reset_index()
                sku_pivot = sku_units.pivot(index="Product SKU", columns="Custom_Week_Year", values="Order Quantity")
                sku_pivot = sku_pivot.fillna(0).astype(int)
                st.markdown("##### SKU Breakdown (Units Sold by Custom Week Year)")
                st.dataframe(sku_pivot, use_container_width=True)
            else:
                st.info("No 'Order Quantity' data available to show units sold.")

# -------------------------------
# Tab 5: Pivot Table: Revenue by Week
# -------------------------------
with tabs[4]:
    st.markdown("### Pivot Table: Revenue by Week")
    with st.expander("Pivot Table Filters", expanded=False):
        pivot_years = st.multiselect("Select Year(s) for Pivot Table", options=available_custom_years, default=default_current_year, key="pivot_years", help="Default is the current custom week year.")
        pivot_quarters = st.multiselect("Select Quarter(s)", options=["Q1", "Q2", "Q3", "Q4"], default=["Q1", "Q2", "Q3", "Q4"], key="pivot_quarters", help="Select one or more quarters to filter by.")
        pivot_channels = st.multiselect("Select Sales Channel(s)", options=sorted(df["Sales Channel"].dropna().unique()), default=[], key="pivot_channels", help="Select one or more channels to filter. If empty, all channels are shown.")
        pivot_listings = st.multiselect("Select Listing(s)", options=sorted(df["Listing"].dropna().unique()), default=[], key="pivot_listings", help="Select one or more listings to filter. If empty, all listings are shown.")
        if pivot_listings and len(pivot_listings) == 1:
            pivot_product_options = sorted(df[df["Listing"] == pivot_listings[0]]["Product"].dropna().unique())
        else:
            pivot_product_options = sorted(df["Product"].dropna().unique())
        pivot_products = st.multiselect("Select Product(s)", options=pivot_product_options, default=[], key="pivot_products", help="Select one or more products to filter (only applies if a specific listing is selected).")
    grouping_key = "Product" if (pivot_listings and len(pivot_listings) == 1) else "Listing"
    effective_products = pivot_products if grouping_key == "Product" else []
    pivot = create_pivot_table(df, selected_years=pivot_years, selected_quarters=pivot_quarters, selected_channels=pivot_channels, selected_listings=pivot_listings, selected_products=effective_products, grouping_key=grouping_key)
    if len(pivot_years) == 1:
        year_for_date = int(pivot_years[0])
        new_columns = []
        for col in pivot.columns:
            if col == "Total Revenue":
                new_columns.append((col, "Total Revenue"))
            else:
                try:
                    week_number = int(col.split()[1])
                    mon, fri = get_custom_week_date_range(year_for_date, week_number)
                    date_range = f"{mon.strftime('%d %b')} - {fri.strftime('%d %b')}" if mon and fri else ""
                    new_columns.append((col, date_range))
                except Exception:
                    new_columns.append((col, ""))
        pivot.columns = pd.MultiIndex.from_tuples(new_columns)
    st.dataframe(pivot, use_container_width=True)

# -------------------------------
# Tab 6: Unrecognised Sales
# -------------------------------
with tabs[5]:
    st.markdown("### Unrecognised Sales")
    unrecognised_sales = df[df["Listing"].str.contains("unrecognised", case=False, na=False)]
    columns_to_drop = ["Year", "Weekly Sales Value (£)", "YOY Growth (%)"]
    unrecognised_sales = unrecognised_sales.drop(columns=columns_to_drop, errors='ignore')
    if unrecognised_sales.empty:
        st.info("No unrecognised sales found.")
    else:
        st.dataframe(unrecognised_sales, use_container_width=True)
