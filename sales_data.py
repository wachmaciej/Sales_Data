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

def create_revenue_drop_report(df, marketplace="Amazon US", num_listings=5, num_products=3):
    if marketplace:
        marketplace_df = df[df["Sales Channel"] == marketplace].copy()
    else:
        marketplace_df = df.copy()
    
    if marketplace_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    current_year = marketplace_df["Custom_Week_Year"].max()
    last_year = current_year - 1
    
    if last_year not in marketplace_df["Custom_Week_Year"].unique():
        return pd.DataFrame([{"Error": f"No data available for {last_year} for comparison"}]), pd.DataFrame()
    
    current_year_data = marketplace_df[marketplace_df["Custom_Week_Year"] == current_year]
    last_week_number = current_year_data["Week"].max()
    
    if current_year_data["Week"].nunique() < 4:
        return pd.DataFrame([{"Error": "Not enough weeks available for analysis (need at least 4 weeks)"}]), pd.DataFrame()
    
    last_4_weeks = sorted(current_year_data["Week"].unique())[-4:]
    
    last_week_current = marketplace_df[(marketplace_df["Custom_Week_Year"] == current_year) & 
                                       (marketplace_df["Week"] == last_week_number)]
    last_week_previous = marketplace_df[(marketplace_df["Custom_Week_Year"] == last_year) & 
                                        (marketplace_df["Week"] == last_week_number)]
    
    last_4_weeks_current = marketplace_df[(marketplace_df["Custom_Week_Year"] == current_year) & 
                                          (marketplace_df["Week"].isin(last_4_weeks))]
    last_4_weeks_previous = marketplace_df[(marketplace_df["Custom_Week_Year"] == last_year) & 
                                           (marketplace_df["Week"].isin(last_4_weeks))]
    
    last_week_current_listings = last_week_current.groupby("Listing")["Sales Value (£)"].sum().reset_index()
    last_week_previous_listings = last_week_previous.groupby("Listing")["Sales Value (£)"].sum().reset_index()
    
    last_4_weeks_current_listings = last_4_weeks_current.groupby("Listing")["Sales Value (£)"].sum().reset_index()
    last_4_weeks_previous_listings = last_4_weeks_previous.groupby("Listing")["Sales Value (£)"].sum().reset_index()
    
    listings_last_week = pd.merge(
        last_week_current_listings, 
        last_week_previous_listings, 
        on="Listing", 
        how="outer", 
        suffixes=("_current", "_previous")
    ).fillna(0)
    
    listings_last_4_weeks = pd.merge(
        last_4_weeks_current_listings, 
        last_4_weeks_previous_listings, 
        on="Listing", 
        how="outer", 
        suffixes=("_current", "_previous")
    ).fillna(0)
    
    listings_last_week["Revenue_Change_Week"] = listings_last_week["Sales Value (£)_current"] - listings_last_week["Sales Value (£)_previous"]
    listings_last_week["Revenue_Change_Week_Pct"] = listings_last_week.apply(
        lambda row: (row["Revenue_Change_Week"] / row["Sales Value (£)_previous"] * 100) 
        if row["Sales Value (£)_previous"] > 0 else 0, 
        axis=1
    )
    
    listings_last_4_weeks["Revenue_Change_4Weeks"] = listings_last_4_weeks["Sales Value (£)_current"] - listings_last_4_weeks["Sales Value (£)_previous"]
    listings_last_4_weeks["Revenue_Change_4Weeks_Pct"] = listings_last_4_weeks.apply(
        lambda row: (row["Revenue_Change_4Weeks"] / row["Sales Value (£)_previous"] * 100) 
        if row["Sales Value (£)_previous"] > 0 else 0, 
        axis=1
    )
    
    listings_analysis = pd.merge(
        listings_last_week[["Listing", "Sales Value (£)_current", "Sales Value (£)_previous", 
                            "Revenue_Change_Week", "Revenue_Change_Week_Pct"]],
        listings_last_4_weeks[["Listing", "Sales Value (£)_current", "Sales Value (£)_previous", 
                              "Revenue_Change_4Weeks", "Revenue_Change_4Weeks_Pct"]],
        on="Listing",
        suffixes=("_week", "_4weeks")
    )
    
    listings_analysis = listings_analysis.rename(columns={
        "Sales Value (£)_current_week": "Last_Week_Revenue",
        "Sales Value (£)_previous_week": "Last_Week_Previous_Year",
        "Sales Value (£)_current_4weeks": "Last_4Weeks_Revenue",
        "Sales Value (£)_previous_4weeks": "Last_4Weeks_Previous_Year"
    })
    
    listings_analysis = listings_analysis.sort_values("Revenue_Change_Week")
    
    top_drop_listings = listings_analysis.head(num_listings)
    
    products_analysis = []
    
    for _, listing_row in top_drop_listings.iterrows():
        listing_name = listing_row["Listing"]
        listing_data = marketplace_df[marketplace_df["Listing"] == listing_name]
        
        last_week_current_products = listing_data[
            (listing_data["Custom_Week_Year"] == current_year) & 
            (listing_data["Week"] == last_week_number)
        ].groupby("Product")["Sales Value (£)"].sum().reset_index()
        
        last_week_previous_products = listing_data[
            (listing_data["Custom_Week_Year"] == last_year) & 
            (listing_data["Week"] == last_week_number)
        ].groupby("Product")["Sales Value (£)"].sum().reset_index()
        
        last_4_weeks_current_products = listing_data[
            (listing_data["Custom_Week_Year"] == current_year) & 
            (listing_data["Week"].isin(last_4_weeks))
        ].groupby("Product")["Sales Value (£)"].sum().reset_index()
        
        last_4_weeks_previous_products = listing_data[
            (listing_data["Custom_Week_Year"] == last_year) & 
            (listing_data["Week"].isin(last_4_weeks))
        ].groupby("Product")["Sales Value (£)"].sum().reset_index()
        
        products_last_week = pd.merge(
            last_week_current_products,
            last_week_previous_products,
            on="Product",
            how="outer",
            suffixes=("_current", "_previous")
        ).fillna(0)
        
        products_last_4_weeks = pd.merge(
            last_4_weeks_current_products,
            last_4_weeks_previous_products,
            on="Product",
            how="outer",
            suffixes=("_current", "_previous")
        ).fillna(0)
        
        products_last_week["Revenue_Change_Week"] = (
            products_last_week["Sales Value (£)_current"] - products_last_week["Sales Value (£)_previous"]
        )
        products_last_week["Revenue_Change_Week_Pct"] = products_last_week.apply(
            lambda row: (row["Revenue_Change_Week"] / row["Sales Value (£)_previous"] * 100) 
            if row["Sales Value (£)_previous"] > 0 else 0, 
            axis=1
        )
        
        products_last_4_weeks["Revenue_Change_4Weeks"] = (
            products_last_4_weeks["Sales Value (£)_current"] - products_last_4_weeks["Sales Value (£)_previous"]
        )
        products_last_4_weeks["Revenue_Change_4Weeks_Pct"] = products_last_4_weeks.apply(
            lambda row: (row["Revenue_Change_4Weeks"] / row["Sales Value (£)_previous"] * 100) 
            if row["Sales Value (£)_previous"] > 0 else 0, 
            axis=1
        )
        
        products_combined = pd.merge(
            products_last_week[["Product", "Sales Value (£)_current", "Sales Value (£)_previous", 
                              "Revenue_Change_Week", "Revenue_Change_Week_Pct"]],
            products_last_4_weeks[["Product", "Sales Value (£)_current", "Sales Value (£)_previous", 
                                 "Revenue_Change_4Weeks", "Revenue_Change_4Weeks_Pct"]],
            on="Product",
            suffixes=("_week", "_4weeks")
        )
        
        products_combined = products_combined.rename(columns={
            "Sales Value (£)_current_week": "Last_Week_Revenue",
            "Sales Value (£)_previous_week": "Last_Week_Previous_Year",
            "Sales Value (£)_current_4weeks": "Last_4Weeks_Revenue",
            "Sales Value (£)_previous_4weeks": "Last_4Weeks_Previous_Year"
        })
        
        products_combined = products_combined.sort_values("Revenue_Change_Week")
        
        top_drop_products = products_combined.head(num_products)
        
        top_drop_products["Listing"] = listing_name
        
        products_analysis.append(top_drop_products)
    
    if products_analysis:
        all_products_analysis = pd.concat(products_analysis).reset_index(drop=True)
        all_products_analysis = all_products_analysis[[ 
            "Listing", "Product", 
            "Last_Week_Revenue", "Last_Week_Previous_Year", "Revenue_Change_Week", "Revenue_Change_Week_Pct",
            "Last_4Weeks_Revenue", "Last_4Weeks_Previous_Year", "Revenue_Change_4Weeks", "Revenue_Change_4Weeks_Pct"
        ]]
    else:
        all_products_analysis = pd.DataFrame()
    
    return top_drop_listings, all_products_analysis

def format_revenue_drop_report(listings_df, products_df):
    if "Error" in listings_df.columns:
        return listings_df, products_df
    
    listings_styled = listings_df.copy()
    
    money_cols = [
        "Last_Week_Revenue", "Last_Week_Previous_Year", "Revenue_Change_Week",
        "Last_4Weeks_Revenue", "Last_4Weeks_Previous_Year", "Revenue_Change_4Weeks"
    ]
    for col in money_cols:
        if col in listings_styled.columns:
            listings_styled[col] = listings_styled[col].round(2)
    
    pct_cols = ["Revenue_Change_Week_Pct", "Revenue_Change_4Weeks_Pct"]
    for col in pct_cols:
        if col in listings_styled.columns:
            listings_styled[col] = listings_styled[col].round(1)
    
    products_styled = products_df.copy() if not products_df.empty else pd.DataFrame()
    
    if not products_styled.empty:
        for col in money_cols:
            if col in products_styled.columns:
                products_styled[col] = products_styled[col].round(2)
        for col in pct_cols:
            if col in products_styled.columns:
                products_styled[col] = products_styled[col].round(1)
    
    # Build a single row "Total" summary row by summing across all listings
    summary_row = {
        "Listing": "Total",
        "Last_4Weeks_Revenue": listings_styled["Last_4Weeks_Revenue"].sum(),
        "Last_4Weeks_Previous_Year": listings_styled["Last_4Weeks_Previous_Year"].sum(),
        "Last_Week_Revenue": listings_styled["Last_Week_Revenue"].sum(),
        "Last_Week_Previous_Year": listings_styled["Last_Week_Previous_Year"].sum(),
        "Revenue_Change_Week": listings_styled["Revenue_Change_Week"].sum(),
        "Revenue_Change_4Weeks": listings_styled["Revenue_Change_4Weeks"].sum()
    }
    total_last4 = summary_row["Last_4Weeks_Previous_Year"]
    total_last_week = summary_row["Last_Week_Previous_Year"]
    summary_row["Revenue_Change_4Weeks_Pct"] = (summary_row["Revenue_Change_4Weeks"] / total_last4 * 100) if total_last4 != 0 else 0
    summary_row["Revenue_Change_Week_Pct"] = (summary_row["Revenue_Change_Week"] / total_last_week * 100) if total_last_week != 0 else 0
    desired_order = ["Listing", "Last_4Weeks_Revenue", "Last_4Weeks_Previous_Year",
                     "Revenue_Change_4Weeks", "Revenue_Change_4Weeks_Pct",
                     "Last_Week_Revenue", "Last_Week_Previous_Year",
                     "Revenue_Change_Week", "Revenue_Change_Week_Pct"]
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
        "Last_4Weeks_Revenue": "{:,}",
        "Last_4Weeks_Previous_Year": "{:,}",
        "Revenue_Change_4Weeks": "{:,.0f}",
        "Revenue_Change_4Weeks_Pct": "{:.1f}%",
        "Last_Week_Revenue": "{:,}",
        "Last_Week_Previous_Year": "{:,}",
        "Revenue_Change_Week": "{:,.0f}",
        "Revenue_Change_Week_Pct": "{:.1f}%"
    }).applymap(color_diff, subset=["Revenue_Change_4Weeks", "Revenue_Change_4Weeks_Pct", "Revenue_Change_Week", "Revenue_Change_Week_Pct"])\
      .set_properties(**{'font-weight': 'bold'})
    
    # Only return the single row total summary
    return styled_total, None

###############################################################################
# UPDATED FUNCTION: Adds "Total Revenue Change (Last 4 Weeks)" + negative highlight
###############################################################################
def export_report_to_excel(listings_df, products_df, marketplace=None):
    import io
    import pandas as pd
    from datetime import datetime
    
    output = io.BytesIO()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Identify money and percentage columns
    money_cols = [
        "Last_Week_Revenue", "Last_Week_Previous_Year", "Revenue_Change_Week",
        "Last_4Weeks_Revenue", "Last_4Weeks_Previous_Year", "Revenue_Change_4Weeks"
    ]
    pct_cols = ["Revenue_Change_Week_Pct", "Revenue_Change_4Weeks_Pct"]
    
    # 1. Convert percentage columns to decimal fractions BEFORE writing to Excel
    #    (e.g., -91.8% becomes -0.918 in the DataFrame)
    if "Error" not in listings_df.columns and not listings_df.empty:
        for pct_col in pct_cols:
            if pct_col in listings_df.columns:
                listings_df[pct_col] = listings_df[pct_col] / 100.0
    
    if not products_df.empty and "Error" not in listings_df.columns:
        for pct_col in pct_cols:
            if pct_col in products_df.columns:
                products_df[pct_col] = products_df[pct_col] / 100.0
    
    # Calculate totals for 4-week summary
    last_4w_current = listings_df["Last_4Weeks_Revenue"].sum() if "Last_4Weeks_Revenue" in listings_df.columns else 0
    last_4w_previous = listings_df["Last_4Weeks_Previous_Year"].sum() if "Last_4Weeks_Previous_Year" in listings_df.columns else 0
    last_4w_change = last_4w_current - last_4w_previous
    if last_4w_previous != 0:
        last_4w_change_pct = last_4w_change / last_4w_previous * 100
    else:
        last_4w_change_pct = 0
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Write Listings Summary
        if "Error" not in listings_df.columns and not listings_df.empty:
            listings_df.to_excel(writer, sheet_name='Listings Summary', index=False)
            workbook = writer.book
            worksheet = writer.sheets['Listings Summary']
            money_format = workbook.add_format({'num_format': '£#,##0.00'})
            pct_format = workbook.add_format({'num_format': '0.0%'})
            header_format = workbook.add_format({
                'bold': True,
                'text_wrap': True,
                'valign': 'top',
                'fg_color': '#D7E4BC',
                'border': 1
            })
            
            # Write headers with styling
            for col_num, value in enumerate(listings_df.columns.values):
                worksheet.write(0, col_num, value, header_format)
            
            # Format columns
            for money_col in money_cols:
                if money_col in listings_df.columns:
                    col_idx = listings_df.columns.get_loc(money_col)
                    worksheet.set_column(col_idx, col_idx, 15, money_format)
            for pct_col in pct_cols:
                if pct_col in listings_df.columns:
                    col_idx = listings_df.columns.get_loc(pct_col)
                    worksheet.set_column(col_idx, col_idx, 12, pct_format)
            
            worksheet.set_column(0, 0, 25)
            # Conditional formatting for numeric diffs (weekly/4-week columns in Listings Summary)
            for col_name in ["Revenue_Change_Week", "Revenue_Change_4Weeks"]:
                if col_name in listings_df.columns:
                    col_idx = listings_df.columns.get_loc(col_name)
                    worksheet.conditional_format(1, col_idx, len(listings_df), col_idx, {
                        'type': 'cell',
                        'criteria': '<',
                        'value': 0,
                        'format': workbook.add_format({'bg_color': '#FFC7CE'})
                    })
                    worksheet.conditional_format(1, col_idx, len(listings_df), col_idx, {
                        'type': 'cell',
                        'criteria': '>',
                        'value': 0,
                        'format': workbook.add_format({'bg_color': '#C6EFCE'})
                    })
        
        # Write Products Detail
        if not products_df.empty and "Error" not in listings_df.columns:
            products_df.to_excel(writer, sheet_name='Products Detail', index=False)
            workbook = writer.book
            worksheet = writer.sheets['Products Detail']
            money_format = workbook.add_format({'num_format': '£#,##0.00'})
            pct_format = workbook.add_format({'num_format': '0.0%'})
            header_format = workbook.add_format({
                'bold': True,
                'text_wrap': True,
                'valign': 'top',
                'fg_color': '#D7E4BC',
                'border': 1
            })
            
            # Write headers with styling
            for col_num, value in enumerate(products_df.columns.values):
                worksheet.write(0, col_num, value, header_format)
            
            # Format columns
            for money_col in money_cols:
                if money_col in products_df.columns:
                    col_idx = products_df.columns.get_loc(money_col)
                    worksheet.set_column(col_idx, col_idx, 15, money_format)
            for pct_col in pct_cols:
                if pct_col in products_df.columns:
                    col_idx = products_df.columns.get_loc(pct_col)
                    worksheet.set_column(col_idx, col_idx, 12, pct_format)
            
            worksheet.set_column(0, 0, 25)
            worksheet.set_column(1, 1, 25)
            # Conditional formatting for numeric diffs (weekly/4-week columns in Products Detail)
            for col_name in ["Revenue_Change_Week", "Revenue_Change_4Weeks"]:
                if col_name in products_df.columns:
                    col_idx = products_df.columns.get_loc(col_name)
                    worksheet.conditional_format(1, col_idx, len(products_df), col_idx, {
                        'type': 'cell',
                        'criteria': '<',
                        'value': 0,
                        'format': workbook.add_format({'bg_color': '#FFC7CE'})
                    })
                    worksheet.conditional_format(1, col_idx, len(products_df), col_idx, {
                        'type': 'cell',
                        'criteria': '>',
                        'value': 0,
                        'format': workbook.add_format({'bg_color': '#C6EFCE'})
                    })
        
        # Write Report Summary if no error
        if "Error" not in listings_df.columns:
            # Similarly get last week totals
            last_week_current = listings_df["Last_Week_Revenue"].sum() if "Last_Week_Revenue" in listings_df.columns else 0
            last_week_previous = listings_df["Last_Week_Previous_Year"].sum() if "Last_Week_Previous_Year" in listings_df.columns else 0
            last_week_change = last_week_current - last_week_previous
            if last_week_previous != 0:
                last_week_change_pct = last_week_change / last_week_previous * 100
            else:
                last_week_change_pct = 0

            # Create the summary data structure
            summary_data = {
                "Report Parameter": [
                    "Marketplace",
                    "Report Date",
                    "Report Period - Last Week",
                    "Report Period - Last 4 Weeks",
                    "Total Listings Analyzed",
                    "Total Products Analyzed",
                    "Total Last Week Revenue Current Year",
                    "Total Last Week Revenue Previous Year",
                    "Total Revenue Change (Last Week)",
                    "Revenue Change Percentage (Last Week)",
                    "Total Last 4 Weeks Revenue Current Year",
                    "Total Last 4 Weeks Revenue Previous Year",
                    "Total Revenue Change (Last 4 Weeks)",
                    "Revenue Change Percentage (Last 4 Weeks)"
                ],
                "Value": [
                    marketplace if marketplace else "All Marketplaces",
                    datetime.now().strftime("%Y-%m-%d %H:%M"),
                    f"Week {listings_df['Last_Week_Revenue'].name if hasattr(listings_df['Last_Week_Revenue'], 'name') else 'Current'}",
                    "Last 4 Weeks",
                    len(listings_df),
                    len(products_df) if not products_df.empty else 0,
                    f"£{last_week_current:,.2f}",
                    f"£{last_week_previous:,.2f}",
                    f"£{last_week_change:,.2f}",
                    f"{last_week_change_pct:.1f}%",
                    f"£{last_4w_current:,.2f}",
                    f"£{last_4w_previous:,.2f}",
                    f"£{last_4w_change:,.2f}",
                    f"{last_4w_change_pct:.1f}%"
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            
            # Create a version with raw numeric values for proper conditional formatting
            numeric_values = [
                "",  # Marketplace
                "",  # Report Date
                "",  # Report Period - Last Week
                "",  # Report Period - Last 4 Weeks
                len(listings_df),  # Total Listings Analyzed
                len(products_df) if not products_df.empty else 0,  # Total Products Analyzed
                last_week_current,  # Total Last Week Revenue Current Year
                last_week_previous,  # Total Last Week Revenue Previous Year
                last_week_change,  # Total Revenue Change (Last Week)
                last_week_change_pct / 100,  # Revenue Change Percentage (Last Week)
                last_4w_current,  # Total Last 4 Weeks Revenue Current Year
                last_4w_previous,  # Total Last 4 Weeks Revenue Previous Year
                last_4w_change,  # Total Revenue Change (Last 4 Weeks)
                last_4w_change_pct / 100  # Revenue Change Percentage (Last 4 Weeks)
            ]
            
            # Write the summary DataFrame to Excel
            summary_df.to_excel(writer, sheet_name='Report Summary', index=False)
            
            workbook = writer.book
            worksheet = writer.sheets['Report Summary']
            header_format = workbook.add_format({
                'bold': True,
                'text_wrap': True,
                'valign': 'top',
                'fg_color': '#D7E4BC',
                'border': 1
            })
            for col_num, value in enumerate(summary_df.columns.values):
                worksheet.write(0, col_num, value, header_format)
            
            worksheet.set_column(0, 0, 40)
            worksheet.set_column(1, 1, 35)
            
            # Format cells with currency and percentage formats
            currency_format = workbook.add_format({'num_format': '£#,##0.00'})
            pct_format = workbook.add_format({'num_format': '0.0%'})
            
            # Write raw numeric values directly to cells for proper conditional formatting
            # First set general formatting for the Value column
            for row in range(1, len(summary_data["Report Parameter"]) + 1):
                # Write the formatted values from the DataFrame
                worksheet.write(row, 1, summary_df.iloc[row-1, 1])
                
            # Now write the numeric values without formatting for specific rows that need conditional formatting
            # Total Revenue Change (Last Week) - row 9
            worksheet.write_number(9, 1, last_week_change, currency_format)
            # Revenue Change Percentage (Last Week) - row 10
            worksheet.write_number(10, 1, last_week_change_pct / 100, pct_format)
            # Total Revenue Change (Last 4 Weeks) - row 13
            worksheet.write_number(13, 1, last_4w_change, currency_format)
            # Revenue Change Percentage (Last 4 Weeks) - row 14
            worksheet.write_number(14, 1, last_4w_change_pct / 100, pct_format)
            
            # Format specific cells with the correct format
            for row in [7, 8, 11, 12]:  # Currency rows (exclude rows we directly wrote)
                worksheet.set_row(row, None, currency_format)
            
            # Apply conditional formatting to the raw numeric cells
            # Total Revenue Change (Last Week)
            worksheet.conditional_format(9, 1, 9, 1, {
                'type': 'cell',
                'criteria': '<',
                'value': 0,
                'format': workbook.add_format({'bg_color': '#FFC7CE'})
            })
            worksheet.conditional_format(9, 1, 9, 1, {
                'type': 'cell',
                'criteria': '>',
                'value': 0,
                'format': workbook.add_format({'bg_color': '#C6EFCE'})
            })
            
            # Total Revenue Change (Last 4 Weeks)
            worksheet.conditional_format(13, 1, 13, 1, {
                'type': 'cell',
                'criteria': '<',
                'value': 0,
                'format': workbook.add_format({'bg_color': '#FFC7CE'})
            })
            worksheet.conditional_format(13, 1, 13, 1, {
                'type': 'cell',
                'criteria': '>',
                'value': 0,
                'format': workbook.add_format({'bg_color': '#C6EFCE'})
            })
            
            # Also highlight the percentage rows
            worksheet.conditional_format(10, 1, 10, 1, {
                'type': 'cell',
                'criteria': '<',
                'value': 0,
                'format': workbook.add_format({'bg_color': '#FFC7CE'})
            })
            worksheet.conditional_format(10, 1, 10, 1, {
                'type': 'cell',
                'criteria': '>',
                'value': 0,
                'format': workbook.add_format({'bg_color': '#C6EFCE'})
            })
            
            worksheet.conditional_format(14, 1, 14, 1, {
                'type': 'cell',
                'criteria': '<',
                'value': 0,
                'format': workbook.add_format({'bg_color': '#FFC7CE'})
            })
            worksheet.conditional_format(14, 1, 14, 1, {
                'type': 'cell',
                'criteria': '>',
                'value': 0,
                'format': workbook.add_format({'bg_color': '#C6EFCE'})
            })

    output.seek(0)
    return output

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
    "Unrecognised Sales",
    "Reporting"
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
            st.markdown("##### Total Summary")
            st.dataframe(styled_total, use_container_width=True)
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

# -------------------------------
# Tab 7: Reporting
# -------------------------------
with tabs[6]:
    st.markdown("### Revenue Analysis Report")
    st.markdown("Generate reports to analyze listings with the highest revenue drops and contributing products.")
    if "report_marketplace" not in st.session_state:
        st.session_state["report_marketplace"] = None
    if "generate_report" not in st.session_state:
        st.session_state["generate_report"] = False
    if "last_report_data" not in st.session_state:
        st.session_state["last_report_data"] = {"generated": False}
    custom_marketplace = st.selectbox(
        "Select marketplace:",
        options=["All"] + sorted(df["Sales Channel"].dropna().unique().tolist()),
        index=0
    )
    if st.button("Generate Report", key="custom_report"):
        st.session_state["report_marketplace"] = None if custom_marketplace == "All" else custom_marketplace
        st.session_state["generate_report"] = True
    st.markdown("#### Report Configuration")
    col1, col2 = st.columns(2)
    with col1:
        num_listings = st.number_input(
            "Number of listings to analyze",
            min_value=1, max_value=20, value=5, step=1,
            help="Number of top listings with highest revenue drop to show"
        )
    with col2:
        num_products = st.number_input(
            "Products per listing",
            min_value=1, max_value=10, value=3, step=1,
            help="Number of products per listing with highest drop to show"
        )
    if st.session_state["generate_report"]:
        marketplace = st.session_state["report_marketplace"]
        with st.spinner("Generating report..."):
            listings_df, products_df = create_revenue_drop_report(
                df, 
                marketplace=marketplace, 
                num_listings=num_listings, 
                num_products=num_products
            )
            st.session_state["last_report_data"] = {
                "generated": True,
                "listings_df": listings_df.copy() if "Error" not in listings_df.columns else pd.DataFrame(),
                "products_df": products_df.copy() if not products_df.empty else pd.DataFrame(),
                "marketplace": marketplace,
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            styled_total, _ = format_revenue_drop_report(listings_df, products_df)
    st.markdown("---")
    
    with st.markdown("Export Options"):
        export_col1, export_col2 = st.columns(2)
        with export_col1:
            st.markdown("##### Export Current Report")
            export_disabled = not st.session_state["last_report_data"].get("generated", False)
            if export_disabled:
                st.info("Generate a report first to enable export")
            if st.button("Export to Excel", key="export_excel", disabled=export_disabled):
                listings_df = st.session_state["last_report_data"].get("listings_df", pd.DataFrame())
                products_df = st.session_state["last_report_data"].get("products_df", pd.DataFrame())
                marketplace = st.session_state["last_report_data"].get("marketplace", "All")
                excel_data = export_report_to_excel(listings_df, products_df, marketplace)
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                file_name = f"Revenue_Report_{marketplace}_{timestamp}.xlsx"
                st.download_button(
                    label="📥 Download Excel Report",
                    data=excel_data,
                    file_name=file_name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
