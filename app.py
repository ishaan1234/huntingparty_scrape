import streamlit as st
import pandas as pd
from homeharvest import scrape_property
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional
import json

st.set_page_config(
    page_title="HuntingParty Real Estate Scraper",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏠 HuntingParty Real Estate Scraper")
st.markdown("Scrape real estate property data from Realtor.com")

with st.sidebar:
    st.header("📍 Search Parameters")

    location = st.text_input(
        "Location*",
        placeholder="e.g., San Diego, CA or 92101",
        help="Enter ZIP code, city name, full address, neighborhood, or county"
    )

    listing_type = st.selectbox(
        "Listing Type*",
        options=["for_sale", "for_rent", "sold", "pending"],
        help="Select the type of property listing"
    )

    st.subheader("🔧 Advanced Filters")

    with st.expander("Property Filters", expanded=False):
        property_types = st.multiselect(
            "Property Types",
            options=["single_family", "townhomes", "condos", "multi_family", "condo_townhome", "condo_townhome_rowhome_coop", "duplex_triplex", "farm", "land", "mobile"],
            help="Select specific property types (leave empty for all)"
        )

        col1, col2 = st.columns(2)
        with col1:
            min_price = st.number_input("Min Price ($)", min_value=0, value=0, step=10000)
        with col2:
            max_price = st.number_input("Max Price ($)", min_value=0, value=0, step=10000)

        col1, col2 = st.columns(2)
        with col1:
            min_sqft = st.number_input("Min Sq Ft", min_value=0, value=0, step=100)
        with col2:
            max_sqft = st.number_input("Max Sq Ft", min_value=0, value=0, step=100)

        col1, col2 = st.columns(2)
        with col1:
            min_beds = st.number_input("Min Beds", min_value=0, max_value=10, value=0)
        with col2:
            max_beds = st.number_input("Max Beds", min_value=0, max_value=10, value=0)

        col1, col2 = st.columns(2)
        with col1:
            min_baths = st.number_input("Min Baths", min_value=0.0, max_value=10.0, value=0.0, step=0.5)
        with col2:
            max_baths = st.number_input("Max Baths", min_value=0.0, max_value=10.0, value=0.0, step=0.5)

    with st.expander("Search Options", expanded=False):
        radius = st.slider(
            "Search Radius (miles)",
            min_value=0,
            max_value=50,
            value=0,
            help="Search radius from location center (0 = no radius filter)"
        )

        mls_only = st.checkbox("MLS Listings Only", value=False)
        foreclosure = st.checkbox("Foreclosure Properties Only", value=False)

        limit = st.number_input(
            "Max Results",
            min_value=1,
            max_value=10000,
            value=100,
            step=10,
            help="Maximum number of properties to fetch"
        )

    with st.expander("Date Filters", expanded=False):
        use_date_filter = st.checkbox("Enable Date Filter")

        if use_date_filter:
            date_filter_type = st.radio(
                "Filter Type",
                ["Past Days", "Date Range"]
            )

            if date_filter_type == "Past Days":
                past_days = st.number_input(
                    "Past Days",
                    min_value=1,
                    max_value=365,
                    value=30,
                    help="Fetch properties from the last N days"
                )
                date_from = None
                date_to = None
            else:
                col1, col2 = st.columns(2)
                with col1:
                    date_from = st.date_input("From Date")
                with col2:
                    date_to = st.date_input("To Date")
                past_days = None
        else:
            past_days = None
            date_from = None
            date_to = None

    with st.expander("Output Options", expanded=False):
        return_type = st.selectbox(
            "Return Type",
            ["pandas", "pydantic", "raw"],
            help="Format of returned data"
        )

        show_raw_data = st.checkbox("Show Raw Data Table", value=True)
        show_visualizations = st.checkbox("Show Visualizations", value=True)
        show_statistics = st.checkbox("Show Statistics", value=True)

    scrape_button = st.button("🔍 Scrape Properties", type="primary", use_container_width=True)

if scrape_button:
    if not location:
        st.error("Please enter a location to search")
    else:
        try:
            with st.spinner(f"Scraping {listing_type} properties in {location}..."):

                kwargs = {
                    "location": location,
                    "listing_type": listing_type,
                    "return_type": return_type,
                    "limit": limit
                }

                if property_types:
                    kwargs["property_type"] = property_types

                if radius > 0:
                    kwargs["radius"] = radius

                if mls_only:
                    kwargs["mls_only"] = True

                if foreclosure:
                    kwargs["foreclosure"] = True

                if past_days:
                    kwargs["past_days"] = past_days
                elif date_from and date_to:
                    kwargs["date_from"] = date_from.strftime("%Y-%m-%d")
                    kwargs["date_to"] = date_to.strftime("%Y-%m-%d")

                properties = scrape_property(**kwargs)

                if return_type == "pandas":
                    df = properties
                elif return_type == "pydantic":
                    df = pd.DataFrame([prop.dict() for prop in properties])
                else:
                    df = pd.DataFrame(properties)

                if max_price > 0:
                    df = df[df['list_price'] <= max_price] if 'list_price' in df.columns else df
                if min_price > 0:
                    df = df[df['list_price'] >= min_price] if 'list_price' in df.columns else df

                if max_sqft > 0 and 'sqft' in df.columns:
                    df = df[df['sqft'] <= max_sqft]
                if min_sqft > 0 and 'sqft' in df.columns:
                    df = df[df['sqft'] >= min_sqft]

                if max_beds > 0 and 'beds' in df.columns:
                    df = df[df['beds'] <= max_beds]
                if min_beds > 0 and 'beds' in df.columns:
                    df = df[df['beds'] >= min_beds]

                if max_baths > 0 and 'baths' in df.columns:
                    df = df[df['baths'] <= max_baths]
                if min_baths > 0 and 'baths' in df.columns:
                    df = df[df['baths'] >= min_baths]

                st.success(f"Successfully scraped {len(df)} properties!")

                col1, col2, col3 = st.columns(3)

                with col1:
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv,
                        file_name=f"properties_{location.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )

                with col2:
                    excel_buffer = pd.ExcelWriter('temp.xlsx', engine='xlsxwriter')
                    df.to_excel(excel_buffer, index=False)
                    excel_data = excel_buffer.book
                    st.download_button(
                        label="📥 Download Excel",
                        data=csv,
                        file_name=f"properties_{location.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

                with col3:
                    if return_type == "raw":
                        json_str = json.dumps(properties, indent=2)
                    else:
                        json_str = df.to_json(orient='records', indent=2)
                    st.download_button(
                        label="📥 Download JSON",
                        data=json_str,
                        file_name=f"properties_{location.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )

                if show_statistics and not df.empty:
                    st.header("📊 Property Statistics")

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        if 'list_price' in df.columns:
                            avg_price = df['list_price'].mean()
                            st.metric("Average Price", f"${avg_price:,.0f}" if pd.notna(avg_price) else "N/A")

                    with col2:
                        if 'sqft' in df.columns:
                            avg_sqft = df['sqft'].mean()
                            st.metric("Average Sq Ft", f"{avg_sqft:,.0f}" if pd.notna(avg_sqft) else "N/A")

                    with col3:
                        if 'beds' in df.columns:
                            avg_beds = df['beds'].mean()
                            st.metric("Average Beds", f"{avg_beds:.1f}" if pd.notna(avg_beds) else "N/A")

                    with col4:
                        if 'price_per_sqft' in df.columns:
                            avg_ppsf = df['price_per_sqft'].mean()
                            st.metric("Avg $/Sq Ft", f"${avg_ppsf:.0f}" if pd.notna(avg_ppsf) else "N/A")

                if show_visualizations and not df.empty:
                    st.header("📈 Data Visualizations")

                    viz_cols = st.columns(2)

                    with viz_cols[0]:
                        if 'list_price' in df.columns:
                            price_df = df[df['list_price'].notna()]
                            if not price_df.empty:
                                fig = px.histogram(
                                    price_df,
                                    x='list_price',
                                    nbins=20,
                                    title="Price Distribution",
                                    labels={'list_price': 'Price ($)', 'count': 'Number of Properties'}
                                )
                                st.plotly_chart(fig, use_container_width=True)

                    with viz_cols[1]:
                        if 'property_style' in df.columns:
                            style_counts = df['property_style'].value_counts().head(10)
                            if not style_counts.empty:
                                fig = px.pie(
                                    values=style_counts.values,
                                    names=style_counts.index,
                                    title="Property Types Distribution"
                                )
                                st.plotly_chart(fig, use_container_width=True)

                    if 'beds' in df.columns and 'list_price' in df.columns:
                        beds_price_df = df[['beds', 'list_price']].dropna()
                        if not beds_price_df.empty:
                            fig = px.box(
                                beds_price_df,
                                x='beds',
                                y='list_price',
                                title="Price by Number of Bedrooms",
                                labels={'beds': 'Number of Bedrooms', 'list_price': 'Price ($)'}
                            )
                            st.plotly_chart(fig, use_container_width=True)

                    if 'sqft' in df.columns and 'list_price' in df.columns:
                        scatter_df = df[['sqft', 'list_price']].dropna()
                        if not scatter_df.empty:
                            fig = px.scatter(
                                scatter_df,
                                x='sqft',
                                y='list_price',
                                title="Price vs Square Footage",
                                labels={'sqft': 'Square Footage', 'list_price': 'Price ($)'},
                                trendline="ols"
                            )
                            st.plotly_chart(fig, use_container_width=True)

                    if 'latitude' in df.columns and 'longitude' in df.columns:
                        map_df = df[['latitude', 'longitude', 'list_price']].dropna()
                        if not map_df.empty:
                            st.subheader("🗺️ Property Map")
                            st.map(map_df[['latitude', 'longitude']])

                if show_raw_data:
                    st.header("📋 Raw Property Data")

                    display_cols = st.multiselect(
                        "Select columns to display",
                        options=df.columns.tolist(),
                        default=df.columns.tolist()
                    )

                    if display_cols:
                        st.dataframe(
                            df[display_cols],
                            use_container_width=True,
                            hide_index=True
                        )
                    else:
                        st.dataframe(df, use_container_width=True, hide_index=True)

                    st.info(f"Showing {len(df)} properties with {len(df.columns)} columns")

        except Exception as e:
            st.error(f"Error occurred: {str(e)}")
            st.info("Please check your search parameters and try again.")

with st.expander("📖 How to Use"):
    st.markdown("""
    ### Basic Usage:
    1. **Enter Location**: ZIP code, city name, or full address
    2. **Select Listing Type**: for_sale, for_rent, sold, or pending
    3. **Click "Scrape Properties"** to fetch data

    ### Advanced Features:
    - **Property Filters**: Filter by type, price, size, beds/baths
    - **Search Options**: Set radius, MLS-only, foreclosure properties
    - **Date Filters**: Get recent listings or specific date ranges
    - **Output Options**: Choose data format and visualization preferences

    ### Data Export:
    - Download results as CSV, Excel, or JSON
    - View interactive visualizations
    - Analyze property statistics
    """)

with st.expander("🔍 Example Searches"):
    st.markdown("""
    - **Recent Sales**: Location: "Austin, TX", Type: "sold", Past Days: 30
    - **Rental Search**: Location: "10001", Type: "for_rent", Property Type: "apartments"
    - **Investment Properties**: Location: "Miami, FL", Type: "for_sale", Foreclosure: ✓
    - **Luxury Homes**: Location: "Beverly Hills, CA", Type: "for_sale", Min Price: $1,000,000
    """)
