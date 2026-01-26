from pathlib import Path

import folium
import geopandas as gpd
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "data" / "CleanupDataSet" / "final_model.csv"
GEOJSON_FILE = BASE_DIR / "GeoJsontoZip" / "areasMunich.json"

@st.cache_data
def load_datasets():
    df = pd.read_csv(DATA_FILE)

    gdf = gpd.read_file(GEOJSON_FILE)
    gdf["district_number"] = gdf["sb_nummer"].astype(int)
    gdf = gdf.to_crs(epsg=4326)

    agg = (
        df.groupby(["district_number", "year"], as_index=False)
        .agg(
            panel_area_m2=("panel_area_m2", "sum"),
            square_meters_with_solar_m2=("square_meters_with_solar_m2", "sum"),
            total_rooftops=("total_rooftops", "sum"),
            rooftops_without_solar=("rooftops_without_solar", "sum"),
            Total_Population=("Total_Population", "mean"),
            Unemployment_Rate=("Unemployment_Rate", "mean"),
            Average_Age=("Average_Age", "mean"),
        )
    )

    return agg, gdf

def build_map(merged_gdf, metric_key):
    merged_gdf[metric_key] = merged_gdf[metric_key].fillna(0)

    center = merged_gdf.geometry.centroid
    lat = center.y.mean()
    lon = center.x.mean()

    m = folium.Map(location=[lat, lon], zoom_start=11, tiles="cartodbpositron")

    folium.Choropleth(
        geo_data=merged_gdf,
        data=merged_gdf,
        columns=["district_number", metric_key],
        key_on="feature.properties.district_number",
        fill_color="YlOrRd",
        fill_opacity=0.8,
        line_opacity=0.3,
        nan_fill_color="#f0f0f0",
        legend_name=metric_key,
    ).add_to(m)

    folium.GeoJson(
        merged_gdf,
        style_function=lambda _: {"color": "#444", "weight": 1, "fillOpacity": 0},
        tooltip=folium.features.GeoJsonTooltip(
            fields=["name", metric_key],
            aliases=["District", metric_key],
            localize=True,
        ),
    ).add_to(m)

    return m

def main():
    st.set_page_config(page_title="Munich Solar Rooftops", layout="wide")
    st.title("Munich Solar Rooftops per District")
    st.markdown(
        "Select a year and metric to see district-level values drawn from the aggregated dataset."
    )

    agg, gdf = load_datasets()

    years = sorted(agg["year"].unique())
    metric_options = {
        "panel_area_m2": "Installed panel area (m²)",
        "square_meters_with_solar_m2": "Solar-covered roof area (m²)",
        "total_rooftops": "Total rooftops",
        "rooftops_without_solar": "Rooftops without solar",
        "Total_Population": "Population (avg)",
        "Unemployment_Rate": "Unemployment rate (avg)",
        "Average_Age": "Average age (avg)",
    }

    col1, col2 = st.columns(2)
    with col1:
        selected_year = st.selectbox("Year", years, index=len(years) - 1)
    with col2:
        metric_key = st.selectbox("Metric", list(metric_options.keys()), format_func=metric_options.get)

    current = agg[agg["year"] == selected_year]
    merged = gdf.merge(current, on="district_number", how="left")

    m = build_map(merged, metric_key)
    st_folium(m, width=None, height=700)

    st.markdown("**Data sources**: final_model.csv (aggregated per district/year) and areasMunich.json (district boundaries).")

if __name__ == "__main__":
    main()
