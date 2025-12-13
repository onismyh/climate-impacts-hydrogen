import geopandas as gpd
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# Matplotlib font settings (for potential Chinese text in your data)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign rendering

from shapely.geometry import Point, Polygon

import matplotlib.colors as mcolors
import matplotlib.cm as cm

import warnings
warnings.filterwarnings("ignore")

def wrap_lon_to_180(ds, lon_name="lon", keep_pos_180=True):
    """
    Convert longitude from 0–360 to −180–180.

    Notes:
    - When keep_pos_180=True, values mapped to -180 are reassigned to +180
      to avoid duplicated endpoints / sorting artifacts.
    """
    lon = ds[lon_name]
    # Map to [-180, 180)
    new_lon = ((lon + 180) % 360) - 180
    if keep_pos_180:
        # Optional: move -180 to +180 to avoid duplicates at the dateline
        new_lon = xr.where(np.isclose(new_lon, -180), 180.0, new_lon)

    ds2 = ds.assign_coords({lon_name: new_lon}).sortby(lon_name)
    return ds2


# --- Load project database ---
df = pd.read_excel(
    r"../DATA/Hydrogen Production Projects Database - September 2025.xlsx",
    sheet_name="Projects",
    header=[0, 1, 2]
)

df.columns = [
    "Ref", "Project name", "Country", "Date online", "Decomission date", "Status",
    "Technology", "Technology_details", "Technology_electricity", "Technology_electricity_details",
    "Product", "Refining", "Ammonia", "Methanol", "Iron&Steel", "Other Ind", "Mobility", "Power",
    "Grid inj.", "CHP", "Domestic heat", "Biofuels", "Synfuels", "CH4 grid inj.", "CH4 mobility",
    "Announced Size", "Capacity_MWel", "Capacity_Nm³ H₂/h", "Capacity_kt H2/y",
    "Capacity_t CO₂ captured/y", "IEA zero-carbon estimated normalized capacity [Nm³ H₂/hour]",
    "References", "Location", "Latitude", "Longitude"
]

df = df[
    [
        "Project name", "Country", "Date online", "Decomission date", "Status",
        "Technology", "Technology_details", "Technology_electricity",
        "Technology_electricity_details", "Product", "Announced Size", "Capacity_MWel",
        "Capacity_Nm³ H₂/h", "Capacity_kt H2/y",
        "IEA zero-carbon estimated normalized capacity [Nm³ H₂/hour]",
        "References", "Location", "Latitude", "Longitude",
    ]
]

# Keep electrolysis projects only
df = df[df["Technology"].isin(["Other Electrolysis", "PEM", "ALK", "SOEC", "AEM", "Electrolysis", "Other electrolysis"])]

# Keep electricity sourcing modes of interest
df = df[df["Technology_electricity"].isin(["Dedicated renewable", "Grid+Renewables"])]

# Keep relevant project statuses
df = df[df["Status"].isin(["Feasibility study", "Concept", "FID/Construction", "Operational", "DEMO", "Various"])]

df = df.reset_index(drop=True)

# Build GeoDataFrame (WGS84)
gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df.Longitude, df.Latitude),
    crs="EPSG:4326"
)
gdf


# --- Load climate impact layers ---
da = xr.open_dataset("monte_carlo_hydrogen_price1.nc").median(dim="realization")["hydrogen_price"]  # median over realizations
da_BY = xr.open_dataset("monte_carlo_hydrogen_price_BY.nc").median(dim="realization")["hydrogen_price"]

# Relative difference
da_diff = (da - da_BY) / da_BY
da_diff = wrap_lon_to_180(da_diff, lon_name="lon")


import folium
from branca.colormap import LinearColormap
from sklearn.neighbors import BallTree

# --- 1) Core step: sample climate-layer values at project locations (optimized) ---
print("Preparing spatial index for the climate grid...")

# Step 1: Preprocess climate grid (drop NaNs, keep valid cells only)
# Convert to DataFrame to filter NaNs easily
df_grid = da_diff.to_dataframe(name="value").reset_index()

# Drop missing values (critical: only search among valid cells)
df_grid_valid = df_grid.dropna(subset=["value"])
if len(df_grid_valid) == 0:
    raise ValueError("The climate layer contains only NaNs; cannot match any project points.")

# Step 2: Build BallTree in radians for fast haversine nearest-neighbor queries
# Note: sklearn expects [lat, lon] order for haversine usage
grid_coords_radians = np.radians(df_grid_valid[["lat", "lon"]].values)
tree = BallTree(grid_coords_radians, metric="haversine")

# Earth radius and search radius
EARTH_RADIUS_KM = 6371.0
SEARCH_RADIUS_KM = 20.0  # ~1 grid cell (adjust as needed)
RADIUS_RAD_THRESHOLD = SEARCH_RADIUS_KM / EARTH_RADIUS_KM  # km -> radians

print(f"Index built. Valid grid points: {len(df_grid_valid)}. Matching projects...")

def get_climate_value_radius(row):
    """
    For each project point, find the nearest valid climate-grid cell.
    Return None if the nearest cell is farther than SEARCH_RADIUS_KM.
    """
    try:
        project_coord = np.radians([[row["Latitude"], row["Longitude"]]])

        # Query the nearest neighbor (k=1)
        dists, indices = tree.query(project_coord, k=1)

        nearest_dist_rad = dists[0][0]
        nearest_idx = indices[0][0]

        # Enforce distance threshold
        if nearest_dist_rad > RADIUS_RAD_THRESHOLD:
            return None

        # Retrieve the matched grid value
        return df_grid_valid.iloc[nearest_idx]["value"]

    except Exception as e:
        print(f"Error matching point {row.get('Project name', '<unknown>')}: {e}")
        return None

# Apply matching to all projects
gdf["climate_impact"] = gdf.apply(get_climate_value_radius, axis=1)

# Keep only successfully matched projects
gdf_clean = gdf.dropna(subset=["climate_impact"])

print("Done:")
print(f"- Total projects: {len(gdf)}")
print(f"- Matched projects: {len(gdf_clean)} (found valid values within {SEARCH_RADIUS_KM} km)")


# --- 2) Create an interactive map ---
# min_zoom prevents zooming out too far (avoids duplicated world display in some contexts)
# max_bounds constrains panning so users stay on a single world view
m = folium.Map(
    location=[20, 0],
    zoom_start=2,
    tiles="CartoDB positron",
    min_zoom=1.4,
    max_bounds=True
)

vmin = gdf_clean["climate_impact"].min()
vmax = gdf_clean["climate_impact"].max()

# Colormap: green -> yellow -> red
cmap = LinearColormap(
    colors=["green", "yellow", "red"],
    vmin=vmin,
    vmax=vmax,
    caption="Climate Change Impact on LCOH (Rel. Diff)"
)
m.add_child(cmap)


# --- 3) Add circle markers with styled tooltips/popups ---
for _, row in gdf_clean.iterrows():
    impact_val = row["climate_impact"]
    impact_percent = f"{impact_val:+.2%}"
    color = cmap(impact_val)

    # Larger, card-style tooltip/popup
    tooltip_html = f"""
    <div style="font-family: 'Helvetica Neue', Arial, sans-serif; min-width: 300px; font-size: 14px;">
        <h3 style="margin: 0 0 5px 0; font-size: 18px; color: #333;">🏗️ {row['Project name']}</h3>
        <span style="color: #666; font-size: 14px;">📍 {row['Country']}</span>
        <hr style="margin: 10px 0; border: 0; border-top: 1px solid #ddd;">

        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 5px;">
            <b>Status:</b> <span>{row['Status']}</span>
            <b>Capacity:</b> <span>{row['Capacity_MWel']} MW</span>
            <b>Tech:</b> <span style="grid-column: span 2;">{row['Technology']}</span>
        </div>

        <div style="margin-top: 15px; background-color: #f9f9f9; padding: 10px; border-radius: 5px; text-align: center;">
            <div style="font-size: 12px; color: #555; margin-bottom: 5px;">Climate Impact on Price</div>
            <span style="font-size: 24px; font-weight: bold; color: {color};">
                {impact_percent}
            </span>
        </div>
    </div>
    """

    folium.CircleMarker(
        location=[row["Latitude"], row["Longitude"]],
        radius=6,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.8,
        tooltip=tooltip_html,
        popup=tooltip_html
    ).add_to(m)


output_file = "Hydrogen_Projects_Climate_Impact.html"
m.save(output_file)
print(f"🎉 Map updated: single-world view + larger info cards -> {output_file}")
