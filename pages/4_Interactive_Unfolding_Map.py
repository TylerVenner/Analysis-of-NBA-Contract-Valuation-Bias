import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit.components.v1 as components
import os
from pathlib import Path

# --- 1. SETUP & CONFIG ---
st.set_page_config(layout="wide", page_title="Interactive Bias Map")

MAP_HTML_PATH = Path("data/app_data/final_3d_map.html")
PLAYER_PATH = Path("data/app_data/player_db.csv")
COEF_PATH = Path("data/app_data/final_ols_coefficients.csv")

# --- 2. DATA LOADING & PREP ---
@st.cache_data
def load_data():
    if not PLAYER_PATH.exists() or not COEF_PATH.exists():
        st.error("⚠️ Data missing. Run `prepare_deployment_data.py` locally.")
        st.stop()

    # Load Players
    df = pd.read_csv(PLAYER_PATH)
    
    # Load Coefficients
    df_coef = pd.read_csv(COEF_PATH)
    # Standardize coef columns
    rename_map = {"Gamma (Price)": "coef", "P-Value": "p_value", "std_err": "std_err"}
    df_coef = df_coef.rename(columns=rename_map)
    
    # Set index to variable name
    if "Variable" in df_coef.columns:
        df_coef = df_coef.set_index("Variable")
    elif "Unnamed: 0" in df_coef.columns:
        df_coef = df_coef.set_index("Unnamed: 0")

    # --- CALCULATE STRUCTURAL BIAS SCORE ON THE FLY ---
    # Score = Sum(Player_Z_Value * Gamma_Z)
    # This approximates the "Structural Bias" without needing residuals
    
    df["Calculated_Bias_Score"] = 0.0
    
    # List of Z columns to check
    z_cols = ["Age", "DRAFT_NUMBER", "OWNER_NET_WORTH_B", "Capacity", "Followers", "is_USA"]
    
    for col in z_cols:
        if col in df.columns and col in df_coef.index:
            # Ensure numeric
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            gamma = df_coef.loc[col, "coef"]
            
            # Add to score
            df["Calculated_Bias_Score"] += df[col] * gamma

    return df

df = load_data()

# --- 3. THE 3D ATTRIBUTION MAP ---
st.title("The Bias Attribution Map (3D)")

st.markdown("""
This 3D visualization represents the **"Topology of Economic Bias."** It clusters players based on which structural forces (Age, Draft, Market Size) dominate their contract valuation.

* **💎 Diamonds (Anchors):** The bias factors themselves.
* **• Dots (Players):** Players positioned by their exposure to these biases.
* **Proximity:** A player near the "Age" diamond is paid largely due to their veteran status.
""")

if MAP_HTML_PATH.exists():
    with open(MAP_HTML_PATH, 'r', encoding='utf-8') as f:
        html_content = f.read()
    components.html(html_content, height=850, scrolling=False)
else:
    st.warning("⚠️ 3D Map HTML not found. Please run the deployment script.")

st.divider()

# --- 4. GEOGRAPHIC BIAS MAP ---
st.header("Geographic Analysis: Who is Structurally 'Overpaid'?")

st.markdown("""
This map visualizes the **Structural Bias Score**.
* **Red (Positive):** The model predicts a salary **premium** based on context (e.g., Veteran in a Big Market).
* **Blue (Negative):** The model predicts a salary **penalty** based on context (e.g., Late Pick in a Small Market).
""")

# --- Sidebar Filters ---
st.sidebar.header("Map Filters")

# Handle missing Salary column gracefully
if "Salary" in df.columns:
    df["Salary"] = pd.to_numeric(df["Salary"], errors="coerce").fillna(0)
    min_sal, max_sal = int(df["Salary"].min()), int(df["Salary"].max())
    salary_range = st.sidebar.slider("Salary Range", min_sal, max_sal, (min_sal, max_sal))
    df = df[(df["Salary"] >= salary_range[0]) & (df["Salary"] <= salary_range[1])]

# View Mode
view_mode = st.sidebar.radio("View Level", ["Individual Players", "Team Average"])

# --- Coordinate Mapping ---
# Basic Lat/Lon Dictionary for NBA Cities
CITY_COORDS = {
    "Atlanta": (33.7490, -84.3880), "Boston": (42.3601, -71.0589), "Brooklyn": (40.6782, -73.9442),
    "Charlotte": (35.2271, -80.8431), "Chicago": (41.8781, -87.6298), "Cleveland": (41.4993, -81.6944),
    "Dallas": (32.7767, -96.7970), "Denver": (39.7392, -104.9903), "Detroit": (42.3314, -83.0458),
    "Houston": (29.7604, -95.3698), "Indiana": (39.7684, -86.1581), "Los Angeles": (34.0430, -118.2673),
    "Memphis": (35.1495, -90.0490), "Miami": (25.7617, -80.1918), "Milwaukee": (43.0389, -87.9065),
    "Minnesota": (44.9778, -93.2650), "New Orleans": (29.9511, -90.0715), "New York": (40.7505, -73.9934),
    "Oklahoma City": (35.4676, -97.5164), "Orlando": (28.5383, -81.3792), "Philadelphia": (39.9526, -75.1652),
    "Phoenix": (33.4484, -112.0740), "Portland": (45.5051, -122.6750), "Sacramento": (38.5816, -121.4944),
    "San Antonio": (29.4241, -98.4936), "Golden State": (37.7680, -122.3877), "Toronto": (43.6532, -79.3832),
    "Utah": (40.7608, -111.8910), "Washington": (38.9072, -77.0369)
}

# Helper to find coords based on partial match
def get_coords(team_name):
    # Default to center of US if not found
    default = (39.8283, -98.5795) 
    if not isinstance(team_name, str): return default
    
    for city, coords in CITY_COORDS.items():
        if city in team_name:
            return coords
    return default

# Map Coords
if "lat" not in df.columns:
    df["lat"] = df["TEAM_NAME"].apply(lambda x: get_coords(x)[0])
    df["lon"] = df["TEAM_NAME"].apply(lambda x: get_coords(x)[1])

# Add Jitter for individual view so dots don't overlap
if view_mode == "Individual Players":
    # Jitter logic
    np.random.seed(42)
    df["lat_j"] = df["lat"] + np.random.normal(0, 0.15, len(df))
    df["lon_j"] = df["lon"] + np.random.normal(0, 0.15, len(df))
    
    fig = px.scatter_mapbox(
        df,
        lat="lat_j",
        lon="lon_j",
        color="Calculated_Bias_Score",
        hover_name="PLAYER_NAME",
        hover_data=["TEAM_NAME", "Salary", "Calculated_Bias_Score"],
        color_continuous_scale="RdBu_r", # Red = High (Premium), Blue = Low (Penalty)
        color_continuous_midpoint=0,
        zoom=3,
        height=600,
        size_max=15
    )
else:
    # Aggregated View
    team_df = df.groupby(["TEAM_NAME", "lat", "lon"]).agg({
        "Calculated_Bias_Score": "mean",
        "Salary": "mean",
        "PLAYER_NAME": "count"
    }).reset_index()
    
    fig = px.scatter_mapbox(
        team_df,
        lat="lat",
        lon="lon",
        color="Calculated_Bias_Score",
        size="Salary",
        hover_name="TEAM_NAME",
        color_continuous_scale="RdBu_r",
        color_continuous_midpoint=0,
        zoom=3,
        height=600
    )

fig.update_layout(mapbox_style="carto-positron", margin={"r":0,"t":0,"l":0,"b":0})
st.plotly_chart(fig, use_container_width=True)

# --- 5. TOP/BOTTOM LISTS ---
st.subheader("Leaderboard: Structural Winners & Losers")

cols = ["PLAYER_NAME", "TEAM_NAME", "Calculated_Bias_Score", "Age", "DRAFT_NUMBER"]

col1, col2 = st.columns(2)

with col1:
    st.markdown("##### 📈 Top Structural Premiums")
    st.dataframe(
        df.sort_values("Calculated_Bias_Score", ascending=False).head(10)[cols].style.format({"Calculated_Bias_Score": "{:.4f}"}),
        hide_index=True,
        use_container_width=True
    )

with col2:
    st.markdown("##### 📉 Top Structural Penalties")
    st.dataframe(
        df.sort_values("Calculated_Bias_Score", ascending=True).head(10)[cols].style.format({"Calculated_Bias_Score": "{:.4f}"}),
        hide_index=True,
        use_container_width=True
    )