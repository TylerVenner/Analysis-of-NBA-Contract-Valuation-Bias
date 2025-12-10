import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit.components.v1 as components
import os
from pathlib import Path
from datetime import datetime

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
    rename_map = {"Gamma (Price)": "coef", "P-Value": "p_value", "std_err": "std_err", "coef": "coef"}
    df_coef = df_coef.rename(columns=rename_map)
    
    # Set index
    if "Variable" in df_coef.columns:
        df_coef = df_coef.set_index("Variable")
    elif "Unnamed: 0" in df_coef.columns:
        df_coef = df_coef.set_index("Unnamed: 0")

    # --- CALCULATE STRUCTURAL BIAS SCORE ---
    df["Calculated_Bias_Score"] = 0.0
    
    # Factors to include in the "Structural" score
    z_cols = ["Age", "DRAFT_NUMBER", "OWNER_NET_WORTH_B", "Capacity", "Followers", "is_USA"]
    
    for col in z_cols:
        if col in df.columns and col in df_coef.index:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            gamma = df_coef.loc[col, "coef"]
            # Accumulate Log-Dollar Impact
            df["Calculated_Bias_Score"] += df[col] * gamma

    # --- CRITICAL FIX: CENTER THE SCORE ---
    # We subtract the median so 0 represents the "Average Player".
    # Positive = Premium relative to average. Negative = Penalty relative to average.
    median_bias = df["Calculated_Bias_Score"].median()
    df["Calculated_Bias_Score"] -= median_bias

    return df

df = load_data()

# --- 3. HEADER & TIMESTAMP ---
st.title("🗺️ The Topology of Bias")

# [cite_start]Get timestamp of the map file to verify freshness [cite: 53]
if MAP_HTML_PATH.exists():
    mod_time = datetime.fromtimestamp(MAP_HTML_PATH.stat().st_mtime)
    st.caption(f"Analysis Snapshot Generated: {mod_time.strftime('%Y-%m-%d %H:%M')}")

# --- 4. THE 3D ATTRIBUTION MAP ---
with st.expander("💎 Expand 3D Topology Map", expanded=True):
    st.markdown("""
    This map is the main output. Feel free to find your favorite player and examine the structural bias.
                
    **How to use the map:**
    * Click and hold, and move your mouse to fly around the map
    * Use your scroll wheel to zoom in and out
    * Hover over dots to see individual players
                
    **How to read the map:**
    * **Clusters:** Players are pulled towards the factors that determine their salary.
    * **"The Void":** Players in the empty center are paid based on pure performance (no structural bias).
    """)
    if MAP_HTML_PATH.exists():
        with open(MAP_HTML_PATH, 'r', encoding='utf-8') as f:
            html_content = f.read()
        components.html(html_content, height=800, scrolling=False)
    else:
        st.warning("⚠️ 3D Map HTML not found.")

st.divider()

# --- 5. GEOGRAPHIC BIAS MAP ---
st.header("Geographic Analysis: The 'Overpaid' vs 'Underpaid'")

col_map, col_context = st.columns([3, 1])

with col_context:
    st.info("""
    **What is the Bias Score?**
    
    It measures the **Relative Structural Advantage** compared to the league median.
    
    * **🔴 Red (> 0):** Structural Premium. This player benefits more from context (Age, Fame) than the average player.
    * **🔵 Blue (< 0):** Structural Penalty. This player faces more headwinds (Draft, Small Market) than average.
    * **⚪ White (~ 0):** Neutral.
    """)
    
    # Filters
    st.subheader("Filters")
    if "Salary" in df.columns:
        df["Salary"] = pd.to_numeric(df["Salary"], errors="coerce").fillna(0)
        min_sal, max_sal = int(df["Salary"].min()), int(df["Salary"].max())
        salary_range = st.slider("Salary ($)", min_sal, max_sal, (min_sal, max_sal), format="$%d")
        df_filtered = df[(df["Salary"] >= salary_range[0]) & (df["Salary"] <= salary_range[1])]
    else:
        df_filtered = df

    view_mode = st.radio("View Mode", ["Team Average", "Individual Players"])

with col_map:
    # --- MAP LOGIC ---
    CITY_COORDS = {
        "Atlanta": (33.7, -84.3), "Boston": (42.3, -71.0), "Brooklyn": (40.6, -73.9), "Charlotte": (35.2, -80.8),
        "Chicago": (41.8, -87.6), "Cleveland": (41.4, -81.6), "Dallas": (32.7, -96.7), "Denver": (39.7, -104.9),
        "Detroit": (42.3, -83.0), "Houston": (29.7, -95.3), "Indiana": (39.7, -86.1), "Los Angeles": (34.0, -118.2),
        "Memphis": (35.1, -90.0), "Miami": (25.7, -80.1), "Milwaukee": (43.0, -87.9), "Minnesota": (44.9, -93.2),
        "New Orleans": (29.9, -90.0), "New York": (40.7, -73.9), "Oklahoma City": (35.4, -97.5), "Orlando": (28.5, -81.3),
        "Philidelphia": (39.9, -75.1), "Pheonix": (33.4, -112.0), "Portland": (45.5, -122.6), "Sacramento": (38.5, -121.4),
        "San Antonio": (29.4, -98.4), "Golden State": (37.7, -122.4), "Toronto": (43.6, -79.3), "Utah": (40.7, -111.8),
        "Washington": (38.9, -77.0), "LA": (34.0, -118.2)
    }

    def get_coords(team_name):
        if not isinstance(team_name, str): return (39.8, -98.5)
        for city, coords in CITY_COORDS.items():
            if city in team_name: return coords
        return (39.8, -98.5)

    if "lat" not in df_filtered.columns:
        df_filtered["lat"] = df_filtered["TEAM_NAME"].apply(lambda x: get_coords(x)[0])
        df_filtered["lon"] = df_filtered["TEAM_NAME"].apply(lambda x: get_coords(x)[1])

    if view_mode == "Individual Players":
        # Add Jitter
        df_filtered["lat_j"] = df_filtered["lat"] + np.random.normal(0, 0.15, len(df_filtered))
        df_filtered["lon_j"] = df_filtered["lon"] + np.random.normal(0, 0.15, len(df_filtered))
        
        fig = px.scatter_mapbox(
            df_filtered, lat="lat_j", lon="lon_j", color="Calculated_Bias_Score",
            hover_name="PLAYER_NAME", hover_data=["TEAM_NAME", "Salary"],
            # Fixed Color Scale: Red=Premium, Blue=Penalty, White=0
            color_continuous_scale="RdBu_r", 
            color_continuous_midpoint=0,
            zoom=3, height=600, size_max=15, title="Individual Structural Bias"
        )
    else:
        # Aggregate by Team
        team_df = df_filtered.groupby(["TEAM_NAME", "lat", "lon"]).agg({
            "Calculated_Bias_Score": "mean", "Salary": "mean", "Age": "mean", 
            "DRAFT_NUMBER": "mean", "PLAYER_NAME": "count"
        }).reset_index()
        
        fig = px.scatter_mapbox(
            team_df, lat="lat", lon="lon", color="Calculated_Bias_Score", size="Salary",
            hover_name="TEAM_NAME", 
            hover_data={"Calculated_Bias_Score":":.2f", "Age":":.1f", "DRAFT_NUMBER":":.1f"},
            color_continuous_scale="RdBu_r", 
            color_continuous_midpoint=0,
            zoom=3, height=600, title="Average Team Structural Bias"
        )

    fig.update_layout(mapbox_style="carto-positron", margin={"r":0,"t":40,"l":0,"b":0})
    st.plotly_chart(fig, use_container_width=True)

# --- 6. VISUAL LEADERBOARD ---
st.header("Structural Leaderboard")

# Prepare Data for Display
display_cols = ["PLAYER_NAME", "TEAM_NAME", "Calculated_Bias_Score", "Age", "DRAFT_NUMBER"]
leader_df = df_filtered[display_cols].copy()
leader_df = leader_df.rename(columns={"Calculated_Bias_Score": "Bias Score", "DRAFT_NUMBER": "Draft Pick"})

col_win, col_loss = st.columns(2)

# Common Column Configuration for the Bar Chart visual
bias_col_config = st.column_config.ProgressColumn(
    "Structural Impact",
    help="Red = Premium, Blue = Penalty",
    format="%.2f",
    min_value=-1.5, # Approximate range after centering
    max_value=1.5,
)

with col_win:
    st.markdown("#### Top Structural Premiums")
    st.caption("These players benefit most from Age, Fame, and Market Size.")
    top_df = leader_df.sort_values("Bias Score", ascending=False).head(10)
    
    st.dataframe(
        top_df,
        column_config={"Bias Score": bias_col_config},
        hide_index=True,
        use_container_width=True
    )

with col_loss:
    st.markdown("#### Top Structural Penalties")
    st.caption("These players are dragged down by Youth, Draft History, or Small Markets.")
    bot_df = leader_df.sort_values("Bias Score", ascending=True).head(10)
    
    st.dataframe(
        bot_df,
        column_config={"Bias Score": bias_col_config},
        hide_index=True,
        use_container_width=True
    )