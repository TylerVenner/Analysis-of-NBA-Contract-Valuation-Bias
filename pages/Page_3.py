import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
import plotly.express as px
import os
import glob
from pathlib import Path

# --- Page Configuration ---
st.set_page_config(
    page_title="PULS Bias Analysis",
    page_icon="📉",
    layout="wide"
)

DATA_PATH = "data/processed/master_dataset_advanced_v2.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

df = load_data()

# --- Helper Functions ---
def get_latest_bias_map_path(base_dir="reports/maps"):
    """
    Finds the most recently generated bias map HTML file.
    Assumes structure: reports/maps/run_{timestamp}/bias_attribution_map_3d.html
    """
    # Search for all html files matching the pattern
    # Adjust 'reports/maps' relative to where you run 'streamlit run app.py'
    # If app.py is in root, use 'reports/maps'. If in src, use '../reports/maps'
    
    # Let's try to be robust and look relative to this script file
    current_dir = Path(__file__).parent.resolve()
    
    # Check if we are in 'src' or root. Assuming reports is in root.
    # Adjust this path if your folder structure is different!
    project_root = current_dir  # Modify this based on where app.py lives vs reports
    
    # Example: If app.py is in project root
    search_pattern = os.path.join("reports", "maps", "run_*", "bias_attribution_map_3d.html")
    
    files = glob.glob(search_pattern)
    
    if not files:
        # Try searching one level up if not found (in case app.py is in a subfolder)
        search_pattern = os.path.join("..", "reports", "maps", "run_*", "bias_attribution_map_3d.html")
        files = glob.glob(search_pattern)

    if not files:
        return None
        
    # Get the latest file based on creation time or name (timestamp in name ensures sort order)
    latest_file = max(files, key=os.path.getctime)
    return latest_file

def render_interactive_map(file_path):
    """Reads an HTML file and renders it as a Streamlit component."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Height should match the height set in visualizer.py (900px)
        components.html(html_content, height=900, scrolling=False)
        st.success(f"Loaded map from: `{file_path}`")
    except Exception as e:
        st.error(f"Error loading map: {e}")

# --- Main Layout ---

st.title("NBA Economic Bias Analysis")

# 1. Introduction / Context
st.markdown("""
This dashboard visualizes the latent structure of salary bias. 
It integrates the **Double Machine Learning (DML)** residuals with the **Latent Space Mapping** engine.
""")

st.divider()

# 2. Bias Models Section (Your previous code)
st.subheader("1. Bias/Treatment Models (h)")
st.markdown("""
Analysis of how well player performance predicts contextual bias factors.
""")

z_cols = [
    "DRAFT_NUMBER", "active_cap", "dead_cap",
    "OWNER_NET_WORTH_B", "Capacity", "STADIUM_YEAR_OPENED", "STADIUM_COST",
    "Followers", "Age", "is_USA"
]
col1, col2 = st.columns([1, 3])
with col1:
    selected_factor = st.selectbox("Select Bias Factor (Z):", z_cols)
with col2:
    st.markdown("### Relationship Between Bias and Performance Factors")
    perf_col = st.selectbox(
        "Select Performance Factor (X):",
        ["OFF_RATING", "DEF_RATING", "NET_RATING", "AST_PCT", "AST_TO", 
         "AST_RATING", "OREB_PCT", "REB_PCT", "DREB_PCT", "TM_TOV_PCT", 
        "EFG_PCT", "TS_PCT", "PACE", "PIE", "USG_PCT", "POSS", "FGM_PG", 
        "FGA_PG", "GP", "MIN", "AVG_SPEED", "DIST_MILES", "ISO_PTS", 
        "POST_PTS", "CLUTCH_PTS", "CLUTCH_GP", "RIM_DFG_PCT"]
    )
    plot_df = df[[perf_col, selected_factor]].dropna()
    if plot_df.empty:
        st.warning("Not enough data to plot this relationship.")
    else:
        # Correlation
        corr_val = plot_df[perf_col].corr(plot_df[selected_factor])

        st.metric(
            label="Correlation (Bias ~ Performance)",
            value=f"{corr_val:.3f}"
        )
        # Scatter w/ OLS trend
        fig = px.scatter(
            plot_df,
            x=perf_col,
            y=selected_factor,
            trendline="ols",
            opacity=0.65,
            labels={
                perf_col: "Performance Metric",
                selected_factor: "Bias / Context Variable"
            },
            title=f"{perf_col} vs. {selected_factor}"
        )

        fig.update_layout(
            height=420,
            margin=dict(l=0, r=0, t=40, b=0),
        )

        st.plotly_chart(fig, use_container_width=True)

st.divider()

# 3. The 3D Latent Map Section (NEW)
st.subheader("2. Latent Structure of Bias (The Map)")
st.markdown("""
The interactive 3D map below reveals clusters of players affected by similar economic forces.
- **Diamonds:** Bias Factors (Anchors).
- **Dots:** Players.
- **Proximity:** Represents the strength of the bias driver.
- **Color:** Contract Type.
""")

# Automatically find and load the latest map
latest_map_path = get_latest_bias_map_path()

if latest_map_path and os.path.exists(latest_map_path):
    render_interactive_map(latest_map_path)
else:
    st.warning("No generated bias map found. Please run `src/scripts/run_bias_mapping.py` first.")
    
    # Fallback: Manual Upload
    st.write("Or upload a map HTML file manually:")
    uploaded_file = st.file_uploader("Choose an HTML file", type="html")
    if uploaded_file is not None:
        string_data = uploaded_file.getvalue().decode("utf-8")
        components.html(string_data, height=900)