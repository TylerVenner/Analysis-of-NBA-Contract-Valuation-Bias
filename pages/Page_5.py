import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# STREAMLIT CONFIG
st.set_page_config(
    page_title="NBA Salary Bias Map",
    layout="wide"
)

# DATA LOADING
DATA_PATH = "data/processed/streamlit_bias_map.csv"

def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

df = load_data()

# REQUIRED COLUMNS CHECK
REQUIRED_COLS = [
    "PLAYER_NAME",
    "City",
    "Salary",
    "Player_bias_effect",
    "TEAM_NAME",
    "Age",
    "YOS"
]

missing = [c for c in REQUIRED_COLS if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

df.columns = df.columns.str.strip()
df["Salary"] = pd.to_numeric(df["Salary"], errors="coerce")
df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
df["YOS"] = pd.to_numeric(df["YOS"], errors="coerce")
df["Player_bias_effect"] = pd.to_numeric(df["Player_bias_effect"], errors="coerce")

# CITY TO LAT / LON
CITY_COORDS = {
    "Atlanta, GA": (33.7490, -84.3880),
    "Boston, MA": (42.3601, -71.0589),
    "Brooklyn, NY": (40.6782, -73.9442),
    "Charlotte, NC": (35.2271, -80.8431),
    "Chicago, IL": (41.8781, -87.6298),
    "Cleveland, OH": (41.4993, -81.6944),
    "Dallas, TX": (32.7767, -96.7970),
    "Denver, CO": (39.7392, -104.9903),
    "Detroit, MI": (42.3314, -83.0458),
    "Houston, TX": (29.7604, -95.3698),
    "Indianapolis, IN": (39.7684, -86.1581),
    "Los Angeles, CA": (34.0430, -118.2673),
    "Inglewood, CA": (33.9618, -118.3534),
    "Memphis, TN": (35.1495, -90.0490),
    "Miami, FL": (25.7617, -80.1918),
    "Milwaukee, WI": (43.0389, -87.9065),
    "Minneapolis, MN": (44.9778, -93.2650),
    "New Orleans, LA": (29.9511, -90.0715),
    "New York, NY": (40.7505, -73.9934),
    "Oklahoma City, OK": (35.4676, -97.5164),
    "Orlando, FL": (28.5383, -81.3792),
    "Philadelphia, PA": (39.9526, -75.1652),
    "Phoenix, AZ": (33.4484, -112.0740),
    "Portland, OR": (45.5051, -122.6750),
    "Sacramento, CA": (38.5816, -121.4944),
    "San Antonio, TX": (29.4241, -98.4936),
    "San Francisco, CA": (37.7680, -122.3877),
    "Toronto, ON": (43.6532, -79.3832),
    "Salt Lake City, UT": (40.7608, -111.8910),
    "Washington, D.C.": (38.9072, -77.0369)
}

df["lat"] = df["City"].map(lambda x: CITY_COORDS.get(x, (None, None))[0])
df["lon"] = df["City"].map(lambda x: CITY_COORDS.get(x, (None, None))[1])

# JITTER FOR PLAYER VIEW
np.random.seed(13)

jitter_scale = 0.18
df["lat_jitter"] = df["lat"] + np.random.normal(0, 0.08, len(df))
df["lon_jitter"] = df["lon"] + np.random.normal(0, 0.08, len(df))

# SIDEBAR FILTERS
st.sidebar.header("Filters")

min_salary, max_salary = st.sidebar.slider(
    "Salary Range",
    int(df["Salary"].min()),
    int(df["Salary"].max()),
    (int(df["Salary"].min()), int(df["Salary"].max()))
)

min_age, max_age = st.sidebar.slider(
    "Age Range",
    int(df["Age"].min()),
    int(df["Age"].max()),
    (int(df["Age"].min()), int(df["Age"].max()))
)

size_by_salary = st.sidebar.toggle(
    "Scale dot size by Salary",
    value=False
)

df = df[
    (df["Salary"] >= min_salary) &
    (df["Salary"] <= max_salary) &
    (df["Age"] >= min_age) &
    (df["Age"] <= max_age)
]
view_mode = st.sidebar.radio(
    "Map View",
    ["Team Average", "Individual Players"]
)

# MAIN TITLE
st.title("NBA Salary Bias Map")
st.subheader("How Context Shapes Player Pay Beyond On-Court Performance")

st.markdown("""
This visualization shows how much a player’s salary is influenced by **contextual/bias factors** after controlling for on-court performance using Double Machine Learning (DML).

The quantity displayed, *Bias Effect*, captures the portion of salary that comes from where a player plays, who owns the team, how visible they are, and how they entered the league, etc, not how well they perform on the court.
""")

st.markdown("---")

st.markdown("""
### How to Read This Map

Each dot represents a player located at their team’s city.

- **Red** → Player is **paid more than expected** after controlling for performance. 
- **Blue** → Player is **paid less than expected** after controlling for performance. 
- **White / Neutral** → Salary closely matches expected value.
""")

st.markdown("---")

st.markdown("""
### What Does a Positive Bias Effect Mean?

A **positive Bias Effect** means a player is being **paid more than their on-court performance alone would predict**.  
In these cases, **context is boosting salary**.

Common contributing factors include:
- Playing in a big-market city (e.g., New York, Los Angeles)
- A wealthy team owner
- High draft pedigree
- Strong media presence or social media following
""")

st.markdown("---")

st.markdown("""
### What Does a Negative Bias Effect Mean?

A **negative Bias Effect** means a player is being **paid less than their performance alone would predict**.  
Here, **context is suppressing salary**.

Common contributing factors include:
- Playing in a small-market city
- Being a late draft pick or undrafted
- Limited media exposure
- International background
- A low-visibility role
""")

st.markdown("---")

st.caption("""
Important: This map does **not** measure how good a player is.  
It isolates the **non-performance portion of salary** driven by external context.
""")

# TEAM AGGREGATION
team_df = df.groupby(
    ["TEAM_NAME", "City", "lat", "lon"], as_index=False
).agg(
    Avg_Bias=("Player_bias_effect", "mean"),
    Avg_Salary=("Salary", "mean"),
    Player_Count=("PLAYER_NAME", "count")
)

# COLOR SETTING
if view_mode == "Team Average":
    plot_df = team_df.copy()
    color_col = "Avg_Bias"
else:
    plot_df = df.copy()
    color_col = "Player_bias_effect"

# CREATE MAP
if view_mode == "Team Average":
    fig = px.scatter_mapbox(
        plot_df,
        lat="lat",
        lon="lon",
        color="Avg_Bias",
        hover_name="TEAM_NAME",
        hover_data={
            "City": True,
            "Player_Count": True,
            "Avg_Salary": ":,.0f",
            "Avg_Bias": ":.3f",
            "lat": False,
            "lon": False
        },
        zoom=3,
        height=720,
        color_continuous_scale="RdBu_r",
        color_continuous_midpoint=0,
        size_max=50,
    )
else:
    fig = px.scatter_mapbox(
        plot_df.sort_values(by="Player_bias_effect"),
        lat="lat_jitter",
        lon="lon_jitter",
        color="Player_bias_effect",
        hover_name="PLAYER_NAME",
        hover_data={
            "TEAM_NAME": True,
            "Age": ":.0f",
            "Salary": ":,.0f",
            "YOS": True,
            "Player_bias_effect": ":.3f",
            "lat_jitter": False,
            "lon_jitter": False
        },
        zoom=3,
        height=720,
        color_continuous_scale="RdBu_r",
        color_continuous_midpoint=0,
        size_max=30,
    )

if size_by_salary:
    if view_mode == "Team Average":
        fig.update_traces(marker=dict(
            size=(plot_df["Avg_Salary"] / plot_df["Avg_Salary"].max()) * 40 + 10
        ))
    else:
        fig.update_traces(marker=dict(
            size=(plot_df["Salary"] / plot_df["Salary"].max()) * 30 + 6
        ))
else:
    if view_mode == "Team Average":
        fig.update_traces(marker=dict(size=20)) # slightly larger for teams
    else:
        fig.update_traces(marker=dict(size=10)) # players

fig.update_layout(
    mapbox_style="carto-positron",
    margin={"r": 0, "t": 0, "l": 0, "b": 0},
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.header("Most Overpaid & Underpaid Players")

# Clean display dataframe
display_cols = [
    "PLAYER_NAME",
    "TEAM_NAME",
    "City",
    "Salary",
    "Player_bias_effect",
    "Age",
    "YOS"
]

display_df = df[display_cols].copy()

# Formatting
display_df["Salary"] = display_df["Salary"].map("${:,.0f}".format)
display_df["Player_bias_effect"] = display_df["Player_bias_effect"].round(4)

# Top 10 Overpaid
top_overpaid = display_df.sort_values(
    by="Player_bias_effect", ascending=False
).head(10)

# Top 10 Underpaid
top_underpaid = display_df.sort_values(
    by="Player_bias_effect", ascending=True
).head(10)

# Display side-by-side
col1, col2 = st.columns(2)

with col1:
    st.subheader("Top 10 Most Overpaid")
    st.dataframe(
        top_overpaid,
        use_container_width=True,
        hide_index=True
    )

with col2:
    st.subheader("Top 10 Most Underpaid")
    st.dataframe(
        top_underpaid,
        use_container_width=True,
        hide_index=True
    )

