import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="Analysis",
    page_icon="📈",
    layout="wide"
)

st.title("something")

# CONFIG
DATA_PATH = "data/processed/master_dataset_cleaned.csv"

X_COLS = [
    "OFF_RATING", "DEF_RATING", "NET_RATING", "AST_PCT", "AST_TO",
    "AST_RATIO", "OREB_PCT", "REB_PCT", "DREB_PCT", "TM_TOV_PCT",
    "EFG_PCT", "TS_PCT", "PACE", "PIE", "USG_PCT",
    "POSS", "FGM_PG", "FGA_PG"
]

Z_COLS = [
    "DRAFT_NUMBER", "active_cap", "avg_team_age", "dead_cap",
    "OWNER_NET_WORTH_B", "Capacity", "STADIUM_YEAR_OPENED", "STADIUM_COST"
]

Y_COL = "Salary"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    if "DRAFT_NUMBER" in df.columns:
        df["DRAFT_NUMBER"] = df["DRAFT_NUMBER"].replace("Undrafted", 61)
        df["DRAFT_NUMBER"] = pd.to_numeric(df["DRAFT_NUMBER"], errors="coerce")
    df = df.dropna(subset=[Y_COL] + X_COLS + Z_COLS)
    df["log_salary"] = np.log(df[Y_COL])
    return df

df = load_data()

corr = df[X_COLS + Z_COLS + ["log_salary"]].corr()
fig = px.imshow(corr, text_auto=False, aspect="auto", title="Correlation Heatmap", color_continuous_scale="Reds")
st.plotly_chart(fig, use_container_width=True)

feature = st.selectbox("Select a Feature", X_COLS + Z_COLS)

fig = px.scatter(
    df, x=feature, y="log_salary", hover_data=["PLAYER_NAME"],
    title=f"log(Salary) vs {feature}"
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Distribution")
fig2 = px.histogram(df, x=feature, nbins=40)
st.plotly_chart(fig2, use_container_width=True)

st.write("---")

st.subheader("Select a Player to View Details")

players_df = pd.read_csv(DATA_PATH)
# Subset the dataframe
columns_to_keep = [
    "PLAYER_NAME","TEAM_NAME","Salary","Followers","BIRTHDATE","COUNTRY","DRAFT_YEAR",
    "DRAFT_ROUND","DRAFT_NUMBER","OFF_RATING","DEF_RATING","NET_RATING","AST_PCT",
    "AST_TO","AST_RATIO","OREB_PCT","DREB_PCT","REB_PCT","TM_TOV_PCT","EFG_PCT","TS_PCT",
    "USG_PCT","PACE","PACE_PER40","PIE","POSS","FGM_PG","FGA_PG"]
players_df = players_df[columns_to_keep]
# Clean up formatting for display
players_df["BIRTHDATE"] = pd.to_datetime(players_df["BIRTHDATE"]).dt.date

search_term = st.text_input("Search players by name:")
if search_term:
    players_df = players_df[players_df["PLAYER_NAME"].str.contains(search_term, case=False, na=False)]

# ADVANCED FILTERS
with st.expander("Advanced Filters"):
    # Numeric filters
    numeric_cols = players_df.select_dtypes(include=['float64','int64']).columns
    filter_col = st.selectbox("Filter by numeric stat (optional):", [None] + list(numeric_cols))

    if filter_col:
        min_val, max_val = float(players_df[filter_col].min()), float(players_df[filter_col].max())
        selected_range = st.slider(
            f"Select range for {filter_col}",
            min_val,
            max_val,
            (min_val, max_val)
        )
        players_df = players_df[
            (players_df[filter_col] >= selected_range[0]) &
            (players_df[filter_col] <= selected_range[1])
        ]

    # Categorical filters
    cat_cols = players_df.select_dtypes(include=['object']).columns
    cat_filter_col = st.selectbox("Filter by categorical stat (optional):", [None] + list(cat_cols))

    if cat_filter_col:
        unique_vals = sorted(players_df[cat_filter_col].dropna().unique())
        selected_val = st.selectbox(f"Select value for {cat_filter_col}", unique_vals)
        players_df = players_df[players_df[cat_filter_col] == selected_val]

# PLAYER SELECTOR
player_name = st.selectbox(
    "Choose a player:",
    sorted(players_df["PLAYER_NAME"].unique()) if "PLAYER_NAME" in players_df.columns else []
)

if player_name:
    player_row = players_df[players_df["PLAYER_NAME"] == player_name].iloc[0]

    st.markdown(f"## **{player_name}**")
    st.write("### Player Information")

    cols = st.columns(2)
    for i, (col_name, value) in enumerate(player_row.items()):
        with cols[i % 2]:
            st.markdown(f"**{col_name}:** {value}")


st.write("### Player Comparison Tool")
# Use the filtered DataFrame everywhere below
current_df = players_df  # rename for clarity if you like

# Build player options from the filtered data
player_options = sorted(current_df["PLAYER_NAME"].unique())

comp_col1, comp_col2 = st.columns(2)

with comp_col1:
    player_a = st.selectbox(
        "Player A",
        player_options,
        key="compA_selectbox"
    )

with comp_col2:
    player_b = st.selectbox(
        "Player B",
        player_options,
        key="compB_selectbox"
    )

# Validate selections
valid_a = player_a in current_df["PLAYER_NAME"].values
valid_b = player_b in current_df["PLAYER_NAME"].values

if not player_options:
    st.warning("No players available after filters. Adjust filters to see comparison.")
elif not (valid_a and valid_b):
    st.warning("Selected player is not in the filtered results. Please pick from the current list.")
elif player_a and player_b:
    a_rows = current_df[current_df["PLAYER_NAME"] == player_a]
    b_rows = current_df[current_df["PLAYER_NAME"] == player_b]

    # Extra safety: ensure we have at least one row for each
    if a_rows.empty or b_rows.empty:
        st.warning("One of the selected players has no rows after filters. Adjust filters or selection.")
    else:
        a_row = a_rows.iloc[0]
        b_row = b_rows.iloc[0]

        st.write(f"Comparing **{player_a}** and **{player_b}**")

        comp_df = pd.DataFrame({
            "Stat": a_row.index,
            player_a: a_row.values,
            player_b: b_row.values
        })

        st.dataframe(comp_df, use_container_width=True)
