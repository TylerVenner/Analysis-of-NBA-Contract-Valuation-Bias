import streamlit as st
import pandas as pd
from configs.vis_utils import load_data

st.title("Player Details")

df, residuals_Y, residuals_Z, final_ols_results = load_data()

# --- Player details ---
st.subheader("Select a Player to View Details")

columns_to_keep = [
    "PLAYER_NAME","TEAM_NAME","Salary","Followers","BIRTHDATE","COUNTRY","DRAFT_YEAR",
    "DRAFT_ROUND","DRAFT_NUMBER","OFF_RATING","DEF_RATING","NET_RATING","AST_PCT",
    "AST_TO","AST_RATIO","OREB_PCT","DREB_PCT","REB_PCT","TM_TOV_PCT","EFG_PCT","TS_PCT",
    "USG_PCT","PACE","PACE_PER40","PIE","POSS","FGM_PG","FGA_PG"
]
players_df = df[columns_to_keep].copy()
players_df["BIRTHDATE"] = pd.to_datetime(players_df["BIRTHDATE"]).dt.date
players_df["DRAFT_NUMBER"] = players_df["DRAFT_NUMBER"].astype(str).replace("61", "Undrafted")

search_term = st.text_input("Search players by name:")
if search_term:
    players_df = players_df[players_df["PLAYER_NAME"].str.contains(search_term, case=False, na=False)]

# --- Advanced filters ---
with st.expander("Advanced Filters"):
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

    cat_cols = players_df.select_dtypes(include=['object']).columns
    cat_filter_col = st.selectbox("Filter by categorical stat (optional):", [None] + list(cat_cols))

    if cat_filter_col:
        unique_vals = sorted(players_df[cat_filter_col].dropna().unique())
        selected_val = st.selectbox(f"Select value for {cat_filter_col}", unique_vals)
        players_df = players_df[players_df[cat_filter_col] == selected_val]

# --- Player selector ---
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

# --- Player comparison tool ---
st.write("### Player Comparison Tool")
current_df = players_df
player_options = sorted(current_df["PLAYER_NAME"].unique())

comp_col1, comp_col2 = st.columns(2)
with comp_col1:
    player_a = st.selectbox("Player A", player_options, key="compA_selectbox")
with comp_col2:
    player_b = st.selectbox("Player B", player_options, key="compB_selectbox")

valid_a = player_a in current_df["PLAYER_NAME"].values
valid_b = player_b in current_df["PLAYER_NAME"].values

if not player_options:
    st.warning("No players available after filters. Adjust filters to see comparison.")
elif not (valid_a and valid_b):
    st.warning("Selected player is not in the filtered results. Please pick from the current list.")
elif player_a and player_b:
    a_rows = current_df[current_df["PLAYER_NAME"] == player_a]
    b_rows = current_df[current_df["PLAYER_NAME"] == player_b]
    if a_rows.empty or b_rows.empty:
        st.warning("One of the selected players has no rows after filters.")
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