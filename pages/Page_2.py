import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Import pipeline function
from scripts.main import run_pipeline

st.set_page_config(
    page_title="Analysis",
    page_icon="📈",
    layout="wide"
)

st.title("something")

# CONFIG
X_COLS = [
    "OFF_RATING", "DEF_RATING", "NET_RATING", "AST_PCT", "AST_TO",
    "AST_RATIO", "OREB_PCT", "REB_PCT", "DREB_PCT", "TM_TOV_PCT",
    "EFG_PCT", "TS_PCT", "PACE", "PIE", "USG_PCT",
    "POSS", "FGM_PG", "FGA_PG"
]

Z_COLS = [
    "DRAFT_NUMBER", 
    "active_cap", 
    "dead_cap", 
    "OWNER_NET_WORTH_B", 
    "Capacity", 
    "STADIUM_YEAR_OPENED",
    "STADIUM_COST",
    "Followers",
    "Age",
    "is_USA"
]

Y_COL = "Salary"

# --- Run pipeline once and cache results ---
@st.cache_data
def run_pipeline_once():
    df_clean, residuals_Y, residuals_Z, final_ols_results = run_pipeline()
    # Add log_salary if not already present
    if "log_salary" not in df_clean.columns:
        df_clean["log_salary"] = np.log(df_clean[Y_COL])
    return df_clean, residuals_Y, residuals_Z, final_ols_results

df, residuals_Y, residuals_Z, final_ols_results = run_pipeline_once()


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

st.write("---")

# --- Correlation heatmap ---
corr = df[X_COLS + Z_COLS + ["log_salary"]].corr()
fig = px.imshow(
    corr, text_auto=False, aspect="auto",
    title="Correlation Heatmap", color_continuous_scale="Reds"
)
st.plotly_chart(fig, use_container_width=True)

# --- Scatter plot ---
feature = st.selectbox("Select a Feature", X_COLS + Z_COLS)
fig = px.scatter(
    df, x=feature, y="log_salary", hover_data=["PLAYER_NAME"],
    title=f"log(Salary) vs {feature}"
)
st.plotly_chart(fig, use_container_width=True)

# --- Distribution ---
st.subheader("Distribution")
fig2 = px.histogram(df, x=feature, nbins=40)
st.plotly_chart(fig2, use_container_width=True)

# --- Scatter matrix ---
team_options = ["None"] + sorted(df["TEAM_NAME"].unique())
team_name = st.selectbox("Select a Team", team_options)

# If "None" is selected, use the full dataframe
if team_name == "None":
    team_df = df
else:
    team_df = df[df["TEAM_NAME"] == team_name]

# Combine X and Z features
all_features = X_COLS + Z_COLS

# Keep only numeric columns
numeric_features = [
    col for col in all_features 
    if pd.api.types.is_numeric_dtype(df[col])
]

# Let the user choose which numeric features to plot
selected_features = st.multiselect(
    "Select numeric features for scatter matrix",
    options=numeric_features,
    default=numeric_features[:5]  # show a few by default
)

if selected_features:
    fig = px.scatter_matrix(
        team_df,
        dimensions=selected_features,
        color="TEAM_NAME",
        title="Scatter Matrix" if team_name == "None" else f"Scatter Matrix for {team_name}"
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Please select at least one numeric feature to display the scatter matrix.")

# --- Numeric vs Categorical Z Visualizations ---

# Identify categorical Z features (including binary-coded ones like is_USA)
categorical_zcols = []
for col in Z_COLS:
    if not pd.api.types.is_numeric_dtype(df[col]):
        categorical_zcols.append(col)
    else:
        # Special case: binary flags like is_USA
        if df[col].nunique() <= 5:  # treat small unique sets as categorical
            categorical_zcols.append(col)

# Identify numeric features (X, Z, Y)
numeric_cols = [
    col for col in df.columns 
    if pd.api.types.is_numeric_dtype(df[col]) and col not in categorical_zcols
]

if categorical_zcols and numeric_cols:
    cat_feature = st.selectbox("Select a categorical Z feature", categorical_zcols)
    num_feature = st.selectbox("Select a numeric feature", numeric_cols)

    # Map is_USA to labels only for plotting
    plot_df = df.copy()
    if cat_feature == "is_USA":
        plot_df["is_USA_label"] = plot_df["is_USA"].map({1: "USA", 0: "Non-USA"})
        cat_feature_plot = "is_USA_label"
    else:
        cat_feature_plot = cat_feature

    # Boxplot
    fig_box = px.box(
        plot_df, x=cat_feature_plot, y=num_feature,
        hover_data=["PLAYER_NAME"],
        title=f"{num_feature} vs. {cat_feature}"
    )
    st.plotly_chart(fig_box, use_container_width=True)

    # Violin Plot
    fig_violin = px.violin(
        plot_df, x=cat_feature_plot, y=num_feature,
        box=True, points="all",
        hover_data=["PLAYER_NAME"],
        title=f"Distribution of {num_feature} vs. {cat_feature}"
    )
    fig_violin.update_traces(pointpos=-0)
    st.plotly_chart(fig_violin, use_container_width=True)

    # Bar Chart (group averages)
    avg_df = plot_df.groupby(cat_feature_plot)[num_feature].mean().reset_index()
    fig_bar = px.bar(
        avg_df, x=cat_feature_plot, y=num_feature,
        title=f"Average {num_feature} vs. {cat_feature}",
        color=num_feature, color_continuous_scale="Blues"
    )
    st.plotly_chart(fig_bar, use_container_width=True)
else:
    st.info("No categorical Z features or numeric features available for comparison.")

# --- Parallel Coordinates ---
team_options = ["None"] + sorted(df["TEAM_NAME"].unique())
team_name = st.selectbox("Select a Team (for parallel coordinates)", team_options)

# If "None" is selected, use the full dataframe
if team_name == "None":
    team_df = df
else:
    team_df = df[df["TEAM_NAME"] == team_name]

# Let the user choose which X_COLS to plot
selected_xcols = st.multiselect(
    "Select features for parallel coordinates",
    options=X_COLS,
    default=X_COLS[:5]   # show a few by default
)

if selected_xcols:
    fig = px.parallel_coordinates(
        team_df,
        dimensions=selected_xcols,
        color="log_salary",
        color_continuous_scale=px.colors.diverging.Tealrose,
        title="Parallel Coordinates" if team_name == "None" else f"Parallel Coordinates for {team_name}"
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Please select at least one feature to display the parallel coordinates plot.")

st.write("---")

# --- Residuals Visualization ---
st.header("Residuals Analysis")

st.subheader("Residuals vs Fitted Values")

# Get fitted values from final OLS
fitted_vals = final_ols_results.fittedvalues
residuals = final_ols_results.resid

fig = px.scatter(
    x=fitted_vals, y=residuals,
    labels={"x": "Fitted Values", "y": "Residuals"},
    title="Residuals vs Fitted Values",
    hover_data=[df["PLAYER_NAME"]]
)
fig.add_hline(y=0, line_dash="dash", line_color="red")
st.plotly_chart(fig, use_container_width=True)

# Residuals correlation heatmap
corr_resid = residuals_Z.corr()
fig_corr_resid = px.imshow(
    corr_resid,
    text_auto=False,
    aspect="auto",
    title="Correlation Heatmap of Z Residuals",
    color_continuous_scale="RdBu"
)
st.plotly_chart(fig_corr_resid, use_container_width=True)

# Scatter plot: residuals_Y vs one residual Z feature
resid_feature = st.selectbox("Select a Residual Feature", residuals_Z.columns)

fig_scatter_resid = px.scatter(
    residuals_Z,
    x=resid_feature,
    y=residuals_Y,
    title=f"Residuals: Salary vs {resid_feature}",
    labels={"y": "Residual Salary", "x": f"Residual {resid_feature}"}
)
st.plotly_chart(fig_scatter_resid, use_container_width=True)

# Distribution of residuals
st.subheader("Residual Distribution")
fig_hist_resid = px.histogram(residuals_Y, nbins=40, title="Distribution of Salary Residuals")
st.plotly_chart(fig_hist_resid, use_container_width=True)

# Parallel coordinates on residuals
st.subheader("Parallel Coordinates of Residuals")
selected_resid_cols = st.multiselect(
    "Select residual features",
    options=list(residuals_Z.columns),
    default=list(residuals_Z.columns[:5])
)

if selected_resid_cols:
    fig_parallel_resid = px.parallel_coordinates(
        residuals_Z,
        dimensions=selected_resid_cols,
        color=residuals_Y,
        color_continuous_scale=px.colors.diverging.Tealrose,
        title="Parallel Coordinates of Residuals"
    )
    st.plotly_chart(fig_parallel_resid, use_container_width=True)