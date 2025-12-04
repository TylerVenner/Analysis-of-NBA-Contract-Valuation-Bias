import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from configs.vis_utils import load_data

st.title("Analysis")

# CONFIG
X_COLS = [
    "OFF_RATING", "DEF_RATING", "NET_RATING", "AST_PCT", "AST_TO",
    "AST_RATIO", "OREB_PCT", "REB_PCT", "DREB_PCT", "TM_TOV_PCT",
    "EFG_PCT", "TS_PCT", "PACE", "PIE", "USG_PCT",
    "POSS", "FGM_PG", "FGA_PG"
]

Z_COLS = [
    "DRAFT_ROUND",
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

df, residuals_Y, residuals_Z, final_ols_results = load_data()

# --- Correlation heatmap ---
corr = df[X_COLS + Z_COLS + ["log_salary"]].corr()
fig = px.imshow(
    corr, text_auto=False, aspect="auto",
    title="Correlation Heatmap", color_continuous_scale="Reds"
)
st.plotly_chart(fig, use_container_width=True)

# --- Scatter plot (X_COLS) ---
feature = st.selectbox("Select a Feature", X_COLS)
fig = px.scatter(
    df, x=feature, y="log_salary", hover_data=["PLAYER_NAME"],
    title=f"log(Salary) vs {feature}"
)
st.plotly_chart(fig, use_container_width=True)

st.write("##### Distribution")
fig2 = px.histogram(df, x=feature, nbins=40)
st.plotly_chart(fig2, use_container_width=True)

# --- Scatter plot (Z_COLS) ---
feature = st.selectbox("Select a Feature", Z_COLS)
fig = px.scatter(
    df, x=feature, y="log_salary", hover_data=["PLAYER_NAME"],
    title=f"log(Salary) vs {feature}"
)
st.plotly_chart(fig, use_container_width=True)

st.write("##### Distribution")
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
