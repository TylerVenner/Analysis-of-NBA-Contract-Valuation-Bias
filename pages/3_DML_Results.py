import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import plotly.express as px
import os
import glob
from pathlib import Path
DATA_PATH = "data/processed/master_dataset_advanced_v2.csv"
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

df = load_data()

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
         "AST_RATIO", "OREB_PCT", "REB_PCT", "DREB_PCT", "TM_TOV_PCT",
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



st.title("📈 Final Coefficients")
st.subheader("Final OLS Results & Interpretation")

st.markdown("""
This page presents the **final debiased OLS coefficients** that come from our  
backend analysis pipeline (run in `main.py`), where we use:

- Gradient Boosting models to remove nonlinear performance effects  
- Treatment models to orthogonalize contextual variables  
- A final OLS regression on the debiased residuals (HC3 robust)  

The results represent the **market price of structural bias factors** after controlling
for performance.
""")

OLS_PATH = Path("data/processed/final_ols_table.csv")

@st.cache_data
def load_ols_table(path: Path):
    return pd.read_csv(path)

if not OLS_PATH.exists():
    st.error("OLS table not found! Run `python main.py` to generate `final_ols_table.csv`.")
    st.stop()

df_coef = load_ols_table(OLS_PATH)

df_coef = df_coef.rename(columns={
    "Variable": "variable",
    "coef": "coef",
    "std_err": "std_err",
    "p_value": "p_value",
    "CI_0.025": "CI_lower",
    "CI_0.975": "CI_upper"
})

df_coef = df_coef.set_index("variable")

# 2. DISPLAY RAW TABLE
st.header("1. Final OLS Regression Table")

st.markdown("""
These coefficients quantify the **impact of each bias factor** on the remaining  
salary unexplained by performance (ε_Y).  

- **Positive coefficient → salary premium**  
- **Negative coefficient → salary penalty**  
- **p < 0.05 → statistically significant**  
""")

st.dataframe(
    df_coef.style.format({
        "coef": "{:.5f}",
        "std_err": "{:.5f}",
        "p_value": "{:.3f}"
    }),
    use_container_width=True
)

# 3. COEFFICIENT VISUALIZATION (BAR PLOTS)
st.header("2. Visualizing Effect Sizes")

viz_df = df_coef.copy()
viz_df["abs_coef"] = viz_df["coef"].abs()

# Signed coefficients
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Signed Coefficients (Premium vs Penalty)")
    chart_signed = (
        alt.Chart(viz_df.reset_index())
        .mark_bar()
        .encode(
            x=alt.X("coef:Q", title="Coefficient (log-salary)"),
            y=alt.Y("variable:N", sort="-x"),
            tooltip=["variable", "coef", "std_err", "p_value"]
        )
    )
    st.altair_chart(chart_signed, use_container_width=True)

# 3. PLAYER STORY & INTERPRETATION SECTION
st.header("3. Player-Specific Interpretation")

st.markdown("""
Select a player to see how our structural bias coefficients **interact with their situation**.  
We compare the player to **league averages** on Age, Draft position, owner wealth, and market size,  
and interpret what a **positive or negative coefficient** means for them.

We **do not treat the coefficient times the raw value as a literal salary change**.  
Instead, we use the **sign** of the coefficient and where the player sits relative to the league  
to tell a qualitative story about premiums or penalties.
""")

# Load player-level data
PLAYER_PATH = Path("data/processed/master_dataset_advanced_v2.csv")

@st.cache_data
def load_player_data(path: Path):
    df = pd.read_csv(path)
    df = df.replace("Undrafted", 61)
    df["DRAFT_NUMBER"] = pd.to_numeric(df["DRAFT_NUMBER"], errors="coerce")
    # make sure numeric where possible
    for col in ["Age", "OWNER_NET_WORTH_B", "Capacity", "Followers"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

player_df = load_player_data(PLAYER_PATH)

# Use the correct name column
PLAYER_COL = "PLAYER_NAME"

if PLAYER_COL not in player_df.columns:
    st.error(f"Expected a column '{PLAYER_COL}' in the player dataset.")
    st.write("Available columns:", list(player_df.columns))
    st.stop()

# league medians for context
league_medians = {}
for col in ["Age", "DRAFT_NUMBER", "OWNER_NET_WORTH_B", "Capacity"]:
    if col in player_df.columns:
        league_medians[col] = player_df[col].median()

# Player dropdown
player_name = st.selectbox(
    "Choose a player to analyze:",
    sorted(player_df[PLAYER_COL].dropna().unique())
)

player_row = player_df[player_df[PLAYER_COL] == player_name].iloc[0]

# Small snapshot table
snapshot_cols = [c for c in [
    PLAYER_COL, "Team", "Age", "DRAFT_NUMBER", "DRAFT_ROUND",
    "Salary", "OWNER_NET_WORTH_B", "Capacity", "Followers", "is_USA"
] if c in player_df.columns]

st.subheader(f"Context for **{player_name}**")
st.dataframe(player_row[snapshot_cols].to_frame().T)

st.markdown("### How the bias coefficients apply to this player")

stories = []

def add_story(text: str):
    stories.append(text.strip())

# AGE
if "Age" in player_row.index and "Age" in df_coef.index and "Age" in league_medians:
    age = player_row["Age"]
    med_age = league_medians["Age"]
    gamma_age = df_coef.loc["Age", "coef"]
    sig_age = df_coef.loc["Age", "p_value"] < 0.05

    if pd.notna(age):
        if gamma_age > 0:
            if age > med_age:
                add_story(f"""
**Age (Veteran premium)**  
- {player_name} is **{age:.1f} years old**, older than the league median of **{med_age:.1f}**.  
- Our OLS coefficient for Age is **{gamma_age:.3f}** ({'significant' if sig_age else 'not significant'} and positive).  
- This suggests that, *holding performance fixed*, **older veterans tend to be paid more than younger players**.  
  In other words, players like {player_name} benefit from a **veteran premium** in salary negotiations.
""")
            else:
                add_story(f"""
**Age (Waiting for the veteran bump)**  
- {player_name} is **{age:.1f} years old**, younger than the league median of **{med_age:.1f}**.  
- Age has a **positive coefficient ({gamma_age:.3f})**, meaning teams usually pay **more** for veterans even after controlling for stats.  
- Young players like {player_name} **do not yet fully benefit from that veteran premium**, even if their performance is strong.
""")
        elif gamma_age < 0:
            add_story(f"""
**Age (Veteran discount)**  
- The coefficient on Age is **negative ({gamma_age:.3f})**, meaning that, after controlling for performance,  
  **older players tend to be paid less** than younger ones.  
- This would indicate a league-wide tendency to **undervalue aging players** relative to their box-score impact.
""")

# DRAFT NUMBER
if "DRAFT_NUMBER" in player_row.index and "DRAFT_NUMBER" in df_coef.index and "DRAFT_NUMBER" in league_medians:
    draft_no = player_row["DRAFT_NUMBER"]
    med_draft = league_medians["DRAFT_NUMBER"]
    gamma_draft = df_coef.loc["DRAFT_NUMBER", "coef"]
    sig_draft = df_coef.loc["DRAFT_NUMBER", "p_value"] < 0.05

    if pd.notna(draft_no):
        if gamma_draft < 0:
            if draft_no > med_draft:
                add_story(f"""
**Draft position (Late-pick penalty)**  
- {player_name} was drafted **#{int(draft_no)}**, later than the median pick of **#{int(med_draft)}**.  
- The coefficient on draft number is **negative ({gamma_draft:.3f})** ({'significant' if sig_draft else 'not significant'}).  
- This means that, for the same performance, **later picks tend to earn less** than early picks.  
  Players like {player_name} face a **structural late-pick penalty** in residual salary.
""")
            else:
                add_story(f"""
**Draft position (Pedigree premium)**  
- {player_name} was drafted **#{int(draft_no)}**, earlier than the median pick of **#{int(med_draft)}**.  
- With a **negative draft coefficient ({gamma_draft:.3f})**, early picks are rewarded with **higher salaries**  
  even after accounting for their on-court performance.  
  {player_name} benefits from this **draft pedigree premium**.
""")
        elif gamma_draft > 0:
            add_story(f"""
**Draft position (Unusual pattern)**  
- The draft coefficient is **positive ({gamma_draft:.3f})**, which would mean later picks earn *more* than early picks  
  after controlling for performance. That would be an unusual league pattern and may indicate model instability.
""")

# OWNER WEALTH
if "OWNER_NET_WORTH_B" in player_row.index and "OWNER_NET_WORTH_B" in df_coef.index and "OWNER_NET_WORTH_B" in league_medians:
    own_w = player_row["OWNER_NET_WORTH_B"]
    med_own = league_medians["OWNER_NET_WORTH_B"]
    gamma_own = df_coef.loc["OWNER_NET_WORTH_B", "coef"]

    if pd.notna(own_w):
        if gamma_own > 0:
            if own_w > med_own:
                add_story(f"""
**Owner wealth (Deep-pocket premium)**  
- {player_name}'s team owner is worth about **${own_w:.1f}B**, above the league median of **${med_own:.1f}B**.  
- The **positive coefficient on owner wealth ({gamma_own:.3f})** suggests rich owners are willing to **overpay**  
  relative to pure performance. Players on these teams, like {player_name}, enjoy a **deep-pocket premium**.
""")
            else:
                add_story(f"""
**Owner wealth (Budget constraint)**  
- {player_name}'s team owner wealth (~**${own_w:.1f}B**) is below the league median of **${med_own:.1f}B**.  
- Since the coefficient on owner wealth is **positive ({gamma_own:.3f})**, players on smaller-budget teams  
  face a mild **structural headwind** in salary, compared to equally productive players on richer teams.
""")

# MARKET SIZE / CAPACITY
if "Capacity" in player_row.index and "Capacity" in df_coef.index and "Capacity" in league_medians:
    cap = player_row["Capacity"]
    med_cap = league_medians["Capacity"]
    gamma_cap = df_coef.loc["Capacity", "coef"]

    if pd.notna(cap):
        if gamma_cap > 0:
            if cap > med_cap:
                add_story(f"""
**Market size (Big-market boost)**  
- {player_name} plays in an arena with capacity **{int(cap):,}**, above the league median of **{int(med_cap):,}** seats.  
- A **positive coefficient on Capacity ({gamma_cap:.3f})** means big markets tend to **push salaries up**  
  beyond what performance alone would justify. {player_name} benefits from a **big-market boost**.
""")
            else:
                add_story(f"""
**Market size (Small-market discount)**  
- {player_name}'s arena capacity **({int(cap):,})** is smaller than the league median of **{int(med_cap):,}**.  
- With a **positive Capacity coefficient ({gamma_cap:.3f})**, players on small-market teams  
  tend to face a **small-market discount** in salary residuals.
""")

# NATIONALITY
if "is_USA" in player_row.index and "is_USA" in df_coef.index:
    nat_flag = player_row["is_USA"]
    gamma_nat = df_coef.loc["is_USA", "coef"]

    if nat_flag == 1:
        nat_label = "American"
    else:
        nat_label = "international"

    if gamma_nat < 0:
        add_story(f"""
**Nationality (International penalty)**  
- {player_name} is **{nat_label}**.  
- The coefficient on `is_USA` is **negative ({gamma_nat:.3f})**, suggesting that, after controlling for performance,  
  **international players earn slightly less** than otherwise similar American players.
""")
    elif gamma_nat > 0:
        add_story(f"""
**Nationality (American premium)**  
- {player_name} is **{nat_label}**.  
- A **positive `is_USA` coefficient ({gamma_nat:.3f})** would mean American players tend to receive a small  
  **structural salary premium** over similar international players.
""")

# Fallback if no stories
if not stories:
    st.info("Not enough contextual information is available for this player to generate a narrative.")
else:
    for s in stories:
        st.markdown(s)
        st.markdown("---")


# 4. Structural Bias Score by Factor (Simple Visualization)

st.header("4. Structural Bias Score by Factor")

st.markdown("""
This chart summarizes how each structural factor contributes to a **salary premium or penalty**
for the selected player, relative to the league median.

For each factor, we compute a simple score:

> (Player value − League median) × OLS coefficient

- **Positive score → premium** (player is structurally favored)
- **Negative score → penalty** (player is structurally disadvantaged)

This is not an exact dollar amount, but a **directional index** of how context and coefficients
combine for this player.
""")

factor_map = [
    ("Age", "Age"),
    ("Draft Number", "DRAFT_NUMBER"),
    ("Owner Net Worth (B$)", "OWNER_NET_WORTH_B"),
    ("Arena Capacity", "Capacity"),
    ("Followers", "Followers"),
    ("Nationality (is_USA)", "is_USA"),
]

score_rows = []
for label, col in factor_map:
    if col in player_row.index and col in league_medians and col in df_coef.index:
        val = player_row[col]
        med = league_medians[col]
        gamma = df_coef.loc[col, "coef"]

        if pd.notna(val) and pd.notna(med) and pd.notna(gamma):
            try:
                val = float(val)
                med = float(med)
                gamma = float(gamma)
            except Exception:
                continue

            delta = val - med
            score = delta * gamma

            if abs(score) < 1e-8:
                direction = "Neutral"
            elif score > 0:
                direction = "Premium"
            else:
                direction = "Penalty"

            score_rows.append({
                "Factor": label,
                "Score": score,
                "Delta_vs_Median": delta,
                "Coefficient": gamma,
                "Direction": direction,
            })

score_df = pd.DataFrame(score_rows)

if score_df.empty:
    st.info("Not enough information to compute structural bias scores for this player.")
else:
    # Sort by absolute impact
    score_df = score_df.sort_values("Score", ascending=True)

    st.markdown("### Structural bias index by factor")

    chart = (
        alt.Chart(score_df)
        .mark_bar()
        .encode(
            x=alt.X("Score:Q", title="Structural bias score"),
            y=alt.Y("Factor:N", sort=list(score_df["Factor"])),
            color=alt.Color("Direction:N", scale=alt.Scale(domain=["Penalty", "Neutral", "Premium"]),
                            legend=alt.Legend(title="Effect")),
            tooltip=[
                "Factor",
                alt.Tooltip("Score:Q", format=".4f", title="Bias score"),
                alt.Tooltip("Delta_vs_Median:Q", format=".2f", title="Player - median"),
                alt.Tooltip("Coefficient:Q", format=".4f", title="OLS coefficient"),
                "Direction",
            ]
        )
        .properties(height=250)
    )

    # Add zero line for reference
    zero_rule = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(strokeDash=[4, 2]).encode(x="x:Q")

    st.altair_chart(chart + zero_rule, use_container_width=True)

    st.markdown(f"""
For **{player_name}**, bars to the **right** of zero indicate structural **premiums**,  
while bars to the **left** indicate structural **penalties**, after controlling for performance.

- A large positive bar means the player is **above the league median on a factor with a positive coefficient**,  
  or **below the median on a factor with a negative coefficient** (both create a premium).
- A large negative bar means the opposite: that contextual factor pushes the player toward a **salary discount**.

This visualization connects the **OLS coefficients** directly to the **player's own situation**,
showing which structural factors help or hurt them the most.
""")
