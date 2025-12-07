import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# --- 1. SETUP & DATA LOADING ---
st.set_page_config(page_title="Statistical vs. Deterministic", layout="wide")

DATA_PATH = Path("data/app_data/player_db.csv")

@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        st.error(f"⚠️ Data file not found at {DATA_PATH}. Please run `prepare_deployment_data.py` locally first.")
        st.stop()
    
    df = pd.read_csv(DATA_PATH)
    
    if "Contract_Type" not in df.columns:
        df["Contract_Type"] = "Free_Market"
    
    # Ensure Salary is numeric
    df["Salary"] = pd.to_numeric(df["Salary"], errors="coerce")
    return df

df = load_data()

# --- DEFINING YOUR X FEATURES ---
# This matches the list you provided from your backend
X_COLS = [
    "OFF_RATING", "DEF_RATING", "NET_RATING", "AST_TO", 
    "AST_RATING", "REB_PCT", "TM_TOV_PCT", 
    "EFG_PCT", "TS_PCT", "PACE", "PIE", "USG_PCT", 
    "POSS", "FGM_PG", "FGA_PG", 
    "GP", "MIN", 
    "AVG_SPEED", "DIST_MILES",
    "ISO_PTS", "POST_PTS",
    "CLUTCH_PTS", "CLUTCH_GP", "RIM_DFG_PCT"
]

# Filter X_COLS to only include those present in the loaded dataframe
available_metrics = [col for col in X_COLS if col in df.columns]

# --- 2. PAGE HEADER & CONTEXT ---

st.title("Statistical vs. Deterministic Players")

col_intro, col_def = st.columns([1.5, 1])

with col_intro:
    st.markdown("""
    **The core challenge of NBA Analytics:** Not every salary is a market signal.
    
    If you run a standard regression on NBA salaries, it will fail. Why? Because nearly **50% of the league** is paid based on rigid formulas (Rules), not their actual on-court value (Markets).
    
    We categorize players into two worlds:
    """)
    st.info("""
    1. **Deterministic (The Noise):** Salaries fixed by the Collective Bargaining Agreement (CBA).
       * *Examples:* Rookie Scale, Max Contracts, Minimums.
    2. **Statistical (The Signal):** Salaries negotiated in the open market.
       * *Examples:* Mid-level veterans, unrestricted free agents.
    """)

with col_def:
    st.markdown("### The Archetypes")
    
    # helper to find example player
    def get_example(ctype, default):
        try:
            return df[df["Contract_Type"] == ctype].sort_values("Salary", ascending=False).iloc[0]["PLAYER_NAME"]
        except:
            return default

    ex_rookie = get_example("Rookie_Scale", "Victor Wembanyama")
    ex_max = get_example("Max_Contract", "Nikola Jokic")
    ex_free = get_example("Free_Market", "Fred VanVleet")

    st.markdown(f"""
    | Contract Type | Example Player | Market Logic? |
    | :--- | :--- | :--- |
    | **Max Contract** | **{ex_max}** | Capped |
    | **Rookie Scale** | **{ex_rookie}** | Fixed |
    | **Free Market** | **{ex_free}** | Negotiated |
    """)

st.markdown("---")

# --- 3. VISUALIZATION SECTION ---

st.header("Visualizing the Structural Barriers")
st.markdown("""
Use the chart below to see the "Walls" and "Ceilings" of the NBA economy.
**Try removing 'Rookie Scale' and 'Max Contract' to see how clean the Free Market relationship is.**
""")

# --- CONTROLS ---
c1, c2, c3 = st.columns([1, 1, 2])

with c1:
    # UPDATED: Uses the full list of available metrics
    default_ix = available_metrics.index("PIE") if "PIE" in available_metrics else 0
    
    x_metric = st.selectbox(
        "Performance Metric (X-Axis)",
        options=available_metrics,
        index=default_ix
    )

with c2:
    # FILTER: Select Contract Types
    all_types = sorted(df["Contract_Type"].unique())
    selected_types = st.multiselect(
        "Filter Contract Types",
        all_types,
        default=all_types # Select all by default
    )

with c3:
    # Metric Dictionary for Context
    metric_desc = {
        "PIE": "Player Impact Estimate: A holistic measure of contribution.",
        "OFF_RATING": "Points produced per 100 possessions.",
        "DEF_RATING": "Points allowed per 100 possessions.",
        "NET_RATING": "Point differential per 100 possessions.",
        "TS_PCT": "True Shooting % (adjusts for 3s and Free Throws).",
        "EFG_PCT": "Effective Field Goal % (adjusts for 3s).",
        "USG_PCT": "Usage Percentage (plays used while on court).",
        "AST_TO": "Assist to Turnover Ratio.",
        "CLUTCH_PTS": "Points scored in last 5 min when game is close.",
        "PACE": "Possessions per 48 minutes.",
        "ISO_PTS": "Points scored in Isolation plays.",
        "POST_PTS": "Points scored in Post-Up plays."
    }
    desc = metric_desc.get(x_metric, "Performance Metric")
    st.caption(f"ℹ️ **{x_metric}**: {desc}")

# --- FILTER DATA ---
if not selected_types:
    st.warning("Please select at least one contract type.")
    st.stop()

filtered_df = df[df["Contract_Type"].isin(selected_types)]

# --- CHART ---
contract_colors = {
    "Free_Market": "#FFA500",   
    "Rookie_Scale": "#EF553B",  
    "Max_Contract": "#636EFA", 
    "Minimum": "#00CC96",       
    "Unknown": "#888888"
}

fig = px.scatter(
    filtered_df,
    x=x_metric,
    y="Salary",
    color="Contract_Type",
    log_y=True,
    hover_name="PLAYER_NAME",
    hover_data=["Age", "TEAM_NAME", "Contract_Type"],
    color_discrete_map=contract_colors,
    title=f"Salary vs. {x_metric} (Log Scale)",
    height=600
)

# ANNOTATIONS
# Only add if the player is actually in the filtered view
highlight_players = ["Victor Wembanyama", "Stephen Curry", "LeBron James", "Tyrese Haliburton"]

for player in highlight_players:
    player_row = filtered_df[filtered_df["PLAYER_NAME"] == player]
    if not player_row.empty:
        row = player_row.iloc[0]
        # Only annotate if data exists
        if pd.notna(row[x_metric]) and pd.notna(row["Salary"]):
            fig.add_annotation(
                x=row[x_metric],
                y=row["Salary"],
                text=f"<b>{player}</b>",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="#333333",
                ax=0,
                ay=-40,
                bgcolor="#ffffff",
                opacity=0.9
            )

st.plotly_chart(fig, use_container_width=True)

# --- 4. NARRATIVE: WEMBY VS CURRY ---

st.markdown("---")

col_thought_L, col_thought_R = st.columns([1, 1])

with col_thought_L:
    st.markdown("### A Thought Experiment")
    st.markdown("""
    **Victor Wembanyama** vs. **Stephen Curry**
    
    Both have massive on-court impact. Both sell out arenas. 
    But looking at the chart above, they inhabit different economic universes.
    
    * **Wembanyama (Red Cluster):** His salary is suppressed by the **Rookie Scale**. No matter how well he plays, he cannot earn his market value yet.
    * **Curry (Blue Cluster):** His salary is capped by the **Max Contract**. He essentially earns the "Maximum Allowable," which might still be less than his true value to the Warriors.
    """)

with col_thought_R:
    st.success("""
    **The Solution: Stratified DML**
    
    Because "Rookies" and "Max Players" break the relationship between Performance and Pay, **we cannot learn market bias from them.**
    
    1. We **filter** the data to keeping only the **Orange Cluster (Free Market)**.
    2. We **learn** the true price of performance and bias from this clean signal.
    3. We **counterfactually apply** those prices to the Rookies and Stars to see what they *should* make in an open market.
    """)