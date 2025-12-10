import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path
import numpy as np

# --- 1. SETUP & CONFIG ---
st.set_page_config(page_title="DML Results: Market Prices", layout="wide")

COEF_PATH = Path("data/app_data/final_ols_coefficients.csv")
PLAYER_PATH = Path("data/app_data/player_db.csv")

# --- 2. DATA LOADING & TRANSFORMATION ---
@st.cache_data
def load_artifacts():
    if not COEF_PATH.exists() or not PLAYER_PATH.exists():
        st.error("⚠️ Artifacts missing. Run `prepare_deployment_data.py` locally.")
        st.stop()

    # A. Load Players
    df_players = pd.read_csv(PLAYER_PATH)
    cols_to_numeric = ["Age", "DRAFT_NUMBER", "OWNER_NET_WORTH_B", "Capacity", "Followers", 
                       "active_cap", "dead_cap", "STADIUM_COST", "STADIUM_YEAR_OPENED"]
    for col in cols_to_numeric:
        if col in df_players.columns:
            df_players[col] = pd.to_numeric(df_players[col], errors="coerce")

    # B. Load Coefficients
    df_coef = pd.read_csv(COEF_PATH)
    
    rename_map = {
        "Gamma (Price)": "coef", "coef": "coef", 
        "P-Value": "p_value", "p_value": "p_value",
        "std_err": "std_err", "bse": "std_err"
    }
    df_coef = df_coef.rename(columns=rename_map)
    
    if "Variable" in df_coef.columns:
        df_coef = df_coef.set_index("Variable")
    elif "Unnamed: 0" in df_coef.columns:
        df_coef = df_coef.set_index("Unnamed: 0")
    
    if "const" in df_coef.index:
        df_coef = df_coef.drop("const")
        
    # Standardized Impact Calculation
    df_coef["std_dev"] = 1.0 
    for idx in df_coef.index:
        if idx in df_players.columns:
            sd = df_players[idx].std()
            if pd.notna(sd) and sd != 0:
                df_coef.loc[idx, "std_dev"] = sd
                
    df_coef["std_coef"] = df_coef["coef"] * df_coef["std_dev"]
    df_coef["pct_impact"] = (np.exp(df_coef["std_coef"]) - 1) * 100

    return df_coef, df_players

df_coef, df_players = load_artifacts()

# --- 3. PAGE HEADER ---
st.title("The Market Price of Bias")
st.markdown("""
We used **Stratified Double Machine Learning** to isolate the "Shadow Price" of contextual factors. 

The chart below shows the **Standardized Impact**: How much a player's salary changes when a factor increases by **one standard deviation** (a "typical" increase).
""")

# --- 4. COEFFICIENT VISUALIZATION ---
st.header("1. Premiums vs. Penalties (Standardized)")
st.caption("Bars are shaded by statistical certainty. Solid = Certain, Faded = Uncertain.")

col1, col2 = st.columns([2, 1])

with col1:
    # Prepare data
    viz_df = df_coef.reset_index()
    viz_df.rename(columns={viz_df.columns[0]: "Factor"}, inplace=True)
    
    viz_df["Impact Label"] = viz_df["pct_impact"].map("{:+.2f}%".format)
    viz_df["One SD Amount"] = viz_df["std_dev"].map("{:,.2f}".format)
    viz_df["Color"] = viz_df["pct_impact"].apply(lambda x: "Premium" if x > 0 else "Penalty")

    # --- CONTINUOUS OPACITY SCALE ---
    # Domain [0.20, 0.0]:
    #   p=0.00 -> Opacity 1.0 (Fully Solid)
    #   p=0.10 -> Opacity 0.65 (Somewhat Faded)
    #   p>=0.20 -> Opacity 0.3 (Ghostly)
    
    chart = alt.Chart(viz_df).mark_bar().encode(
        x=alt.X("pct_impact:Q", title="Salary Impact per 1 Standard Deviation (%)"),
        y=alt.Y("Factor:N", sort="-x", title="Bias Factor"),
        color=alt.Color("Color", scale=alt.Scale(domain=['Premium', 'Penalty'], range=['#2ca02c', '#d62728'])),
        
        # CONTINUOUS FADING LOGIC
        opacity=alt.Opacity(
            'p_value', 
            scale=alt.Scale(domain=[0.20, 0.0], range=[0.3, 1.0], clamp=True),
            legend=None
        ),
        
        tooltip=["Factor", "Impact Label", "p_value", "One SD Amount"]
    ).properties(height=400)
    
    st.altair_chart(chart, use_container_width=True)

with col2:
    st.info("💡 **Why Standardize?**")
    st.markdown("""
    Some factors (like **Active Cap**) are measured in hundreds of millions, so the impact of $1 is tiny. 
    
    By looking at **Standard Deviations**, we can compare them fairly:
    
    * **Age:** Impact of being ~4 years older.
    * **Cap:** Impact of the team having ~$20M more space.
    * **Draft:** Impact of being drafted ~15 spots later.
    """)
    
    st.markdown("---")
    # Show the raw table
    st.write("##### Statistical Significance")
    st.dataframe(
        df_coef[["coef", "p_value"]].style.apply(
            lambda x: ['background-color: #d4edda' if v < 0.05 else '' for v in x], 
            subset=['p_value']
        ).format("{:.5f}"),
        use_container_width=True,
        height=300
    )

st.divider()

# --- 5. PLAYER STORY ENGINE ---
st.header("2. Player Case Studies")
st.markdown("Select a player to see their structural bias.")

# Context
league_medians = df_players[["Age", "DRAFT_NUMBER", "OWNER_NET_WORTH_B", "Capacity", "Followers"]].median()

# Selector
player_list = sorted(df_players["PLAYER_NAME"].dropna().unique())
default_ix = player_list.index("Stephen Curry") if "Stephen Curry" in player_list else 0
selected_player = st.selectbox("Analyze Player:", player_list, index=default_ix)
player_row = df_players[df_players["PLAYER_NAME"] == selected_player].iloc[0]

# --- CALCULATE NET BIAS ---
net_bias_score = 0
valid_factors = []
z_cols = ["Age", "DRAFT_NUMBER", "Capacity", "OWNER_NET_WORTH_B", "Followers"]

for factor in z_cols:
    if factor in df_coef.index and factor in player_row:
        val = player_row[factor]
        med = league_medians.get(factor, np.nan)
        coef = df_coef.loc[factor, "coef"]
        
        if pd.notna(val) and pd.notna(med):
            diff = val - med
            net_bias_score += diff * coef
            valid_factors.append(factor)

net_bias_pct = (np.exp(net_bias_score) - 1) * 100

# --- DISPLAY METRICS ---
col_metric, col_narrative = st.columns([1, 2])

with col_metric:
    st.subheader("Net Structural Bias")
    if net_bias_pct > 0:
        st.metric(label="Systemic Advantage", value=f"+{net_bias_pct:.1f}%", delta="Premium")
        st.caption("This player's context boosts their value above their raw performance.")
    else:
        st.metric(label="Systemic Disadvantage", value=f"{net_bias_pct:.1f}%", delta="Penalty", delta_color="inverse")
        st.caption("This player faces structural headwinds.")

with col_narrative:
    st.subheader("The Narrative")
    
    if "Age" in df_coef.index:
        age_impact = df_coef.loc["Age", "std_coef"]
        age_val = player_row.get("Age")
        if age_val > league_medians["Age"] and age_impact > 0:
            st.success(f"**👴 Veteran Status:** At {age_val:.1f} years old, they command a tenure premium.")
        elif age_val < league_medians["Age"] and age_impact > 0:
            st.warning(f"**👶 Youth Penalty:** At {age_val:.1f}, they haven't unlocked the veteran salary tier yet.")

    if "DRAFT_NUMBER" in df_coef.index:
        draft_val = player_row.get("DRAFT_NUMBER")
        if draft_val < 10:
             st.success(f"**Draft Pedigree:** Being a top pick (#{int(draft_val)}) acts as a salary safety net.")
        elif draft_val > 40:
             st.error(f"**Late-Pick Drag:** The market penalizes their draft slot (#{int(draft_val)}).")

    if "Followers" in df_coef.index:
        fame_impact = df_coef.loc["Followers", "coef"]
        fame_val = player_row.get("Followers", 0)
        med_fame = league_medians.get("Followers", 0)
        if fame_val > med_fame * 2 and fame_impact > 0:
            st.info(f"**📸 The Fame Boost:** With massive social reach, this player is a marketing asset.")

st.markdown("---")

# --- 6. DETAILED BREAKDOWN CHART ---
st.subheader("Detailed Factor Breakdown")
st.caption("Which specific factors are driving this player's Structural Bias?")

scores = []
for factor in valid_factors:
    val = player_row[factor]
    med = league_medians[factor]
    coef = df_coef.loc[factor, "coef"]
    score = (val - med) * coef
    scores.append({"Factor": factor, "Log Impact": score})

score_df = pd.DataFrame(scores)

if not score_df.empty:
    c = alt.Chart(score_df).mark_bar().encode(
        x=alt.X("Log Impact:Q", title="Relative Contribution to Salary (Log $)"),
        y=alt.Y("Factor:N", sort="-x"),
        color=alt.condition(alt.datum["Log Impact"] > 0, alt.value("#2ca02c"), alt.value("#d62728")),
        tooltip=["Factor", alt.Tooltip("Log Impact", format=".4f")]
    )
    st.altair_chart(c, use_container_width=True)
else:
    st.warning("Insufficient data to generate breakdown.")