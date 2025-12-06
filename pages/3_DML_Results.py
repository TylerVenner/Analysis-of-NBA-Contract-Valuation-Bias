import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path
import numpy as np

# --- 1. SETUP & CONFIG ---
st.set_page_config(page_title="DML Results: Market Prices", layout="wide")

COEF_PATH = Path("data/app_data/final_ols_coefficients.csv")
PLAYER_PATH = Path("data/app_data/player_db.csv")

# --- 2. DATA LOADING (ROBUST) ---
@st.cache_data
def load_artifacts():
    """
    Loads the coefficients and player database from the static deployment folder.
    """
    if not COEF_PATH.exists() or not PLAYER_PATH.exists():
        st.error("⚠️ Deployment artifacts missing. Please run `prepare_deployment_data.py` locally.")
        st.stop()

    # A. Load Coefficients
    df_coef = pd.read_csv(COEF_PATH)
    
    # Standardize column names (handle variations from statsmodels export)
    rename_map = {
        "Gamma (Price)": "coef",
        "coef": "coef",
        "P-Value": "p_value",
        "p_value": "p_value",
        "P>|t|": "p_value",
        "Std. Error": "std_err",
        "std_err": "std_err",
        "bse": "std_err"
    }
    df_coef = df_coef.rename(columns=rename_map)
    
    # Handle Index Logic Robustly
    # We want the bias factor names to be the index
    if "Unnamed: 0" in df_coef.columns:
        df_coef = df_coef.set_index("Unnamed: 0")
    elif "Variable" in df_coef.columns:
        df_coef = df_coef.set_index("Variable")
    
    # Drop 'const' if it exists (usually not interesting for the plot)
    if "const" in df_coef.index:
        df_coef = df_coef.drop("const")

    # B. Load Player Data (for Context)
    df_players = pd.read_csv(PLAYER_PATH)
    
    # Numeric conversions
    cols_to_numeric = ["Age", "DRAFT_NUMBER", "OWNER_NET_WORTH_B", "Capacity", "Followers"]
    for col in cols_to_numeric:
        if col in df_players.columns:
            df_players[col] = pd.to_numeric(df_players[col], errors="coerce")

    return df_coef, df_players

df_coef, df_players = load_artifacts()

# --- 3. PAGE HEADER ---
st.title("The Market Price of Bias")
st.markdown("""
We used **Stratified Double Machine Learning (DML)** to isolate the impact of contextual factors on salary, 
independent of on-court performance. 

The results below represent the **"Shadow Price"** of these attributes in the open market.
""")

# --- 4. COEFFICIENT VISUALIZATION ---
st.header("1. Premiums vs. Penalties")

col1, col2 = st.columns([2, 1])

with col1:
    # Prepare data for Altair
    viz_df = df_coef.reset_index()
    
    # --- ROBUST FIX: Force rename the first column to 'Factor' ---
    # This ensures Altair finds the column regardless of whether it was named "index", "Variable", or "Unnamed: 0"
    viz_df.rename(columns={viz_df.columns[0]: "Factor"}, inplace=True)
    
    # Classify for coloring
    viz_df["Effect"] = viz_df["coef"].apply(lambda x: "Premium (Positive)" if x > 0 else "Penalty (Negative)")
    viz_df["Significance"] = viz_df["p_value"].apply(lambda x: "Significant" if x < 0.05 else "Not Sig.")
    
    # Bar Chart
    chart = alt.Chart(viz_df).mark_bar().encode(
        x=alt.X("coef:Q", title="Marginal Effect on Log-Salary"),
        y=alt.Y("Factor:N", sort="-x", title="Bias Factor"),
        color=alt.Color("Effect", scale=alt.Scale(domain=['Premium (Positive)', 'Penalty (Negative)'], range=['#2ca02c', '#d62728'])),
        opacity=alt.condition(alt.datum.p_value < 0.05, alt.value(1.0), alt.value(0.3)),
        tooltip=["Factor", "coef", "p_value", "std_err"]
    ).properties(height=400)
    
    st.altair_chart(chart, use_container_width=True)

with col2:
    st.info("""
    **How to read this chart:**
    
    * **Green Bars (Right):** Factors that boost salary (e.g., Age/Experience).
    * **Red Bars (Left):** Factors that reduce salary (e.g., International Status).
    * **Faded Bars:** Factors that are not statistically significant (random noise).
    """)
    
    # Show the raw table
    st.write("##### The Numbers")
    st.dataframe(
        df_coef[["coef", "p_value"]].style.apply(
            lambda x: ['background-color: #d4edda' if v < 0.05 else '' for v in x], 
            subset=['p_value']
        ).format("{:.4f}"),
        use_container_width=True,
        height=300
    )

st.divider()

# --- 5. PLAYER STORY ENGINE ---
st.header("2. Player Case Studies")
st.markdown("""
How do these market forces affect specific individuals? Select a player to see how the 
**general market biases** interact with their **specific situation**.
""")

# Setup Context (League Medians)
league_medians = df_players[["Age", "DRAFT_NUMBER", "OWNER_NET_WORTH_B", "Capacity", "Followers"]].median()

# Player Selector
player_list = sorted(df_players["PLAYER_NAME"].dropna().unique())
# Default to a generic player if available, else first in list
default_ix = player_list.index("Stephen Curry") if "Stephen Curry" in player_list else 0
selected_player = st.selectbox("Analyze Player:", player_list, index=default_ix)

# Get Player Data
player_row = df_players[df_players["PLAYER_NAME"] == selected_player].iloc[0]

# --- NARRATIVE GENERATION ---
st.subheader(f"Structural Context: {selected_player}")

# Show Snapshot
# Check which columns actually exist to avoid KeyErrors
snap_cols = [c for c in ["Team", "Age", "DRAFT_NUMBER", "Contract_Type"] if c in player_row.index]
st.dataframe(player_row[snap_cols].to_frame().T, hide_index=True)

col_narrative, col_chart = st.columns([1, 1])

with col_narrative:
    st.markdown("#### The Narrative")
    
    # 1. AGE STORY
    if "Age" in df_coef.index:
        coef = df_coef.loc["Age", "coef"]
        age_val = player_row.get("Age", None)
        med_age = league_medians.get("Age", None)
        
        if pd.notna(age_val) and pd.notna(med_age):
            if coef > 0 and age_val > med_age:
                st.success(f"**👴 The Veteran Premium:** At {age_val:.1f} years old, {selected_player} benefits from the league-wide preference for experience. Our model estimates a salary boost purely for tenure.")
            elif coef > 0 and age_val < med_age:
                st.warning(f"**👶 The Youth Discount:** At {age_val:.1f} years old, {selected_player} has not yet unlocked the 'Veteran Premium' that older peers enjoy.")

    # 2. DRAFT STORY
    if "DRAFT_NUMBER" in df_coef.index:
        coef = df_coef.loc["DRAFT_NUMBER", "coef"]
        draft_val = player_row.get("DRAFT_NUMBER", None)
        
        if pd.notna(draft_val):
            # Note: If coef is POSITIVE, later picks earn MORE (unusual). 
            # If NEGATIVE, later picks earn LESS (Pedigree Bias).
            if coef < 0 and draft_val < 15: # Lottery pick
                st.success(f"**💎 Draft Pedigree:** Being a top pick (#{int(draft_val)}) provides a structural salary floor. The market 'remembers' draft status years later.")
            elif coef < 0 and draft_val > 30:
                st.error(f"**📉 Late-Pick Drag:** Despite current performance, the market penalizes the late draft slot (#{int(draft_val)}), valuing the player lower than lottery picks with similar stats.")

    # 3. MARKET/OWNER STORY
    if "Capacity" in df_coef.index:
        coef = df_coef.loc["Capacity", "coef"]
        cap_val = player_row.get("Capacity", None)
        med_cap = league_medians.get("Capacity", None)
        
        if pd.notna(cap_val) and pd.notna(med_cap):
            if coef > 0 and cap_val > med_cap:
                 st.info(f"**🏟️ Big Market Boost:** Playing in a large arena ({int(cap_val):,} seats) correlates with a salary premium, likely due to higher team revenue.")

with col_chart:
    st.markdown("#### Structural Bias Score")
    st.caption("How much each factor helps (Right) or hurts (Left) this player's valuation relative to the league average.")
    
    # Calculate Scores: (PlayerValue - Median) * Coefficient
    scores = []
    for factor in ["Age", "DRAFT_NUMBER", "Capacity", "OWNER_NET_WORTH_B"]:
        if factor in df_coef.index and factor in player_row:
            val = player_row[factor]
            med = league_medians.get(factor, np.nan)
            
            if pd.notna(val) and pd.notna(med):
                diff = val - med
                score = diff * df_coef.loc[factor, "coef"]
                scores.append({"Factor": factor, "Score": score})
            
    score_df = pd.DataFrame(scores)
    
    if not score_df.empty:
        score_chart = alt.Chart(score_df).mark_bar().encode(
            x=alt.X("Score:Q", title="Relative Impact"),
            y=alt.Y("Factor:N", sort="-x"),
            color=alt.condition(
                alt.datum.Score > 0,
                alt.value("#2ca02c"),  # Green
                alt.value("#d62728")   # Red
            )
        )
        st.altair_chart(score_chart, use_container_width=True)
    else:
        st.warning("Not enough data to calculate bias scores.")

st.markdown("---")
st.caption("Note: These narratives interpret the coefficients from our DML model. They describe statistical tendencies, not the specific thoughts of a General Manager.")