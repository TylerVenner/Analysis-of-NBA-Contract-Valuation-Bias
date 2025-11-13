import streamlit as st
import pandas as pd

# Set the page configuration
st.set_page_config(
    page_title="NBA Contract Analysis",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Main Page Content ---

st.title("🏀 Analysis of NBA Contract Valuation")
st.subheader("STA 160 Capstone Project - Group 5")

st.markdown("""
Welcome to our project! This app presents our analysis of potential biases in NBA contract valuations.
We use a Double Machine Learning (DML) model to isolate the effect of non-performance factors (like draft number, owner net worth, etc.) on player salaries, after accounting for on-court performance.

### How to Use This App
Use the sidebar on the left to navigate between the different sections of our analysis:
- **Home:** You are here.
- **Data Overview:** (Example Page) A look at the raw data we collected.
- **Model Results:** (Example Page) The final output from our DML model.
- **[Teammate Pages]:** Explorations and specific analyses from each team member.

---
""")

st.header("Project Overview")
st.info("Our goal is to determine if a player's salary (Y) is influenced by contextual 'bias' factors (Z) even after controlling for their performance (X).")

st.markdown("""
### Methodology
1.  **Model 1 (Outcome):** We predict Salary based on Performance (`Y ~ X`).
2.  **Model 2 (Bias):** We predict each Bias Factor based on Performance (`Z ~ X`).
3.  **DML:** We get the residuals (the unexplained parts) from both models.
4.  **Final OLS:** We run a final regression on these residuals (`Residual_Y ~ Residual_Z`). The coefficients from this final model show us the "debiased" effect of the contextual factors on salary.
""")

st.image("https://placehold.co/1200x400/000000/FFFFFF?text=High-Level+Model+Architecture+Diagram",
         caption="A simplified view of our Double Machine Learning (DML) pipeline.",
         use_column_width=True)

st.sidebar.header("About")
st.sidebar.info("This app was created by Alberto, Gary, Leo, Macy, and Tyler for STA 160.")