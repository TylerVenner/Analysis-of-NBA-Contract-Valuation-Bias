import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="something else",
    page_icon="📉",
    layout="wide"
)

st.title("a page")
st.subheader("Bias/Treatment Models (h) - Bias ~ Performance")

st.markdown("""
This page could focus on the **Bias Models (h)**, showing how well player performance can predict the contextual/bias factors (e.g., `Draft_Number ~ Performance`).

### Example: Interactive Selection
```python
# Get the list of Z_COLS (bias factors)
z_cols = [
    "DRAFT_NUMBER", "active_cap", "avg_team_age", "dead_cap",
    "OWNER_NET_WORTH_B", "Capacity", "STADIUM_YEAR_OPENED", "STADIUM_COST"
]
selected_factor = st.selectbox("Select a Bias Factor (Z) to Analyze:", z_cols)

st.write(f"Showing analysis for the model: **{selected_factor} ~ X (Performance)**")
```

""")

# Placeholder for content
st.header("My Analysis Section")
st.info("Coming soon: Analysis of the `Z ~ X` models and their residuals.")

# You can add more sections
st.header("Another Section")
st.write("More content will go here.")