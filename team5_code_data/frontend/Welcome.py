import streamlit as st
from pathlib import Path
from PIL import Image

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="NBA Economic Bias Mapper",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- 2. HEADER SECTION ---
st.title("🏀 Mapping the Latent Structure of Economic Bias | NBA 2024-25")
st.subheader("STA 160 Capstone Project - Group 5")
st.markdown("**University of California, Davis**")

# Team Credits
col_team1, col_team2, col_team3, col_team4, col_team5 = st.columns(5)
col_team1.markdown("**Tyler Venner**")
col_team2.markdown("**Macy Chen**")
col_team3.markdown("**Leonel Garibay-Estrada**")
col_team4.markdown("**Jiarui Hou**")
col_team5.markdown("**Alberto Ramirez**")

st.divider()

# --- 3. THE HOOK (HERO SECTION) ---
# We try to load a hero image. If not found, we skip it gracefully.
HERO_IMAGE_PATH = Path("data/app_data/landing_image.png") # User needs to add this!

col_text, col_img = st.columns([1.5, 1])

with col_text:
    st.markdown("### The Research Question")
    st.info("""
    **What does economic bias *look like* when you map it?**
    
    Traditional sports analytics can tell us *that* bias exists—that veterans get paid more, or that market size matters. 
    But they can't tell us the **geometry** of that bias.
    
    Our project uses **Double Machine Learning (DML)** and **Latent Space Mapping** to uncover the hidden "neighborhoods" 
    of the NBA economy, revealing which players are valued for their stats, and which are defined by hype, 
    draft pedigree, or market structure.
    """)

with col_img:
    if HERO_IMAGE_PATH.exists():
        image = Image.open(HERO_IMAGE_PATH)
        st.image(image, caption="The Topology of NBA Salaries (3D Output)", use_container_width=True)

# --- 4. CONTEXT & SOLUTION (Two Column Layout) ---
st.markdown("---")

c1, c2 = st.columns(2)

with c1:
    st.markdown("### The Problem: A Wall of Coefficients")
    st.markdown("""
    Econometric analysis often ends with a static table of coefficients. 
    While statistically correct, this approach fails to reveal the **system**.
    
    * It doesn't show how biases cluster together.
    * It struggles with **"Deterministic Contracts"** (Rookies/Max deals) that defy free-market logic.
    * It leaves decision-makers guessing about *why* a specific player is an outlier.
    """)

with c2:
    st.markdown("### Our Solution: A Geometric Map")
    st.markdown("""
    We developed a **Stratified Learn-Apply Protocol**:
    
    1.  **Isolate the Signal:** We learn "fair market value" *only* from free-market contracts.
    2.  **Calculate the Distortion:** We apply those prices to the whole league to find "Structural Bias".
    3.  **Map the Space:** We treat bias as a geometric force, clustering players who are affected by the same hidden factors.
    """)

# --- 5. NAVIGATION GUIDE ---
st.markdown("---")
st.subheader("How to Explore This App")

st.markdown("""
Use the sidebar to navigate our analysis pipeline:

1.  **Statistical vs. Deterministic Players:** *See why we can't just run a simple regression on the whole NBA. We visualize the structural "walls" of Rookie Scale and Max contracts.*
    
2.  **The DML Pipeline:** *A high-level overview of our "Learn-Apply" methodology. How we filter the noise to find the signal.*
    
3.  **DML Results (The Prices):** *The "Shadow Prices" of bias. Exactly how much is a #1 Draft Pick worth in salary years later? What is the "Veteran Premium"?*
    
4.  **The Interactive Maps:** *The core of our project. Explore the **3D Topology** of bias and the **Geographic Map** of structural winners and losers.*
    
5.  **Assumptions & Conclusion:** *Limitations, implications for General Managers, and final thoughts.*
""")