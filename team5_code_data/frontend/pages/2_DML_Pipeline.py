import streamlit as st
import graphviz

st.set_page_config(page_title="The Methodology", layout="wide")

st.title("The Engine: Stratified Double Machine Learning")
st.subheader("How we isolated the signal from the noise.")

# --- 1. THE PIPELINE VISUALIZATION (Hero Image) ---
st.markdown("### The 'Learn-Apply' Protocol")
st.markdown("""
We built a custom pipeline that treats "Free Market" players differently from "Fixed Contract" players. 
This prevents the rules of the CBA (Collective Bargaining Agreement) from confusing our economic model.
""")

# --- SCALING FIX: Use Columns to constrain width ---
# [Spacer, Content, Spacer] -> Middle column is 4/6ths (66%) of the screen
col_left, col_center, col_right = st.columns([1, 1, 1])

with col_center:
    # Create a professional flow diagram using Graphviz
    graph = graphviz.Digraph()
    # rankdir='TB' (Top-to-Bottom) for vertical stacking
    # splines='ortho' for square lines
    graph.attr(rankdir='TB', splines='ortho', nodesep='0.5', ranksep='0.5', fontname='Arial')

    # Nodes
    # We use 'box' shape with rounded corners for a cleaner look vertically
    graph.attr('node', shape='box', style='filled', fontname='Arial', fontsize='12', margin='0.2')

    # 1. Raw Data
    graph.node('Data', 'Raw NBA Data\n(Performance + Contracts)', fillcolor='#E0E0E0', shape='cylinder')

    # 2. Filter (Orange)
    graph.node('Filter', 'Stratification:\nIsolate Free Market Contracts', fillcolor='#FFA500')

    # 3. DML (Blue)
    graph.node('DML', 'Double Machine Learning\n(Remove Performance Signal)', fillcolor='#87CEFA')

    # 4. Coefficients (Green)
    graph.node('Coeffs', 'Learn Market Prices\n(Gamma Coefficients)', fillcolor='#90EE90', shape='parallelogram')

    # 5. Apply (Yellow)
    graph.node('Apply', 'Counterfactual Application\n(Apply Prices to ALL Players)', fillcolor='#FFD700')

    # 6. Map (Pink)
    graph.node('Map', '3D Bias Map\n(Topology)', fillcolor='#FF69B4', shape='component')

    # Edges
    graph.attr('edge', fontname='Arial', fontsize='10', color='#333333', penwidth='1.5')
    graph.edge('Data', 'Filter')
    graph.edge('Filter', 'DML', label=' Train on Negotiated Deals ')
    graph.edge('DML', 'Coeffs', label=' Extract Bias Impact ')
    graph.edge('Coeffs', 'Apply')
    graph.edge('Data', 'Apply', label=' Full Roster ')
    graph.edge('Apply', 'Map', label=' Clustering ')

    # Render
    st.graphviz_chart(graph, use_container_width=True)

st.divider()

# --- 2. DEEP DIVE TABS ---
tab1, tab2, tab3 = st.tabs(["1. The Problem", "2. The Solution (DML)", "3. The Transformation"])

with tab1:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("### The 'Deterministic' Trap")
        st.info("""
        If you run a standard regression on NBA salaries, it fails.
        
        **Why?**
        Because a huge chunk of the league is paid by **Rule**, not by **Market**.
        """)
        st.markdown("""
        * **Rookies:** Paid based on draft slot (Fixed).
        * **Superstars:** Paid the 'Max' (Capped).
        * **Veterans:** Paid the 'Minimum' (Floor).
        """)
    with col2:
        st.markdown("### The Consequence")
        st.markdown("""
        If we trained on everyone, the model would think:
        > *"Being a #1 Pick is bad for your salary relative to your performance."*
        
        ...simply because **Victor Wembanyama** is underpaid by rule.
        
        **Our Fix:** We exclude these players from the *Learning* phase so they don't contaminate the definition of "Market Value."
        """)

with tab2:
    st.markdown("### Double Machine Learning (DML)")
    st.markdown("We use DML to mathematically 'subtract' the performance stats from the salary.")
    
    col_demo1, col_demo2, col_demo3 = st.columns(3)
    
    with col_demo1:
        st.markdown("**Step A: Predict Salary**")
        st.caption("Model $f(X)$")
        st.markdown("We train a model to predict salary based *only* on stats (Points, PER, etc).")
        st.metric("Actual Salary", "$20M")
        st.metric("Predicted based on Stats", "$15M")
        st.metric("Residual (Y - Y_hat)", "+$5M", delta_color="normal")
        st.caption("This $5M is 'Unexplained'.")

    with col_demo2:
        st.markdown("**Step B: Predict Bias**")
        st.caption("Model $h(X)$")
        st.markdown("We predict bias factors (e.g., Market Size) based on stats. (Are good players usually in big markets?)")
        st.metric("Actual Market Size", "Huge (NY)")
        st.metric("Predicted from Stats", "Average")
        st.metric("Residual (Z - Z_hat)", "+Big Market", delta_color="normal")
    
    with col_demo3:
        st.markdown("**Step C: Regress Residuals**")
        st.caption("The Final Step")
        st.markdown("""
        We compare the **Salary Residual** ($5M) to the **Bias Residual** (Big Market).
        
        If they correlate, we have found a **Market Bias**, independent of player skill.
        """)

with tab3:
    st.markdown("### Creating the Map")
    st.markdown("""
    Once we have the **Coefficients** (The Price of Bias), we don't stop there.
    
    We take those prices and apply them to **every player in the league**, even the Rookies we ignored earlier.
    
    1.  **LSM (Latent Space Mapping):** We treat these bias impacts as coordinates.
    2.  **The Result:** A 3D map where players cluster not by *how good* they are, but by *how their contract is structured*.
    """)

st.divider()

# --- 3. FOR THE STATISTICIANS (Collapsible) ---
with st.expander("🤓 Technical Note: The Frisch-Waugh-Lovell Theorem"):
    st.markdown("""
    Our approach relies on the **Frisch-Waugh-Lovell (FWL)** theorem. 
    
    In a partially linear model:
    $$Y = \\alpha + \\gamma Z + f(X) + \\epsilon$$
    
    We cannot estimate $\\gamma$ (the bias price) directly because $Z$ (Bias) and $X$ (Performance) are correlated.
    
    DML solves this by orthogonalizing both sides:
    1. $\\tilde{Y} = Y - E[Y|X]$ (Remove performance signal from salary)
    2. $\\tilde{Z} = Z - E[Z|X]$ (Remove performance signal from bias factors)
    3. $\\tilde{Y} = \\gamma \\tilde{Z} + \\nu$ (Regress the residuals)
    
    We implement this using **Gradient Boosting Regressors** for the nuisance functions $f(X)$ and $h(X)$, trained via 5-fold cross-fitting to prevent overfitting.
    """)