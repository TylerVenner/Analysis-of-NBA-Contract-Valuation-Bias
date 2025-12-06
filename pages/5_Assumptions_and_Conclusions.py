import streamlit as st

st.set_page_config(page_title="Conclusion & Assumptions", page_icon="📘", layout="wide")

# --- HERO SECTION ---
st.title("📘 Conclusions & Future Directions")
st.markdown("""
**The Big Idea:** We moved from asking *"Does bias exist?"* to asking *"What is the **geometry** of bias?"*
""")

st.divider()

# --- 1. THE PARADIGM SHIFT (Before vs After) ---
st.header("1. The Shift: From Lists to Maps")

col_old, col_new = st.columns(2)

with col_old:
    st.error("The Old Way: 'Wall of Coefficients'")
    st.markdown("""
    * **Output:** A static table of p-values.
    * **Limit:** Tells you *if* a factor matters, but not how it connects to others.
    * **Failure:** Breaks when facing **Deterministic Contracts** (Rookies/Max deals).
    * **Result:** A fragmented view of the market.
    """)

with col_new:
    st.success("Our Solution: 'Bias Topology'")
    st.markdown("""
    * **Output:** An interactive 3D Map and Geographic Heatmap.
    * **Insight:** Reveals structural "Neighborhoods" (e.g., The Veteran Cluster).
    * **Fix:** Uses **Stratified DML** to learn prices purely from the free market.
    * **Result:** A unified theory of NBA value.
    """)

st.divider()

# --- 2. CRITICAL ASSUMPTIONS (Tabs for Organization) ---
st.header("2. Under the Hood: Assumptions & Limitations")
st.caption("Every model is a simplification of reality. Here is where ours simplifies.")

tab_assume, tab_limit, tab_future = st.tabs(["⚠️ The Core Assumption", "🚧 Limitations", "🔮 Future Work"])

with tab_assume:
    col_text, col_viz = st.columns([2, 1])
    with col_text:
        st.markdown("### The 'Transportability' Leap")
        st.info("""
        **We assume that the 'Price of Bias' learned from the Free Market also applies to Rookies and Max Players.**
        """)
        st.markdown("""
        * **What this means:** If the market pays a 5% premium for "Big Market Exposure" to a veteran free agent, we assume it *would* pay that same 5% premium to a Rookie if the Rookie Scale didn't exist.
        * **The Risk:** It is possible that teams value attributes differently for 19-year-olds vs. 30-year-olds.
        * **The Defense:** This is the only way to disentangle "Rule-based Salary" from "Market-based Salary." We could get around this by simply using free market players. But this is not as interesting.
        """)
    with col_viz:
        st.caption("Visualizing the Leap")
        st.markdown("""
        `Free Market` -> **Learn** $\gamma$  
        `Rookies` -> **Apply** $\gamma$
        """)

with tab_limit:
    st.markdown("""
    1.  **The "Free Market" is Small:** Only ~50% of the league negotiates freely in any given year. This reduces our sample size.
    2.  **Performance Snapshots:** We use box-score stats (PER, Win Shares). We cannot measure "Locker Room Leadership" or "Jersey Sales Potential" directly, unless captured by our proxy variables (Age, Followers).
    3.  **Dimensionality Loss:** squashing 10+ bias factors into a 3D map inevitably loses some information. The map is a *guide*, not a GPS.
    """)

with tab_future:
    st.markdown("""
    * **Longitudinal Study:** Track how the "Geometry of Bias" changes over 10 years. Does the "Age Premium" shrink as the game gets faster?
    * **Other Domains:** Apply this `DML + Unfolding` framework to **Real Estate** (Market Price vs. Zoning Laws) or **Tech Hiring** (Skills vs. Pedigree).
    """)

st.divider()

# --- 3. PRACTICAL IMPLICATIONS (Action Cards) ---
st.header("3. So What? (The GM Decision Deck)")
st.markdown("""
How can a General Manager use this map to build a winning roster? 
**Select a strategic objective below** to see how our model reveals specific market inefficiencies.
""")

# The "Drop Down" Interaction
gm_strategy = st.selectbox(
    "🎯 Select a Strategic Goal:",
    [
        "Find Efficient 'Moneyball' Assets",
        "Avoid 'Bad Contract' Traps",
        "Exploit the 'Pedigree' Arbitrage"
    ]
)

st.divider()

# Dynamic Content based on selection
if gm_strategy == "Find Efficient 'Moneyball' Assets":
    col_icon, col_text = st.columns([1, 4])
    with col_icon:
        st.markdown("# 🎯")
    with col_text:
        st.subheader("Target: The 'Null Space'")
        st.markdown("""
        **The Insight:** Players located in the **geometric center** of our Bias Map (the "Null Space") are valued almost entirely on production, with near-zero structural distortion.
        
        **The GM Playbook:**
        1.  Filter for players with a **Structural Bias Score near 0**.
        2.  **Sign them.** These are "Pure Performance" assets. Every dollar you spend goes toward on-court production, not "Hype" or "Market Size."
        """)
        st.info("💡 **Why it works:** You avoid paying the 'Status Tax' that usually inflates the cost of stars.")

elif gm_strategy == "Avoid 'Bad Contract' Traps":
    col_icon, col_text = st.columns([1, 4])
    with col_icon:
        st.markdown("# 🚩")
    with col_text:
        st.subheader("Flag: High Structural Bias Scores")
        st.markdown("""
        **The Insight:** A high positive Bias Score means a player's salary is heavily propped up by factors *other* than winning basketball games (e.g., Age, Fame, Market Size).
        
        **The GM Playbook:**
        1.  Before signing a veteran, check their **Age Premium** on our map.
        2.  **Audit the cost:** Ask, *"Am I paying for future production, or am I paying a 'Legacy Tax' for what they did 5 years ago?"*
        """)
        st.warning("⚠️ **Risk Warning:** High structural scores often indicate 'Transient Factors' (like Hype) rather than sustainable value.")

elif gm_strategy == "Exploit the 'Pedigree' Arbitrage":
    col_icon, col_text = st.columns([1, 4])
    with col_icon:
        st.markdown("# 📉")
    with col_text:
        st.subheader("Target: The 'Draft Drag'")
        st.markdown("""
        **The Insight:** Our model quantifies a "Draft Drag" effect where late-round picks must perform significantly *better* than lottery picks just to earn the same market respect.
        
        **The GM Playbook:**
        1.  Identify 2nd-Round picks who have "broken out" statistically.
        2.  **Extension Target:** The market likely still prices them lower than a Top-5 pick with identical numbers. Sign them before the market corrects its bias.
        """)
        st.success("🚀 **The Arbitrage:** You get Lottery-level production at a 2nd-Round price point.")

# --- 4. FINAL FOOTER ---
st.markdown("### 🔗 Project Resources")
col_github, col_paper = st.columns([1, 5])
with col_github:
    st.markdown("[**📂 GitHub Repository**](https://github.com/TylerVenner/Analysis-of-NBA-Contract-Valuation-Bias)") # Add your actual link if you want
with col_paper:
    st.caption("Developed by Group 5 for STA 160. University of California, Davis.")