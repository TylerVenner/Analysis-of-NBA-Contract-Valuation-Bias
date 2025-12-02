import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Conclusion & Assumptions", page_icon="📘", layout="wide")

st.title("Conclusion & Model Assumptions")

# 1. Why this model is important

st.header("1. Why This Model Matters")

st.markdown(
    """
### A. Traditional statistical models miss the bigger picture
Most salary studies stop at a **wall of coefficients**.  
These tables can tell you *which* factors matter (age, draft number, market size),  
but they cannot tell you:

- how these forces are connected,  
- which factors act together as a larger structural force,  
- or which types of players are shaped by the same kinds of bias.

---

### B. Our model solves a fundamental structural problem
NBA salaries include **deterministic contracts**:

- Rookie Scale deals  
- Maximum contracts  
- Minimum deals  

These salaries are set by rules, not the free market.  
Including them in a regular regression **breaks the model**.

Our framework fixes this by using a **Learn–Apply protocol**:

1. **Learn Phase** — study only free-market contracts to learn the true market prices of bias.  
2. **Apply Phase** — take those learned prices and apply them to the *entire* league  
   (including rookies and max players) to understand how the same forces *would* affect them  
   under an open market.

This avoids contamination and gives us a cleaner picture of how the market values context.

---

### C. Turning coefficients into a structural map
Coefficients alone cannot reveal deeper patterns.  
We convert them into an **Attribution Matrix**, which measures:

 How strongly each bias factor influences each player,
 all in the same unit — "unexplained salary influence."

This lets us build a **Bias Map**, a geometric visualization that shows:

- which bias factors cluster together,  
- which players share similar structural forces,  
- and where “pure performance” players sit (those whose salary is mostly explained by stats).  

---
"""
)



# 2. Key assumptions (written for normal audience)

st.header("2. Core Assumptions")

st.markdown(
    """
Our model is powerful, but relies on two important assumptions.

### A. Transportability (the biggest assumption)
We assume the prices of bias we learn from free-market contracts  
also apply to players with fixed salaries (rookies, max players).

This is necessary to avoid model errors,  
but it assumes that teams value attributes like **draft pedigree** or **market size**  
in a similar way across different contract types.

In reality, fixed contracts behave differently —  
so this assumption, while reasonable, should be used carefully.

---

### B. Overlap in performance profiles
When we apply our models to rookies or low-minute players,  
their performance stats may look different from free-market veterans.

If their stats fall outside the range of the training data,  
the counterfactual predictions may be less accurate.

This does **not break** the model,  
but it means some players may have noisier bias estimates.

---
"""
)


# 3. Limitations — written clearly and aligned with report

st.header("3. Limitations")

st.markdown(
    """
### A. Free-market data is limited  
Only part of the league negotiates in the open market.  
This reduces sample size and may increase noise.

### B. Bias prices may shift year to year  
The bias structure could change with the CBA, cap rules, or stylistic trends.

### C. Performance snapshots can miss context  
Our model uses available statistics,  
but cannot capture leadership, injuries, off-court factors, or locker-room value.

### D. Dimension reduction introduces uncertainty  
The Bias Map intentionally represents each player and factor  
as a **probability distribution**, not a single point.  
This protects against overconfidence, but also means some placements are fuzzy.

None of these limitations invalidate the insights —  
they simply remind us to interpret the map as a **structured guide**,  
not an absolute ranking.

---
"""
)


# 4. Implications

st.header("4. Impact and Implications")

st.markdown(
    """
### Practical implications for real decision-makers

- It becomes easy to spot **“pure performance” players** — those who land near the center of the Bias Map, where salaries are mostly explained by basketball performance rather than structural forces. 

- It also highlights players who are shaped by **strong structural distortions**. 

For **general managers**, this helps identify **market inefficiencies not just by size, but by type**:

- Is a player’s valuation driven by sustainable performance?  
- Or by transient factors like hype, big-market exposure, or the wealth of the ownership group?


This makes the model valuable not only for analysts,  
but also for **general managers, agents, and league decision-makers**  
seeking a deeper understanding of fairness and inefficiency in the NBA contract ecosystem.
"""
)

