import streamlit as st




st.markdown("""
##  Statistical vs. Deterministic Players

Not every NBA salary is created the same way.

Some players are paid by **pure formula** – their contracts are locked in by league rules, not by what a team is willing to offer in an open negotiation. These are the **deterministic players**: rookies on the fixed Rookie Scale, stars on pre-defined Max deals, and veterans on minimum contracts. Their salaries are mostly decided before a single shot is taken.

Other players live in the **statistical world**. Their contracts are negotiated in the open market, where performance, timing, leverage, and perception all collide. These are the players our models can truly “listen to” when learning how the NBA values different skills and attributes.

---

##  What Our Salary Data Lets Us See

Our dataset combines:
- **Player salary** (how much they actually get paid),
- **On-court performance stats** (points, shooting, defense, impact metrics),
- **Contextual factors** like age, draft position, team market, and popularity signals.

By separating **deterministic contracts** from **free-market contracts**, we can:
- Learn how the market prices different skills *only* from players whose salaries are truly negotiable.
- Then **apply** those learned prices to everyone else, including rookies and max players, to see where the system itself creates distortions.

In other words, we ask:
> “If this rookie or max-contract player were paid like a normal free-market player, what would their salary look like?”

That counterfactual lets us see where the rules of the system, not just performance, are driving big gaps.

---

##  A Simple Question: Wemby vs. Curry

Take a thought experiment:

- **Victor Wembanyama**: a young player who puts up elite defensive stats, blocks a ton of shots, and warps the floor on defense.
- **Stephen Curry**: a veteran whose block numbers are tiny, but whose shooting, gravity, and global popularity are off the charts.

On paper, you might ask:
> “If Wemby has way more blocks, rim protection, and defensive impact, why does Curry get paid so much more?”

Our project doesn’t just shrug and say “because he’s a star.”  
We try to break that down into **measurable pieces**:

- Which **stats actually move salary** once you control for everything else?
- How much of a player’s paycheck is tied to **age**, **experience**, or **draft history**, instead of pure performance?
- How big is the **popularity premium** — things like social media following, brand power, and being the face of a franchise?
- And for rookies like Wemby, how much is their value **held back** by deterministic contract rules, even when their performance already looks elite?

---

## What You’ll See on This Page

Below, we highlight preliminary visualizations that start to answer these questions, such as:

- **Salary vs. Performance Scatterplots**  
  Showing how free-market players align (or don’t) with their production.

- **Deterministic vs. Statistical Groups**  
  Comparing rookies, max players, and open-market veterans to see where the biggest gaps emerge.

- **Case Study Points**  
  Marking players like Wembanyama and Curry on these plots to visualize just how differently the system treats them.

These visuals are the first step in moving from “Curry just gets paid more” to a clearer, data-driven story:  
**Which attributes the NBA really rewards, how the rules distort that value, and who ends up caught in the middle.**
""")






















st.set_page_config(
    page_title="Modeling Pipeline",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Double-Residual Machine Learning (DML) Pipeline")
st.write("""
This page documents the full modeling workflow used in our project, including the 
four core modules that implement the **Double-Residual Machine Learning (DML)** method.
This pipeline ensures unbiased estimation of the effect of contextual (bias) variables 
on NBA contract value after controlling for on-court performance.
""")

st.markdown("---")

# =========================================
# SECTION 1 — HIGH-LEVEL PIPELINE
# =========================================
st.header("📐 Overview of the Modeling Workflow")

st.write("""
Our analysis follows the Double-Residual Machine Learning (DML) framework.  
The full pipeline consists of four modules:

1. **Outcome Model (Module 1)** — Predict Salary from Performance  
2. **Treatment Models (Module 2)** — Predict Bias Factors from Performance  
3. **DML Cross-Fitting Engine (Module 3)** — Generate out-of-sample residuals  
4. **Final OLS Bias Model (Module 4)** — Regress residuals on residuals to estimate bias  

This structure ensures that contextual variables (nationality, draft status, age, etc.)  
are tested **only on the part of salary they cannot explain by performance alone**.
""")

st.markdown("---")

# =========================================
# MODULE 1
# =========================================
st.header("📘 Module 1 — Outcome Model  \n *f(X) = Salary ~ Performance*")

st.write("""
**Goal:** Predict a player's market salary based only on their performance metrics (X).  
This isolates the "expected" salary given on-court value.

**Input:**  
- `X_train` → Performance statistics  
- `y_train` → Salary (log-transformed or raw)

**Output:**  
- A trained model with `.predict()`  
- Used to compute **epsilon_Y = Y - ŷ** (salary residuals)

**Current Model:**  
- Random Forest Regressor (simple to tune, handles nonlinearities)  
- Wrapped inside our `train_f_model` function  
- Tuned using GridSearchCV (but can be swapped for linear/Ridge/Lasso later)

**Why this matters:**  
This residual (epsilon_Y) represents **salary mispricing** — the under/overpay relative to a player’s performance.
""")

st.code("""
def train_f_model(X_train, y_train):
    \"\"\"Trains outcome model f: Y ~ X (Salary ~ Performance).\"\"\"
    return model_f
""", language="python")

st.markdown("---")

# =========================================
# MODULE 2
# =========================================
st.header("📙 Module 2 — Treatment Models  \n *h_j(X) = Bias Factors ~ Performance*")

st.write("""
**Goal:** Predict each contextual/bias variable (Zᱼ) using only performance metrics (X).  
Examples of Z include:
- Draft Position  
- Age  
- Nationality  
- Team Market Size  
- Role/Position  
- Minutes per game context  
- Team salary cap / owner wealth

Each Zᱼ gets **its own model**, trained independently.

**Input:**  
- `X_train` → Performance features  
- `Z_train` → Bias/contextual variables  

**Output:**  
- Dictionary of models `{Z_j: model_h_j}`  
- Used to compute **epsilon_Z_j = Z_j - Ẑ_j**

**Why this matters:**  
This step removes the part of each bias factor that is *explained* by performance.  
What remains is the “pure” bias component.
""")

st.code("""
def train_h_models(X_train, Z_train):
    \"\"\"Trains one treatment model h_j for each bias factor Z_j.\"\"\"
    return models_h
""", language="python")

st.markdown("---")

# =========================================
# MODULE 3
# =========================================
st.header("⚙️ Module 3 — DML Cross-Fitting Engine")

st.write("""
This is the **core engine** of the pipeline.  
It implements the K-fold cross-fitting algorithm described in Section 6 of our methodology.

**What it does:**

1. Creates K folds (default K=5)  
2. For each fold:  
   - Trains f and hᱼ models on the training split  
   - Predicts f(X) and hᱼ(X) on the out-of-sample (OOS) split  
3. Stores residuals only from OOS predictions  
4. After all folds, concatenates residuals into full-series vectors  

**Output:**  
- εᵧ (OOS salary residuals)  
- ε𝑍 (OOS bias residuals, one column per Z variable)

These “clean” residuals feed into the final OLS step.
""")

st.code("""
def generate_dml_residuals(X, Y, Z, model_f_trainer, model_h_trainer, k_folds=5):
    \"\"\"Main residual-generation engine for DML cross-fitting.\"\"\"
    return residuals_Y_oos, residuals_Z_oos
""", language="python")

st.markdown("---")

# =========================================
# MODULE 4
# =========================================
st.header("📗 Module 4 — Final Debiased OLS Regression")

st.write("""
The final step estimates the effect of contextual variables (Z) on mispricing (Y residuals):

### **εᵧ = β₀ + β₁ εZ₁ + β₂ εZ₂ + ... + βₖ εZₖ + u**

Because both sides are **residualized**, the resulting coefficients are *debiased*  
and represent the true causal contribution of each factor to contract misvaluation.

**Output:**  
- Statsmodels regression object  
- Coefficients, p-values, standard errors  
- Interpretability for bias analysis  

**This is where the “bias effects” are measured.**
""")

st.code("""
def run_final_ols(residuals_Y, residuals_Z):
    \"\"\"Runs final OLS: epsilon_Y ~ epsilon_Z.\"\"\"
    return results
""", language="python")

st.markdown("---")

# =========================================
# SUMMARY BLOCK
# =========================================
st.header("✅ Summary")

st.write("""
Our DML pipeline ensures:

- ✔ Correct handling of high-dimensional performance metrics  
- ✔ Separation of performance effects from contextual variables  
- ✔ Unbiased estimation of contract bias  
- ✔ Modular design (each model easily replaceable)  
- ✔ Full reproducibility through cross-fitting  

This modeling system forms the backbone of our project’s analysis of NBA salary inefficiency 
and structural bias.
""")
