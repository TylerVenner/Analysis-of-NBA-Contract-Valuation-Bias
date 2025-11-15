import streamlit as st

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
- Used to compute **epsilon_Z_j = Z_j - Ẑ_j**

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
