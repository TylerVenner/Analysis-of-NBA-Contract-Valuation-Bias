import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
import re
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="Final Model Interpretation",
    page_icon="⚖️",
    layout="wide"
)

st.title("Final Coefficients")
st.subheader("Final DML Results & Interpretation")
# ------------------------------------------------------
# 1. CACHE THE FULL PIPELINE
# ------------------------------------------------------
@st.cache_data
def run_dml_pipeline():

    st.info("Running DML Pipeline... (This may take a moment)")

    # Load data
    df = pd.read_csv("master_dataset_cleaned.csv")
    df = df.replace("Undrafted", 61)
    df["DRAFT_NUMBER"] = pd.to_numeric(df["DRAFT_NUMBER"])

    # Define variables
    Y = np.log(df["Salary"])
    
    X_cols = ["OFF_RATING", "DEF_RATING", "NET_RATING", "AST_PCT", "AST_TO", "AST_RATIO",
              "OREB_PCT", "REB_PCT", "DREB_PCT", "TM_TOV_PCT", "EFG_PCT", "TS_PCT",
              "PACE", "PIE", "USG_PCT", "POSS", "FGM_PG", "FGA_PG"]

    Z_cols = ["DRAFT_NUMBER", "active_cap", "avg_team_age", "dead_cap",
              "OWNER_NET_WORTH_B", "Capacity", "STADIUM_YEAR_OPENED", "STADIUM_COST"]

    X = df[X_cols].dropna()
    Z = df[Z_cols].dropna()
    Y = Y.loc[X.index]

    # --- Helper functions
    def f_model(Xt, yt):
        pipe = Pipeline([("s", StandardScaler()), ("m", LinearRegression())])
        pipe.fit(Xt, yt)
        return pipe

    def h_models(Xt, Zt):
        models = {}
        for col in Zt.columns:
            pipe = Pipeline([("s", StandardScaler()), ("m", LinearRegression())])
            pipe.fit(Xt, Zt[col])
            models[col] = pipe
        return models

    # --- K-fold cross-fitting
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    resY = pd.Series(np.nan, index=Y.index)
    resZ = pd.DataFrame(np.nan, index=Y.index, columns=Z.columns)

    for tr, te in kf.split(X):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        Ytr, Yte = Y.iloc[tr], Y.iloc[te]
        Ztr, Zte = Z.iloc[tr], Z.iloc[te]

        f = f_model(Xtr, Ytr)
        h = h_models(Xtr, Ztr)

        resY.loc[Yte.index] = Yte - f.predict(Xte)

        for col in Z.columns:
            resZ.loc[Zte.index, col] = Zte[col] - h[col].predict(Xte)

    # Final OLS
    idx = resY.dropna().index
    resY = resY.loc[idx]
    resZ = resZ.loc[idx]

    Zc = sm.add_constant(resZ)
    results = sm.OLS(resY, Zc).fit(cov_type='HC3')

    return results


# ------------------------------------------------------
# 2. RUN THE MODEL (CACHED)
# ------------------------------------------------------
results = run_dml_pipeline()

# Display OLS Summary
st.header("Final DML OLS Regression Summary")
st.text(results.summary().as_text())

# ------------------------------------------------------
# 3. Extract coefficients into a DataFrame
# ------------------------------------------------------
params = results.params
bse = results.bse
pvals = results.pvalues
conf = results.conf_int()

df_coef = pd.DataFrame({
    "Coefficient": params,
    "Std Error": bse,
    "p-value": pvals,
    "CI Lower": conf[0],
    "CI Upper": conf[1],
})

st.header("Extracted Coefficient Table")
st.dataframe(df_coef)

# ------------------------------------------------------
# 4. Visualization
# ------------------------------------------------------
st.header("📈 Effect Sizes (Gamma Coefficients)")
fig, ax = plt.subplots()
ax.bar(df_coef.index, df_coef["Coefficient"], color="skyblue")
ax.axhline(0, color="black")
ax.set_xticklabels(df_coef.index, rotation=45, ha="right")
st.pyplot(fig)


st.subheader("Confidence Interval Plot")
fig, ax = plt.subplots(figsize=(8, 5))
ax.errorbar(
    df_coef.index,
    df_coef["Coefficient"],
    yerr=[
        df_coef["Coefficient"] - df_coef["CI Lower"],
        df_coef["CI Upper"] - df_coef["Coefficient"]
    ],
    fmt="o",
    capsize=5,
)
ax.axhline(0, color="gray", linestyle="--")
plt.xticks(rotation=45, ha="right")
ax.set_title("Coefficient Estimates with 95% CI")
st.pyplot(fig)


st.subheader("Volcano Plot (Effect Size vs Significance)")
df_coef["neg_log_p"] = -np.log10(df_coef["p-value"])

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(
    df_coef["Coefficient"],
    df_coef["neg_log_p"],
    c=df_coef["p-value"].apply(lambda p: "red" if p < 0.05 else "gray"),
    s=80
)
ax.axvline(0, color="black", linestyle="--")
ax.axhline(-np.log10(0.05), color="red", linestyle="--")

for name in df_coef.index:
    ax.text(df_coef.loc[name, "Coefficient"], df_coef.loc[name, "neg_log_p"], name)

ax.set_xlabel("Coefficient (Effect Size)")
ax.set_ylabel("-log10(p-value)")
ax.set_title("Volcano Plot")
st.pyplot(fig)


st.subheader("Actual vs Predicted Residuals (Model Fit)")
actual = results.model.endog
predicted = results.fittedvalues

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(actual, predicted, alpha=0.6)
ax.axline((0, 0), slope=1, color="red", linestyle="--")
ax.set_xlabel("Actual ε_Y")
ax.set_ylabel("Predicted ε_Y")
ax.set_title("Actual vs Predicted Residuals")
st.pyplot(fig)
# ------------------------------------------------------
# 5. Interpretation
# ------------------------------------------------------
st.header("Interpretation")

for name, coef in results.params.items():
    p = results.pvalues[name]
    if name == "const":
        continue

    sig = "statistically significant (p < 0.05)" if p < 0.05 else "not statistically significant"

    st.write(
        f"**{name}**: A 1-unit increase in `{name}` changes performance-adjusted log-salary "
        f"by **{coef:.5f}**. This effect is **{sig}**."
    )
