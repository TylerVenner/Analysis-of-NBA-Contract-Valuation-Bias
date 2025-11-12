#!/usr/bin/env python
# coding: utf-8

# In[12]:


get_ipython().system('pip install statsmodels')
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

# Load data
df = pd.read_csv("master_dataset_cleaned.csv")
df = df.replace("Undrafted", 61)

# --- Define variables ---
Y = np.log(df["Salary"])  # log-transform for stability
X_cols = ["OFF_RATING", "DEF_RATING", "NET_RATING", "AST_PCT", "AST_TO", "AST_RATIO", "OREB_PCT", "REB_PCT", "DREB_PCT", "TM_TOV_PCT", "EFG_PCT",  
          "TS_PCT", "PACE", "PIE", "USG_PCT", "POSS", "FGM_PG", "FGA_PG" ]
Z_cols = ["DRAFT_NUMBER", "active_cap", "avg_team_age", "dead_cap", "OWNER_NET_WORTH_B", "Capacity", "STADIUM_YEAR_OPENED", "STADIUM_COST"]

X = df[X_cols]
Z = df[Z_cols]

# Standardize X and Z
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
Z_scaled = pd.DataFrame(scaler.fit_transform(Z), columns=Z.columns)

def generate_dml_residuals(X, Y, Z, k_folds=5):
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    y_resid = np.zeros(len(Y))
    Z_resid = np.zeros_like(Z)

    for train_idx, test_idx in kf.split(X):
        # Train models
        f_model = LinearRegression().fit(X.iloc[train_idx], Y.iloc[train_idx])
        h_models = {col: LinearRegression().fit(X.iloc[train_idx], Z[col].iloc[train_idx]) for col in Z.columns}

        # Predict and get residuals
        y_pred = f_model.predict(X.iloc[test_idx])
        y_resid[test_idx] = Y.iloc[test_idx] - y_pred

        for col in Z.columns:
            z_pred = h_models[col].predict(X.iloc[test_idx])
            Z_resid[test_idx, list(Z.columns).index(col)] = Z[col].iloc[test_idx] - z_pred

    return pd.Series(y_resid, name="eps_Y"), pd.DataFrame(Z_resid, columns=Z.columns)

resid_Y, resid_Z = generate_dml_residuals(X_scaled, Y, Z_scaled)
# Add constant term
X_bias = sm.add_constant(resid_Z)
final_model = sm.OLS(resid_Y, X_bias).fit()

print(final_model.summary())



# In[ ]:




