import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from typing import Callable, Tuple


def generate_dml_residuals(
    X: pd.DataFrame,
    Y: pd.Series,
    Z: pd.DataFrame,
    model_f_trainer: Callable,
    model_h_trainer: Callable,
    k_folds: int = 5,
    random_state: int = 42
) -> Tuple[pd.Series, pd.DataFrame]:


    print(f"Starting DML Residual Generation with {k_folds}-fold cross-fitting...")

    if isinstance(Z, pd.Series):
        Z = Z.to_frame()

    # Initialize storage
    n = len(Y)
    m = Z.shape[1]
    residuals_Y_oos = np.zeros(n)
    residuals_Z_oos = np.zeros((n, m))

    # Create K folds
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)

    # Enumerate folds
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
        print(f"\n--- Fold {fold_idx}/{k_folds} ---")

        # Split data
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        Y_train, Y_test = Y.iloc[train_idx], Y.iloc[test_idx]
        Z_train, Z_test = Z.iloc[train_idx], Z.iloc[test_idx]

        # Train Stage 1 model: f̂ (Y ~ X)
        f_model = model_f_trainer(X_train, Y_train)
        Y_pred = f_model.predict(X_test)
        residuals_Y_oos[test_idx] = Y_test - Y_pred

        # Train Stage 2 models: ĥ_j (Z_j ~ X)
        h_models = model_h_trainer(X_train, Z_train)

        for j, col in enumerate(Z.columns):
            Z_pred = h_models[col].predict(X_test)
            residuals_Z_oos[test_idx, j] = Z_test[col] - Z_pred

    # Wrap in pandas objects
    residuals_Y_oos = pd.Series(residuals_Y_oos, index=Y.index, name="epsilon_Y")
    residuals_Z_oos = pd.DataFrame(residuals_Z_oos, index=Z.index, columns=Z.columns)

    print("\n✅ DML residual generation complete.")
    return residuals_Y_oos, residuals_Z_oos

from sklearn.linear_model import LinearRegression

# --- Stage 1: Trainer for Y ~ X ---
def fake_train_f_model(X_train, Y_train):
    """Train f̂ model for Y ~ X"""
    model = LinearRegression()
    model.fit(X_train, Y_train)
    return model  # ✅ single model, not dict

# --- Stage 2: Trainer for Z_j ~ X ---
def fake_train_h_models(X_train, Z_train):
    """Train a separate ĥ_j model for each Z column"""
    models = {}
    # ensure DataFrame form
    if isinstance(Z_train, pd.Series):
        Z_train = Z_train.to_frame()

    for col in Z_train.columns:
        m = LinearRegression()
        m.fit(X_train, Z_train[col])
        models[col] = m
    return models  # ✅ dict of models, one per column





DATA_PATH = "/Users/albertoramirez/Downloads/master_dataset_cleaned.csv"

Y_COL = "Salary"

X_COLS = [ 'OFF_RATING',
 'DEF_RATING',
 'NET_RATING',
 'AST_PCT',
 'AST_TO',
 'AST_RATIO',
 'OREB_PCT',
 'DREB_PCT',
 'REB_PCT',
 'TM_TOV_PCT',
 'EFG_PCT',
 'TS_PCT',
 'USG_PCT',
 'PACE',
 'PACE_PER40',
 'PIE',
 'POSS',
 'FGM_PG',
 'FGA_PG'

]

Z_COLS_RAW = [
'Followers',
 'TEAM_ABBREVIATION',
 'record',
 'active_players',
 'avg_team_age',
 'total_cap_used',
 'remaining_cap_space',
 'active_cap',
 'active_top_3',
 'dead_cap',
 'TEAM_NAME',
 'OWNER_NET_WORTH_B',
 'Stadium_Name',
 'Capacity',
 'City',
 'STADIUM_YEAR_OPENED',
 'STADIUM_COST'
]

df = pd.read_csv(DATA_PATH)

needed_cols = [Y_COL] + X_COLS + Z_COLS_RAW
df = df.dropna(subset=needed_cols).copy()

# --- Step 2. Define Y, X, and Z as actual data, not lists ---
Y = df[Y_COL].copy()
X = df[X_COLS].copy()
Z_raw = df[Z_COLS_RAW].copy()

# Split into numeric and categorical
Z_num = Z_raw.select_dtypes(include=[np.number])
Z_cat = Z_raw.select_dtypes(exclude=[np.number])

# One-hot encode categoricals
if not Z_cat.empty:
    Z_cat_dummies = pd.get_dummies(Z_cat, drop_first=True)
    Z = pd.concat([Z_num, Z_cat_dummies], axis=1)
else:
    Z = Z_num

Z = Z.astype(float)


resY, resZ = generate_dml_residuals(
    X = X,
    Y = Y,
    Z = Z,
    model_f_trainer=fake_train_f_model,
    model_h_trainer=fake_train_h_models,
    k_folds=5

)




