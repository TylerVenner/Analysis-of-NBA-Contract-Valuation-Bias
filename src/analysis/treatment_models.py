import re
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

# CONFIG
PROJECT_ROOT = Path().resolve().parent.parent

numeric_contextual_vars = [
    'age', 'draft_number', 'salary', 'followers',
    'owner_net_worth_b', 'total_cap_used', 'capacity', 'stadium_cost']

categorical_contextual_vars = ['country', 'draft_round']

performance_vars = [
    'off_rating','def_rating','net_rating','ast_pct','ast_to','ast_ratio','oreb_pct','dreb_pct',
    'reb_pct','tm_tov_pct','efg_pct','ts_pct','usg_pct','pace','pace_per40','pie','poss','fgm_pg','fga_pg']

# DATA CLEANING
def clean_colname(name: str) -> str:
    """Normalize column names: lowercase, underscores, no specials."""
    name = name.strip().lower()
    name = re.sub(r'[^a-z0-9]+', '_', name)
    name = re.sub(r'_+', '_', name)
    return name.strip('_')


def compute_age(birthdate):
    """Convert birthdate string to age in years."""
    try:
        return (datetime.now() - pd.to_datetime(birthdate)).days / 365
    except:
        return np.nan


def remove_outliers_iqr(df: pd.DataFrame, column: str, lower_q=0.01, upper_q=0.99) -> pd.DataFrame:
    """Remove outliers in a column based on IQR."""
    Q1 = df[column].quantile(lower_q)
    Q3 = df[column].quantile(upper_q)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# FEATURE REDUCTION
def reduce_vif(X: pd.DataFrame, vif_threshold: float) -> List[str]:
    """Iteratively remove variables with VIF > threshold."""
    variables = X.columns.tolist()
    while True:
        vif_data = pd.DataFrame({
            "feature": variables,
            "VIF": [variance_inflation_factor(X[variables].values, i)
                    for i in range(len(variables))]
        })
        max_vif = vif_data["VIF"].max()
        if max_vif <= vif_threshold:
            break
        drop_var = vif_data.sort_values("VIF", ascending=False)["feature"].iloc[0]
        print(f"Dropping {drop_var} (VIF={max_vif:.2f})")
        variables.remove(drop_var)
    print(f"Reduced from {X.shape[1]} -> {len(variables)} features.")
    return variables

# MODEL TRAINING
def train_h_models(X_df: pd.DataFrame, Z_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Trains the treatment models h: Z_j ~ X (Bias Factor ~ Performance)
    for each bias factor j in Z.
    """
    print(f"Training h_models for {list(Z_df.columns)}...")
    models_h = {}
    
    scaler = StandardScaler()
    scaler.fit(X_df)

    for var_name in Z_df.columns:
        y = Z_df[var_name].dropna()
        Xy = X_df.loc[y.index]
        Xy_scaled = scaler.transform(Xy)

        # Regression
        if var_name in numeric_contextual_vars:
            y_log = np.log(y.replace(0, np.nan).dropna())
            Xy_scaled = Xy_scaled[:len(y_log)]

            X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
                Xy_scaled, y_log, test_size=0.2, random_state=42
            )

            model = LinearRegression().fit(X_train_split, y_train_split)
            r2_train = r2_score(y_train_split, model.predict(X_train_split))
            r2_test = r2_score(y_test_split, model.predict(X_test_split))

            models_h[var_name] = {
                "type": "Regression",
                "model": model,
                "r2_train": r2_train,
                "r2_test": r2_test,
            }

        # Classification
        else:
            le = LabelEncoder()
            y_encoded = le.fit_transform(y.astype(str))

            X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
                Xy_scaled, y_encoded, test_size=0.2, random_state=42
            )

            model = LogisticRegression(max_iter=1000)
            model.fit(X_train_split, y_train_split)
            acc_train = model.score(X_train_split, y_train_split)
            acc_test = model.score(X_test_split, y_test_split)

            models_h[var_name] = {
                "type": "Classification",
                "model": model,
                "label_encoder": le,
                "accuracy_train": acc_train,
                "accuracy_test": acc_test,
            }

    print("h_models training complete.")
    return models_h

def get_h_models(
    csv_path: str,
    vif_threshold: float = 10.0) -> Dict[str, Any]:
    """
    Full pipeline from raw CSV → cleaned data → VIF reduction → trained h_models.
    """
    print("Loading and preprocessing data...")

    df = pd.read_csv(csv_path)
    df.columns = [clean_colname(c) for c in df.columns]

    # Compute age if birthdate exists
    if 'birthdate' in df.columns:
        df['age'] = df['birthdate'].apply(compute_age)

    # Convert numeric contextual vars
    for var in numeric_contextual_vars:
        if var in df.columns:
            df[var] = pd.to_numeric(df[var], errors='coerce')

    # Drop missing performance data
    df = df.dropna(subset = performance_vars)

    # VIF reduction
    X_full = df[performance_vars].copy()
    reduced_vars = reduce_vif(X_full, vif_threshold = vif_threshold)
    X_reduced = df[reduced_vars]

    # Combine contextual vars
    all_contextual_vars = numeric_contextual_vars + categorical_contextual_vars
    Z = df[all_contextual_vars].dropna(how='all')

    # Train h models
    h_models = train_h_models(X_reduced.loc[Z.index], Z)

    print("Full pipeline complete.")
    return h_models

def main():
    h_models = get_h_models(
        csv_path = PROJECT_ROOT / "data" / "processed" / "master_dataset_cleaned.csv"
    )
    print(h_models)

if __name__ == "__main__":
    main()