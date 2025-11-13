import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from typing import Callable, Tuple, Any

def generate_dml_residuals(
    X: pd.DataFrame,
    Y: pd.Series,
    Z: pd.DataFrame,
    model_f_trainer: Callable[[pd.DataFrame, pd.Series], Any],
    model_h_trainer: Callable[[pd.DataFrame, pd.DataFrame], dict[str, Any]],
    k_folds: int = 5,
    random_state: int = 42) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Implements the DML cross-fitting algorithm (Section 6 of the doc).

    This function orchestrates the training and prediction across K-folds
    to generate out-of-sample residuals for both the outcome (Y) and
    the treatment/bias factors (Z).

    Args:
        X: Full DataFrame of performance features.
        Y: Full Series of the outcome (log_salary).
        Z: Full DataFrame of contextual/bias factors.
        model_f_trainer: The function to train model f (i.e., train_f_model).
        model_h_trainer: The function to train models h (i.e., train_h_models).
        k_folds: Number of folds for cross-fitting.
        random_state: Seed for KFold shuffling.

    Returns:
        A tuple containing:
        - residuals_Y_oos (pd.Series): The out-of-sample residuals for Y (epsilon_Y).
        - residuals_Z_oos (pd.DataFrame): The out-of-sample residuals for Z (epsilon_Z).
    """

    print(f"Starting DML Residual Generation with {k_folds}-fold cross-fitting...")

    # Initialize storage arrays
    # We use .loc/.iloc indices for robust assignment
    residuals_Y_oos = pd.Series(np.nan, index=Y.index, name="epsilon_Y")
    residuals_Z_oos = pd.DataFrame(columns=Z.columns, index=Z.index, dtype=float)

    # Create K folds
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)

    # Enumerate folds
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
        print(f"\n--- Fold {fold_idx}/{k_folds} ---")

        # Split data using .iloc for integer-based indexing
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        Y_train, Y_test = Y.iloc[train_idx], Y.iloc[test_idx]
        Z_train, Z_test = Z.iloc[train_idx], Z.iloc[test_idx]

        # --- 1. Train f_model (Y ~ X) ---
        f_model = model_f_trainer(X_train, Y_train)
        
        # --- 2. Train h_models (Z ~ X) ---
        h_models = model_h_trainer(X_train, Z_train)
        
        # --- 3. Predict on test fold (out-of-sample) ---
        Y_pred = f_model.predict(X_test)
        
        # --- 4. Store Y residuals ---
        # Use .loc to assign based on original DataFrame index
        residuals_Y_oos.loc[Y_test.index] = Y_test - Y_pred

        # --- 5. Predict and store Z residuals ---
        for col in Z.columns:
            if col in h_models:
                Z_pred = h_models[col].predict(X_test)
                residuals_Z_oos.loc[Z_test.index, col] = Z_test[col] - Z_pred
            else:
                print(f"  Warning: No model found for Z column '{col}' in fold {fold_idx}.")

    print("\nDML residual generation complete.")
    
    # Check for any missing residuals (e.g., from failed folds)
    if residuals_Y_oos.isnull().any() or residuals_Z_oos.isnull().values.any():
        print("Warning: Some residuals are NaN. Check for missing data or model failures.")
        
    return residuals_Y_oos.dropna(), residuals_Z_oos.dropna()