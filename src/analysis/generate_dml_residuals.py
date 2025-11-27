import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from typing import Callable, Tuple, Any, Dict, List
from sklearn.metrics import mean_squared_error, r2_score, log_loss, roc_auc_score, accuracy_score
from typing import Callable, Tuple, Any, Dict, List

def _compute_metrics(
    y_true: pd.Series, 
    model: Any, 
    X_test: pd.DataFrame, 
    target_name: str) -> Dict[str, float]:
    """
    Computes performance metrics. Automatically detects if the task is 
    Regression (continuous) or Classification (binary/categorical).
    """
    metrics = {"target": target_name}
    
    has_proba = hasattr(model, "predict_proba")
    
    # If target is numeric but only has 2 unique values, it's likely binary classification
    # even if encoded as 0/1 ints.
    is_binary = (y_true.dropna().nunique() <= 2)
    is_numeric = pd.api.types.is_numeric_dtype(y_true)

    if is_numeric and not is_binary:
        # REGRESSION MODE
        y_pred = model.predict(X_test)
        metrics["type"] = "regression"
        metrics["r2"] = r2_score(y_true, y_pred)
        metrics["rmse"] = np.sqrt(mean_squared_error(y_true, y_pred))
    else:
        # CLASSIFICATION MODE
        metrics["type"] = "classification"
        
        # Accuracy (Hard Predictions)
        y_pred = model.predict(X_test)
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        
        # Cross-Entropy / Log Loss (Probabilistic Predictions)
        if has_proba:
            try:
                # Get probability of the positive class
                # Handle cases where model classes might be [0, 1] or just [0] or [1] in rare folds
                classes = model.classes_
                if len(classes) == 2:
                    y_prob = model.predict_proba(X_test)[:, 1]
                    metrics["log_loss"] = log_loss(y_true, y_prob)
                    
                    if len(np.unique(y_true)) > 1:
                         metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
            except Exception as e:
                print(f"    Notice: Could not compute proba metrics for {target_name}: {e}")

    return metrics

def generate_dml_residuals(
    X: pd.DataFrame,
    Y: pd.Series,
    Z: pd.DataFrame,
    model_f_trainer: Callable[[pd.DataFrame, pd.Series], Any],
    model_h_trainer: Callable[[pd.DataFrame, pd.DataFrame], dict[str, Any]],
    k_folds: int = 5,
    random_state: int = 1) -> Tuple[pd.Series, pd.DataFrame]:
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

    performance_records = [] # store residuals

    # Create K folds
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)

    # Enumerate folds
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
        print(f"\n--- Fold {fold_idx}/{k_folds} ---")

        # Split data using .iloc for integer-based indexing
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        Y_train, Y_test = Y.iloc[train_idx], Y.iloc[test_idx]
        Z_train, Z_test = Z.iloc[train_idx], Z.iloc[test_idx]

        # 1. Train f_model (Y ~ X)
        f_model = model_f_trainer(X_train, Y_train)
        
        # store metrics
        f_metrics = _compute_metrics(Y_test, f_model, X_test, target_name="Salary_Model")
        f_metrics["fold"] = fold_idx
        performance_records.append(f_metrics)

        # 2. Train h_models (Z ~ X)
        h_models = model_h_trainer(X_train, Z_train)
        
        # 3. Predict on test fold (out-of-sample)
        Y_pred = f_model.predict(X_test)
        
        # 4. Store Y residuals
        # Use .loc to assign based on original DataFrame index
        residuals_Y_oos.loc[Y_test.index] = Y_test - Y_pred

        # 5. Predict and store Z residuals
        for col in Z.columns:
            if col in h_models:

                model_h = h_models[col]
                
                # compute h metrics
                h_metrics = _compute_metrics(Z_test[col], model_h, X_test, target_name=col)
                h_metrics["fold"] = fold_idx
                performance_records.append(h_metrics)

                # predict prob of classification, rather than binary. this is needed for log loss.
                Z_pred = model_h.predict_proba(X_test)[:, 1]

                residuals_Z_oos.loc[Z_test.index, col] = Z_test[col] - Z_pred
            else:
                print(f"  Warning: No model found for Z column '{col}' in fold {fold_idx}.")

    print("\nDML residual generation complete.")
    
    perf_df = pd.DataFrame(performance_records)

    metrics_summary = perf_df.groupby(['target', 'type']).agg({
        'r2': ['mean', 'std'],
        'rmse': ['mean', 'std'],
        'accuracy': ['mean', 'std'],
        'roc_auc': ['mean', 'std'],
        'log_loss': ['mean', 'std']
    }).round(4)

    metrics_summary.columns = ['_'.join(col).strip() for col in metrics_summary.columns.values]

    # Check for any missing residuals (e.g., from failed folds)
    if residuals_Y_oos.isnull().any() or residuals_Z_oos.isnull().values.any():
        print("Warning: Some residuals are NaN. Check for missing data or model failures.")
        
    return residuals_Y_oos.dropna(), residuals_Z_oos.dropna(), metrics_summary