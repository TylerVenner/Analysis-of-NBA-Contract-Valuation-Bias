import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from typing import Dict, Any

def train_h_models(X_train: pd.DataFrame, Z_train: pd.DataFrame) -> Dict[str, Any]:
    """
    Trains the treatment models h: Z_j ~ X (Bias Factor ~ Performance)
    for each bias factor j in Z, as specified in the DML pipeline (Module 2).

    This function is designed to be called *inside* the cross-fitting
    loop, so it only performs training on the given fold.

    It automatically detects if Z_j is numeric (uses LinearRegression)
    or categorical (uses LogisticRegression).

    Args:
        X_train: DataFrame of performance features (training fold).
        Z_train: DataFrame of contextual/bias factors (training fold).

    Returns:
        A dictionary where keys are Z column names (e.g., 'Draft_Status')
        and values are the corresponding trained *model objects* (Pipelines).
    """
    print(f"  -> Training h_models (Z ~ X) for {list(Z_train.columns)}...")
    models_h = {}

    # Iterate over each bias/contextual factor
    for col_name in Z_train.columns:
        Z_j = Z_train[col_name]

        # Check if it's effectively binary (<= 2 unique values)
        # This catches 'is_USA' even though it is int64.
        is_binary = (Z_j.dropna().nunique() <= 2)

        # Check if it's numeric
        is_numeric = pd.api.types.is_numeric_dtype(Z_j)

        # 1. Determine model type based on Z_j's data type
        if is_numeric and not is_binary:
            # Continuous Factor (e.g. Age) -> REGRESSION
            model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=3,
                random_state=42
            )
        else:
            # Binary Factor (e.g. is_USA) -> CLASSIFICATION
            # GradientBoostingClassifier provides better probability estimates
            # than Logistic Regression for complex decision boundaries.
            model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=3,
                random_state=42
            )

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])

        # 3. Fit the pipeline
        # We must handle potential NaNs in this *specific* Z_j column
        if Z_j.isnull().any():
            # Fit only on the rows where this Z_j is not null
            not_null_idx = Z_j.notnull()
            pipeline.fit(X_train.loc[not_null_idx], Z_j.loc[not_null_idx])
        else:
            # This column is clean, fit on all data
            pipeline.fit(X_train, Z_j)
            
        # 4. Store the trained pipeline (model object)
        models_h[col_name] = pipeline

    return models_h