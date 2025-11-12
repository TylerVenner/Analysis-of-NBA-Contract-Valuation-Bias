import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
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

        # 1. Determine model type based on Z_j's data type
        if pd.api.types.is_numeric_dtype(Z_j):
            # It's a continuous variable, use Linear Regression
            model = LinearRegression()
        else:
            # It's a categorical variable, use Logistic Regression
            # We add 'class_weight' to help with imbalanced classes
            model = LogisticRegression(
                max_iter=1000, 
                random_state=1, 
                class_weight='balanced'
            )

        # 2. Build a pipeline to scale X and then fit the model
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