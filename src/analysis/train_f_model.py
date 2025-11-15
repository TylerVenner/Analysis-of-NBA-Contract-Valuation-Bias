import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from typing import Any

def train_f_model(X_train: pd.DataFrame, y_train: pd.Series) -> Any:
    """
    Trains the outcome model f: Y ~ X (Salary ~ Performance)
    as specified in the DML pipeline (Module 1).

    This function is designed to be called *inside* the cross-fitting
    loop, so it only performs training on the given fold.

    Args:
        X_train: DataFrame of performance features (training fold).
        y_train: Series of the outcome (log_salary) (training fold).

    Returns:
        A trained model object with a .predict() method.
    """
    print(f"  -> Training f_model (Y ~ X) on {X_train.shape[0]} samples...")

    # Per the spec, we use a simple model.
    # A Pipeline is best practice, as it bundles scaling and regression.
    # This prevents data leakage and simplifies prediction.
    model_f = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])

    # Fit the model on the training data from the fold
    model_f.fit(X_train, y_train)

    return model_f