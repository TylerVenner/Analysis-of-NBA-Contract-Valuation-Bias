import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
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

    model_f = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', GradientBoostingRegressor(
            n_estimators=200,       # More trees for complex relationships
            learning_rate=0.05,     # Slower learning prevents overfitting
            max_depth=4,            # Depth 4 captures 3-way interactions
            subsample=0.8,          # Stochastic boosting for robustness
            random_state=42
        ))
    ])

    model_f.fit(X_train, y_train)

    return model_f