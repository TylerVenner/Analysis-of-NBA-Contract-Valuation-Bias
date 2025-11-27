import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from typing import Any

def train_f_model(X_train: pd.DataFrame, y_train: pd.Series) -> Any:
    """
    Trains the outcome model f: Y ~ X (Salary ~ Performance).
    
    Uses RandomizedSearchCV to automatically find the best hyperparameters
    that balance model complexity (Variance) vs. generalizability (Bias).
    """
    
    # 1. Define the Pipeline
    # We still need scaling, though trees handle unscaled data okay, 
    # it's safer for the loss function convergence.
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', GradientBoostingRegressor(random_state=42))
    ])

    # 2. Define the Hyperparameter Space
    # This grid covers the "Regularization" parameters we discussed.
    param_grid = {
        # Depth: Shallower trees (2-3) prevent overfitting on small N
        'regressor__max_depth': [2, 3, 4],
        
        # Min Samples: Higher values force the model to find broader trends
        'regressor__min_samples_leaf': [3, 5, 10, 15],
        
        # Subsample: Training on random subsets adds robustness (Stochastic Boosting)
        'regressor__subsample': [0.5, 0.7, 0.9],
        
        # Learning Rate: Slower is generally better but requires more trees
        'regressor__learning_rate': [0.01, 0.05, 0.1],
        
        # Trees: Enough to converge, but not infinite
        'regressor__n_estimators': [100, 200, 300]
    }

    # 3. Setup Cross-Validation Search
    # n_iter=20 means it will try 20 random combinations from the grid above.
    # cv=5 means it validates each combo 5 times (Robust!).
    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_grid,
        n_iter=20,          # Budget: Try 20 candidates
        scoring='r2',       # Optimize for R-Squared
        cv=5,               # 5-Fold Internal CV
        n_jobs=-1,          # Use all CPU cores
        random_state=42,
        verbose=0
    )

    # 4. Find the Best Model
    # This might take 10-20 seconds to run depending on your CPU
    search.fit(X_train, y_train)

    # Optional: Print what it found so you know "what works"
    # print(f"  -> Best Params: {search.best_params_}")
    # print(f"  -> Best Internal CV Score: {search.best_score_:.3f}")

    # Return the best estimator (already refitted on full X_train)
    return search.best_estimator_