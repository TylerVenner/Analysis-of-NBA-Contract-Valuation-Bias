import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV # comment out for speed
from typing import Any

def train_f_model(X_train: pd.DataFrame, y_train: pd.Series) -> Any:
    """
    Trains the outcome model f: Y ~ X (Salary ~ Performance).
    
    ACADEMIC NOTE:
    Hyperparameters were selected via RandomizedSearchCV (5-fold CV)
    optimizing for R^2 on the Free Market subset. 
    The search code is preserved below but commented out for deployment performance.
    """
    
    param_grid = {
        'regressor__max_depth': [2, 3, 4],
        'regressor__min_samples_leaf': [3, 5, 10, 15],
        'regressor__subsample': [0.5, 0.7, 0.9],
        'regressor__learning_rate': [0.01, 0.05, 0.1],
        'regressor__n_estimators': [100, 200, 300]
    }
    
    pipe = Pipeline([('scaler', StandardScaler()), ('regressor', GradientBoostingRegressor(random_state=42))])
    
    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_grid,
        n_iter=20, cv=5, scoring='r2', n_jobs=-1, random_state=42
    )
    search.fit(X_train, y_train)
    return search.best_estimator_

    # model_f = Pipeline([
    #     ('scaler', StandardScaler()),
    #     ('regressor', GradientBoostingRegressor(
    #         # these are optimal hyperparams
    #         n_estimators=200,       
    #         learning_rate=0.05,     
    #         max_depth=3,            
    #         min_samples_leaf=5,    
    #         subsample=0.5,
            
    #         random_state=1,
    #         loss='huber'
    #     ))
    # ])

    # model_f.fit(X_train, y_train)

    # return model_f