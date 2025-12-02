import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

# Load the cleaned master dataset
df = pd.read_csv("/Users/leonelgaribay/Analysis-of-NBA-Contract-Valuation-Bias/data/processed/master_dataset_cleaned.csv")

# Quick check
print(df.shape)
df.head()


def train_f_model(df):
    """
    Trains the Outcome Model (f̂): Salary ~ Performance
    Returns: trained model and test performance metrics
    """

    # --- 1. Select features (X) and target (y) ---
    performance_features = [
        'OFF_RATING', 'DEF_RATING', 'NET_RATING', 'AST_PCT', 'AST_TO',
        'AST_RATIO', 'OREB_PCT', 'DREB_PCT', 'REB_PCT', 'EFG_PCT', 
        'TS_PCT', 'USG_PCT', 'PIE', 'FGM_PG', 'FGA_PG'
    ]

    # Drop any rows missing these values
    df = df.dropna(subset=performance_features + ['Salary'])

    X = df[performance_features]
    y = df['Salary']

    # --- 2. Split data into train/test sets ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --- 3. Train Random Forest model with GridSearchCV ---
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [5, 10, None],
        'min_samples_leaf': [1, 3]
    }

    grid_search = GridSearchCV(
        estimator=RandomForestRegressor(random_state=42),
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        scoring='neg_mean_squared_error'
    )

    grid_search.fit(X_train, y_train)
    model = grid_search.best_estimator_

    # --- 4. Evaluate model ---
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    print(f"R² score: {r2:.3f}")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Best Parameters: {grid_search.best_params_}")

    return model, r2, mse


if __name__ == "__main__":
    train_f_model(df)
