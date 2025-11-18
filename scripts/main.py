import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
import sys
from datetime import datetime

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.analysis.train_f_model import train_f_model
from src.analysis.treatment_models import train_h_models
from src.analysis.generate_dml_residuals import generate_dml_residuals
from src.analysis.run_final_ols import run_final_ols

# --- Configuration ---
DATA_PATH = "../data/processed/master_dataset_cleaned.csv"

# Y (Outcome)
Y_COL = "Salary"

# X (Performance) - 18 features
X_COLS = [
    "OFF_RATING", "DEF_RATING", "NET_RATING", "AST_PCT", "AST_TO", 
    "AST_RATING", "OREB_PCT", "REB_PCT", "DREB_PCT", "TM_TOV_PCT", 
    "EFG_PCT", "TS_PCT", "PACE", "PIE", "USG_PCT", 
    "POSS", "FGM_PG", "FGA_PG"
]

# Z (Contextual/Bias) - REFINED SET
# Keeping all variables as requested by team lead
Z_COLS = [
    "DRAFT_NUMBER", 
    "active_cap", 
    "dead_cap", 
    "OWNER_NET_WORTH_B", 
    "Capacity", 
    "STADIUM_YEAR_OPENED",
    "STADIUM_COST",     # <-- RE-ADDED per your request
    "Followers",        # <-- RE-ADDED per your request
    "Age",              # <-- NEW
    "is_USA"            # <-- NEW
]

# Helper function
def compute_age(birthdate):
    """Convert birthdate string to age in years."""
    try:
        return (datetime.now() - pd.to_datetime(birthdate)).days / 365.25
    except:
        return np.nan

def main():
    """
    Main function to run the entire DML pipeline.
    """
    # --- This is the better way ---
    # Create local, mutable copies of the global constants.
    # This avoids all 'global' keywords and UnboundLocalErrors.
    x_cols_local = X_COLS.copy()
    z_cols_local = Z_COLS.copy()
    
    print("Starting DML Pipeline (Refined Z-Set)...")
    
    # --- 1. Load Data ---
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"ERROR: Data file not found at {DATA_PATH}")
        return

    print(f"Loaded {df.shape[0]} rows from {DATA_PATH}")

    # --- 2. Preprocess Data ---
    
    # Handle 'Undrafted' for DRAFT_NUMBER
    if "DRAFT_NUMBER" in df.columns:
        df["DRAFT_NUMBER"] = df["DRAFT_NUMBER"].replace("Undrafted", 61)
        df["DRAFT_NUMBER"] = pd.to_numeric(df["DRAFT_NUMBER"], errors='coerce')

    # Handle 'BIRTHDATE' -> 'Age'
    if "BIRTHDATE" in df.columns:
        df["Age"] = df["BIRTHDATE"].apply(compute_age)
        
    # Handle 'COUNTRY' -> 'is_USA' (1 if USA, 0 if not)
    if "COUNTRY" in df.columns:
        df["is_USA"] = np.where(df["COUNTRY"] == "USA", 1, 0)
    
    # Define all columns needed for the analysis (using local lists)
    all_needed_cols = [Y_COL] + x_cols_local + z_cols_local

    # Check for missing columns
    missing_cols = [col for col in all_needed_cols if col not in df.columns]
    if missing_cols:
        # Check for AST_RATING specifically, as it was mis-named
        if "AST_RATING" in missing_cols and "AST_RATIO" in df.columns:
            print("Info: 'AST_RATING' not found, using 'AST_RATIO' instead.")
            
            # Modify the *local* list, not the global one.
            x_cols_local = [col if col != "AST_RATING" else "AST_RATIO" for col in x_cols_local]
            
            # Re-run the check
            all_needed_cols = [Y_COL] + x_cols_local + z_cols_local
            missing_cols = [col for col in all_needed_cols if col not in df.columns]

        if missing_cols:
            print(f"ERROR: Dataset is missing required columns: {missing_cols}")
            print("Please check Z_COLS list and data file.")
            return

    # Drop any rows missing *any* of our key variables
    df_clean = df.dropna(subset=all_needed_cols).copy()
    
    print(f"Retained {df_clean.shape[0]} complete rows after dropping NaNs.")

    # --- 3. Define Final Y, X, Z variables ---
    # Use the local lists to create the final DataFrames
    Y = np.log(df_clean[Y_COL])
    X = df_clean[x_cols_local] # Use the (potentially modified) local list
    Z = df_clean[z_cols_local]
    
    print(f"Final Z features: {list(Z.columns)}")

    # --- 4. Run DML Residual Generation (Module 3) ---
    residuals_Y, residuals_Z = generate_dml_residuals(
        X=X,
        Y=Y,
        Z=Z,
        model_f_trainer=train_f_model,
        model_h_trainer=train_h_models,
        k_folds=10
    )
    
    print(f"\nGenerated {len(residuals_Y)} out-of-sample residuals.")

    # --- 5. Check for Multicollinearity in Z-Residuals ---
    print("\n" + "="*80)
    print(" Correlation Matrix of Z-Residuals (epsilon_Z) ")
    print(" Looking for values close to 1.0 or -1.0")
    print("="*80)
    
    residuals_corr = residuals_Z.corr()
    corr_pairs = residuals_corr.unstack()
    
    high_corr_pairs = corr_pairs[
        (corr_pairs.abs() > 0.9) & (corr_pairs.abs() < 1.0)
    ].sort_values(ascending=False, key=abs)
    
    if high_corr_pairs.empty:
        print("No highly correlated pairs (abs > 0.9) found.")
    else:
        print("WARNING: Found highly correlated pairs (abs > 0.9).")
        print("Consider dropping one variable from each pair:")
        print(high_corr_pairs.drop_duplicates())
        
    print("="*80)

    # --- 6. Run Final OLS (Module 4) ---
    final_ols_results = run_final_ols(residuals_Y, residuals_Z)

    # --- 7. Show Final Results ---
    print("\n" + "="*80)
    print(" DML FINAL OLS RESULTS (Refined Z-Set) ")
    print("="*80)
    print(final_ols_results.summary())
    print("\nPipeline complete.")

if __name__ == "__main__":
    main()