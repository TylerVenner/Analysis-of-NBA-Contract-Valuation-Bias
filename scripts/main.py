import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
import sys

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
    "AST_RATIO", "OREB_PCT", "REB_PCT", "DREB_PCT", "TM_TOV_PCT", 
    "EFG_PCT", "TS_PCT", "PACE", "PIE", "USG_PCT", 
    "POSS", "FGM_PG", "FGA_PG"
]

# Z (Contextual/Bias) - 8 features
Z_COLS = [
    "DRAFT_NUMBER", "active_cap", "avg_team_age", "dead_cap", 
    "OWNER_NET_WORTH_B", "Capacity", "STADIUM_YEAR_OPENED", "STADIUM_COST"
]

def main():
    """
    Main function to run the entire DML pipeline.
    """
    print("Starting DML Pipeline...")
    
    # --- 1. Load Data ---
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"ERROR: Data file not found at {DATA_PATH}")
        return

    print(f"Loaded {df.shape[0]} rows from {DATA_PATH}")

    # --- 2. Preprocess Data ---
    
    # Handle 'Undrafted' for DRAFT_NUMBER
    # We'll set Undrafted to 61 (as seen in Week_6_Module_4.py)
    if "DRAFT_NUMBER" in df.columns:
        df["DRAFT_NUMBER"] = df["DRAFT_NUMBER"].replace("Undrafted", 61)
        df["DRAFT_NUMBER"] = pd.to_numeric(df["DRAFT_NUMBER"], errors='coerce')

    # Define the final set of columns needed
    all_needed_cols = [Y_COL] + X_COLS + Z_COLS

    # Drop any rows missing *any* of our key variables
    df_clean = df.dropna(subset=all_needed_cols).copy()
    
    print(f"Retained {df_clean.shape[0]} complete rows after dropping NaNs.")

    # --- 3. Define Final Y, X, Z variables ---
    # We use .copy() to avoid SettingWithCopyWarning
    
    # Y: Log-transform Salary for stability (as in Week_6_Module_4.py)
    Y = np.log(df_clean[Y_COL])
    
    # X: Performance features
    X = df_clean[X_COLS]
    
    # Z: Contextual/Bias features
    Z = df_clean[Z_COLS]

    # --- 4. Run DML Residual Generation (Module 3) ---
    # This is the core engine. It calls Module 1 (train_f_model)
    # and Module 2 (train_h_models) inside its cross-fitting loop.
    residuals_Y, residuals_Z = generate_dml_residuals(
        X=X,
        Y=Y,
        Z=Z,
        model_f_trainer=train_f_model,
        model_h_trainer=train_h_models,
        k_folds=10
    )
    
    print(f"\nGenerated {len(residuals_Y)} out-of-sample residuals.")

    # --- 5. Run Final OLS (Module 4) ---
    # This runs the final debiased regression on the residuals.
    final_ols_results = run_final_ols(residuals_Y, residuals_Z)

    # --- 6. Show Final Results ---
    print("\n" + "="*80)
    print(" DML FINAL OLS RESULTS (epsilon_Y ~ epsilon_Z) ")
    print("="*80)
    print(final_ols_results.summary())
    print("="*80)
    print("\nPipeline complete.")

if __name__ == "__main__":
    main()