import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.analysis.train_f_model import train_f_model
from src.analysis.treatment_models import train_h_models
from src.analysis.generate_dml_residuals import generate_dml_residuals
from src.analysis.run_final_ols import run_final_ols

from src.integration.attribution import BiasAttributor
from src.integration.visualizer import plot_bias_map_3d
from src.core.fitters import BiasMapFitter

# Paths
DATA_PATH = os.path.join(project_root, "data", "processed", "master_dataset_cleaned.csv")
OUTPUT_DIR = os.path.join(project_root, "reports", "maps")

# Feature Config (Matching main.py)
Y_COL = "Salary"

X_COLS = [
    "OFF_RATING", "DEF_RATING", "NET_RATING", "AST_PCT", "AST_TO", 
    "AST_RATING", "OREB_PCT", "REB_PCT", "DREB_PCT", "TM_TOV_PCT", 
    "EFG_PCT", "TS_PCT", "PACE", "PIE", "USG_PCT", 
    "POSS", "FGM_PG", "FGA_PG"
]

# Bias Factors 
Z_COLS = [
    "DRAFT_NUMBER", 
    "active_cap", 
    "dead_cap", 
    "OWNER_NET_WORTH_B", 
    "Capacity", 
    "STADIUM_YEAR_OPENED",
    "STADIUM_COST",
    "Followers",
    "Age",
    "is_USA"
]

# Helper for preprocessing (Same main.py)
def preprocess_data(df):
    """Replicates the preprocessing logic from your original main.py"""
    if "DRAFT_NUMBER" in df.columns:
        df["DRAFT_NUMBER"] = df["DRAFT_NUMBER"].replace("Undrafted", 61)
        df["DRAFT_NUMBER"] = pd.to_numeric(df["DRAFT_NUMBER"], errors='coerce')

    if "BIRTHDATE" in df.columns:
        def compute_age(birthdate):
            try:
                return (datetime.now() - pd.to_datetime(birthdate)).days / 365.25
            except:
                return np.nan
        df["Age"] = df["BIRTHDATE"].apply(compute_age)
        
    if "COUNTRY" in df.columns:
        df["is_USA"] = np.where(df["COUNTRY"] == "USA", 1, 0)
        
    # Handle missing columns gracefully (e.g. AST_RATING vs AST_RATIO)
    x_cols_final = X_COLS.copy()
    if "AST_RATING" not in df.columns and "AST_RATIO" in df.columns:
        print("Info: Swapping AST_RATING for AST_RATIO")
        x_cols_final = [c if c != "AST_RATING" else "AST_RATIO" for c in x_cols_final]
        
    # Filter complete cases
    all_cols = [Y_COL] + x_cols_final + Z_COLS
    df_clean = df.dropna(subset=all_cols).copy()
    
    return df_clean, x_cols_final

def main():
    # 1. Setup Output Directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(OUTPUT_DIR, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"--- Starting Bias Mapping Pipeline ---")
    print(f"Output Directory: {run_dir}")

    # 2. Load and Preprocess Data
    print("\n[1/5] Loading Data...")
    try:
        df_raw = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"Error: Could not find data at {DATA_PATH}")
        return

    df_clean, x_cols_final = preprocess_data(df_raw)
    print(f"Loaded {len(df_clean)} players.")

    # 3. Run DML Pipeline (The "Learn" Phase)
    print("\n[2/5] Running DML Residualization...")
    # Note: We use a copy to ensure index alignment
    X = df_clean[x_cols_final]
    Y = np.log(df_clean[Y_COL]) # Log Salary is crucial!
    Z = df_clean[Z_COLS]

    # Call your existing module
    residuals_Y, residuals_Z = generate_dml_residuals(
        X=X, Y=Y, Z=Z,
        model_f_trainer=train_f_model,
        model_h_trainer=train_h_models,
        k_folds=5 # 5-fold is sufficient for map generation
    )

    # Run Final OLS to get the "Prices" (Gammas)
    print("\n[3/5] Estimating Price of Bias (Gamma)...")
    ols_results = run_final_ols(residuals_Y, residuals_Z)
    gamma_coefficients = ols_results.params
    
    # Save econometric results
    with open(os.path.join(run_dir, "econometric_summary.txt"), "w") as f:
        f.write(ols_results.summary().as_text())
    print("Saved OLS summary.")

    # 4. Construct Attribution Matrix (The Bridge)
    print("\n[4/5] Constructing Attribution Matrix (L)...")
    attributor = BiasAttributor(gamma_coefficients, residuals_Z)
    
    # Get the L matrix (L_ij = gamma_j * epsilon_Z_ij)
    # BiasAttributor automatically aligns gamma with residuals_Z columns.
    # Since residuals_Z does not have 'const', it is automatically dropped inside the class.
        
    L_matrix = attributor.get_attribution_matrix(normalize=True)
    print(f"Matrix Shape: {L_matrix.shape} (Players x Bias Factors)")
    
    # Save the matrix for inspection
    L_matrix.to_csv(os.path.join(run_dir, "attribution_matrix.csv"))

    # 5. Fit the Bias Map (The Engine)
    print("\n[5/5] Fitting 3D Latent Map (JAX Engine)...")
    
    fitter = BiasMapFitter(
        n_dimensions=3,
        n_cycles=10,      # Alternating cycles
        alt_steps=200,
        polish_steps=2000,
        learning_rate=0.02
    )
    
    # We pass the raw numpy array to JAX
    fitter.fit(L_matrix.values)

    # 6. Visualization
    print("\nGenerating Interactive HTML...")
    
    # Extract Metadata (e.g., Contract Type if available, or Position)
    # For now, let's try to use 'Contract_Type' if you engineered it, 
    # otherwise default to 'Position' or 'Unknown'
    meta_col = 'CONTRACT_TYPE' if 'CONTRACT_TYPE' in df_clean.columns else 'Unknown'
    if meta_col == 'Unknown' and 'POSITION' in df_clean.columns:
        meta_col = 'POSITION'
        
    player_meta = attributor.get_player_metadata(df_clean, contract_col=meta_col)
    
    plot_path = os.path.join(run_dir, "bias_attribution_map_3d.html")
    
    plot_bias_map_3d(
        fitter=fitter,
        attribution_matrix=L_matrix,
        bias_labels=L_matrix.columns.tolist(),
        player_metadata=player_meta,
        title="Bias Attribution Map (DML-PULS Fusion)",
        output_path=plot_path
    )
    
    print(f"\nSUCCESS! Map generated at: {plot_path}")

if __name__ == "__main__":
    main()