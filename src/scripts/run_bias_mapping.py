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
DATA_PATH = os.path.join(project_root, "data", "processed", "master_dataset_advanced.csv")
OUTPUT_DIR = os.path.join(project_root, "reports", "maps")

# Feature Config (Matching main.py)
Y_COL = "Salary"
X_COLS = [
    "OFF_RATING", "DEF_RATING", "NET_RATING", "AST_PCT", "AST_TO", 
    "AST_RATING", "OREB_PCT", "REB_PCT", "DREB_PCT", "TM_TOV_PCT", 
    "EFG_PCT", "TS_PCT", "PACE", "PIE", "USG_PCT", 
    "POSS", "FGM_PG", "FGA_PG", 
    "GP", "MIN", "PTS",
    "AVG_SPEED", "DIST_MILES", "ALL_STAR_APPEARANCES"
]
Z_COLS = [
    "DRAFT_NUMBER", "active_cap", "dead_cap", "OWNER_NET_WORTH_B", 
    "Capacity", "STADIUM_YEAR_OPENED", "STADIUM_COST", "Followers", 
    "Age", "is_USA"
]

def preprocess_data(df):
    """Clean and prepare data, ensuring Contract_Type is preserved."""
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
        
    x_cols_final = X_COLS.copy()
    if "AST_RATING" not in df.columns and "AST_RATIO" in df.columns:
        print("Info: Swapping AST_RATING for AST_RATIO")
        x_cols_final = [c if c != "AST_RATING" else "AST_RATIO" for c in x_cols_final]
    
    # Ensure Contract_Type exists, default to 'Free_Market' if missing (for safety)
    if "Contract_Type" not in df.columns:
        print("Warning: Contract_Type column missing. Defaulting all to 'Free_Market'.")
        df["Contract_Type"] = "Free_Market"

    if "Followers" in df.columns:
        # log1p(x) = log(x + 1) to handle zeros safely
        df["Followers"] = np.log1p(df["Followers"])
        
    if "OWNER_NET_WORTH_B" in df.columns:
        # Net worth is also heavily skewed (Ballmer vs others)
        df["OWNER_NET_WORTH_B"] = np.log1p(df["OWNER_NET_WORTH_B"])
        
    # Filter complete cases for ANALYSIS columns (Y, X, Z)
    # We keep Contract_Type for splitting logic
    all_cols = [Y_COL] + x_cols_final + Z_COLS + ["Contract_Type"]
    df_clean = df.dropna(subset=all_cols).copy()
    
    return df_clean, x_cols_final

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(OUTPUT_DIR, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"--- Starting Stratified Bias Mapping Pipeline ---")

    # 1. Load Data
    print("\n[1/5] Loading Data...")
    try:
        df_raw = pd.read_csv(DATA_PATH)
        df_raw = df_raw.set_index('PLAYER_NAME')
    except FileNotFoundError:
        print(f"Error: Could not find data at {DATA_PATH}")
        return

    df_clean, x_cols_final = preprocess_data(df_raw)
    print(f"Loaded {len(df_clean)} complete player records.")

    # 2. Stratification (Learn vs. Apply)
    print("\n[2/5] Stratifying Data...")
    df_fm = df_clean[df_clean['Contract_Type'] == 'Free_Market'].copy()
    df_fixed = df_clean[df_clean['Contract_Type'] != 'Free_Market'].copy()
    
    print("Top 5 Salaries in Clean Data:")
    print(df_fm.sort_values(Y_COL, ascending=False)[[Y_COL, 'PTS']].head())

    print(f"  - Free Market Players (Training Set): {len(df_fm)}")
    print(f"  - Fixed Contract Players (Application Set): {len(df_fixed)}")

    # 3. Train Models on Free Market Data
    print("\n[3/5] Training Models on Free Market...")
    
    # Prepare Training Data
    X_train = df_fm[x_cols_final]
    Y_train = np.log(df_fm[Y_COL])
    Z_train = df_fm[Z_COLS]
    
    import statsmodels.api as sm
    
    print("\n[Diagnostic] Checking Feature Significance (Linear Proxy)...")
    print("  (Note: These p-values are from a simple OLS. The actual Gradient Boosting model captures more complex non-linearities.)")
    
    try:
        # 1. Add Constant for Intercept
        X_diag = sm.add_constant(X_train.astype(float))
        
        # 2. Fit Simple OLS
        model_diag = sm.OLS(Y_train, X_diag).fit()
        
        # 3. Create Clean Summary Table
        feature_summary = pd.DataFrame({
            "Coef (Impact)": model_diag.params,
            "P-Value": model_diag.pvalues,
            "t-stat": model_diag.tvalues
        })
        
        # 4. Sort by Most Significant (Lowest P-Value)
        print(feature_summary.sort_values("P-Value").round(4))
        print("-" * 60)
        
    except Exception as e:
        print(f"  Warning: Could not run diagnostic OLS: {e}")

    res_Y_dml, res_Z_dml, metrics_df = generate_dml_residuals(
        X=X_train,
        Y=Y_train,
        Z=Z_train,
        model_f_trainer=train_f_model,
        model_h_trainer=train_h_models,
        k_folds=5
    )

    metrics_path = os.path.join(run_dir, "model_performance.csv")
    metrics_df.to_csv(metrics_path)
    print(f"  -> Model metrics saved to: {metrics_path}")
    print("  -> Snapshot of performance:")
    print(metrics_df[['r2_mean', 'rmse_mean', 'log_loss_mean']])

    print("  - Estimating Gamma Coefficients (OLS on DML Residuals)...")
    ols_results = run_final_ols(res_Y_dml, res_Z_dml)

    gamma_summary = pd.DataFrame({
        "Gamma (Price)": ols_results.params,
        "P-Value": ols_results.pvalues,
        "Std. Error": ols_results.bse  # Standard Error is also very useful
    })

    print("\n--- Learned Market Prices (Gamma & Significance) ---")
    # Sort by P-Value so the most significant factors appear at the top
    print(gamma_summary.sort_values("P-Value").round(5)) 
    
    # Store just the coefficients for the attribution step
    gamma_coefficients = ols_results.params
    
    # Counterfactual attribution
    print("\n[4/5] Phase 2: Generating Attribution Map for Full League...")

    # Step A: LEARN (Train on Full Free Market Dataset)
    # We retrain to get the strongest possible predictive model
    print(f"  - Training production models on {len(df_fm)} Free Market players...")
    prod_model_f = train_f_model(X_train, Y_train) 
    prod_models_h = train_h_models(X_train, Z_train)

    # Step B: APPLY (Predict on ALL Players)
    print(f"  - Applying models to generate residuals for all {len(df_clean)} players...")
    
    # Prepare Full Data (D_ALL)
    X_all = df_clean[x_cols_final]
    Y_all = np.log(df_clean[Y_COL])
    Z_all = df_clean[Z_COLS]

    Y_pred_all = prod_model_f.predict(X_all)
    residuals_Y_all = Y_all - Y_pred_all
    
    # 2. Generate Treatment Residuals (Epsilon_Z)
    residuals_Z_all = pd.DataFrame(index=df_clean.index, columns=Z_COLS)

    for col in Z_COLS:
        if col in prod_models_h:
            model = prod_models_h[col]
            
            # If the model is a classifier (e.g. for 'is_USA'), we want the residual
            # to be (Actual - Probability), which aligns with Cross-Entropy/Log Loss optimization.
            if hasattr(model, "predict_proba"):
                try:
                    # Get probability of the positive class (1)
                    Z_pred = model.predict_proba(X_all)[:, 1]
                except:
                    # Fallback
                    Z_pred = model.predict(X_all)
            else:
                # Regression (continuous variables like Age)
                Z_pred = model.predict(X_all)
                
            residuals_Z_all[col] = Z_all[col] - Z_pred

    # Construct Attribution Matrix (L) for ALL Players
    # L = Gamma * Residuals_All
    attributor = BiasAttributor(gamma_coefficients, residuals_Z_all)
    L_matrix = attributor.get_attribution_matrix(normalize=True)
    
    print(f"  - Final Matrix Shape: {L_matrix.shape}")
    L_matrix.to_csv(os.path.join(run_dir, "attribution_matrix.csv"))

    # Finally, fit the map
    print("\n[5/5] Fitting 3D Latent Map...")
    fitter = BiasMapFitter(
        n_dimensions=3,
        n_cycles=20,
        alt_steps=200,
        polish_steps=2000,
        learning_rate=0.02
    )
    fitter.fit(L_matrix.values)

    # 8. Visualization
    print("\nGenerating Interactive HTML...")
    
    # Get Metadata from the CLEAN dataframe (which aligns with L_matrix)
    # We specifically want 'Contract_Type'
    player_meta = df_clean['Contract_Type']
    
    plot_path = os.path.join(run_dir, "bias_attribution_map_3d.html")
    
    plot_bias_map_3d(
        fitter=fitter,
        attribution_matrix=L_matrix,
        bias_labels=L_matrix.columns.tolist(),
        player_metadata=player_meta,
        title="Bias Attribution Map (Stratified DML-PULS)",
        output_path=plot_path
    )
    
    print(f"\nSUCCESS! Map generated at: {plot_path}")

if __name__ == "__main__":
    main()