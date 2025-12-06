import sys
from pathlib import Path
import pandas as pd
import shutil
import os

# Add src to path so we can import backend
sys.path.append(str(Path(__file__).parent))

# Import your heavy machinery
from src.scripts.run_bias_mapping import main as run_heavy_pipeline

def main():
    print("STARTING DEPLOYMENT PREP...")
    
    # 1. Run the Heavy Pipeline (JAX + XGBoost)
    # This generates the timestamped folder in reports/maps/run_XXXX/
    print("--- Running Heavy Analysis (this may take a minute) ---")
    run_heavy_pipeline()
    
    # 2. Find the output
    maps_dir = Path("reports/maps")
    # Get latest run folder
    latest_run = max([d for d in maps_dir.iterdir() if d.is_dir()], key=os.path.getmtime)
    print(f"✅ Analysis finished. Using outputs from: {latest_run}")
    
    # 3. Create 'app_data' folder for Streamlit
    # This is the ONLY folder the website will look at.
    app_data_dir = Path("data/app_data")
    app_data_dir.mkdir(parents=True, exist_ok=True)
    
    # 4. Copy and Rename Artifacts
    
    # A. The Interactive 3D Map HTML
    html_source = latest_run / "bias_attribution_map_3d.html"
    shutil.copy(html_source, app_data_dir / "final_3d_map.html")
    print("   -> Copied 3D Map HTML")
    
    # B. The Attribution Matrix (CSV)
    attr_source = latest_run / "attribution_matrix.csv"
    shutil.copy(attr_source, app_data_dir / "attribution_matrix.csv")
    print("   -> Copied Attribution Matrix")

    # C. The Model Performance Metrics (for Page 2 context if needed)
    perf_source = latest_run / "model_performance.csv"
    if perf_source.exists():
        shutil.copy(perf_source, app_data_dir / "model_performance.csv")
    
    # D. Final OLS Coefficients (for Page 3)
    # Note: run_bias_mapping usually prints this to stdout, 
    # we need to ensure it saved a CSV or we extract it from the pipeline.
    # If run_bias_mapping doesn't save "final_ols_table.csv", we might need to 
    # manually locate it or update run_bias_mapping.py to save it.
    # Assuming it exists in processed/ or the run folder:
    
    # Check processed folder first (main.py output) or run folder
    ols_source_1 = Path("data/processed/final_ols_table.csv") 
    if ols_source_1.exists():
        shutil.copy(ols_source_1, app_data_dir / "final_ols_coefficients.csv")
        print("   -> Copied OLS Coefficients")
    else:
        print("WARNING: final_ols_table.csv not found. Check run_bias_mapping.py")

    # E. The Player Metadata (for Page 4 Geographic Map)
    # This needs to be the dataset aligned with the artifacts
    meta_source = Path("data/processed/master_dataset_advanced_v2.csv")
    df = pd.read_csv(meta_source)
    
    # We need to merge the 'Player_bias_effect' onto this if it's not there.
    # For now, let's copy the raw file, but ideally, we calculate the bias effect here.
    shutil.copy(meta_source, app_data_dir / "player_db.csv")
    print("   -> Copied Player Database")
    
    print("\nDONE! All website assets are in 'data/app_data/'.")

if __name__ == "__main__":
    main()