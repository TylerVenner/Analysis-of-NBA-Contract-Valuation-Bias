import pandas as pd
from pathlib import Path
import sys
import os
import re
import unicodedata

# --- File Configuration ---

# Source files (Originals)
STATS_FILE = 'Player_Performance_raw.csv' # has team id and player id
CONTEXT_FILE = 'raw_player_context.csv' # has player id
SALARY_FILE = 'raw_player_salaries.csv' # has player name

# Manually-edited files (Your team needs to create these)
# IMPORTANT: Save your team's edited files with these EXACT names
CAPS_FILE = 'raw_player_caps.csv' # has abbreviated team names (currently adding team id)
OWNERS_FILE = 'Owner Net Worth in Billions .csv' # has team names (currently adding team id)

# Final output file
OUTPUT_FILE = 'master_dataset_v1.csv'

# --- End Configuration ---


def standardize_player_name(name: str) -> str:
    """
    Cleans and standardizes player names so merges work across data sources.
    (Copied from your cleaning_helpers.py)
    """
    if not isinstance(name, str):
        return ""
    
    # Normalize unicode (e.g., accents)
    try:
        name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode()
    except Exception as e:
        print(f"Warning: Could not normalize name '{name}'. Error: {e}")
        pass # Continue with the original name if normalization fails

    # Replace punctuation and collapse spaces
    name = re.sub(r'[^a-zA-Z\s]', '', name)
    name = re.sub(r'\s+', ' ', name)
    
    return name.strip().lower()


def load_data(filename: str, required_cols: list = None, dtype: dict = None) -> pd.DataFrame:
    """
    Loads a CSV file with robust debugging and error checking.
    """
    print(f"Attempting to load file: '{filename}'...")
    
    # 1. Check if file exists
    if not os.path.exists(filename):
        print(f"--- 🔴 ERROR: File not found! ---")
        print(f"Script stopped. Could not find file: {filename}")
        if filename in [CAPS_FILE, OWNERS_FILE]:
            print("Reminder: Your team members must manually create and save this file.")
            print(f"Please make sure it is saved in the same directory with the name '{filename}'.")
        sys.exit(1) # Stop the script
        
    # 2. Load the data
    try:
        df = pd.read_csv(filename, dtype=dtype)
        print(f"✅ Successfully loaded '{filename}'. Found {len(df)} rows.")
    except Exception as e:
        print(f"--- 🔴 ERROR: Could not read file! ---")
        print(f"Could not load {filename}. Error: {e}")
        sys.exit(1)

    # 3. Check for required columns
    if required_cols:
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"--- 🔴 ERROR: Missing required columns in '{filename}'! ---")
            print(f"Expected columns: {required_cols}")
            print(f"Missing columns: {missing_cols}")
            if 'TEAM_ID' in missing_cols and filename in [CAPS_FILE, OWNERS_FILE]:
                print("Reminder: Your team members must manually add the 'TEAM_ID' column to this file.")
            sys.exit(1)
            
    return df


def main():
    """
    Main function to run the entire merge workflow.
    """
    print("--- Starting Full Merge Process ---")

    # --- Part 1: Load Player Data ---
    print("\n--- [Step 1] Loading Player Data ---")
    df_stats = load_data(STATS_FILE, ['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID'])
    df_context = load_data(CONTEXT_FILE, ['PLAYER_ID', 'PLAYER_NAME', 'BIRTHDATE'])
    df_salary = load_data(SALARY_FILE, ['Player_Name', 'Salary'])

    # --- Part 2: Merge Stats + Context ---
    print("\n--- [Step 2] Merging Stats and Context (on PLAYER_ID) ---")
    rows_stats = len(df_stats)
    rows_context = len(df_context)
    
    # Using 'inner' merge as requested
    df_player_base = pd.merge(
        df_stats, 
        df_context, 
        on="PLAYER_ID", 
        how="inner",
        suffixes=('_stats', '_context')
    )
    
    print(f"Stats rows: {rows_stats} | Context rows: {rows_context}")
    print(f"Merge complete. Result rows: {len(df_player_base)}")
    
    if len(df_player_base) == 0:
        print("--- 🔴 ERROR: Merge 1 (Stats + Context) resulted in 0 rows. ---")
        print("Please check PLAYER_ID columns in both files.")
        sys.exit(1)
    
    # --- Part 3: Merge Salaries ---
    print("\n--- [Step 3] Merging Salaries (on Standardized Name) ---")
    
    # Standardize names for merging
    # Using PLAYER_NAME_stats because it's the one from the performance file.
    df_player_base['merge_key'] = df_player_base['PLAYER_NAME_stats'].apply(standardize_player_name)
    df_salary['merge_key'] = df_salary['Player_Name'].apply(standardize_player_name)
    
    rows_before_salary = len(df_player_base)
    rows_salary = len(df_salary)
    
    # Using 'inner' merge as requested
    df_player_master = pd.merge(
        df_player_base, 
        df_salary, 
        on="merge_key", 
        how="inner"
    )
    
    print(f"Player Base rows: {rows_before_salary} | Salary rows: {rows_salary}")
    print(f"Merge complete. Result rows: {len(df_player_master)}")
    print(f"⚠️ Players dropped (no salary match): {rows_before_salary - len(df_player_master)}")
    
    if len(df_player_master) == 0:
        print("--- 🔴 ERROR: Merge 2 (Salaries) resulted in 0 rows. ---")
        print("Check name standardization or if salary file is correct.")
        sys.exit(1)

    # --- Part 4: Load Team Data (with manual TEAM_ID) ---
    print("\n--- [Step 4] Loading Manually-Edited Team Files ---")
    # This assumes your team has created these files and added TEAM_ID
    # We will load TEAM_ID as a string to avoid int/float mismatch, then clean.
    df_caps = load_data(CAPS_FILE, ['TEAM_ID'], dtype={'TEAM_ID': str})
    df_owners = load_data(OWNERS_FILE, ['TEAM_ID'], dtype={'TEAM_ID': str})

    # --- Part 5: Merge Team Caps ---
    print("\n--- [Step 5] Merging Team Caps (on TEAM_ID) ---")
    
    # Clean TEAM_ID for merging (convert all to integers)
    try:
        df_player_master['TEAM_ID'] = pd.to_numeric(df_player_master['TEAM_ID'], errors='coerce').astype('Int64')
        df_caps['TEAM_ID'] = pd.to_numeric(df_caps['TEAM_ID'], errors='coerce').astype('Int64')
    except Exception as e:
        print(f"--- 🔴 ERROR: Could not convert TEAM_ID to a number for merging. ---")
        print(f"Error: {e}")
        sys.exit(1)
        
    # Drop rows where TEAM_ID became <NA> (missing)
    df_player_master = df_player_master.dropna(subset=['TEAM_ID'])
    df_caps = df_caps.dropna(subset=['TEAM_ID'])
    
    rows_before_caps = len(df_player_master)
    rows_caps = len(df_caps)
    
    # Using 'inner' merge as requested
    df_merged_caps = pd.merge(
        df_player_master, 
        df_caps, 
        on="TEAM_ID", 
        how="inner"
    )
    
    print(f"Player Master rows: {rows_before_caps} | Caps rows: {rows_caps}")
    print(f"Merge complete. Result rows: {len(df_merged_caps)}")
    print(f"⚠️ Players dropped (no team cap match): {rows_before_caps - len(df_merged_caps)}")

    if len(df_merged_caps) == 0:
        print("--- 🔴 ERROR: Merge 3 (Caps) resulted in 0 rows. ---")
        print("Check TEAM_ID columns in your player data and caps_with_id.csv.")
        sys.exit(1)

    # --- Part 6: Merge Owner Worth ---
    print("\n--- [Step 6] Merging Owner Net Worth (on TEAM_ID) ---")
    
    # Clean TEAM_ID for merging
    df_owners['TEAM_ID'] = pd.to_numeric(df_owners['TEAM_ID'], errors='coerce').astype('Int64')
    df_owners = df_owners.dropna(subset=['TEAM_ID'])

    rows_before_owners = len(df_merged_caps)
    rows_owners = len(df_owners)
    
    # Using 'inner' merge as requested
    df_final = pd.merge(
        df_merged_caps,
        df_owners,
        on="TEAM_ID",
        how="inner"
    )

    print(f"Previous Merge rows: {rows_before_owners} | Owners rows: {rows_owners}")
    print(f"Merge complete. Final rows: {len(df_final)}")
    print(f"⚠️ Players dropped (no owner match): {rows_before_owners - len(df_final)}")

    if len(df_final) == 0:
        print("--- 🔴 WARNING: Final merge resulted in 0 rows. ---")
        print("Check TEAM_ID columns in owners_with_id.csv.")
    
    # --- Part 7: Final Cleanup and Save ---
    print("\n--- [Step 7] Cleaning and Saving Final Dataset ---")
    
    # Optional: Drop duplicate columns that might have come from merges
    # e.g., if 'PLAYER_NAME_context' or 'merge_key' are still around
    df_final = df_final.loc[:, ~df_final.columns.duplicated()]
    
    # Clean up salary/cap numbers (example)
    for col in ['Salary', 'total_cap_used', 'active_cap', 'dead_cap']:
        if col in df_final.columns:
            df_final[col] = (
                df_final[col]
                .astype(str)
                .str.replace(r"[^\d.]", "", regex=True)
                .replace("", None)
                .astype(float)
            )
            
    # Clean Owner Net Worth (example)
    if ' Owner Net Worth in Billions ' in df_final.columns:
        df_final.rename(columns={' Owner Net Worth in Billions ': 'Owner_Net_Worth_Billions'}, inplace=True)
        df_final['Owner_Net_Worth_Billions'] = (
            df_final['Owner_Net_Worth_Billions']
            .astype(str)
            .str.replace(r"[^\d.]", "", regex=True)
            .replace("", None)
            .astype(float)
        )
        
    
    # Save to output file
    output_path = Path(OUTPUT_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(output_path, index=False)
    
    print(f"\n✅ --- PROCESS COMPLETE --- ✅")
    print(f"Final dataset with {len(df_final)} rows and {len(df_final.columns)} columns saved to:")
    print(f"{output_path.resolve()}")


if __name__ == "__main__":
    main()
