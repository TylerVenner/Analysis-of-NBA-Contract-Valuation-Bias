import pandas as pd
from pathlib import Path
import sys
import os
import re
import unicodedata

# --- File Configuration ---

# Source files (Originals)
STATS_FILE = 'Player_Performance_raw.csv'
CONTEXT_FILE = 'raw_player_context.csv'
SALARY_FILE = 'raw_player_salaries.csv'

# --- NEW FILES ---
POPULARITY_FILE = 'nba_player_popularity.csv'
STADIUMS_FILE = 'nba_stadiums.csv'

# Manually-edited files (Your team is editing these)
# IMPORTANT: These must have a 'TEAM_ID' column added manually.
CAPS_FILE = 'raw_salary_caps.csv' 
OWNERS_FILE = 'Owner Net Worth in Billions .csv'

# Final output file
OUTPUT_FILE = 'master_dataset_v3.csv'

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
        # Handle potential empty strings or NaNs that become float
        name = str(name)
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
            print("Reminder: Your team members must manually create and save this file,")
            print(f"or at least add a 'TEAM_ID' column to '{filename}'.")
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
            print(f"All columns found: {list(df.columns)}")
            if 'TEAM_ID' in missing_cols and filename in [CAPS_FILE, OWNERS_FILE]:
                print("Reminder: Your team members must manually add the 'TEAM_ID' column to this file.")
            sys.exit(1)
            
    return df

def clean_team_id(df: pd.DataFrame, col_name='TEAM_ID') -> pd.DataFrame:
    """Converts TEAM_ID column to a standardized integer format for merging."""
    if col_name not in df.columns:
        print(f"--- 🔴 ERROR: Tried to clean '{col_name}' but column does not exist.")
        sys.exit(1)
    
    try:
        df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
        # Drop rows where TEAM_ID could not be converted (became <NA>)
        rows_before = len(df)
        df = df.dropna(subset=[col_name])
        rows_after = len(df)
        if rows_before > rows_after:
            print(f"⚠️ Dropped {rows_before - rows_after} rows with invalid/missing TEAM_ID.")
        
        df[col_name] = df[col_name].astype('Int64')
        return df
    
    except Exception as e:
        print(f"--- 🔴 ERROR: Could not convert '{col_name}' to a number for merging. ---")
        print(f"Error: {e}")
        sys.exit(1)
        

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
    df_popularity = load_data(POPULARITY_FILE, ['Player', 'Followers'])

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
    # Note: Use the PLAYER_NAME from the context file if it exists and is cleaner
    df_player_base['merge_key'] = df_player_base['PLAYER_NAME_context'].apply(standardize_player_name)
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
        
    # --- Part 4: Merge Player Popularity (NEW) ---
    print("\n--- [Step 4] Merging Player Popularity (on Standardized Name) ---")
    df_popularity['merge_key'] = df_popularity['Player'].apply(standardize_player_name)
    
    rows_before_pop = len(df_player_master)
    rows_pop = len(df_popularity)
    
    # Use LEFT merge to keep all players, even if not on popularity list
    df_player_master = pd.merge(
        df_player_master,
        df_popularity.drop(columns=['Player'], errors='ignore'), # Drop original name col
        on="merge_key",
        how="left"
    )
    
    print(f"Player Master rows: {rows_before_pop} | Popularity rows: {rows_pop}")
    print(f"Left merge complete. Result rows: {len(df_player_master)}")
    print(f"Players kept (from left merge): {len(df_player_master)}")

    # --- Part 5: Load and Merge Team Data ---
    print("\n--- [Step 5] Loading and Merging All Team Data ---")
    
    # Load all team files
    # We load TEAM_ID as string to avoid int/float mismatch, then clean.
    df_caps = load_data(CAPS_FILE, ['TEAM_ID', 'team'], dtype={'TEAM_ID': str})
    df_owners = load_data(OWNERS_FILE, ['TEAM_ID'], dtype={'TEAM_ID': str})
    df_stadiums = load_data(STADIUMS_FILE, ['TEAM_ABBREVIATION'])

    # Clean TEAM_ID columns for merging
    df_caps = clean_team_id(df_caps, 'TEAM_ID')
    df_owners = clean_team_id(df_owners, 'TEAM_ID')
    
    # Merge Caps + Owners
    print("Merging Team Caps + Owner Worth (on TEAM_ID)...")
    rows_caps = len(df_caps)
    rows_owners = len(df_owners)
    
    df_team_base = pd.merge(
        df_caps,
        df_owners,
        on="TEAM_ID",
        how="inner" # Inner merge as they should both have all 30 teams
    )
    print(f"Caps rows: {rows_caps} | Owners rows: {rows_owners}")
    print(f"Team base merge complete. Result rows: {len(df_team_base)}")
    
    # Merge Stadiums
    print("Merging Stadium Data (on Team Abbreviation)...")
    rows_team_base = len(df_team_base)
    rows_stadiums = len(df_stadiums)
    
    df_team_master = pd.merge(
        df_team_base,
        df_stadiums,
        left_on="team", # Abbreviation from raw_salary_caps.csv
        right_on="TEAM_ABBREVIATION",
        how="left" # Left merge to keep teams even if stadium data is missing
    )
    print(f"Team Base rows: {rows_team_base} | Stadiums rows: {rows_stadiums}")
    print(f"Team master merge complete. Result rows: {len(df_team_master)}")


    # --- Part 6: Final Merge: Players + Teams ---
    print("\n--- [Step 6] Final Merge: Players + Teams (on TEAM_ID) ---")
    
    # Clean player TEAM_ID
    df_player_master = clean_team_id(df_player_master, 'TEAM_ID')
    
    rows_players = len(df_player_master)
    rows_teams = len(df_team_master)
    
    # Using 'inner' merge as requested
    df_final = pd.merge(
        df_player_master,
        df_team_master,
        on="TEAM_ID",
        how="inner"
    )
    
    print(f"Player Master rows: {rows_players} | Team Master rows: {rows_teams}")
    print(f"Merge complete. Final rows: {len(df_final)}")
    print(f"⚠️ Players dropped (no team match): {rows_players - len(df_final)}")

    if len(df_final) == 0:
        print("--- 🔴 WARNING: Final merge resulted in 0 rows. ---")
        print("Check that TEAM_IDs exist in both player and team files.")
    
    # --- Part 7: Final Cleanup and Save ---
    print("\n--- [Step 7] Cleaning and Saving Final Dataset ---")
    
    # Per your request: Keep ALL columns.
    # Pandas merge automatically adds suffixes (_x, _y) if col names conflict.
    
    # Clean up salary/cap/numeric columns
    # List of columns that might contain non-numeric characters like '$' or ','
    numeric_cols_to_clean = [
        'Salary', 'total_cap_used', 'active_cap', 'dead_cap',
        'Followers', 'Capacity', 'Construction_Cost'
    ]
    
    # Clean the 'Owner Net Worth' column (which has a weird name)
    owner_col_name = ' Owner Net Worth in Billions '
    if owner_col_name in df_final.columns:
        df_final.rename(columns={owner_col_name: 'Owner_Net_Worth_Billions'}, inplace=True)
        numeric_cols_to_clean.append('Owner_Net_Worth_Billions')
    
    for col in numeric_cols_to_clean:
        if col in df_final.columns:
            df_final[col] = (
                df_final[col]
                .astype(str)
                .str.replace(r"[^\d.]", "", regex=True) # Keep digits and decimals
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