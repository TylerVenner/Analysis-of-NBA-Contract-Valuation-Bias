import pandas as pd
import sys
import os
from pathlib import Path
from nba_api.stats.endpoints import leaguedashplayerstats
import time

# --- SETUP ---
# Adjust this to match your project structure if needed
project_root = Path(__file__).resolve().parents[2]
INPUT_PATH = os.path.join(project_root, "data", "processed", "master_dataset_with_contracts.csv")
OUTPUT_PATH = os.path.join(project_root, "data", "processed", "master_dataset_final.csv")

def main():
    print("--- NBA Volume Stats Fetcher ---")
    
    # 1. Load your existing data
    print(f"Loading local dataset: {INPUT_PATH}")
    if not os.path.exists(INPUT_PATH):
        print("Error: Input file not found.")
        return
    
    df_local = pd.read_csv(INPUT_PATH)
    print(f"  -> Found {len(df_local)} players.")

    # 2. Fetch NBA Data (2024-25 Season)
    # We use '2024-25' because your report is Nov 2025, so we want the last FULL season of performance.
    target_season = '2024-25' 
    print(f"\nFetching official stats from NBA API for season {target_season}...")
    
    try:
        # LeagueDashPlayerStats gets totals for every player
        stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season=target_season,
            per_mode_detailed='Totals'  # We want TOTAL Minutes/Points, not Per Game
        )
        df_nba = stats.get_data_frames()[0]
        
        # Keep only what we need
        # GP: Games Played, MIN: Total Minutes, PTS: Total Points
        cols_to_keep = ['PLAYER_NAME', 'GP', 'MIN', 'PTS']
        df_volume = df_nba[cols_to_keep].copy()
        
        print(f"  -> Successfully fetched stats for {len(df_volume)} players.")
        print(f"  -> Sample: {df_volume.head(2).to_dict()}")
        
    except Exception as e:
        print(f"Error fetching from NBA API: {e}")
        return

    # 3. Merge Data
    print("\nMerging datasets on 'PLAYER_NAME'...")
    
    # We use a left join to keep all your original players
    # Note: This relies on names matching exactly (e.g. 'Luka Doncic').
    df_merged = pd.merge(df_local, df_volume, on='PLAYER_NAME', how='left')
    
    # Check for missing volume stats (mismatched names)
    missing = df_merged[df_merged['MIN'].isnull()]
    if not missing.empty:
        print(f"  Warning: {len(missing)} players did not match NBA API names.")
        print(f"  (First 5 missing: {missing['PLAYER_NAME'].head(5).tolist()})")
    
    # 4. Save
    df_merged.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSuccess! New dataset saved to:\n{OUTPUT_PATH}")
    print("Columns added: GP, MIN, PTS")

if __name__ == "__main__":
    main()