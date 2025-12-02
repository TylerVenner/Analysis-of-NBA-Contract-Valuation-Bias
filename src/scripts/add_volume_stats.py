import pandas as pd
import sys
import os
import unicodedata
import re
from pathlib import Path
from nba_api.stats.endpoints import leaguedashplayerstats

# --- CONFIGURATION ---
# Adjust these paths to match your exact folder structure
project_root = Path(__file__).resolve().parents[2]
INPUT_PATH = os.path.join(project_root, "data", "processed", "master_dataset_with_contracts.csv")
OUTPUT_PATH = os.path.join(project_root, "data", "processed", "master_dataset_final.csv")

def normalize_name(name):
    """
    Robust name normalization to match NBA API names with Contract Data names.
    - Removes accents (Jokić -> Jokic)
    - Removes suffixes (Jr., III)
    - Removes punctuation (., ')
    - Lowercases and strips whitespace
    """
    if pd.isna(name):
        return ""
    
    # 1. Unicode Normalization (Splits characters from accents)
    # NFD form decomposes characters (e.g., 'ć' becomes 'c' + '´')
    nfkd_form = unicodedata.normalize('NFD', str(name))
    
    # Filter out non-spacing mark characters (the accents) and encode back to ASCII
    name_ascii = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
    
    # 2. String Cleaning
    name_clean = name_ascii.lower()
    
    # Remove suffixes (Jr, Sr, II, III, IV)
    # We use regex to ensure we match " jr" at end of string or " jr."
    name_clean = re.sub(r'\s+(jr\.?|sr\.?|ii|iii|iv)(\s|$)', '', name_clean)
    
    # Remove special chars (periods, apostrophes)
    name_clean = re.sub(r"[.']", "", name_clean)
    
    return name_clean.strip()

def main():
    print("--- NBA Volume Stats Fetcher (Superstar Fix) ---")
    
    # 1. Load Local Contract Data
    print(f"Loading local dataset: {INPUT_PATH}")
    if not os.path.exists(INPUT_PATH):
        print(f"Error: Input file not found at {INPUT_PATH}")
        return
    
    df_local = pd.read_csv(INPUT_PATH)
    print(f"  -> Local records: {len(df_local)}")

    # 2. Fetch NBA 2024-25 Data
    target_season = '2024-25'
    print(f"\nFetching official stats from NBA API for {target_season}...")
    
    try:
        # Fetch TOTALS (not Per Game) to capture volume / durability
        stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season=target_season,
            per_mode_detailed='Totals' 
        )
        df_nba = stats.get_data_frames()[0]
        
        # Keep only essential columns
        df_volume = df_nba[['PLAYER_NAME', 'GP', 'MIN', 'PTS']].copy()
        print(f"  -> API records fetched: {len(df_volume)}")
        
    except Exception as e:
        print(f"CRITICAL ERROR fetching NBA API data: {e}")
        return

    # 3. Create Normalization Keys
    print("\nNormalizing names to catch Superstars (e.g., Dončić -> Doncic)...")
    
    df_local['name_norm'] = df_local['PLAYER_NAME'].apply(normalize_name)
    df_volume['name_norm'] = df_volume['PLAYER_NAME'].apply(normalize_name)
    
    # Diagnostic: Check specific tricky names
    print("  -> Diagnostic Check:")
    for check_name in ["Luka Doncic", "Nikola Jokic", "Stephen Curry"]:
        norm = normalize_name(check_name)
        in_local = norm in df_local['name_norm'].values
        in_api = norm in df_volume['name_norm'].values
        print(f"     '{check_name}' ({norm}): Local={in_local}, API={in_api}")

    # 4. Merge
    print("\nMerging datasets...")
    
    # Left merge on Normalized Name to preserve local Contract Data
    df_merged = pd.merge(
        df_local, 
        df_volume[['name_norm', 'GP', 'MIN', 'PTS']], 
        on='name_norm', 
        how='left'
    )
    
    # Drop the temporary norm column
    df_merged.drop(columns=['name_norm'], inplace=True)

    # 5. Validation
    # Check if the "Missing Superstars" are populated now
    top_earners = df_merged.sort_values('Salary', ascending=False).head(5)
    print("\nTop 5 Highest Paid Players in Final Dataset:")
    print(top_earners[['PLAYER_NAME', 'Salary', 'PTS']])
    
    # Check for missing stats
    missing_stats = df_merged[df_merged['PTS'].isnull()]
    if not missing_stats.empty:
        print(f"\nWarning: {len(missing_stats)} players still missing volume stats.")
        # print(missing_stats['PLAYER_NAME'].head().tolist())
    else:
        print("\nSuccess: All players matched!")

    # 6. Save
    df_merged.to_csv(OUTPUT_PATH, index=False)
    print(f"\nDataset saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()