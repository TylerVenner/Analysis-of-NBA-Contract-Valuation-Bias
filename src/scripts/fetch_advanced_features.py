import pandas as pd
import time
import os
from pathlib import Path
from nba_api.stats.endpoints import leaguedashplayerstats, leaguedashptstats, playerawards
from nba_api.stats.static import players as static_players

project_root = Path(__file__).resolve().parents[2]
INPUT_PATH = os.path.join(project_root, "data", "processed", "master_dataset_final.csv")
OUTPUT_PATH = os.path.join(project_root, "data", "processed", "master_dataset_advanced.csv")
TARGET_SEASON = '2024-25'

def normalize_name(name):
    """Simple normalizer to match your existing dataset"""
    return name.lower().replace(".", "").strip()

def main():
    print("--- Fetching Advanced NBA Features ---")
    
    if not os.path.exists(INPUT_PATH):
        print("Error: Run add_volume_stats.py first to generate the base file.")
        return
        
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} players.")

    # ---------------------------------------------------------
    # 1. Fetch Advanced Stats (PIE, Pace, Usage, Def Rating)
    # ---------------------------------------------------------
    print("\n[1/3] Fetching Advanced Stats...")
    try:
        adv_stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season=TARGET_SEASON,
            measure_type_detailed_defense='Advanced'
        ).get_data_frames()[0]
        
        # Keep only high-value columns
        adv_cols = ['PLAYER_NAME', 'OFF_RATING', 'DEF_RATING', 'PIE', 'USG_PCT', 'TS_PCT']
        df_adv = adv_stats[adv_cols].copy()
        
        # Merge
        df = pd.merge(df, df_adv, on='PLAYER_NAME', how='left', suffixes=('', '_new'))
        
        # Fill missing old cols if they exist, or create new ones
        for col in adv_cols:
            if col != 'PLAYER_NAME':
                if col + '_new' in df.columns:
                    df[col] = df[col + '_new'].fillna(df.get(col, 0))
                    df.drop(columns=[col + '_new'], inplace=True)
                
        print(f"  -> Added Advanced Stats for {len(df_adv)} players.")
        
    except Exception as e:
        print(f"  -> Error fetching Advanced Stats: {e}")

    # ---------------------------------------------------------
    # 2. Fetch Tracking Data (Speed & Distance)
    # ---------------------------------------------------------
    print("\n[2/3] Fetching Tracking Data (Speed/Distance)...")
    try:
        # PtMeasureType='SpeedDistance' gives us AVG_SPEED, DIST_MILES
        track_stats = leaguedashptstats.LeagueDashPtStats(
            season=TARGET_SEASON,
            pt_measure_type='SpeedDistance',
            player_or_team='Player'
        ).get_data_frames()[0]
        
        track_cols = ['PLAYER_NAME', 'AVG_SPEED', 'DIST_MILES']
        df_track = track_stats[track_cols].copy()
        
        # Merge
        df = pd.merge(df, df_track, on='PLAYER_NAME', how='left')
        print(f"  -> Added Tracking Stats for {len(df_track)} players.")

    except Exception as e:
        print(f"  -> Error fetching Tracking Data: {e}")

    # ---------------------------------------------------------
    # 3. Fetch Accolades (All-Star Selections) - SLOW but VALUABLE
    # ---------------------------------------------------------
    print("\n[3/3] Fetching Career Accolades (Iterative - This takes time)...")
    
    # We need to map Player Name -> Player ID to call the Awards endpoint
    nba_players = static_players.get_players()
    name_to_id = {normalize_name(p['full_name']): p['id'] for p in nba_players}
    
    all_star_counts = []
    
    # Optimization: Only fetch for players in our dataset
    unique_names = df['PLAYER_NAME'].unique()
    
    print(f"  -> Querying awards for {len(unique_names)} players (approx 2-3 mins)...")
    
    count = 0
    for name in unique_names:
        norm_name = normalize_name(name)
        pid = name_to_id.get(norm_name)
        
        n_all_star = 0
        if pid:
            try:
                # Add a tiny delay to be nice to the API
                time.sleep(0.3) 
                awards = playerawards.PlayerAwards(player_id=pid).get_data_frames()[0]
                
                # Count "All-Star" selections
                if not awards.empty:
                    n_all_star = len(awards[awards['DESCRIPTION'].str.contains('All-Star', case=False)])
            except:
                pass # If call fails, assume 0
        
        all_star_counts.append({'PLAYER_NAME': name, 'ALL_STAR_APPEARANCES': n_all_star})
        
        count += 1
        if count % 50 == 0:
            print(f"     ... processed {count} players")

    df_awards = pd.DataFrame(all_star_counts)
    df = pd.merge(df, df_awards, on='PLAYER_NAME', how='left')

    # 4. Save Final Super-Dataset
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSuccess! Enhanced dataset saved to: {OUTPUT_PATH}")
    print("New Columns: AVG_SPEED, DIST_MILES, ALL_STAR_APPEARANCES, + Advanced Stats")

if __name__ == "__main__":
    main()