import pandas as pd
import time
import os
import unicodedata
import re
from pathlib import Path

from nba_api.stats.endpoints import (
    LeagueDashPtStats,          
    PlayerAwards,               
    SynergyPlayTypes,           
    LeagueDashPlayerClutch,     
    LeagueDashPtDefend          
)
from nba_api.stats.static import players as static_players

project_root = Path(__file__).resolve().parents[2]
INPUT_PATH = os.path.join(project_root, "data", "processed", "master_dataset_advanced.csv")
OUTPUT_PATH = os.path.join(project_root, "data", "processed", "master_dataset_advanced_v2.csv")
TARGET_SEASON = '2024-25'

def normalize_name(name):
    if pd.isna(name): return ""
    nfkd = unicodedata.normalize('NFD', str(name))
    ascii_name = "".join([c for c in nfkd if not unicodedata.combining(c)])
    clean = ascii_name.lower()
    clean = re.sub(r'\s+(jr\.?|sr\.?|ii|iii|iv)(\s|$)', '', clean)
    clean = re.sub(r"[.']", "", clean)
    return clean.strip()

def main():
        
    df = pd.read_csv(INPUT_PATH)
    df['name_norm'] = df['PLAYER_NAME'].apply(normalize_name)
    print(f"Loaded {len(df)} players.")

    # 1. SYNERGY PLAY TYPES (Iso, P&R)
    print("\n[1/4] Fetching Synergy Play Types...")
    play_types = [
        ('Isolation', 'ISO'), 
        ('PRBallHandler', 'PNR'), 
        ('Postup', 'POST')
    ]
    
    for pt_name, prefix in play_types:
        try:
            time.sleep(0.6) 
            syn = SynergyPlayTypes(
                season=TARGET_SEASON,
                play_type_nullable=pt_name,
                type_grouping_nullable='Offensive',
                per_mode_simple='Totals',
                player_or_team_abbreviation='P'
            ).get_data_frames()[0]
            
            syn['name_norm'] = syn['PLAYER_NAME'].apply(normalize_name)
            syn = syn[['name_norm', 'PTS', 'PPP']].rename(columns={
                'PTS': f'{prefix}_PTS',
                'PPP': f'{prefix}_PPP'
            })
            
            df = pd.merge(df, syn, on='name_norm', how='left')
            print(f"  -> Added {pt_name} Stats")
        except Exception as e: 
            print(f"  !! Error fetching {pt_name}: {e}")

    # 2. CLUTCH STATS
    print("\n[2/4] Fetching Clutch Stats...")
    try:
        clutch = LeagueDashPlayerClutch(
            season=TARGET_SEASON,
            per_mode_detailed='Totals'
        ).get_data_frames()[0]
        
        clutch['name_norm'] = clutch['PLAYER_NAME'].apply(normalize_name)
        clutch = clutch[['name_norm', 'PTS', 'GP']].rename(columns={
            'PTS': 'CLUTCH_PTS',
            'GP': 'CLUTCH_GP'
        })
        
        df = pd.merge(df, clutch, on='name_norm', how='left')
        print("  -> Added Clutch Stats")
    except Exception as e: print(f"  !! Error fetching Clutch: {e}")

    # 3. RIM PROTECTION
    print("\n[3/4] Fetching Rim Defense (< 6 FT)...")
    try:
        defense = LeagueDashPtDefend(
            season=TARGET_SEASON,
            defense_category='Less Than 6Ft',  # No space in parameter value
            per_mode_simple='Totals'
        ).get_data_frames()[0]
        
        defense['name_norm'] = defense['PLAYER_NAME'].apply(normalize_name)
        defense = defense[['name_norm', 'LT_06_PCT', 'FREQ']].rename(columns={
            'LT_06_PCT': 'RIM_DFG_PCT',
            'FREQ': 'RIM_CONTEST_FREQ'
        })
        
        df = pd.merge(df, defense, on='name_norm', how='left')
        print("  -> Added Rim Protection Stats")
    except Exception as e: print(f"  !! Error fetching Defense: {e}")

    # 4. AWARDS
    print("\n[4/4] Fetching Career Awards...")
    nba_players = static_players.get_players()
    name_to_id = {normalize_name(p['full_name']): p['id'] for p in nba_players}
    unique_names = df['name_norm'].unique()
    awards_data = []
    
    count = 0
    for name in unique_names:
        pid = name_to_id.get(name)
        n_all_nba = 0
        n_all_star = 0
        if pid:
            try:
                time.sleep(0.3)
                awd = PlayerAwards(player_id=pid).get_data_frames()[0]
                if not awd.empty:
                    n_all_nba = len(awd[awd['DESCRIPTION'].str.contains('All-NBA', case=False)])
                    n_all_star = len(awd[awd['DESCRIPTION'].str.contains('All-Star', case=False)])
            except: pass
        awards_data.append({'name_norm': name, 'ALL_NBA_COUNT': n_all_nba, 'ALL_STAR_COUNT': n_all_star})
        count += 1
        if count % 50 == 0: print(f"     ... processed {count}")

    df_awards = pd.DataFrame(awards_data)
    df = pd.merge(df, df_awards, on='name_norm', how='left')
    
    df.drop(columns=['name_norm'], inplace=True)
    df = df.fillna(0)

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nDATASET SAVED: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()