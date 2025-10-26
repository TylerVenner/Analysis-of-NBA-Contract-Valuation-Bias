{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "4a3b1941-194f-44e6-b0fa-a2de7a2b74b7",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Saved full player stats to data/raw/full_player_stats.csv\n"
     ]
    }
   ],
   "source": [
    "# fetch_full_player_stats.py\n",
    "import pandas as pd\n",
    "from nba_api.stats.endpoints import leaguedashplayerstats\n",
    "from nba_api.stats.static import teams\n",
    "from project_config import SEASON\n",
    "from pathlib import Path\n",
    "\n",
    "# ---------------- Configuration ----------------\n",
    "SEASON = \"2024-25\"\n",
    "OUTPUT_FILE = Path(\"data/raw/full_player_stats.csv\")\n",
    "\n",
    "# ---------------- Functions ----------------\n",
    "def fetch_stats(season: str, measure_type: str) -> pd.DataFrame:\n",
    "    \"\"\"\n",
    "    Fetch NBA stats (Advanced or Base) for a season.\n",
    "    \"\"\"\n",
    "    stats = leaguedashplayerstats.LeagueDashPlayerStats(\n",
    "        season=season,\n",
    "        per_mode_detailed='PerGame',\n",
    "        measure_type_detailed_defense=measure_type\n",
    "    )\n",
    "    df = stats.get_data_frames()[0]\n",
    "    return df\n",
    "\n",
    "def build_full_dataset(season: str) -> pd.DataFrame:\n",
    "    # Fetch Advanced stats\n",
    "    df_advanced = fetch_stats(season, 'Advanced')\n",
    "\n",
    "    # Fetch Base stats\n",
    "    df_base = fetch_stats(season, 'Base')\n",
    "\n",
    "    # Merge Advanced and Base on PLAYER_ID, PLAYER_NAME, TEAM_ID\n",
    "    df_full = pd.merge(df_advanced, df_base, on=['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID'], how='outer')\n",
    "\n",
    "    # Optional: Fetch only the columns you listed\n",
    "    columns_needed = [\n",
    "        'PLAYER_ID', 'PLAYER_NAME', 'NICKNAME', 'TEAM_ID', 'TEAM_ABBREVIATION',\n",
    "        'AGE', 'GP', 'W', 'L', 'W_PCT', 'MIN', 'E_OFF_RATING', 'OFF_RATING',\n",
    "        'sp_work_OFF_RATING', 'E_DEF_RATING', 'DEF_RATING', 'sp_work_DEF_RATING',\n",
    "        'E_NET_RATING', 'NET_RATING', 'sp_work_NET_RATING', 'AST_PCT', 'AST_TO',\n",
    "        'AST_RATIO', 'OREB_PCT', 'DREB_PCT', 'REB_PCT', 'TM_TOV_PCT', 'E_TOV_PCT',\n",
    "        'EFG_PCT', 'TS_PCT', 'USG_PCT', 'E_USG_PCT', 'E_PACE', 'PACE', 'PACE_PER40',\n",
    "        'sp_work_PACE', 'PIE', 'POSS', 'FGM', 'FGA', 'FGM_PG', 'FGA_PG', 'FG_PCT',\n",
    "        'GP_RANK', 'W_RANK', 'L_RANK', 'W_PCT_RANK', 'MIN_RANK', 'E_OFF_RATING_RANK',\n",
    "        'OFF_RATING_RANK', 'sp_work_OFF_RATING_RANK', 'E_DEF_RATING_RANK',\n",
    "        'DEF_RATING_RANK', 'sp_work_DEF_RATING_RANK', 'E_NET_RATING_RANK',\n",
    "        'NET_RATING_RANK', 'sp_work_NET_RATING_RANK', 'AST_PCT_RANK', 'AST_TO_RANK',\n",
    "        'AST_RATIO_RANK', 'OREB_PCT_RANK', 'DREB_PCT_RANK', 'REB_PCT_RANK',\n",
    "        'TM_TOV_PCT_RANK', 'E_TOV_PCT_RANK', 'EFG_PCT_RANK', 'TS_PCT_RANK',\n",
    "        'USG_PCT_RANK', 'E_USG_PCT_RANK', 'E_PACE_RANK', 'PACE_RANK', 'sp_work_PACE_RANK',\n",
    "        'PIE_RANK', 'FGM_RANK', 'FGA_RANK', 'FGM_PG_RANK', 'FGA_PG_RANK', 'FG_PCT_RANK',\n",
    "        'TEAM_COUNT'\n",
    "    ]\n",
    "\n",
    "    # Keep only columns that exist in the DataFrame\n",
    "    columns_to_keep = [col for col in columns_needed if col in df_full.columns]\n",
    "    df_full = df_full[columns_to_keep]\n",
    "\n",
    "    return df_full\n",
    "\n",
    "# ---------------- Main Script ----------------\n",
    "def main():\n",
    "    df_full = build_full_dataset(SEASON)\n",
    "\n",
    "    # Ensure output directory exists\n",
    "    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)\n",
    "\n",
    "    # Save CSV\n",
    "    df_full.to_csv(OUTPUT_FILE, index=False)\n",
    "    print(f\"Saved full player stats to {OUTPUT_FILE}\")\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    main()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ef61dfa5-23ea-42b0-b6a9-ee80a1792e74",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
