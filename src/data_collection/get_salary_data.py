import requests
import pandas as pd
from pathlib import Path
import io
import time

# CONFIG
TEAM_SLUGS = [
    'atlanta-hawks', 'boston-celtics', 'brooklyn-nets', 'charlotte-hornets', 
    'chicago-bulls', 'cleveland-cavaliers', 'dallas-mavericks', 'denver-nuggets',
    'detroit-pistons', 'golden-state-warriors', 'houston-rockets', 'indiana-pacers',
    'la-clippers', 'los-angeles-lakers', 'memphis-grizzlies', 
    'miami-heat', 'milwaukee-bucks', 'minnesota-timberwolves', 'new-orleans-pelicans',
    'new-york-knicks', 'oklahoma-city-thunder', 'orlando-magic', 
    'philadelphia-76ers', 'phoenix-suns', 'portland-trail-blazers', 
    'sacramento-kings', 'san-antonio-spurs', 'toronto-raptors', 'utah-jazz', 
    'washington-wizards'
]
SEASON = "2024-25"
SEASON_START_YEAR = SEASON.split('-')[0]

PROJECT_ROOT = Path().resolve().parent.parent
RAW_SALARY_FILE = PROJECT_ROOT / "data" / "raw" / "raw_player_salaries.csv"

HEADERS = {
    #"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# Core Functions
def fetch_team_salary_table(team_slug: str, season_year: str, headers: dict) -> pd.DataFrame | None:
    """
    Fetch salary table for a given NBA team from Spotrac.
    Returns a pandas DataFrame if successful, or None if an error occurs.
    """
    url = f"https://www.spotrac.com/nba/{team_slug}/payroll/{season_year}/"
    print(f"Fetching {url} ...")

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        tables = pd.read_html(io.StringIO(response.text))
        if not tables:
            print(f"No tables found for {team_slug}")
            return None

        df = tables[0].copy()
        # Find the player column dynamically
        player_col = next((col for col in df.columns if str(col).lower().startswith('player')), None)
        if not player_col:
            print(f"Could not find player column for {team_slug}")
            return None

        df.rename(columns={player_col: 'Player_Name'}, inplace=True)
        df['team_slug'] = team_slug
        return df

    except requests.exceptions.RequestException as e:
        print(f"Error fetching page for {team_slug}: {e}")
        return None

def fetch_all_teams_data(team_slugs: list[str], season_year: str, headers: dict, delay: float = 1.0) -> pd.DataFrame | None:
    """
    Fetch payroll data for all teams and combine into a single DataFrame.
    """
    all_team_dfs = []
    for slug in team_slugs:
        df = fetch_team_salary_table(slug, season_year, headers)
        if df is not None:
            all_team_dfs.append(df)
        time.sleep(delay)  # Be polite to Spotrac's servers

    if not all_team_dfs:
        print("No team data fetched.")
        return None

    combined_df = pd.concat(all_team_dfs, ignore_index=True)
    print(f"Combined {len(all_team_dfs)} teams, total rows: {len(combined_df)}")
    return combined_df

def clean_salary_dataframe(raw_df: pd.DataFrame, player_col: str = 'Player_Name', salary_col: str = 'Cap Hit') -> pd.DataFrame:
    """
    Cleans and normalizes the combined salary data.
    - Flattens multi-index columns if needed
    - Selects player/salary columns
    - Removes duplicates
    """
    df = raw_df.copy()

    # Flatten MultiIndex columns if necessary
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip() for col in df.columns.values]

    print(f"Available columns: {df.columns.tolist()}")

    # Select relevant columns
    if player_col not in df.columns or salary_col not in df.columns:
        raise KeyError(f"Expected columns '{player_col}' or '{salary_col}' not found.")

    cleaned_df = df[[player_col, salary_col]].copy()
    cleaned_df.rename(columns={player_col: 'Player_Name', salary_col: 'Salary'}, inplace=True)

    cleaned_df.drop_duplicates(subset=['Player_Name'], keep='first', inplace=True)
    print(f"Cleaned DataFrame: {len(cleaned_df)} unique players.")
    return cleaned_df

def remove_two_way_players(cleaned_df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes two-way contract players from dataset.
    """
    if 'Salary' not in cleaned_df.columns:
        raise KeyError("Expected salary column not found.")
    
    filtered_df = cleaned_df[~cleaned_df['Salary'].str.contains('two-way', case=False)].copy()

    removed_count = len(cleaned_df) - len(filtered_df)
    print(f"Removed {removed_count} two-way contract players. {len(filtered_df)} unique players remaining.")

    return filtered_df

def save_dataframe_to_csv(df: pd.DataFrame, filepath: Path):
    """
    Saves the given DataFrame to a CSV file.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"Saved DataFrame to {filepath}")

# Main Pipeline
def main():
    print("Starting NBA Salary Scraper ...")

    combined_df = fetch_all_teams_data(TEAM_SLUGS, SEASON_START_YEAR, HEADERS)
    if combined_df is None:
        print("No data collected. Exiting.")
        return

    try:
        cleaned_df = clean_salary_dataframe(combined_df)
    except KeyError as e:
        print(f"Column mismatch error: {e}")
        print("Inspect combined_df columns manually.")
        return
    
    final_df = remove_two_way_players(cleaned_df)

    save_dataframe_to_csv(final_df, RAW_SALARY_FILE)
    print("All done!")

if __name__ == "__main__":
    main()