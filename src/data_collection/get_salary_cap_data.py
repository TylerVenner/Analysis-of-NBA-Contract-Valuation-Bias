import requests
import pandas as pd
from pathlib import Path
import io

# CONFIG
YEAR = "2024"

PROJECT_ROOT = Path().resolve().parent.parent
RAW_SALARY_FILE = PROJECT_ROOT / "data" / "raw" / "raw_team_caps.csv"

HEADERS = {
    #"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# Core Functions
def fetch_team_cap_table(season_year: str, headers: dict) -> pd.DataFrame | None:
    """
    Fetch salary cap table for all teams from Spotrac.
    Returns a pandas DataFrame if successful, or None if an error occurs.
    """
    url = f"https://www.spotrac.com/nba/cap/_/year/{season_year}"
    print(f"Fetching {url} ...")

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        tables = pd.read_html(io.StringIO(response.text))

        if tables:
            df = tables[0].copy()
        else:
            print(f"No tables found.")
        
        return df
            
    except requests.exceptions.RequestException as e:
        print(f"Error fetching page: {e}")

def clean_salary_cap_dataframe(raw_df: pd.DataFrame, rank_col: str = 'rank', team_col: str = 'team') -> pd.DataFrame:
    """
    Cleans the salary cap data.
    - Rename columns
    - Standardize team name
    - Remove unneeded rows and columns
    """
    df = raw_df.copy()
    
    # Rename columns
    if len(df.columns) == 10:
        df.columns = ['rank', 'team', 'record', 'active_players', 'avg_team_age', 
                                'total_cap_used', 'remaining_cap_space', 'active_cap', 
                                'active_top_3', 'dead_cap']
    else:
        raise KeyError(f"Incorrect number of columns. Expected 10, found {len(df.columns)} columns")
    
    # Select relevant columns
    if rank_col not in df.columns or team_col not in df.columns:
        raise KeyError(f"Expected columns '{rank_col}' or '{team_col}' not found.")
    
    # Drop rank column
    cleaned_df = df.drop('rank', axis = 1)
        
    # Clean team column - extract only the first part before space
    cleaned_df['team'] = cleaned_df['team'].str.split().str[0]

    # Drop last two rows
    cleaned_df = cleaned_df.iloc[:-2]

    return cleaned_df

def save_dataframe_to_csv(df: pd.DataFrame, filepath: Path):
    """
    Saves the given DataFrame to a CSV file.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"Saved DataFrame to {filepath}")

# Main Pipeline
def main():
    print("Starting NBA Team Cap Scraper ...")

    salary_cap_df = fetch_team_cap_table(YEAR, HEADERS)
    if salary_cap_df is None:
        print("No data collected. Exiting.")
        return

    try:
        cleaned_df = clean_salary_cap_dataframe(salary_cap_df)
    except KeyError as e:
        print(f"Column mismatch error: {e}")
        print("Inspect columns manually.")
        return

    save_dataframe_to_csv(cleaned_df, RAW_SALARY_FILE)
    print("All done!")

if __name__ == "__main__":
    main()