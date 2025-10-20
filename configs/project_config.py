from pathlib import Path

# ---
# GLOBAL CONSTANTS
# ---
SEASON = "2024-25"  # The season we are analyzing

# ---
# FILE PATHS (relative to the project root)
# ---
# This line automatically finds the root of our project
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Define the paths to our data folders
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed"

# ---
# "CONTRACT" FILENAMES
# These are the exact inputs/outputs for each module
# ---

# Module 1a (get_nba_stats.py) Output
RAW_STATS_FILE = RAW_DATA_PATH / "raw_player_stats.csv"

# Module 1b (get_context_data.py) Output
RAW_CONTEXT_FILE = RAW_DATA_PATH / "raw_player_context.csv"

# Module 2 (get_salary_data.py) Output
RAW_SALARY_FILE = RAW_DATA_PATH / "raw_player_salaries.csv"

# Module 3 (merge_data.py) Output
# This is also the main INPUT for all future modules (EDA, Clustering, etc.)
PROCESSED_MERGED_FILE = PROCESSED_DATA_PATH / "merged_player_data_v1.csv"