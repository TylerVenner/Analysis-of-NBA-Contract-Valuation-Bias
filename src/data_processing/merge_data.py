import pandas as pd
from pathlib import Path
from configs.project_config import RAW_STATS_FILE, RAW_CONTEXT_FILE, RAW_SALARY_FILE, PROCESSED_MERGED_FILE
from src.data_processing.cleaning_helpers import standardize_player_name

def main():
    # >Load datasets<
    
    df_stats = pd.read_csv(RAW_STATS_FILE)
    df_context = pd.read_csv(RAW_CONTEXT_FILE)
    df_salary = pd.read_csv(RAW_SALARY_FILE)
   

    print("Merging stats and context (on PLAYER_ID)...")
    df_api = pd.merge(df_stats, df_context, on="PLAYER_ID", how="inner")

    # --- Standardize player names before merging salary ---
    print("Standardizing player names for salary merge...")
    df_api["merge_key"] = df_api["PLAYER_NAME"].apply(standardize_player_name)
    df_salary["merge_key"] = df_salary["Player_Name"].apply(standardize_player_name)

    # --- Merge salary data on standardized names ---
    print("Merging salary data (on standardized names)...")
    df_merged = pd.merge(df_api, df_salary, on="merge_key", how="left", validate="m:1")

    # --- Clean Salary column ---
    print("Cleaning salary values...")
    df_merged["Salary"] = (
        df_merged["Salary"]
        .astype(str)
        .str.replace(r"[^\d]", "", regex=True)
        .replace("", None)
        .astype(float)
    )

    # --- Calculate AGE ---
    print("Calculating player ages...")
    df_merged["BIRTHDATE"] = pd.to_datetime(df_merged["BIRTHDATE"], errors="coerce")
    df_merged["AGE"] = df_merged["BIRTHDATE"].apply(lambda x: 2024 - x.year if pd.notnull(x) else None)

    # --- Log duplicates (for same names) ---
    print("\nChecking for duplicate merge keys...")
    dupes = df_merged[df_merged.duplicated(subset=["merge_key"], keep=False)]
    if not dupes.empty:
        print(f"⚠️ Found {len(dupes)} duplicate player names — please review:")
        print(dupes[["PLAYER_NAME", "merge_key"]].head(10))

    # --- Log missing salaries ---
    print("\nChecking for players missing salary data...")
    missing_salaries = df_merged[df_merged["Salary"].isna()]
    print(f"⚠️ Players missing salary data: {len(missing_salaries)}")
    if not missing_salaries.empty:
        print(missing_salaries[["PLAYER_NAME", "merge_key"]].head(10))

    # --- Save cleaned output ---
    print(f"\nSaving cleaned dataset to {PROCESSED_MERGED_FILE}")
    Path(PROCESSED_MERGED_FILE).parent.mkdir(parents=True, exist_ok=True)
    df_merged.to_csv(PROCESSED_MERGED_FILE, index=False)
    print("✅ Merge and cleaning complete.")

if __name__ == "__main__":
    main()