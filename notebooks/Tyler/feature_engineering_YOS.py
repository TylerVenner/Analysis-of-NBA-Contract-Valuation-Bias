import pandas as pd
import numpy as np

print("Starting feature engineering script...")

try:
    # 1. Load the dataset
    df = pd.read_csv(r"C:\Users\tyler\School\Learn Statistics\STA 160\Project\notebooks\Tyler\master_dataset_cleaned.csv")
    print("Loaded 'master_dataset_cleaned.csv'.")

    # --- 2. Create Age Column ---
    # Convert BIRTHDATE to datetime objects
    # We parse dates to correctly handle the format
    df['BIRTHDATE'] = pd.to_datetime(df['BIRTHDATE'])
    
    # Calculate Age relative to the start of the 2024-2025 season (using 2024)
    current_year = 2024
    df['Age'] = current_year - df['BIRTHDATE'].dt.year
    print("Created 'Age' column.")

    # --- 3. Clean Draft-Related Columns ---
    # Convert draft columns to numeric, setting 'Undrafted' and other non-numeric values to NaN
    df['DRAFT_YEAR'] = pd.to_numeric(df['DRAFT_YEAR'], errors='coerce')
    df['DRAFT_ROUND'] = pd.to_numeric(df['DRAFT_ROUND'], errors='coerce')
    df['DRAFT_NUMBER'] = pd.to_numeric(df['DRAFT_NUMBER'], errors='coerce')
    print("Cleaned draft columns to numeric, 'Undrafted' is now NaN.")

    # --- 4. Create YOS (Years of Service) Column for Drafted Players ---
    # First, calculate YOS for drafted players. Undrafted will be NaN.
    df['YOS'] = current_year - df['DRAFT_YEAR']
    print("Calculated 'YOS' for drafted players.")

    # --- 5. Apply User's Fill Logic for Undrafted Players ---
    # Per your request: "infer their YOS by the player above them" -> ffill()
    # This works because the file is sorted by DRAFT_YEAR.
    df['YOS'] = df['YOS'].ffill()
    print("Forward-filled 'YOS' for undrafted players.")
    
    # "if that player also is undrafted, then you find the i - 1 and so on"
    # This implies we should also back-fill to catch any NaNs at the top of the file.
    df['YOS'] = df['YOS'].bfill()
    print("Backward-filled any remaining 'YOS' NaNs.")

    # --- 6. Finalize YOS Column ---
    # Ensure YOS is an integer
    df['YOS'] = df['YOS'].astype(int)
    print("Converted 'YOS' to integer.")

    # --- 7. Save the New DataFrame ---
    df.to_csv(r'C:\Users\tyler\School\Learn Statistics\STA 160\Project\notebooks\Tyler\master_dataset_with_yos.csv', index=False)
    print("Successfully created and saved 'master_dataset_with_yos.csv'.")

    # --- 8. Display Results ---
    print("\nFirst 15 rows of the new data (showing relevant columns):")
    print(df[['PLAYER_NAME', 'BIRTHDATE', 'Age', 'DRAFT_YEAR', 'DRAFT_ROUND', 'DRAFT_NUMBER', 'YOS', 'Salary']].head(15))

except FileNotFoundError:
    print("Error: 'master_dataset_cleaned.csv' not found.")
except Exception as e:
    print(f"An error occurred: {e}")