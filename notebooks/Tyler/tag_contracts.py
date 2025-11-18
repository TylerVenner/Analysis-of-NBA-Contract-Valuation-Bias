import pandas as pd
import numpy as np
import sys
import os  # <-- ADDED THIS

# --- FILE PATHS (USER: PLEASE EDIT THESE) ---
# 1. Paste the FULL path to the file you created in the last step
INPUT_FILE_PATH = r"C:\Users\tyler\School\Learn Statistics\STA 160\Project\notebooks\Tyler\master_dataset_with_yos.csv"

# 2. Paste the FULL path for where you want to SAVE the new file
#    (It's easiest to use the same folder, just change the filename)
OUTPUT_FILE_PATH = r'C:\Users\tyler\School\Learn Statistics\STA 160\Project\notebooks\Tyler\master_dataset_with_contracts.csv'
# ----------------------------------------------


# --- 1. DEFINE 2024-2025 CBA RULES ---
# All values are based on the user-confirmed rules.

SALARY_CAP = 141_000_000

# Max salaries are based on YOS tiers
MAX_SALARY_TIERS = {
    '0-6': 35_147_000,  # 25% of cap for 0-6 YOS
    '7-9': 42_176_400,  # 30% of cap for 7-9 YOS
    '10+': 49_205_800   # 35% of cap for 10+ YOS
}

# Veteran minimums are a dictionary mapping YOS to salary
VET_MIN_SALARY = {
    0: 1_157_153,
    1: 1_862_265,
    2: 2_087_519,
    3: 2_162_606,
    4: 2_237_691,
    5: 2_425_403,
    6: 2_613_120,
    7: 2_800_834,
    8: 2_988_550,
    9: 3_003_427,
    10: 3_303_771 # This applies to 10+ YOS
}

# We'll use a 2% tolerance to catch minor variations in contracts
# (e.g., from 8% raises, incentives, or agent fees)
TOLERANCE = 0.02

# --- 2. HELPER FUNCTIONS ---

def is_near(salary, rule_value):
    """Checks if a salary is within TOLERANCE of a rule value."""
    if not rule_value:  # Handle cases where rule_value might be None
        return False
    lower_bound = rule_value * (1 - TOLERANCE)
    upper_bound = rule_value * (1 + TOLERANCE)
    return (salary >= lower_bound) and (salary <= upper_bound)

def get_max_salary(yos):
    """Returns the correct max salary tier for a given YOS."""
    if yos <= 6:
        return MAX_SALARY_TIERS['0-6']
    elif yos <= 9:
        return MAX_SALARY_TIERS['7-9']
    else:
        return MAX_SALARY_TIERS['10+']

def get_min_salary(yos):
    """Returns the correct min salary for a given YOS."""
    # Use 10 as the key for any YOS 10 or greater
    lookup_yos = min(yos, 10)
    return VET_MIN_SALARY.get(lookup_yos)

# --- 3. THE MASTER TAGGER FUNCTION ---

def assign_contract_type(row):
    """
    Assigns a contract type to a player based on their salary and CBA rules.
    The order of these rules is critical for correct categorization.
    """
    salary = row['Salary']
    yos = row['YOS']
    draft_round = row['DRAFT_ROUND']
    
    # RULE 1: ROOKIE SCALE
    # A 1st round pick (1.0) with 0, 1, 2, or 3 YOS is on a rookie scale contract.
    # We add a check to ensure they aren't on a massive extension that
    # already classifies as a Max Contract.
    max_for_rookie_tier = MAX_SALARY_TIERS['0-6']
    if draft_round == 1.0 and yos <= 3 and salary < (max_for_rookie_tier * (1 - TOLERANCE)):
        return 'Rookie_Scale'

    # RULE 2: MAX CONTRACT
    # Check if their salary is at or *above* (for Supermax) the max for their tier.
    # We use a *lower* bound check for this.
    max_for_tier = get_max_salary(yos)
    if salary >= max_for_tier * (1 - TOLERANCE):
        return 'Max_Contract'
        
    # RULE 3: VETERAN MINIMUM
    min_for_tier = get_min_salary(yos)
    if is_near(salary, min_for_tier):
        return 'Vet_Minimum'

    # RULE 4: FREE MARKET
    # If a player is not on a rookie, max, or min contract,
    # they are by definition a "Free_Market" player.
    return 'Free_Market'

# --- 4. MAIN EXECUTION ---

def process_contracts(input_file, output_file):  # <-- EDITED THIS
    """
    Loads the dataset, applies the contract tagger, and saves the new file.
    """
    print(f"Loading '{input_file}'...")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"ERROR: Input file not found: '{input_file}'")
        print("Please make sure you have run the previous feature engineering script first.")
        return
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")
        return
        
    print("Applying contract tagging rules to all players...")
    # Apply the tagger function to every row
    df['Contract_Type'] = df.apply(assign_contract_type, axis=1)

    # Save the final dataset
    df.to_csv(output_file, index=False)
    print(f"\nSuccessfully created '{output_file}'")
    
    # --- 5. Show Results ---
    print("\n--- Contract Type Breakdown ---")
    print(df['Contract_Type'].value_counts())
    
    print("\n--- Examples of each type ---")
    for contract_type in df['Contract_Type'].unique():
        print(f"\n--- {contract_type} Examples ---")
        # Show relevant columns for validation
        print(df[df['Contract_Type'] == contract_type][['PLAYER_NAME', 'Salary', 'YOS', 'DRAFT_ROUND', 'Contract_Type']].head(5))

if __name__ == "__main__":
    # --- EDITED THIS SECTION ---
    # We now use the hardcoded paths from the top of the script
    if not os.path.exists(INPUT_FILE_PATH):
        print(f"ERROR: Cannot find input file at:")
        print(f"{INPUT_FILE_PATH}")
        print("Please edit the 'INPUT_FILE_PATH' variable at the top of the script.")
    else:
        print("Starting contract tagging process...")
        process_contracts(input_file=INPUT_FILE_PATH, output_file=OUTPUT_FILE_PATH)