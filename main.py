# main.py
from src.data_collection import get_nba_data, get_salary_data
from src.data_processing import merge_data
from src.analysis import clustering, salary_modeling, bias_analysis

def main():
    print("PHASE 0: Data Collection...")
    get_nba_data.fetch_player_data()
    get_salary_data.scrape_salary_data()

    print("PHASE 0: Data Processing...")
    processed_data = merge_data.create_merged_dataset()

    print("PHASE 1 & 2: Running Clustering...")
    data_with_roles = clustering.assign_player_roles(processed_data)

    print("PHASE 3: Running Salary Models...")
    data_with_residuals = salary_modeling.run_role_models(data_with_roles)

    print("PHASE 4: Running Bias Analysis...")
    bias_analysis.run_bias_model(data_with_residuals)

if __name__ == "__main__":
    main()