import streamlit as st
import numpy as np
import pandas as pd
from scripts.main import run_pipeline

Y_COL = "Salary"

@st.cache_data
def load_data():
    df_clean, residuals_Y, residuals_Z, final_ols_results = run_pipeline()

    if "log_salary" not in df_clean.columns:
        df_clean["log_salary"] = np.log(df_clean[Y_COL])

    return df_clean, residuals_Y, residuals_Z, final_ols_results