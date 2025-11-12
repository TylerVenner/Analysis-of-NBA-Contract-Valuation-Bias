import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import RegressionResultsWrapper

def run_final_ols(residuals_Y: pd.Series, 
                    residuals_Z: pd.DataFrame) -> RegressionResultsWrapper:
    """
    Runs the final debiased OLS regression on the out-of-sample residuals,
    as specified in the DML pipeline (Module 4).

    This model estimates:  epsilon_Y ~ epsilon_Z
    
    This corresponds to the "All-at-Once" approach (Section 6.1)
    from the methodology document.

    Args:
        residuals_Y: The out-of-sample residuals for Y (epsilon_Y).
        residuals_Z: The out-of-sample residuals for Z (epsilon_Z).

    Returns:
        The statsmodels OLS results object for the final regression.
    """
    
    print("Running final OLS regression on residuals...")
    
    # 1. Add a constant (intercept) to the Z residuals
    # This is crucial for the OLS model (the gamma_0 intercept)
    Z_with_const = sm.add_constant(residuals_Z, prepend=True)
    
    # 2. Run the final OLS model: epsilon_Y ~ const + epsilon_Z
    final_ols_model = sm.OLS(residuals_Y, Z_with_const)
    
    # 3. Fit the model
    final_ols_results = final_ols_model.fit()
    
    print("Final OLS regression complete.")
    
    # 4. Return the results object (which contains .summary(), .params, etc.)
    return final_ols_results