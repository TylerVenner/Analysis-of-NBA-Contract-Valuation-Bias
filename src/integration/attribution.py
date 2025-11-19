import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class BiasAttributor:
    """
    The Bridge Module: Converts DML econometric outputs into a 
    Latent Space Attribution Matrix (L).
    """
    
    def __init__(self, gamma_coefficients: pd.Series, residuals_Z: pd.DataFrame):
        """
        Args:
            gamma_coefficients (pd.Series): The debiased coefficients (prices) from Final OLS.
                                            Index should be bias factor names.
            residuals_Z (pd.DataFrame): The out-of-sample residuals for bias factors.
                                        Index should be Player IDs/Names.
        """
        self.gamma = gamma_coefficients
        self.residuals_Z = residuals_Z
        
        # Align keys (ensure we only use factors present in both)
        common_factors = self.gamma.index.intersection(self.residuals_Z.columns)
        self.gamma = self.gamma[common_factors]
        self.residuals_Z = self.residuals_Z[common_factors]
        
        print(f"Initialized Attributor with {len(common_factors)} bias factors.")

    def get_attribution_matrix(self, normalize=True):
        """
        Calculates the raw attribution: L = gamma * epsilon_Z
        
        Returns:
            pd.DataFrame: The N x M attribution matrix.
        """
        # Element-wise multiplication of column vector (residuals) by scalar (coefficient)
        # We broadcast the gamma coefficients across the rows
        L_raw = self.residuals_Z.multiply(self.gamma, axis=1)
        
        if normalize:
            return self._min_max_scale(L_raw)
        return L_raw

    def _min_max_scale(self, df):
        """
        Scales the attribution matrix to [0, 1] range.
        Also returns metadata about the scaling for interpretation.
        """
        scaler = MinMaxScaler(feature_range=(0, 1))
        
        # We scale the ENTIRE matrix globally to preserve relative magnitude 
        # between different bias factors (e.g. Age vs Draft).
        # Scaling column-by-column would destroy the "Price" information.
        
        raw_values = df.values.flatten().reshape(-1, 1)
        scaler.fit(raw_values)
        
        scaled_values = scaler.transform(df.values.flatten().reshape(-1, 1))
        df_scaled = pd.DataFrame(
            scaled_values.reshape(df.shape),
            index=df.index,
            columns=df.columns
        )
        
        self.scaling_meta = {
            'min': scaler.data_min_[0],
            'max': scaler.data_max_[0],
            'scale': scaler.scale_[0]
        }
        
        return df_scaled

    def get_player_metadata(self, df_original, contract_col='Contract_Type'):
        """
        Retrieves metadata for the players in the residual matrix to help 
        with visualization (e.g., coloring by Contract Type).
        """
        # Filter original df to match the players we actually have residuals for
        matched_meta = df_original.loc[self.residuals_Z.index]
        
        if contract_col not in matched_meta.columns:
            print(f"Warning: {contract_col} not found. Defaulting to 'Unknown'.")
            return pd.Series('Unknown', index=matched_meta.index)
            
        return matched_meta[contract_col]