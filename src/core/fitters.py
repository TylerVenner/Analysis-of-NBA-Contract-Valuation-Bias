import numpy as np
from functools import partial

# --- Core Scientific Computing ---
from sklearn.decomposition import PCA
import jax
import jax.numpy as jnp
import optax
from jax.scipy.stats import norm

from .utils import update_step

class BiasMapFitter:
    """
    A JAX-powered engine for mapping the latent structure of economic bias.
    
    This fitter uses Maximum Likelihood Estimation (MLE) with a single 
    Global Error Sigma to robustly recover the geometric relationships 
    between Players and Bias Factors from the Attribution Matrix (L).
    """
    
    def __init__(self, n_dimensions=2, n_cycles=10, alt_steps=200, polish_steps=5000, 
                 learning_rate=0.01, convergence_tol=1e-7):
        """
        Args:
            n_dimensions (int): Embedding dimension (2 or 3).
            n_cycles (int): Number of Alternating Optimization cycles.
            alt_steps (int): Gradient steps per alternating phase.
            polish_steps (int): Max steps for final simultaneous optimization.
            learning_rate (float): Learning rate for the Adam optimizer.
            convergence_tol (float): Loss change threshold for early stopping.
        """
        if n_dimensions not in [2, 3]:
            raise ValueError("n_dimensions must be 2 or 3.")
            
        self.n_dimensions = n_dimensions
        self.n_cycles = n_cycles
        self.alt_steps = alt_steps
        self.polish_steps = polish_steps
        self.learning_rate = learning_rate
        self.convergence_tol = convergence_tol
        
        # Learned Parameters
        self.player_coords = None   # The 'Consumer' Ideal Points (C)
        self.factor_coords = None   # The 'Product' Locations (P)
        self.player_lvars = None    # Log-variance for players (uncertainty)
        self.factor_lvars = None    # Log-variance for factors (uncertainty)
        self.global_sigma = None    # The single global noise parameter
        self.final_loss = None

    def _unpack_params(self, params, n_players, n_factors, d):
        """Helper to slice the flat parameter vector into matrices."""
        # Param vector structure: [Players | Factors | PlayerVars | FactorVars | GlobalSigma]
        c_end = n_players * d
        p_end = c_end + n_factors * d
        vc_end = p_end + n_players * d
        vp_end = vc_end + n_factors * d
        
        player_coords = params[:c_end].reshape((n_players, d))
        factor_coords = params[c_end:p_end].reshape((n_factors, d))
        player_lvars = params[p_end:vc_end].reshape((n_players, d))
        factor_lvars = params[vc_end:vp_end].reshape((n_factors, d))
        log_sigma = params[-1] # Last element is the scalar global sigma
        
        return player_coords, factor_coords, player_lvars, factor_lvars, log_sigma

    def objective_function(self, params, attribution_matrix, nan_mask, min_val, max_val, 
                           n_players, n_factors, n_dimensions):
        """
        The Negative Log-Likelihood (NLL) objective function.
        """
        (p_coords, f_coords, 
         p_lvars, f_lvars, 
         log_sigma) = self._unpack_params(params, n_players, n_factors, n_dimensions)
        
        # 1. Prepare Tensors for Broadcasting
        # Shape: (n_players, 1, d) and (1, n_factors, d)
        p_coords_exp = p_coords[:, None, :]
        f_coords_exp = f_coords[None, :, :]
        
        # Variances must be positive, so we exponentiate the log-vars
        p_vars_exp = jnp.exp(p_lvars)[:, None, :]
        f_vars_exp = jnp.exp(f_lvars)[None, :, :]
        
        # 2. Compute Latent Geometry (Closed-Form Gaussian Integral)
        # Difference in means
        mu_tensor = p_coords_exp - f_coords_exp 
        # Sum of variances (Convolution of clouds)
        V_diag_tensor = p_vars_exp + f_vars_exp 
        # Precision scaling matrix (I + 2V)
        M_diag_tensor = 1 + 2 * V_diag_tensor 
        
        # 3. Calculate Predicted Attribution (Unscaled [0,1] probability)
        # Log-Determinant term (Penalty for uncertainty)
        log_det_term = -0.5 * jnp.sum(jnp.log(M_diag_tensor), axis=2)
        # Quadratic Distance term (Penalty for mismatch)
        exponent_tensor = -jnp.sum(mu_tensor**2 / M_diag_tensor, axis=2)
        
        # Combine in log-space for stability, then exponentiate
        log_pred = log_det_term + exponent_tensor
        predicted_prob = jnp.exp(jnp.clip(log_pred, -700, 700))
        
        # 4. Rescale Prediction to Data Range
        # This maps the [0,1] probability to the actual attribution values (e.g. $ values)
        scaled_prediction = min_val + (max_val - min_val) * predicted_prob
        
        # 5. Compute NLL (Gaussian Error)
        # We use the single learned global sigma
        error_sigma = jax.nn.softplus(log_sigma) + 1e-6
        
        # Calculate log-likelihood of the observed data given the prediction
        log_probs = norm.logpdf(attribution_matrix, loc=scaled_prediction, scale=error_sigma)
        
        # Mask missing data (if any)
        masked_log_probs = jnp.where(nan_mask, log_probs, 0.0)
        
        # Return Negative Log Likelihood (to be minimized)
        return -jnp.sum(masked_log_probs)

    def _initialize_with_pca(self, data):
        """Uses PCA to find a smart starting configuration."""
        n_players, n_factors = data.shape
        d = self.n_dimensions
        
        # Impute NaNs with column means for PCA initialization
        col_means = np.nanmean(data, axis=0)
        imputed = np.where(np.isnan(data), col_means, data)
        
        pca = PCA(n_components=d)
        init_p_coords = pca.fit_transform(imputed)
        init_f_coords = pca.components_.T * np.sqrt(pca.explained_variance_)

        # Normalize to avoid scale issues at start
        init_p_coords /= (np.std(init_p_coords) + 1e-9)
        init_f_coords /= (np.std(init_f_coords) + 1e-9)
        
        # Initialize variances to be small (log(-2.3) approx 0.1)
        init_p_lvars = np.full((n_players, d), -2.3)
        init_f_lvars = np.full((n_factors, d), -2.3)
        
        # Initialize global sigma (log(0) = 0.0 -> sigma=softplus(0) approx 0.7)
        init_log_sigma = np.array([0.0])
        
        return np.concatenate([
            init_p_coords.flatten(), init_f_coords.flatten(),
            init_p_lvars.flatten(), init_f_lvars.flatten(),
            init_log_sigma
        ])

    def _get_masks(self, n_players, n_factors, d):
        """Creates gradient masks for Alternating Optimization."""
        # Mask structure must match params vector:
        # [Players(1), Factors(0), PlayerVars(1), FactorVars(0), Sigma(1)]
        
        player_mask = jnp.concatenate([
            jnp.ones(n_players * d),   # Update Players
            jnp.zeros(n_factors * d),  # Freeze Factors
            jnp.ones(n_players * d),   # Update Player Vars
            jnp.zeros(n_factors * d),  # Freeze Factor Vars
            jnp.ones(1)                # Update Sigma
        ])
        
        factor_mask = jnp.concatenate([
            jnp.zeros(n_players * d),  # Freeze Players
            jnp.ones(n_factors * d),   # Update Factors
            jnp.zeros(n_players * d),  # Freeze Player Vars
            jnp.ones(n_factors * d),   # Update Factor Vars
            jnp.ones(1)                # Update Sigma
        ])
        return player_mask, factor_mask

    def fit(self, attribution_matrix):
        """
        Main fitting routine.
        attribution_matrix: The n x m matrix L from the DML stage.
        """
        n_players, n_factors = attribution_matrix.shape
        d = self.n_dimensions
        
        print(f"--- Initializing Map ({d}D) ---")
        params = jnp.array(self._initialize_with_pca(attribution_matrix))
        
        # Data Prep
        nan_mask = jnp.array(~np.isnan(attribution_matrix))
        data_filled = jnp.nan_to_num(attribution_matrix)
        min_val, max_val = np.nanmin(attribution_matrix), np.nanmax(attribution_matrix)
        
        # Partial binding for JAX
        obj_fn = partial(
            self.objective_function, 
            attribution_matrix=data_filled, nan_mask=nan_mask,
            min_val=min_val, max_val=max_val,
            n_players=n_players, n_factors=n_factors, n_dimensions=d
        )
        
        # JAX Transformations
        value_and_grad_fn = jax.value_and_grad(obj_fn)
        optimizer = optax.adam(self.learning_rate)
        opt_state = optimizer.init(params)
        
        # Masks for Alternating Opt
        p_mask, f_mask = self._get_masks(n_players, n_factors, d)
        
        print(f"--- Stage 1: Alternating Optimization ({self.n_cycles} cycles) ---")
        for i in range(self.n_cycles):
            # Update Players
            for _ in range(self.alt_steps):
                params, opt_state, _ = update_step(params, opt_state, value_and_grad_fn, optimizer, p_mask)
            # Update Factors
            for _ in range(self.alt_steps):
                params, opt_state, _ = update_step(params, opt_state, value_and_grad_fn, optimizer, f_mask)
            print(f"  Cycle {i+1}/{self.n_cycles} complete.")

        print(f"--- Stage 2: Final Polish ({self.polish_steps} steps max) ---")
        last_loss = jnp.inf
        for step in range(self.polish_steps):
            params, opt_state, loss = update_step(params, opt_state, value_and_grad_fn, optimizer)
            
            if step % 500 == 0:
                # Check convergence
                if jnp.abs(loss - last_loss) < self.convergence_tol:
                    print(f"  Converged at step {step} with loss {loss:.4f}")
                    break
                last_loss = loss
                
        self.final_loss = float(last_loss)
        
        # Store Final Results
        (self.player_coords, self.factor_coords, 
         self.player_lvars, self.factor_lvars, 
         log_sigma) = self._unpack_params(params, n_players, n_factors, d)
         
        self.global_sigma = jax.nn.softplus(float(log_sigma)) + 1e-6
        print(f"Fit Complete. Final Loss: {self.final_loss:.2f}, Global Sigma: {self.global_sigma:.4f}")