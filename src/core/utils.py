from functools import partial
import jax
import optax

@partial(jax.jit, static_argnames=['value_and_grad_fn', 'optimizer'])
def update_step(params, opt_state, value_and_grad_fn, optimizer, gradient_mask=None):
    """
    A pure, JIT-compiled function that performs a single optimization step.
    
    Args:
        params: The current model parameters (DeviceArray).
        opt_state: The internal state of the optimizer (e.g., Adam momentum).
        value_and_grad_fn: The JAX function that computes loss and gradients.
        optimizer: The Optax optimizer instance.
        gradient_mask (optional): A binary mask (1s and 0s) to freeze specific 
                                  parameters during Alternating Optimization.
                                  
    Returns:
        new_params: Updated parameters.
        new_opt_state: Updated optimizer state.
        loss: The loss value at this step.
    """
    # 1. Calculate Loss and Gradients
    loss, grads = value_and_grad_fn(params)
    
    # 2. Apply Gradient Masking (for Alternating Optimization)
    if gradient_mask is not None:
        # Element-wise multiplication: grad * 1 = grad, grad * 0 = 0
        grads = jax.tree_util.tree_map(lambda g, m: g * m, grads, gradient_mask)
        
    # 3. Compute Updates (Adam logic)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    
    # 4. Apply Updates to Parameters
    new_params = optax.apply_updates(params, updates)
    
    return new_params, opt_state, loss