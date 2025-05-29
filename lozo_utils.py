import torch
import numpy as np
from common_utils import zo_forward # Import from common_utils

# -----------------------------------------------------------------------------
# LOZO specific functions
# -----------------------------------------------------------------------------

def get_current_rank(iter_num, max_iters, min_rank, max_rank, rank_adaptive, rank_r):
    """
    Compute current rank using linear scheduling from min_rank to max_rank.
    
    Args:
        iter_num: Current iteration number
        max_iters: Total number of iterations
        min_rank: Starting rank
        max_rank: Ending rank
        rank_adaptive: Boolean, if False, return rank_r
        rank_r: Fixed rank to use if not adaptive
    
    Returns:
        current_rank: Integer rank for current iteration
    """
    if not rank_adaptive:
        return rank_r  # Use fixed rank if adaptive is disabled
    
    # Linear interpolation from min_rank to max_rank
    progress = min(iter_num / max_iters, 1.0)  # Clamp to [0, 1]
    current_rank_float = min_rank + progress * (max_rank - min_rank)
    current_rank = max(min_rank, min(max_rank, int(round(current_rank_float))))
    
    return current_rank

# Function to perturb model parameters with low-rank perturbations
def lowrank_zo_perturb_parameters(model, v_dict, zo_random_seed, step, zo_eps, step_interval, rank_r, scaling_factor=1, current_rank=None):
    """
    Perturb the parameters with random vector uv^t.
    Only update V matrices every step_interval steps.
    Supports adaptive rank by resizing V matrices when needed.
    """
    if current_rank is None:
        current_rank = rank_r  # Use default rank if not specified
        
    torch.manual_seed(zo_random_seed)
    
    # Create a list of named parameters to optimize - refresh on each call to handle compiled models
    named_parameters_to_optim = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Handle _orig_mod prefix from torch.compile
            clean_name = name
            if name.startswith('_orig_mod.'):
                clean_name = name[len('_orig_mod.'):]
            named_parameters_to_optim.append((clean_name, param))
    
    for clean_name, param in named_parameters_to_optim:
        if param.ndim >= 2:
            # For matrices, use low-rank perturbation
            # Create new V matrix at the specified interval or if not present
            need_new_v = (step % step_interval == 0 or 
                         clean_name not in v_dict or 
                         v_dict[clean_name].size(1) != current_rank)  # Rank changed
            
            if need_new_v:
                v = torch.randn(param.size(1), current_rank, device=param.device, dtype=param.dtype)
                v_dict[clean_name] = v
            else:
                v = v_dict[clean_name]
            
            u = torch.randn(param.size(0), current_rank, device=param.device, dtype=param.dtype)
            param.data = param.data + scaling_factor * (u @ v.t()) * zo_eps
        else:
            # For vectors (biases), use Gaussian perturbation
            z = torch.normal(mean=0, std=1, size=param.size(), device=param.device, dtype=param.dtype)
            param.data = param.data + scaling_factor * z * zo_eps
    
    return named_parameters_to_optim

# Function to estimate gradient using LOZO with multiple (q) estimates
def lowrank_zo_step(model, X, Y, v_dict, step, zo_random_seed, zo_eps, step_interval, rank_r, zo_q, ctx_obj, current_rank=None):
    """
    Estimate gradient using LOZO with q-times finite differences.
    
    For q>1:
    1. Initialize g = 0
    2. For each q iteration:
       - Generate a single UV^T low-rank perturbation
       - Compute (f+ - f-)/2*zo_eps
       - Accumulate: g += (scalar coefficient) * UV^T
    3. Average: g = g / q
    """
    if current_rank is None:
        current_rank = rank_r  # Use default rank if not specified
        
    # If q=1, use the original implementation for maximum compatibility
    if zo_q == 1:
        # First function evaluation
        lowrank_zo_perturb_parameters(model, v_dict, zo_random_seed, step, zo_eps, step_interval, rank_r, scaling_factor=1, current_rank=current_rank)
        loss1 = zo_forward(model, X, Y, ctx_obj)

        # Second function evaluation
        lowrank_zo_perturb_parameters(model, v_dict, zo_random_seed, step, zo_eps, step_interval, rank_r, scaling_factor=-2, current_rank=current_rank)
        loss2 = zo_forward(model, X, Y, ctx_obj)

        # Calculate projected gradient
        projected_grad = ((loss1 - loss2) / (2 * zo_eps)).item()

        # Reset model back to its parameters at start of step
        lowrank_zo_perturb_parameters(model, v_dict, zo_random_seed, step, zo_eps, step_interval, rank_r, scaling_factor=1, current_rank=current_rank)
        
        return loss1, projected_grad, zo_random_seed
    
    # For q>1, implement the multiple estimation approach
    # Initialize an empty gradient accumulator for each parameter
    accumulated_grads = {}
    first_loss = None
    
    # Create a list of parameters to optimize
    params_to_optimize = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            clean_name = name
            if name.startswith('_orig_mod.'):
                clean_name = name[len('_orig_mod.'):]
            params_to_optimize.append((clean_name, param))
    
    # For each q estimation
    for q_idx in range(zo_q):
        # Sample a new random seed for this estimation
        current_zo_seed = np.random.randint(1000000000) if q_idx > 0 else zo_random_seed
        torch.manual_seed(current_zo_seed)
        
        # Generate perturbations for each parameter
        perturbations = {}  # Store the perturbations for this iteration
        
        for clean_name, param in params_to_optimize:
            if param.ndim >= 2:
                # For matrices, use low-rank perturbation
                if clean_name in v_dict:
                    v = v_dict[clean_name]
                    u = torch.randn(param.size(0), current_rank, device=param.device, dtype=param.dtype)
                    perturbation = u @ v.t()
                    perturbations[clean_name] = (perturbation, True)  # (perturbation, is_lowrank)
                    
                    # Apply positive perturbation
                    param.data = param.data + perturbation * zo_eps
                else:
                    # For matrices without v vector, use Gaussian
                    z = torch.normal(mean=0, std=1, size=param.size(), device=param.device, dtype=param.dtype)
                    perturbations[clean_name] = (z, False)
                    
                    # Apply positive perturbation
                    param.data = param.data + z * zo_eps
            else:
                # For vectors, use Gaussian perturbation
                z = torch.normal(mean=0, std=1, size=param.size(), device=param.device, dtype=param.dtype)
                perturbations[clean_name] = (z, False)
                
                # Apply positive perturbation
                param.data = param.data + z * zo_eps
        
        # Evaluate f(θ + perturbation)
        f_plus = zo_forward(model, X, Y, ctx_obj)
        
        # Save first loss for reporting
        if q_idx == 0:
            first_loss = f_plus
        
        # Apply negative perturbation (from current state)
        for clean_name, param in params_to_optimize:
            if clean_name in perturbations:
                perturbation, _ = perturbations[clean_name]
                param.data = param.data - 2 * perturbation * zo_eps
        
        # Evaluate f(θ - perturbation)
        f_minus = zo_forward(model, X, Y, ctx_obj)
        
        # Compute scalar coefficient
        coeff = (f_plus - f_minus).item() / (2 * zo_eps)
        
        # Accumulate gradients for each parameter
        for clean_name, param in params_to_optimize:
            if clean_name in perturbations:
                perturbation, is_lowrank = perturbations[clean_name]
                
                # Calculate gradient for this parameter
                grad = coeff * perturbation
                
                # Add to accumulated gradients
                if clean_name not in accumulated_grads:
                    accumulated_grads[clean_name] = grad
                else:
                    accumulated_grads[clean_name] += grad
        
        # Reset parameters to their original values
        for clean_name, param in params_to_optimize:
            if clean_name in perturbations:
                perturbation, _ = perturbations[clean_name]
                param.data = param.data + perturbation * zo_eps
    
    # Average the gradients if q > 1
    if zo_q > 1:
        for clean_name in accumulated_grads:
            accumulated_grads[clean_name] = accumulated_grads[clean_name] / zo_q
    
    # Return the first loss and accumulated gradients
    return first_loss, accumulated_grads, zo_random_seed

# Function to update parameters using LOZO (original version for q=1)
def lowrank_zo_update(model, optimizer, projected_grad, zo_random_seed, v_dict, step, lr, rank_r, weight_decay, master_process, named_parameters_to_optim=None, current_rank=None):
    """
    Update model parameters using the LOZO gradient estimate.
    """
    if current_rank is None:
        current_rank = rank_r  # Use default rank if not specified
        
    torch.manual_seed(zo_random_seed)
    
    # If named_parameters_to_optim is not provided, create it
    if named_parameters_to_optim is None:
        named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                clean_name = name
                if name.startswith('_orig_mod.'):
                    clean_name = name[len('_orig_mod.'):]
                named_parameters_to_optim.append((clean_name, param))
    
    # CRITICAL FIX: Scale learning rate by rank for LOZO low-rank updates
    effective_lr = lr / current_rank
    
    # Debug logging only on first step
    if step == 0 and master_process:
        print(f"v_dict contains {len(v_dict)} parameter entries")
        print(f"First few keys: {list(v_dict.keys())[:5]}")
        print(f"Total parameters to optimize: {len(named_parameters_to_optim)}")
        print(f"LOZO: Using effective_lr = {effective_lr:.6f} (lr={lr:.6f} / rank={current_rank})")
        print(f"projected_grad = {projected_grad}")
    
    missing_params = []
    for clean_name, param in named_parameters_to_optim:
        is_weight = "bias" not in clean_name and "layer_norm" not in clean_name and "layernorm" not in clean_name
        if param.ndim >= 2:
            # For matrices, use low-rank update with rank scaling
            if clean_name not in v_dict:
                missing_params.append(clean_name)
                continue
                
            v = v_dict[clean_name]
            u = torch.randn(param.size(0), current_rank, device=param.device, dtype=param.dtype)
            
            if is_weight:
                param.data = param.data - effective_lr * (projected_grad * (u @ v.t()) + weight_decay * lr * param.data)
            else:
                param.data = param.data - effective_lr * (projected_grad * (u @ v.t()))
        else:
            # For vectors (biases), use Gaussian update with full lr (no rank scaling)
            z = torch.normal(mean=0, std=1, size=param.size(), device=param.device, dtype=param.dtype)
            
            if is_weight:
                param.data = param.data - lr * (projected_grad * z + weight_decay * param.data)
            else:
                param.data = param.data - lr * (projected_grad * z)
    
    # Only print missing parameters once
    if step == 0 and missing_params and len(missing_params) > 0 and master_process:
        print(f"Missing {len(missing_params)}/{len(named_parameters_to_optim)} parameters in v_dict")
        print(f"First few missing: {missing_params[:5]}")

# Function to update parameters using LOZO with direct gradient dictionary
def lowrank_zo_update_direct(model, optimizer, grad_dict, lr, weight_decay, named_parameters_to_optim=None):
    """
    Update model parameters using the pre-computed gradient dictionary from q-times estimation.
    """
    raise NotImplementedError("Direct update functions are deprecated and will be removed. Use standard LOZO update instead.")

# Function to update parameters using LOZO with momentum (LOZO-M)
def lowrank_zo_update_momentum(model, optimizer, projected_grad, zo_random_seed, v_dict, exp_avg_m, v_old_dict, step, lr, rank_r, step_interval, weight_decay, master_process, beta1=0.9, named_parameters_to_optim=None, current_rank=None):
    """
    Update model parameters using the LOZO gradient estimate with momentum.
    This is more memory-efficient than standard momentum and can lead to better performance.
    
    Args:
        model: The model to update
        optimizer: The optimizer (not directly used but kept for API consistency)
        projected_grad: The projected gradient estimate from LOZO
        zo_random_seed: Random seed for reproducibility
        v_dict: Dictionary of V matrices for low-rank updates
        exp_avg_m: Dictionary of exponential moving average for momentum
        v_old_dict: Dictionary of old V matrices (from previous step_interval)
        step: Current optimization step
        lr: Learning rate
        beta1: Momentum coefficient (default: 0.9)
        named_parameters_to_optim: List of (name, parameter) tuples to optimize
        current_rank: Current rank for adaptive rank scheduling
    """
    if current_rank is None:
        current_rank = rank_r  # Use default rank if not specified
        
    torch.manual_seed(zo_random_seed)     
    
    # If named_parameters_to_optim is not provided, create it
    if named_parameters_to_optim is None:
        named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                clean_name = name
                if name.startswith('_orig_mod.'):
                    clean_name = name[len('_orig_mod.'):]
                named_parameters_to_optim.append((clean_name, param))
    
    # CRITICAL FIX: Scale learning rate by rank for LOZO low-rank updates
    effective_lr = lr / current_rank
    
    # Debug logs on first step
    if step == 0 and master_process:
        print(f"v_dict contains {len(v_dict)} parameter entries for LOZO-M")
        print(f"First few keys: {list(v_dict.keys())[:5]}")
        print(f"Total parameters to optimize: {len(named_parameters_to_optim)}")
        print(f"LOZO-M: Using effective_lr = {effective_lr:.6f} (lr={lr:.6f} / rank={current_rank})")
        print(f"projected_grad = {projected_grad}")
    
    missing_params = []
    for clean_name, param in named_parameters_to_optim:
        is_weight = "bias" not in clean_name and "layer_norm" not in clean_name and "layernorm" not in clean_name
        if param.ndim >= 2:
            # For matrices, use low-rank update with momentum
            if clean_name not in v_dict:
                missing_params.append(clean_name)
                continue
                
            v = v_dict[clean_name]
            u = torch.randn(param.size(0), current_rank, device=param.device, dtype=param.dtype)
            
            # Initialize momentum if needed
            if clean_name not in exp_avg_m:
                exp_avg_m[clean_name] = torch.zeros_like(u)
            
            # Update momentum based on step interval
            if step % step_interval == 0:
                if clean_name in v_old_dict:   
                    # Use the transition matrix between old and new V
                    v_old = v_old_dict[clean_name]
                    n = v_old.shape[0]  # Use row dimension as in original LOZO
                    exp_avg_m[clean_name] = beta1 * (exp_avg_m[clean_name] @ v_old.t() @ v / n) + (1 - beta1) * projected_grad * u
                else:
                    # First initialization or no old V available
                    exp_avg_m[clean_name] = projected_grad * u
            elif step % step_interval == step_interval - 1:
                # Store old V matrix before it changes in next step
                v_old_dict[clean_name] = v  # No need for clone() as we only read from it
                exp_avg_m[clean_name] = beta1 * exp_avg_m[clean_name] + (1 - beta1) * projected_grad * u
            else:
                # Regular momentum update
                exp_avg_m[clean_name] = beta1 * exp_avg_m[clean_name] + (1 - beta1) * projected_grad * u
            
            # Apply update with rank scaling
            if is_weight:
                param.data = param.data - effective_lr * (exp_avg_m[clean_name] @ v.t() + weight_decay * lr * param.data)
            else:
                param.data = param.data - effective_lr * (exp_avg_m[clean_name] @ v.t())
        else:
            # For vectors, use Gaussian update with momentum (no rank scaling for vectors)
            z = torch.normal(mean=0, std=1, size=param.size(), device=param.device, dtype=param.dtype)
            
            # Initialize or update momentum
            if clean_name not in exp_avg_m:
                exp_avg_m[clean_name] = projected_grad * z
            else:
                exp_avg_m[clean_name] = beta1 * exp_avg_m[clean_name] + (1 - beta1) * projected_grad * z
            
            # Apply update with full lr (no rank scaling for vectors)
            if is_weight:
                param.data = param.data - lr * (exp_avg_m[clean_name] + weight_decay * param.data)
            else:
                param.data = param.data - lr * exp_avg_m[clean_name]
    
    # Only print missing parameters once
    if step == 0 and missing_params and len(missing_params) > 0 and master_process:
        print(f"Missing {len(missing_params)}/{len(named_parameters_to_optim)} parameters in v_dict")
        print(f"First few missing: {missing_params[:5]}")

# -----------------------------------------------------------------------------
# SVD-LOZO specific functions
# -----------------------------------------------------------------------------

def randomized_svd(A, n_components, n_oversamples=10):
    """
    Compute randomized SVD for efficient approximate decomposition.
    """
    raise NotImplementedError("SVD-based functions are deprecated and will be removed. Use standard LOZO instead.")

def full_svd(A, n_components):
    """
    Compute full SVD and truncate to n_components.
    """
    raise NotImplementedError("SVD-based functions are deprecated and will be removed. Use standard LOZO instead.")

def compute_svd(A, n_components, use_full_svd=False, n_oversamples=10):
    """
    Unified SVD interface - choose between full and randomized SVD.
    """
    raise NotImplementedError("SVD-based functions are deprecated and will be removed. Use standard LOZO instead.")

def svdlozo_perturb_parameters(model, zo_random_seed, step, zo_eps, svd_tau, svd_max_rank, use_full_svd, scaling_factor=1, debug_output=False):
    """
    Perturb model parameters using SVD-guided low-rank optimization.
    """
    raise NotImplementedError("SVD-based functions are deprecated and will be removed. Use standard LOZO instead.")

def svdlozo_step(model, X, Y, step, zo_random_seed, zo_eps, svd_tau, svd_max_rank, use_full_svd, zo_q, ctx_obj):
    """
    Estimate gradient using SVD-LOZO with q-times finite differences.
    """
    raise NotImplementedError("SVD-based functions are deprecated and will be removed. Use standard LOZO instead.")

def svdlozo_update(model, optimizer, projected_grad, zo_random_seed, step, lr, zo_eps, svd_tau, svd_max_rank, use_full_svd, weight_decay, master_process, named_parameters_to_optim=None):
    """
    Update model parameters using SVD-LOZO gradient estimate.
    """
    raise NotImplementedError("SVD-based functions are deprecated and will be removed. Use standard LOZO instead.")

def svdlozo_update_direct(model, optimizer, grad_dict, lr, weight_decay, named_parameters_to_optim=None):
    """
    Update model parameters using the pre-computed gradient dictionary from q-times SVD-LOZO estimation.
    """
    raise NotImplementedError("SVD-based functions are deprecated and will be removed. Use standard LOZO instead.")

# -----------------------------------------------------------------------------
# DiLoZO (Directional Low-Rank Zero-Order) specific functions
# -----------------------------------------------------------------------------

def dilozo_perturb_parameters(model, v_dict, u_seed, step, zo_eps, step_interval, rank_r, scaling_factor=1, eps=None, current_rank=None):
    """Perturb model parameters with U_seed V^T for DiLoZO.
    V matrices are handled similarly to LOZO (updated every step_interval).
    U matrix is generated based on u_seed.
    """ 
    if eps is None:
        eps = zo_eps # Use global zo_eps if not provided
    if current_rank is None:
        current_rank = rank_r # Use default rank if not specified
    if current_rank == 0: # Avoid issues with rank 0
        current_rank = 1

    torch.manual_seed(u_seed) # Ensure U is generated based on this specific seed

    named_parameters_to_optim = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            clean_name = name
            if name.startswith('_orig_mod.'):
                clean_name = name[len('_orig_mod.'):]
            named_parameters_to_optim.append((clean_name, param))

    for clean_name, param in named_parameters_to_optim:
        if param.ndim >= 2:
            # For matrices, use low-rank perturbation U V^T
            need_new_v = (step % step_interval == 0 or
                          clean_name not in v_dict or
                          v_dict[clean_name].size(1) != current_rank) # Rank changed
            
            if need_new_v:
                # Ensure v has the correct dimensions, especially for the second dimension (rank)
                v = torch.randn(param.size(1), int(current_rank), device=param.device, dtype=param.dtype)
                v_dict[clean_name] = v
            else:
                v = v_dict[clean_name]
            
            # Ensure u has the correct dimensions
            u = torch.randn(param.size(0), int(current_rank), device=param.device, dtype=param.dtype)
            param.data = param.data + scaling_factor * (u @ v.t()) * eps
        else:
            # For vectors (biases), use Gaussian perturbation z
            z = torch.normal(mean=0, std=1, size=param.size(), device=param.device, dtype=param.dtype)
            param.data = param.data + scaling_factor * z * eps
    
    return named_parameters_to_optim

def dilozo_step(model, X, Y, v_dict, step, zo_random_seed, directional_q_val, zo_eps, step_interval, rank_r, master_process, ctx_obj, eps=None, current_rank=None):
    """Estimate gradient using DiLoZO (Directional Low-Rank Zero-Order).
    
    Algorithm:
    1. Compute baseline loss: f(θ)
    2. Sample q random low-rank directions U_i V^T for i = 1, ..., q
    3. Evaluate f(θ + ε*U_i V^T) for each direction i
    4. Select U_best V^T = argmin_i f(θ + ε*U_i V^T) (direction that gives lowest loss)
    5. Success = min_loss < baseline_loss
    6. Estimate directional derivative: c = [f(θ + ε*U_best V^T) - f(θ - ε*U_best V^T)]/(2ε)
       This gives the projected gradient ∇f(θ) · (U_best V^T) along the best direction.
       Since U_best V^T decreases loss, c < 0, so θ ← θ - α * c * U_best V^T moves toward U_best V^T.

    Args:
        directional_q_val: Number of directions to try.
    """
    current_directional_q = directional_q_val

    if eps is None:
        eps = zo_eps
    if current_rank is None:
        current_rank = rank_r
    if current_rank == 0: # Avoid issues with rank 0
        current_rank = 1


    baseline_loss = zo_forward(model, X, Y, ctx_obj)
    best_loss_val = float('inf')
    best_u_seed = None

    for i in range(current_directional_q):
        direction_u_seed = zo_random_seed + i
        dilozo_perturb_parameters(model, v_dict, direction_u_seed, step, zo_eps, step_interval, rank_r, scaling_factor=1, eps=eps, current_rank=current_rank)
        current_loss = zo_forward(model, X, Y, ctx_obj)
        if current_loss < best_loss_val:
            best_loss_val = current_loss
            best_u_seed = direction_u_seed
        # Reset parameters to original state before trying next direction
        dilozo_perturb_parameters(model, v_dict, direction_u_seed, step, zo_eps, step_interval, rank_r, scaling_factor=-1, eps=eps, current_rank=current_rank)

    success = best_loss_val < baseline_loss

    # If no direction improved, best_u_seed might be None. Handle this.
    if best_u_seed is None: # This can happen if all directions yield worse or equal loss
        if master_process:
            print(f"DiLoZO Step: No improving direction found out of {current_directional_q}. Using first direction's seed for grad_coeff calculation.")
        best_u_seed = zo_random_seed # Fallback to the first seed

    # Parameter perturbation for f_plus
    dilozo_perturb_parameters(model, v_dict, best_u_seed, step, zo_eps, step_interval, rank_r, scaling_factor=1, eps=eps, current_rank=current_rank)
    # f_plus is ideally best_loss_val if the best_u_seed led to it, otherwise re-evaluate if necessary.
    # If best_u_seed was a fallback, best_loss_val might not correspond to f(theta + eps U_best V^T)
    # Re-evaluating f_plus ensures correctness.
    f_plus = zo_forward(model, X, Y, ctx_obj) 
    
    # Parameter perturbation for f_minus
    # We perturbed by +1*eps*UVT to get f_plus. Now perturb by -2*eps*UVT from current state to get to (theta - eps*UVT)
    dilozo_perturb_parameters(model, v_dict, best_u_seed, step, zo_eps, step_interval, rank_r, scaling_factor=-2, eps=eps, current_rank=current_rank)
    f_minus = zo_forward(model, X, Y, ctx_obj)
    
    grad_coeff = ((f_plus - f_minus) / (2 * eps)).item()
    
    # Reset parameters to original state (theta)
    # We are currently at (theta - eps*UVT). Add back 1*eps*UVT to get to theta
    dilozo_perturb_parameters(model, v_dict, best_u_seed, step, zo_eps, step_interval, rank_r, scaling_factor=1, eps=eps, current_rank=current_rank)

    return best_loss_val, grad_coeff, best_u_seed, success

def dilozo_update(model, optimizer, grad_coeff, best_u_seed, v_dict, step, lr, rank_r, weight_decay, master_process, named_parameters_to_optim=None, current_rank=None):
    """Update model parameters using DiLoZO.
    
    Update rule: θ ← θ - α * (grad_coeff / r) * U_best V_best^T
    
    Mathematical reasoning:
    - grad_coeff = ∇f(θ) · (U_best V_best^T) is the directional derivative
    - Since U_best V_best^T was selected to decrease loss, grad_coeff < 0
    - The update θ ← θ - α * (negative) * U_best V_best^T moves toward U_best V_best^T
    - Division by rank r provides scaling for the low-rank structure
    """
    if current_rank is None:
        current_rank = rank_r
    if current_rank == 0:
        if master_process: print("Warning: DiLoZO update called with current_rank=0. Skipping update.")
        return

    torch.manual_seed(best_u_seed) # Regenerate U_best

    if named_parameters_to_optim is None:
        named_parameters_to_optim = [(name if not name.startswith('_orig_mod.') else name[len('_orig_mod.'):], param)
                                     for name, param in model.named_parameters() if param.requires_grad]

    # CRITICAL FIX: Scale learning rate by rank for LOZO low-rank updates
    effective_lr = lr / current_rank
    
    # Debug logging only on first step
    if step == 0 and master_process:
        print(f"v_dict contains {len(v_dict)} parameter entries")
        print(f"First few keys: {list(v_dict.keys())[:5]}")
        print(f"Total parameters to optimize: {len(named_parameters_to_optim)}")
        print(f"LOZO: Using lr = {lr:.6f}, rank = {current_rank}")
        print(f"projected_grad = {grad_coeff}")
    
    missing_params = []
    for clean_name, param in named_parameters_to_optim:
        is_weight = "bias" not in clean_name and "layer_norm" not in clean_name and "layernorm" not in clean_name
        if param.ndim >= 2:
            # Check if this parameter has a V matrix before accessing
            if clean_name in v_dict:
                u_best = torch.randn(param.size(0), int(current_rank), device=param.device, dtype=torch.float32) # U in float32
                v_best = v_dict[clean_name].float() # V in float32
                
                # Perform update calculation in float32
                update_term = lr * grad_coeff * (u_best @ v_best.t())
                
                if is_weight and weight_decay > 0:
                    param_data_float32 = param.data.float() - (update_term + weight_decay * lr * param.data.float())
                else:
                    param_data_float32 = param.data.float() - update_term
            else:
                # For matrices without V, use Gaussian update like vectors
                z_best = torch.normal(mean=0, std=1, size=param.size(), device=param.device, dtype=torch.float32) # Z in float32
                update_term_mat = lr * grad_coeff * z_best
                
                if is_weight and weight_decay > 0:
                    param_data_float32 = param.data.float() - (update_term_mat + weight_decay * lr * param.data.float())
                else:
                    param_data_float32 = param.data.float() - update_term_mat
        else:
            # For vectors (biases)
            z_best = torch.normal(mean=0, std=1, size=param.size(), device=param.device, dtype=torch.float32) # Use float32 for Z
            update_term_vec = lr * grad_coeff * z_best
            
            if is_weight and weight_decay > 0:
                param_data_float32 = param.data.float() - (update_term_vec + weight_decay * lr * param.data.float())
            else:
                param_data_float32 = param.data.float() - update_term_vec
        
        param.data = param_data_float32.to(param.data.dtype) # Convert back to original dtype

    # Only print missing parameters once
    if step == 0 and missing_params and len(missing_params) > 0 and master_process:
        print(f"Missing {len(missing_params)}/{len(named_parameters_to_optim)} parameters in v_dict")
        print(f"First few missing: {missing_params[:5]}")

def dilozo_update_momentum(model, optimizer, grad_coeff, best_u_seed, v_dict, exp_avg_m, step, lr, rank_r, weight_decay, master_process, beta1=0.9, named_parameters_to_optim=None, current_rank=None):
    """Update model parameters using DiLoZO with momentum.
    
    Momentum update:
    m_t = β1 * m_{t-1} + (1 - β1) * g_t
    θ_t = θ_{t-1} - α * m_t
    
    where g_t = (grad_coeff / r) * U_best V_best^T for matrices, and grad_coeff * Z_best for vectors
    
    Mathematical reasoning:
    - grad_coeff = ∇f(θ) · (U_best V_best^T) is the directional derivative
    - Since U_best V_best^T was selected to decrease loss, grad_coeff < 0
    - The momentum accumulates these negative gradients, creating consistent movement toward good directions
    """
    if current_rank is None:
        current_rank = rank_r
    if current_rank == 0:
        if master_process: print("Warning: DiLoZO Momentum update called with current_rank=0. Skipping update.")
        return

    torch.manual_seed(best_u_seed) # Regenerate U_best / Z_best

    if named_parameters_to_optim is None:
        named_parameters_to_optim = [(name if not name.startswith('_orig_mod.') else name[len('_orig_mod.'):], param)
                                     for name, param in model.named_parameters() if param.requires_grad]

    # For matrix parameters, g_t is scaled by 1/rank. For vector parameters, effectively rank=1.
    # grad_coeff is (f+ - f-)/(2eps)
    
    for clean_name, param in named_parameters_to_optim:
        is_weight = "bias" not in clean_name and "layer_norm" not in clean_name and "layernorm" not in clean_name
        
        original_dtype = param.data.dtype
        param_data_float32 = param.data.float() # Convert param data to float32

        if param.ndim >= 2:
            # Check if this parameter has a V matrix before accessing
            if clean_name in v_dict:
                u_best = torch.randn(param.size(0), int(current_rank), device=param.device, dtype=torch.float32) # U in float32
                v_best = v_dict[clean_name].float() # V in float32
                # g_t = (grad_coeff / current_rank) * (U_best @ V_best^T)
                g_t = (grad_coeff / current_rank) * (u_best @ v_best.t()) 
            else:
                # For matrices without V, use Gaussian update like vectors
                z_best = torch.normal(mean=0, std=1, size=param.size(), device=param.device, dtype=torch.float32) # Z in float32
                # g_t = (grad_coeff / current_rank) * Z_best (scaled by rank for consistency)
                g_t = (grad_coeff / current_rank) * z_best
        else:
            # For vectors (biases), effectively rank is 1.
            z_best = torch.normal(mean=0, std=1, size=param.size(), device=param.device, dtype=torch.float32) # Z in float32
            # g_t = grad_coeff * Z_best
            g_t = grad_coeff * z_best

        if clean_name not in exp_avg_m:
            exp_avg_m[clean_name] = torch.zeros_like(param_data_float32) # Initialize momentum in float32
        
        current_momentum = exp_avg_m[clean_name].float() # Ensure momentum is float32
        current_momentum.mul_(beta1).add_(g_t, alpha=1 - beta1)
        exp_avg_m[clean_name] = current_momentum
        
        update_val = lr * current_momentum # This is m_t * lr

        if is_weight and weight_decay > 0:
            param_data_float32 = param_data_float32 - (update_val + weight_decay * lr * param_data_float32)
        else:
            param_data_float32 = param_data_float32 - update_val
            
        param.data = param_data_float32.to(original_dtype)