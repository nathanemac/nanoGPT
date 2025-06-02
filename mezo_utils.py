import torch
import numpy as np
from common_utils import zo_forward # Import from common_utils

# -----------------------------------------------------------------------------
# MeZO specific functions
# -----------------------------------------------------------------------------

def mezo_step(model, X, Y, step, zo_random_seed, zo_eps, ctx_obj):
    """
    Estimate gradient using MeZO (Memorization-Efficient Zero-Order optimization).
    
    Args:
        model: The model to optimize
        X, Y: Input batch
        step: Current step (unused, kept for compatibility)
        zo_random_seed: Random seed for perturbation
        zo_eps: Perturbation size
        ctx_obj: Context for forward pass
    
    Returns:
        baseline_loss: Loss from unperturbed model
        projected_grad: Scalar gradient coefficient
        named_parameters_to_optim: List of (name, param) tuples
    """
    # Ensure model is in training mode for gradient estimation
    model.train()
    
    # Use the provided random seed for reproducibility
    torch.manual_seed(zo_random_seed)
    
    # Get list of parameters to optimize with clean names
    named_parameters_to_optim = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            clean_name = name
            if name.startswith('_orig_mod.'):
                clean_name = name[len('_orig_mod.'):]
            named_parameters_to_optim.append((clean_name, param))
    
    # Generate and apply positive perturbation
    perturbations = {}
    for clean_name, param in named_parameters_to_optim:
        z = torch.normal(mean=0, std=1, size=param.size(), device=param.device, dtype=param.dtype)
        perturbations[clean_name] = z
        param.data = param.data + z * zo_eps
    
    # Evaluate f(θ + z)
    f_plus = zo_forward(model, X, Y, ctx_obj)
    
    # Apply negative perturbation
    for clean_name, param in named_parameters_to_optim:
        z = perturbations[clean_name]
        param.data = param.data - 2 * z * zo_eps
    
    # Evaluate f(θ - z)
    f_minus = zo_forward(model, X, Y, ctx_obj)
    
    # Compute gradient coefficient
    projected_grad = (f_plus - f_minus).item() / (2 * zo_eps)
    
    # Reset parameters to original values
    for clean_name, param in named_parameters_to_optim:
        z = perturbations[clean_name]
        param.data = param.data + z * zo_eps
    
    # Return baseline loss for reporting (use f_plus as approximation)
    baseline_loss = f_plus
    
    return baseline_loss, projected_grad, named_parameters_to_optim

def mezo_perturb_parameters(model, zo_random_seed, zo_eps, scaling_factor=1):
    """
    Legacy function for compatibility. Not used in current training code.
    """
    pass

def mezo_update(model, optimizer, projected_grad, zo_random_seed, step, lr, weight_decay, master_process, named_parameters_to_optim=None):
    """
    Update model parameters using MeZO gradient estimate.
    
    Args:
        model: The model to update
        optimizer: Optimizer (unused, kept for compatibility)
        projected_grad: Scalar gradient coefficient from mezo_step
        zo_random_seed: Random seed to regenerate perturbations
        step: Current step (unused, kept for compatibility)
        lr: Learning rate
        weight_decay: Weight decay coefficient
        master_process: Whether this is the master process (unused, kept for compatibility)
        named_parameters_to_optim: List of (name, param) tuples
    """
    torch.manual_seed(zo_random_seed)
    
    if named_parameters_to_optim is None:
        named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                clean_name = name
                if name.startswith('_orig_mod.'):
                    clean_name = name[len('_orig_mod.'):]
                named_parameters_to_optim.append((clean_name, param))
    
    # Apply MeZO update: θ ← θ - α * c * z
    for clean_name, param in named_parameters_to_optim:
        z = torch.normal(mean=0, std=1, size=param.size(), device=param.device, dtype=param.dtype)
        
        is_weight = "bias" not in clean_name and "layer_norm" not in clean_name and "layernorm" not in clean_name
        if is_weight and weight_decay > 0:
            param.data = param.data - lr * (projected_grad * z + weight_decay * param.data)
        else:
            param.data = param.data - lr * (projected_grad * z)

def mezo_update_momentum(model, optimizer, projected_grad, zo_random_seed, exp_avg_m, step, lr, weight_decay, master_process, beta1=0.9, named_parameters_to_optim=None):
    """
    Update model parameters using MeZO with momentum (MeZO-M).
    
    Args:
        model: The model to update
        optimizer: Optimizer (unused, kept for compatibility)
        projected_grad: Scalar gradient coefficient from mezo_step
        zo_random_seed: Random seed to regenerate perturbations
        exp_avg_m: Dictionary storing momentum state
        step: Current step (unused, kept for compatibility)
        lr: Learning rate
        weight_decay: Weight decay coefficient
        master_process: Whether this is the master process (unused, kept for compatibility)
        beta1: Momentum coefficient
        named_parameters_to_optim: List of (name, param) tuples
    """
    torch.manual_seed(zo_random_seed)
    
    if named_parameters_to_optim is None:
        named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                clean_name = name
                if name.startswith('_orig_mod.'):
                    clean_name = name[len('_orig_mod.'):]
                named_parameters_to_optim.append((clean_name, param))
    
    # Apply MeZO-M update with momentum
    for clean_name, param in named_parameters_to_optim:
        z = torch.normal(mean=0, std=1, size=param.size(), device=param.device, dtype=param.dtype)
        grad = projected_grad * z
        
        # Initialize momentum if needed
        if clean_name not in exp_avg_m:
            exp_avg_m[clean_name] = torch.zeros_like(grad)
        
        # Update momentum: m_t = β * m_{t-1} + (1-β) * g_t
        exp_avg_m[clean_name] = beta1 * exp_avg_m[clean_name] + (1 - beta1) * grad
        
        # Apply update with momentum
        is_weight = "bias" not in clean_name and "layer_norm" not in clean_name and "layernorm" not in clean_name
        if is_weight and weight_decay > 0:
            param.data = param.data - lr * (exp_avg_m[clean_name] + weight_decay * param.data)
        else:
            param.data = param.data - lr * exp_avg_m[clean_name]

# -----------------------------------------------------------------------------
# DiMeZO (Directional MeZO) specific functions
# -----------------------------------------------------------------------------

def dimezo_step(model, X, Y, step, zo_random_seed, directional_q_global, zo_eps, master_process, ctx_obj, directional_q=None, eps=None, direct_movement=False):
    """
    Estimate gradient using DiMeZO (Directional MeZO) with directional selection.
    
    Algorithm:
    1. Compute baseline loss: f(θ)
    2. Sample q random directions z_1, ..., z_q
    3. Evaluate f(θ + ε*z_i) for each direction i
    4. Select z_best = argmin_zi f(θ + ε*z_i) (direction that gives lowest loss)
    5. Success = min_loss < baseline_loss
    6. If direct_movement=True: return Z_best directly for θ ← θ + α * Z_best
       Else: Estimate directional derivative: c = [f(θ + ε*z_best) - f(θ - ε*z_best)]/(2ε)
    """
    current_directional_q = directional_q if directional_q is not None else directional_q_global
    
    if eps is None:
        eps = zo_eps
    
    # Step 1: Compute baseline loss f(θ)
    baseline_loss = zo_forward(model, X, Y, ctx_obj)
    
    # Phase 1: Directional Selection - try q directions and find the best
    best_loss = float('inf')
    best_seed = None
    
    for i in range(current_directional_q):
        # Generate a unique random seed for this direction
        direction_seed = zo_random_seed + i
        
        # Apply perturbation in this direction
        dimezo_perturb_parameters(model, direction_seed, scaling_factor=1, eps=eps)
        
        # Evaluate loss at θ + ε*z_i
        loss = zo_forward(model, X, Y, ctx_obj)
        
        # For the first direction, always set it as the initial best
        # For subsequent directions, only update if loss is better
        if i == 0 or loss < best_loss:
            best_loss = loss
            best_seed = direction_seed
        
        # Reset model back to original parameters
        dimezo_perturb_parameters(model, direction_seed, scaling_factor=-1, eps=eps)
    
    # Safety check: if no directions were tried (should not happen, but be safe)
    if best_seed is None:
        best_seed = zo_random_seed
        best_loss = baseline_loss
    
    # Determine success: did we find a direction better than baseline?
    success = best_loss < baseline_loss
    
    if direct_movement:
        # Generate and return the best direction dictionary
        torch.manual_seed(best_seed)
        best_direction_dict = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                clean_name = name
                if name.startswith('_orig_mod.'):
                    clean_name = name[len('_orig_mod.'):]
                z = torch.normal(mean=0, std=1, size=param.size(), device=param.device, dtype=param.dtype)
                best_direction_dict[clean_name] = z
        gradient_or_direction = best_direction_dict
    else:
        # Phase 2: Gradient Estimation along best direction
        # Apply perturbation in best direction
        dimezo_perturb_parameters(model, best_seed, scaling_factor=1, eps=eps)
        f_plus = zo_forward(model, X, Y, ctx_obj)  # Recompute f_plus correctly
        
        # Apply negative perturbation: θ - ε*z_best
        dimezo_perturb_parameters(model, best_seed, scaling_factor=-2, eps=eps)
        f_minus = zo_forward(model, X, Y, ctx_obj)
        
        # Calculate projected gradient coefficient
        gradient_or_direction = ((f_plus - f_minus) / (2 * eps)).item()
        
        # Reset model back to original parameters
        dimezo_perturb_parameters(model, best_seed, scaling_factor=1, eps=eps)
    
    return best_loss, gradient_or_direction, best_seed, success

def dimezo_perturb_parameters(model, direction_seed, scaling_factor=1, eps=1e-3):
    """
    Perturb the parameters with standard Gaussian noise for DiMeZO directional selection.
    """
    torch.manual_seed(direction_seed)
    
    named_parameters_to_optim = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            clean_name = name
            if name.startswith('_orig_mod.'):
                clean_name = name[len('_orig_mod.'):]
            named_parameters_to_optim.append((clean_name, param))
    
    for clean_name, param in named_parameters_to_optim:
        z = torch.normal(mean=0, std=1, size=param.size(), device=param.device, dtype=param.dtype)
        param.data = param.data + scaling_factor * z * eps
    
    return named_parameters_to_optim

def dimezo_update(model, optimizer, gradient_or_direction, best_seed, step, lr, weight_decay, master_process, named_parameters_to_optim=None, direct_movement=False):
    """
    Update model parameters using DiMeZO.
    
    Two update modes:
    1. Direct Movement (direct_movement=True): 
       θ ← θ + α * Z_best (move directly toward the best direction)
       
    2. Gradient Descent (direct_movement=False):
       θ ← θ - α * c * Z_best where c = ∇f(θ) · Z_best (directional derivative)
    """
    if named_parameters_to_optim is None:
        named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                clean_name = name
                if name.startswith('_orig_mod.'):
                    clean_name = name[len('_orig_mod.'):]
                named_parameters_to_optim.append((clean_name, param))
    
    if direct_movement:
        # Direct movement: θ ← θ + α * Z_best
        direction_dict = gradient_or_direction
        for clean_name, param in named_parameters_to_optim:
            if clean_name in direction_dict:
                z = direction_dict[clean_name]
                is_weight = "bias" not in clean_name and "layer_norm" not in clean_name and "layernorm" not in clean_name
                if is_weight:
                    # Correct weight decay: θ ← θ * (1 - α * λ) + α * Z_best
                    param.data = param.data * (1 - lr * weight_decay) + lr * z
                else:
                    param.data = param.data + lr * z
    else:
        # Gradient descent mode: θ ← θ - α * c * Z_best
        torch.manual_seed(best_seed)
        projected_grad = gradient_or_direction
        
        for clean_name, param in named_parameters_to_optim:
            z = torch.normal(mean=0, std=1, size=param.size(), device=param.device, dtype=param.dtype)
            
            is_weight = "bias" not in clean_name and "layer_norm" not in clean_name and "layernorm" not in clean_name
            if is_weight:
                param.data = param.data - lr * (projected_grad * z + weight_decay * param.data)
            else:
                param.data = param.data - lr * (projected_grad * z)

def dimezo_update_momentum(model, optimizer, gradient_or_direction, best_seed, exp_avg_m, step, lr, weight_decay, master_process, beta1=0.9, named_parameters_to_optim=None, direct_movement=False):
    """
    Update model parameters using DiMeZO with momentum.
    
    Momentum update:
    m_t = β1 * m_{t-1} + (1 - β1) * g_t
    θ_t = θ_{t-1} - α * m_t
    
    Two update modes:
    1. Direct Movement (direct_movement=True): 
       g_t = Z_best, θ ← θ + α * m_t (move with momentum toward best directions)
       
    2. Gradient Descent (direct_movement=False):
       g_t = c * Z_best where c = ∇f(θ) · Z_best, θ ← θ - α * m_t
    """
    if named_parameters_to_optim is None:
        named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                clean_name = name
                if name.startswith('_orig_mod.'):
                    clean_name = name[len('_orig_mod.'):]
                named_parameters_to_optim.append((clean_name, param))
    
    if direct_movement:
        # Direct movement with momentum: θ ← θ + α * m_t where m_t accumulates Z_best
        direction_dict = gradient_or_direction
        for clean_name, param in named_parameters_to_optim:
            if clean_name in direction_dict:
                z = direction_dict[clean_name]
                
                # Initialize momentum if needed
                if clean_name not in exp_avg_m:
                    exp_avg_m[clean_name] = torch.zeros_like(z)
                
                # Update momentum: m_t = β * m_{t-1} + (1-β) * Z_best
                exp_avg_m[clean_name] = beta1 * exp_avg_m[clean_name] + (1 - beta1) * z
                
                is_weight = "bias" not in clean_name and "layer_norm" not in clean_name and "layernorm" not in clean_name
                if is_weight:
                    # Apply momentum update with weight decay: θ ← θ * (1 - α * λ) + α * m_t
                    param.data = param.data * (1 - lr * weight_decay) + lr * exp_avg_m[clean_name]
                else:
                    param.data = param.data + lr * exp_avg_m[clean_name]
    else:
        # Gradient descent mode with momentum: θ ← θ - α * m_t where m_t accumulates c * Z_best
        torch.manual_seed(best_seed)
        projected_grad = gradient_or_direction
        
        for clean_name, param in named_parameters_to_optim:
            z = torch.normal(mean=0, std=1, size=param.size(), device=param.device, dtype=param.dtype)
            grad = projected_grad * z
            
            # Initialize momentum if needed
            if clean_name not in exp_avg_m:
                exp_avg_m[clean_name] = torch.zeros_like(grad)
            
            # Update momentum: m_t = β * m_{t-1} + (1-β) * g_t
            exp_avg_m[clean_name] = beta1 * exp_avg_m[clean_name] + (1 - beta1) * grad
            
            # Apply update with momentum
            is_weight = "bias" not in clean_name and "layer_norm" not in clean_name and "layernorm" not in clean_name
            if is_weight:
                param.data = param.data - lr * (exp_avg_m[clean_name] + weight_decay * param.data)
            else:
                param.data = param.data - lr * exp_avg_m[clean_name] 