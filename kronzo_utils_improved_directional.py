import torch
import numpy as np
from collections import deque
from common_utils import zo_forward
from kronzo_utils import (
    choose_kron_dims, choose_diverse_kron_dims, 
    kronzo_perturb_parameters
)

class ImprovedDirectionalHistory:
    """
    Maintains history of loss values for conservative update decisions.
    
    CORRECTED ALGORITHM:
    - Keep sliding window of `history_size` previous SUCCESSFUL losses
    - Accept update only if candidate_loss ≤ max{previous successful losses}  
    - Only update the sliding window when an update is actually accepted
    """
    def __init__(self, history_size: int = 10):
        self.history_size = history_size
        self.loss_history = deque(maxlen=history_size)  # Stores successful update losses
    
    def add_loss(self, loss_value: float):
        """Add a successful loss value to the history (only called on successful updates)."""
        self.loss_history.append(loss_value)
    
    def should_accept_update(self, candidate_loss: float, baseline_loss: float) -> bool:
        """
        Determine if we should accept an update based on candidate loss.
        
        CORRECTED LOGIC per user specification:
        - If no history yet: accept any improvement over baseline
        - If have history: accept only if candidate_loss ≤ max{previous successful losses}
        
        Args:
            candidate_loss: Loss value for the candidate update
            baseline_loss: Current baseline loss f(θ) (used only for initialization)
            
        Returns:
            True if update should be accepted
        """
        # During initialization phase (not enough history yet)
        if len(self.loss_history) < self.history_size:
            # Accept if candidate improves over baseline
            return candidate_loss < baseline_loss
            
        # Conservative phase: candidate must beat the worst of our successful history
        max_historical_loss = max(self.loss_history)
        return candidate_loss <= max_historical_loss
    
    def get_best_historical_loss(self) -> float:
        """Get the best (minimum) loss from history."""
        if len(self.loss_history) == 0:
            return float('inf')
        return min(self.loss_history)
    
    def get_worst_historical_loss(self) -> float:
        """Get the worst (maximum) loss from history."""
        if len(self.loss_history) == 0:
            return float('inf')
        return max(self.loss_history)
    
    def get_history_info(self) -> dict:
        """Get information about current history state."""
        if len(self.loss_history) == 0:
            return {
                'count': 0,
                'best': float('inf'),
                'worst': float('inf'),
                'threshold': float('inf'),
                'phase': 'empty'
            }
        
        return {
            'count': len(self.loss_history),
            'best': min(self.loss_history),
            'worst': max(self.loss_history),
            'threshold': max(self.loss_history),
            'phase': 'initialization' if len(self.loss_history) < self.history_size else 'conservative'
        }

def improved_kronzo_step(model, 
                        X, Y, 
                        step: int,
                        zo_random_seed: int,
                        directional_q: int,
                        zo_eps: float,
                        lr: float,
                        ctx_obj,
                        strategy: str = "approx_square",
                        max_factor: int = 32,
                        kronzo_sampling_number: int = 1,
                        b_dict: dict | None = None,
                        step_interval: int = 50,
                        loss_history: ImprovedDirectionalHistory | None = None,
                        get_batch_fn=None):
    """
    Improved directional KronZO step that evaluates actual update candidates.
    
    CORRECTED ALGORITHM per user specification: 
    - Get ONE fresh batch per iteration: (X_eval, Y_eval) = get_batch_fn()
    - Evaluate ALL candidates f(θ + s_i) on the SAME batch (X_eval, Y_eval) for fair comparison
    - Evaluate baseline f(θ) on the SAME batch (X_eval, Y_eval)
    - This avoids overfitting while maintaining fair candidate comparison
    
    Algorithm:
    1. Get ONE fresh evaluation batch for this entire iteration
    2. For each direction i = 1, ..., directional_q:
       a. Sample Z_i (Kronecker: A_i ⊗ B_i) 
       b. Compute projected gradient: c_i = [f(θ + ε*Z_i) - f(θ - ε*Z_i)] / (2ε) on batch (X, Y)
       c. Compute candidate step: s_i = -lr*c_i*Z_i
       d. Evaluate actual update: f(θ + s_i) on the SAME evaluation batch (X_eval, Y_eval)
    3. Evaluate baseline: f(θ) on the SAME evaluation batch (X_eval, Y_eval)
    4. Find best candidate: i_best = argmin_i f(θ + s_i) 
    5. Accept update only if f(θ + s_i_best) ≤ max{previous successful losses} (conservative)
    
    Args:
        model: The model to optimize
        X, Y: Input batch (used for gradient estimation)
        step: Current training step
        zo_random_seed: Random seed for reproducibility
        directional_q: Number of directions to evaluate
        zo_eps: Perturbation size for gradient estimation  
        lr: Learning rate for candidate evaluation
        ctx_obj: Context for forward pass
        strategy: Kronecker factorization strategy
        max_factor: Maximum factor size for factorization
        kronzo_sampling_number: Number of Kronecker products to sample and sum
        b_dict: Dictionary to store B matrices
        step_interval: Interval for updating B matrices
        loss_history: History tracker for conservative updates
        get_batch_fn: Function to get fresh batches (X_new, Y_new) = get_batch_fn()
        
    Returns:
        baseline_loss: Original loss f(θ) on evaluation batch
        best_candidate_loss: Loss of best candidate f(θ + s_best) on evaluation batch
        best_direction_info: Dict with best direction details
        should_update: Whether to accept the update
        candidate_step_data: Data needed for the actual update
    """
    if b_dict is None:
        b_dict = {}
    
    if loss_history is None:
        loss_history = ImprovedDirectionalHistory()
    
    # Step 0: Get ONE fresh evaluation batch for this entire iteration
    if get_batch_fn is not None:
        X_eval, Y_eval = get_batch_fn()
        used_fresh_batch = True
    else:
        # Fallback: use original batch (with warning)
        X_eval, Y_eval = X, Y
        used_fresh_batch = False
        if step % 100 == 0:  # Warn occasionally
            print(f"WARNING: Using same batch for candidate evaluation at step {step}")
    
    # Storage for direction information (memory efficient)
    direction_seeds = []
    projected_grads = []
    candidate_losses = []
    
    best_loss = float('inf')
    best_direction_idx = -1
    
    # Step 1: Compute projected gradients for all directions
    for i in range(directional_q):
        direction_seed = zo_random_seed + i
        direction_seeds.append(direction_seed)
        
        # Phase 1a: Compute projected gradient c_i = [f(θ + ε*Z_i) - f(θ - ε*Z_i)] / (2ε)
        # Use ORIGINAL batch (X, Y) for gradient estimation
        
        # f(θ + ε*Z_i)
        kronzo_perturb_parameters(
            model, direction_seed, step, zo_eps_global=zo_eps, 
            scaling_factor=1.0, eps=zo_eps,
            strategy=strategy, max_factor=max_factor,
            kronzo_sampling_number=kronzo_sampling_number,
            b_dict=b_dict, step_interval=step_interval
        )
        loss_plus = zo_forward(model, X, Y, ctx_obj)
        
        # f(θ - ε*Z_i) 
        kronzo_perturb_parameters(
            model, direction_seed, step, zo_eps_global=zo_eps,
            scaling_factor=-2.0, eps=zo_eps,  # -2 to go from +ε to -ε
            strategy=strategy, max_factor=max_factor,
            kronzo_sampling_number=kronzo_sampling_number,
            b_dict=b_dict, step_interval=step_interval
        )
        loss_minus = zo_forward(model, X, Y, ctx_obj)
        
        # Reset to θ
        kronzo_perturb_parameters(
            model, direction_seed, step, zo_eps_global=zo_eps,
            scaling_factor=1.0, eps=zo_eps,  # +1 to go from -ε back to θ
            strategy=strategy, max_factor=max_factor,
            kronzo_sampling_number=kronzo_sampling_number,
            b_dict=b_dict, step_interval=step_interval
        )
        
        # Compute projected gradient coefficient
        c_i = ((loss_plus - loss_minus) / (2 * zo_eps)).item()
        projected_grads.append(c_i)
    
    # Step 2: Evaluate baseline f(θ) on the SAME evaluation batch
    baseline_loss = zo_forward(model, X_eval, Y_eval, ctx_obj)
    
    # Step 3: Evaluate ALL candidate updates on the SAME evaluation batch
    for i in range(directional_q):
        direction_seed = direction_seeds[i]
        c_i = projected_grads[i]
        
        # Apply candidate step: θ_candidate = θ - lr*c_i*Z_i
        candidate_scaling = -lr * c_i
        
        kronzo_perturb_parameters(
            model, direction_seed, step, zo_eps_global=zo_eps,
            scaling_factor=candidate_scaling, eps=1.0,  # Apply full candidate step directly
            strategy=strategy, max_factor=max_factor,
            kronzo_sampling_number=kronzo_sampling_number,
            b_dict=b_dict, step_interval=step_interval
        )
        
        # Evaluate f(θ + s_i) on the SAME evaluation batch
        candidate_loss = zo_forward(model, X_eval, Y_eval, ctx_obj)
        candidate_losses.append(candidate_loss.item())
        
        # Track best candidate
        if candidate_loss < best_loss:
            best_loss = candidate_loss.item()
            best_direction_idx = i
        
        # Reset model back to θ
        kronzo_perturb_parameters(
            model, direction_seed, step, zo_eps_global=zo_eps,
            scaling_factor=-candidate_scaling, eps=1.0,  # Undo the candidate step exactly
            strategy=strategy, max_factor=max_factor,
            kronzo_sampling_number=kronzo_sampling_number,
            b_dict=b_dict, step_interval=step_interval
        )
    
    # Step 4: Decide whether to accept best candidate
    # Compare candidate loss with baseline loss (both on same evaluation batch)
    should_update = loss_history.should_accept_update(best_loss, baseline_loss.item())
    
    # Prepare return data with enhanced logging information
    improvement_vs_baseline = baseline_loss.item() - best_loss if best_direction_idx >= 0 else 0.0
    
    best_direction_info = {
        'direction_idx': best_direction_idx,
        'direction_seed': direction_seeds[best_direction_idx] if best_direction_idx >= 0 else None,
        'projected_grad': projected_grads[best_direction_idx] if best_direction_idx >= 0 else 0.0,
        'candidate_loss': best_loss,
        'baseline_loss': baseline_loss.item(),
        'improvement': improvement_vs_baseline,
        'improvement_magnitude': abs(improvement_vs_baseline),
        'relative_improvement': abs(improvement_vs_baseline) / baseline_loss.item() if baseline_loss.item() > 0 else 0.0,
        'used_fresh_batch': used_fresh_batch,
        'all_candidate_losses': candidate_losses,
        'all_projected_grads': projected_grads,
        'candidate_loss_range': max(candidate_losses) - min(candidate_losses) if candidate_losses else 0.0,
        'num_candidates': len(candidate_losses)
    }
    
    candidate_step_data = {
        'best_seed': direction_seeds[best_direction_idx] if best_direction_idx >= 0 else None,
        'best_projected_grad': projected_grads[best_direction_idx] if best_direction_idx >= 0 else 0.0,
        'lr': lr,
        'zo_eps': zo_eps,
        'strategy': strategy,
        'max_factor': max_factor,
        'kronzo_sampling_number': kronzo_sampling_number,
        'b_dict': b_dict,
        'step_interval': step_interval,
        'step': step
    }
    
    return (baseline_loss.item(), best_loss, best_direction_info, 
            should_update, candidate_step_data)

def improved_kronzo_update(model,
                          optimizer,  # kept for API compatibility
                          candidate_step_data: dict,
                          master_process: bool = False,
                          weight_decay: float = 0.1,
                          named_parameters_to_optim: list[tuple[str, torch.Tensor]] | None = None):
    """
    Apply the improved KronZO update using pre-computed candidate step data.
    
    CRITICAL: This applies exactly the same step that was evaluated in improved_kronzo_step.
    We use kronzo_perturb_parameters with the same scaling to ensure consistency.
    
    Args:
        model: The model to update
        optimizer: Optimizer (kept for API compatibility)
        candidate_step_data: Data from improved_kronzo_step containing update info
        master_process: Whether this is the master process (for logging)
        weight_decay: Weight decay coefficient
        named_parameters_to_optim: List of parameters to optimize
    """
    best_seed = candidate_step_data['best_seed']
    best_projected_grad = candidate_step_data['best_projected_grad']
    lr = candidate_step_data['lr']
    strategy = candidate_step_data['strategy']
    max_factor = candidate_step_data['max_factor']
    kronzo_sampling_number = candidate_step_data['kronzo_sampling_number']
    b_dict = candidate_step_data['b_dict']
    step_interval = candidate_step_data['step_interval']
    step = candidate_step_data['step']
    
    if best_seed is None:
        # No valid update to apply
        return
    
    # FIXED: Apply exactly the same step that was evaluated
    # Use the same scaling factor: candidate_scaling = -lr * c_best
    candidate_scaling = -lr * best_projected_grad
    
    # CRITICAL FIX: Use zo_eps (not lr) to match the evaluation phase exactly
    zo_eps = candidate_step_data.get('zo_eps', lr)  # Fallback for compatibility
    
    # Apply the evaluated step using kronzo_perturb_parameters (same as evaluation)
    kronzo_perturb_parameters(
        model, best_seed, step, zo_eps_global=zo_eps,  # FIXED: Use zo_eps, not lr
        scaling_factor=candidate_scaling, eps=1.0,  # Exact same call as in evaluation
        strategy=strategy, max_factor=max_factor,
        kronzo_sampling_number=kronzo_sampling_number,
        b_dict=b_dict, step_interval=step_interval
    )
    
    # Apply weight decay separately if needed
    if weight_decay > 0:
        if named_parameters_to_optim is None:
            named_parameters_to_optim = [
                (n if not n.startswith("_orig_mod.") else n[len("_orig_mod."):], p)
                for n, p in model.named_parameters()
                if p.requires_grad
            ]
        
        for clean_name, param in named_parameters_to_optim:
            is_weight = ("bias" not in clean_name and 
                        "layer_norm" not in clean_name and 
                        "layernorm" not in clean_name)
            
            if is_weight:
                param.data.mul_(1 - lr * weight_decay)

def improved_kronzo_update_momentum(model,
                                   optimizer,  # kept for API compatibility  
                                   candidate_step_data: dict,
                                   exp_avg_m: dict[str, torch.Tensor],
                                   beta1: float = 0.9,
                                   master_process: bool = False,
                                   weight_decay: float = 0.1,
                                   named_parameters_to_optim: list[tuple[str, torch.Tensor]] | None = None):
    """
    Apply improved KronZO update with momentum.
    
    CRITICAL: This applies momentum to the exact same step that was evaluated.
    Instead of storing momentum for A matrices, we store momentum for the full step.
    
    Args:
        model: The model to update
        optimizer: Optimizer (kept for API compatibility)
        candidate_step_data: Data from improved_kronzo_step
        exp_avg_m: Dictionary storing momentum states
        beta1: Momentum coefficient
        master_process: Whether this is master process (for logging)
        weight_decay: Weight decay coefficient  
        named_parameters_to_optim: List of parameters to optimize
    """
    best_seed = candidate_step_data['best_seed']
    best_projected_grad = candidate_step_data['best_projected_grad']
    lr = candidate_step_data['lr']
    strategy = candidate_step_data['strategy']
    max_factor = candidate_step_data['max_factor']
    kronzo_sampling_number = candidate_step_data['kronzo_sampling_number']
    b_dict = candidate_step_data['b_dict']
    step_interval = candidate_step_data['step_interval']
    step = candidate_step_data['step']
    
    if best_seed is None:
        # No valid update to apply
        return

    # Step 1: Temporarily apply the evaluated step to compute the update direction
    candidate_scaling = -lr * best_projected_grad
    
    # CRITICAL FIX: Use zo_eps (not lr) to match the evaluation phase exactly
    zo_eps = candidate_step_data.get('zo_eps', lr)  # Fallback for compatibility
    
    # Store original parameters
    if named_parameters_to_optim is None:
        named_parameters_to_optim = [
            (n if not n.startswith("_orig_mod.") else n[len("_orig_mod."):], p)
            for n, p in model.named_parameters()
            if p.requires_grad
        ]
    
    original_params = {}
    for clean_name, param in named_parameters_to_optim:
        original_params[clean_name] = param.data.clone()
    
    # Apply the step to get the update direction
    kronzo_perturb_parameters(
        model, best_seed, step, zo_eps_global=zo_eps,  # FIXED: Use zo_eps, not lr
        scaling_factor=candidate_scaling, eps=1.0,
        strategy=strategy, max_factor=max_factor,
        kronzo_sampling_number=kronzo_sampling_number,
        b_dict=b_dict, step_interval=step_interval
    )
    
    # Compute the update direction: Δθ = θ_new - θ_old
    update_directions = {}
    for clean_name, param in named_parameters_to_optim:
        update_directions[clean_name] = param.data - original_params[clean_name]
    
    # Reset to original parameters
    for clean_name, param in named_parameters_to_optim:
        param.data.copy_(original_params[clean_name])
    
    # Step 2: Apply momentum to the update direction
    for clean_name, param in named_parameters_to_optim:
        update_direction = update_directions[clean_name]
        
        # Initialize momentum if needed
        momentum_key = f"{clean_name}_momentum"
        if momentum_key not in exp_avg_m:
            exp_avg_m[momentum_key] = torch.zeros_like(param.data)
        
        # Update momentum: m_t = β * m_{t-1} + (1-β) * Δθ
        exp_avg_m[momentum_key] = beta1 * exp_avg_m[momentum_key] + (1 - beta1) * update_direction
        
        # Apply momentum update
        param.data.add_(exp_avg_m[momentum_key])
        
        # Apply weight decay
        if weight_decay > 0:
            is_weight = ("bias" not in clean_name and 
                        "layer_norm" not in clean_name and 
                        "layernorm" not in clean_name)
            
            if is_weight:
                param.data.mul_(1 - lr * weight_decay) 