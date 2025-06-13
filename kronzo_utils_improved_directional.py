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
    
    Two modes:
    1. SUCCESS MODE (default): Keep sliding window of previous SUCCESSFUL losses
       - Accept update only if candidate_loss ≤ max{previous successful losses}  
       - Only update the sliding window when an update is actually accepted
    
    2. BASELINE MODE: Keep sliding window of previous BASELINE losses (regardless of acceptance)
       - Accept update only if candidate_loss ≤ max{previous baseline losses}
       - Update sliding window every iteration regardless of acceptance
    """
    def __init__(self, history_size: int = 10, track_baseline_losses: bool = False):
        self.history_size = history_size
        self.track_baseline_losses = track_baseline_losses
        
        if track_baseline_losses:
            self.loss_history = deque(maxlen=history_size)  # Stores baseline losses
        else:
            self.loss_history = deque(maxlen=history_size)  # Stores successful update losses
            
        # Add tracking for verification
        self.total_decisions = 0
        self.accepted_decisions = 0
        self.rejected_due_to_history = 0
        self.accepted_due_to_improvement = 0
    
    def add_loss(self, loss_value: float):
        """Add a successful loss value to the history (only called on successful updates)."""
        if not self.track_baseline_losses:
            self.loss_history.append(loss_value)
        # If tracking baseline losses, this method is ignored (use add_baseline_loss instead)
    
    def add_baseline_loss(self, baseline_loss_value: float):
        """Add a baseline loss value to the history (called every iteration regardless of acceptance)."""
        if self.track_baseline_losses:
            self.loss_history.append(baseline_loss_value)
        # If not tracking baseline losses, this method is ignored (use add_loss instead)
    
    def should_accept_update(self, candidate_loss: float, baseline_loss: float) -> bool:
        """
        Determine if we should accept an update based on candidate loss.
        
        FIXED LOGIC:
        - If no history yet: accept any improvement over baseline (more lenient)
        - If have history: accept if candidate_loss ≤ max{previous successful losses}
        - Also accept if candidate shows significant improvement over baseline (escape hatch)
        
        Args:
            candidate_loss: Loss value for the candidate update
            baseline_loss: Current baseline loss f(θ) (used for initialization and escape hatch)
            
        Returns:
            True if update should be accepted
        """
        self.total_decisions += 1
        
        # During initialization phase (not enough history yet)
        if len(self.loss_history) < self.history_size:
            # Accept if candidate improves over baseline OR is close to baseline
            improvement_threshold = 0.01  # Accept if within 1% of baseline
            should_accept = candidate_loss <= baseline_loss * (1 + improvement_threshold)
            if should_accept:
                self.accepted_decisions += 1
                self.accepted_due_to_improvement += 1
            return should_accept
            
        # Conservative phase: candidate must beat the worst of our successful history
        max_historical_loss = max(self.loss_history)
        
        # Primary criterion: beat historical performance
        should_accept = candidate_loss <= max_historical_loss
        
        # Escape hatch: accept significant improvements over current baseline
        if not should_accept:
            relative_improvement = (baseline_loss - candidate_loss) / baseline_loss
            if relative_improvement > 0.05:  # 5% improvement over current baseline
                should_accept = True
                if self.total_decisions % 50 == 0:  # Log occasionally
                    print(f"  Accepting due to escape hatch: {relative_improvement:.2%} improvement")
        
        if should_accept:
            self.accepted_decisions += 1
        else:
            self.rejected_due_to_history += 1
            
        return should_accept
    
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
    
    def get_acceptance_rate(self) -> float:
        """Get the overall acceptance rate."""
        if self.total_decisions == 0:
            return 0.0
        return self.accepted_decisions / self.total_decisions
    
    def get_history_info(self) -> dict:
        """Get information about current history state."""
        if len(self.loss_history) == 0:
            return {
                'count': 0,
                'best': float('inf'),
                'worst': float('inf'),
                'threshold': float('inf'),
                'phase': 'empty',
                'acceptance_rate': self.get_acceptance_rate(),
                'total_decisions': self.total_decisions,
                'accepted': self.accepted_decisions,
                'rejected_due_to_history': self.rejected_due_to_history,
                'accepted_due_to_improvement': self.accepted_due_to_improvement
            }
        
        return {
            'count': len(self.loss_history),
            'best': min(self.loss_history),
            'worst': max(self.loss_history),
            'threshold': max(self.loss_history),
            'phase': 'initialization' if len(self.loss_history) < self.history_size else 'conservative',
            'acceptance_rate': self.get_acceptance_rate(),
            'total_decisions': self.total_decisions,
            'accepted': self.accepted_decisions,
            'rejected_due_to_history': self.rejected_due_to_history,
            'accepted_due_to_improvement': self.accepted_due_to_improvement
        }

class AdaptiveParameterTracker:
    """
    Tracks gradient quality (CV) and acceptance rate for adaptive parameter tuning.
    
    Maintains sliding windows for:
    - Coefficient of Variation (CV) of projected gradients
    - Acceptance rate of updates
    
    Adaptively adjusts:
    - ε (perturbation size) based on CV
    - α (learning rate) based on acceptance rate and CV
    
    Supports warmup period where parameters remain fixed until warmup is complete.
    """
    def __init__(self, 
                 window_size: int = 100,
                 eps_min: float = 5e-4,
                 eps_max: float = 5e-2,
                 alpha_min: float = 1e-4,
                 alpha_max: float = 3e-2,
                 initial_eps: float = 1e-3,
                 initial_alpha: float = 1e-3,
                 warmup_iters: int = 0):
        self.window_size = window_size
        
        # Parameter bounds
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        
        # Current adaptive parameters
        self.current_eps = initial_eps
        self.current_alpha = initial_alpha
        self.initial_eps = initial_eps  # Store for warmup
        self.initial_alpha = initial_alpha  # Store for warmup
        
        # Warmup settings
        self.warmup_iters = warmup_iters
        self.current_iter = 0
        
        # Sliding windows
        self.cv_history = deque(maxlen=window_size)
        self.acceptance_history = deque(maxlen=window_size)  # True/False for each iteration
        
        # Statistics
        self.total_iterations = 0
        self.eps_updates = 0
        self.alpha_updates = 0
        
    def is_in_warmup(self) -> bool:
        """Check if we are still in the warmup period."""
        return self.current_iter < self.warmup_iters
        
    def set_current_iter(self, iter_num: int):
        """Set the current iteration number for warmup tracking."""
        self.current_iter = iter_num
        
    def set_warmup_alpha(self, warmup_alpha: float):
        """Set the alpha value to use during warmup (from lr schedule)."""
        if self.is_in_warmup():
            self.current_alpha = warmup_alpha
    
    def add_iteration_data(self, projected_grads: list[float], was_accepted: bool):
        """
        Add data from one iteration.
        
        Args:
            projected_grads: List of projected gradient coefficients c_i
            was_accepted: Whether the update was accepted
        """
        self.total_iterations += 1
        
        # Compute CV for this iteration
        if len(projected_grads) > 1:
            grads_array = np.array(projected_grads)
            mean_grad = np.mean(grads_array)
            std_grad = np.std(grads_array, ddof=1)  # Sample std
            
            # CV with stability constant
            eps_num = 1e-7
            cv = std_grad / (abs(mean_grad) + eps_num)
            self.cv_history.append(cv)
        else:
            # Single direction: CV undefined, use special value
            self.cv_history.append(float('nan'))
        
        # Record acceptance
        self.acceptance_history.append(was_accepted)
        
        # Update parameters based on sliding window averages
        # Only if we're past the warmup period
        if not self.is_in_warmup():
            self._update_parameters()
    
    def _update_parameters(self):
        """Update ε and α based on current sliding window statistics."""
        if len(self.cv_history) == 0:
            return
        
        # Calculate current CV average (ignoring NaN values)
        valid_cvs = [cv for cv in self.cv_history if not np.isnan(cv)]
        if len(valid_cvs) > 0:
            avg_cv = np.mean(valid_cvs)
        else:
            avg_cv = float('nan')
        
        # Calculate current acceptance rate
        if len(self.acceptance_history) > 0:
            acceptance_rate = sum(self.acceptance_history) / len(self.acceptance_history)
        else:
            acceptance_rate = 0.0
        
        # Update ε based on CV
        if not np.isnan(avg_cv):
            old_eps = self.current_eps
            if avg_cv > 1.0:
                # Noise dominates: increase ε
                self.current_eps = min(self.current_eps * 1.01, self.eps_max)
            elif avg_cv < 0.4:
                # Likely bias: decrease ε
                self.current_eps = max(self.current_eps * 0.99, self.eps_min)
            
            if abs(self.current_eps - old_eps) > 1e-10:
                self.eps_updates += 1
        
        # Update α based on acceptance rate and CV
        old_alpha = self.current_alpha
        if acceptance_rate < 0.4 and (np.isnan(avg_cv) or avg_cv < 1.0):
            # Step too large: decrease α
            self.current_alpha = max(self.current_alpha * 0.99, self.alpha_min)
        elif acceptance_rate > 0.6:
            # Step too small: increase α
            self.current_alpha = min(self.current_alpha * 1.01, self.alpha_max)
        
        if abs(self.current_alpha - old_alpha) > 1e-10:
            self.alpha_updates += 1
    
    def get_current_cv(self) -> float:
        """Get the current average CV from the sliding window."""
        if len(self.cv_history) == 0:
            return float('nan')
        
        valid_cvs = [cv for cv in self.cv_history if not np.isnan(cv)]
        if len(valid_cvs) == 0:
            return float('nan')
        
        return np.mean(valid_cvs)
    
    def get_current_acceptance_rate(self) -> float:
        """Get the current acceptance rate from the sliding window."""
        if len(self.acceptance_history) == 0:
            return 0.0
        
        return sum(self.acceptance_history) / len(self.acceptance_history)
    
    def get_adaptive_parameters(self) -> tuple[float, float]:
        """Get the current adaptive parameters (eps, alpha)."""
        return self.current_eps, self.current_alpha
    
    def get_statistics(self) -> dict:
        """Get comprehensive statistics about the adaptive tracking."""
        return {
            'total_iterations': self.total_iterations,
            'current_eps': self.current_eps,
            'current_alpha': self.current_alpha,
            'current_cv': self.get_current_cv(),
            'current_acceptance_rate': self.get_current_acceptance_rate(),
            'window_fill': len(self.cv_history),
            'window_size': self.window_size,
            'eps_updates': self.eps_updates,
            'alpha_updates': self.alpha_updates,
            'eps_bounds': (self.eps_min, self.eps_max),
            'alpha_bounds': (self.alpha_min, self.alpha_max),
            'is_in_warmup': self.is_in_warmup(),
            'warmup_iters': self.warmup_iters,
            'current_iter': self.current_iter
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
    
    CORRECTED ALGORITHM: 
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

def adaptive_improved_kronzo_step(model, X, Y, step, zo_random_seed, directional_q, 
                                 lr_adaptive, ctx_obj, strategy, max_factor, 
                                 kronzo_sampling_number, b_dict, step_interval, 
                                 loss_history, adaptive_tracker, get_batch_fn):
    """
    Improved directional KronZO step with adaptive parameter tuning.
    
    Uses coefficient of variation (CV) to adapt ε and acceptance rate to adapt α.
    Supports warmup period where parameters follow fixed schedule until warmup completes.
    
    During warmup:
    - ε remains constant at initial value
    - α follows the provided lr_adaptive (from lr schedule)
    
    After warmup:
    - ε adapts based on CV of projected gradients
    - α adapts based on acceptance rate and CV
    """
    # Get adaptive parameters from tracker
    current_eps, current_alpha = adaptive_tracker.get_adaptive_parameters()
    
    # During warmup, use provided lr_adaptive instead of adaptive alpha
    if adaptive_tracker.is_in_warmup():
        effective_alpha = lr_adaptive  # Use lr schedule during warmup
        effective_eps = adaptive_tracker.initial_eps  # Keep eps constant during warmup
        adaptive_tracker.set_warmup_alpha(lr_adaptive)  # Update tracker for consistency
    else:
        effective_alpha = current_alpha  # Use adaptive alpha after warmup
        effective_eps = current_eps     # Use adaptive eps after warmup
    
    # Call the standard improved kronzo step with effective parameters
    (baseline_loss, best_candidate_loss, best_direction_info, 
     should_update, candidate_step_data) = improved_kronzo_step(
        model, X, Y, step, zo_random_seed, directional_q,
        zo_eps=effective_eps,  # Use effective eps (constant during warmup, adaptive after)
        lr=effective_alpha,    # Use effective alpha (lr schedule during warmup, adaptive after)
        ctx_obj=ctx_obj, strategy=strategy, max_factor=max_factor,
        kronzo_sampling_number=kronzo_sampling_number, b_dict=b_dict,
        step_interval=step_interval, loss_history=loss_history,
        get_batch_fn=get_batch_fn
    )
    
    # Extract gradient information for adaptive tracking
    projected_grads = best_direction_info.get('all_projected_grads', [])
    
    # Update adaptive tracker with this iteration's data
    # This will only update parameters if we're past warmup
    adaptive_tracker.add_iteration_data(projected_grads, should_update)
    
    # Add adaptive information to best_direction_info for logging
    best_direction_info['adaptive_eps'] = effective_eps
    best_direction_info['adaptive_alpha'] = effective_alpha
    best_direction_info['current_cv'] = adaptive_tracker.get_current_cv()
    best_direction_info['current_acceptance_rate'] = adaptive_tracker.get_current_acceptance_rate()
    best_direction_info['is_in_warmup'] = adaptive_tracker.is_in_warmup()
    
    return baseline_loss, best_candidate_loss, best_direction_info, should_update, candidate_step_data

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
    Apply improved KronZO update with PROPER matrix-level momentum that maintains evaluation consistency.
    
    CORRECT APPROACH:
    1. Use the SAME kronzo_perturb_parameters function to generate the exact matrices as evaluation
    2. Apply momentum to the (coefficient * A) matrices per parameter
    3. Reconstruct the full perturbation using stored B matrices
    4. This maintains both consistency AND proper momentum accumulation
    
    Args:
        model: The model to update
        optimizer: Optimizer (kept for API compatibility)
        candidate_step_data: Data from improved_kronzo_step
        exp_avg_m: Dictionary storing momentum states (stores A matrices per parameter)
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

    if named_parameters_to_optim is None:
        named_parameters_to_optim = [
            (n if not n.startswith("_orig_mod.") else n[len("_orig_mod."):], p)
            for n, p in model.named_parameters()
            if p.requires_grad
        ]

    if b_dict is None:
        b_dict = {}

    # Set the random seed to match the evaluation phase exactly
    torch.manual_seed(best_seed)
    
    # Debug logging
    if step % 100 == 0 and master_process:
        print(f"Improved KronZO-M step {step}: coefficient={best_projected_grad:.6f}, "
              f"beta1={beta1}, using matrix-level momentum")

    # Apply matrix-level momentum using the same approach as DiKronZO
    for clean_name, param in named_parameters_to_optim:
        if param.ndim >= 2:  # matrices
            d_out, d_in = param.shape
            
            if kronzo_sampling_number == 1:
                # Standard single Kronecker product with efficient momentum storage
                m1, m2, n1, n2 = choose_kron_dims(d_out, d_in, strategy, max_factor)
                
                # Always sample fresh A matrix (same as evaluation)
                A = torch.randn(m1, n1, device=param.device, dtype=param.dtype)
                
                # B matrix: use stored B from evaluation (must be consistent)
                b_key = f"{clean_name}_B"
                if b_key in b_dict:
                    B = b_dict[b_key]
                else:
                    # Fallback: reconstruct B matrix with same logic as evaluation
                    need_new_b = (step % step_interval == 0 or b_key not in b_dict)
                    if need_new_b:
                        torch.manual_seed(1000000 + step * 1000 + hash(clean_name) % 1000)
                        B = torch.randn(m2, n2, device=param.device, dtype=param.dtype)
                        b_dict[b_key] = B
                    else:
                        B = b_dict[b_key]
                
                # EFFICIENT MOMENTUM: Store only A matrix scaled by gradient coefficient
                momentum_key = f"{clean_name}_A"
                if momentum_key not in exp_avg_m:
                    exp_avg_m[momentum_key] = torch.zeros_like(A)
                
                # Update momentum: m_t = β * m_{t-1} + (1-β) * (c * A)
                grad_scaled_A = best_projected_grad * A
                exp_avg_m[momentum_key] = beta1 * exp_avg_m[momentum_key] + (1 - beta1) * grad_scaled_A
                
                # Apply momentum: A_momentum ⊗ B
                momentum_perturbation = torch.kron(exp_avg_m[momentum_key], B)
                
            else:
                # Multi-sampling: efficient momentum for each factorization
                factorizations = choose_diverse_kron_dims(d_out, d_in, kronzo_sampling_number, strategy, max_factor)
                momentum_perturbation = torch.zeros(d_out, d_in, device=param.device, dtype=param.dtype)
                
                for i, (m1, m2, n1, n2) in enumerate(factorizations):
                    # Always sample fresh A matrix (with different seed for each factorization)
                    torch.manual_seed(best_seed + i + 1)
                    A = torch.randn(m1, n1, device=param.device, dtype=param.dtype)
                    
                    # B matrix: use stored B from evaluation
                    b_key = f"{clean_name}_B_{i}"
                    if b_key in b_dict:
                        B = b_dict[b_key]
                    else:
                        # Fallback: reconstruct B matrix
                        need_new_b = (step % step_interval == 0 or b_key not in b_dict)
                        if need_new_b:
                            torch.manual_seed(1000000 + step * 1000 + hash(clean_name) % 1000 + i)
                            B = torch.randn(m2, n2, device=param.device, dtype=param.dtype)
                            b_dict[b_key] = B
                        else:
                            B = b_dict[b_key]
                    
                    # EFFICIENT MOMENTUM: Store only A matrix scaled by gradient coefficient
                    momentum_key = f"{clean_name}_A_{i}"
                    if momentum_key not in exp_avg_m:
                        exp_avg_m[momentum_key] = torch.zeros_like(A)
                    
                    # Update momentum: m_t = β * m_{t-1} + (1-β) * (c * A)
                    grad_scaled_A = best_projected_grad * A
                    exp_avg_m[momentum_key] = beta1 * exp_avg_m[momentum_key] + (1 - beta1) * grad_scaled_A
                    
                    # Apply momentum: A_momentum ⊗ B
                    kron_momentum = torch.kron(exp_avg_m[momentum_key], B)
                    momentum_perturbation.add_(kron_momentum)
                
                # Scale by 1/kronzo_sampling_number to maintain similar magnitude
                momentum_perturbation.mul_(1.0 / kronzo_sampling_number)

            # Apply the momentum update
            is_weight = ("bias" not in clean_name and 
                        "layer_norm" not in clean_name and 
                        "layernorm" not in clean_name)
            
            if is_weight:
                param.data.sub_(lr * (momentum_perturbation + weight_decay * param.data))
            else:
                param.data.sub_(lr * momentum_perturbation)
                
        else:  # vectors
            z = torch.normal(0.0, 1.0,
                             size=param.size(),
                             device=param.device,
                             dtype=param.dtype)

            # Momentum for vectors (standard approach)
            momentum_key = clean_name
            if momentum_key not in exp_avg_m:
                exp_avg_m[momentum_key] = torch.zeros_like(z)
            exp_avg_m[momentum_key] = (beta1 * exp_avg_m[momentum_key] +
                                       (1 - beta1) * best_projected_grad * z)

            is_weight = ("bias" not in clean_name and 
                        "layer_norm" not in clean_name and 
                        "layernorm" not in clean_name)

            if is_weight:
                param.data.sub_(lr * (exp_avg_m[momentum_key] + weight_decay * param.data))
            else:
                param.data.sub_(lr * exp_avg_m[momentum_key]) 