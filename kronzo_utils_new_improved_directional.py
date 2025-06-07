import torch
import numpy as np
from collections import deque
from common_utils import zo_forward
from kronzo_utils import (
    choose_kron_dims, choose_diverse_kron_dims, 
    kronzo_perturb_parameters, kronzo_update, kronzo_update_momentum
)

class NewImprovedDirectionalHistory:
    """
    Maintains history for conservative update decisions with configurable strategy.
    
    TWO MODES:
    1. BASELINE HISTORY MODE (use_baseline_history=True): 
       - Keep sliding window of previous BASELINE evaluations f(θ_k; batch_k)
       - Accept update only if candidate_loss ≤ max{previous baseline evaluations}  
       - Update history AFTER each decision, regardless of acceptance
    
    2. SUCCESS HISTORY MODE (use_baseline_history=False):
       - Keep sliding window of previous SUCCESSFUL update losses (like improved_kronzo)
       - Accept update only if candidate_loss ≤ max{previous successful losses}
       - Update history ONLY when update is accepted
    """
    def __init__(self, history_size: int = 10, use_baseline_history: bool = True):
        self.history_size = history_size
        self.use_baseline_history = use_baseline_history
        
        if use_baseline_history:
            self.baseline_history = deque(maxlen=history_size)  # Stores baseline losses from each iteration
        else:
            self.loss_history = deque(maxlen=history_size)  # Stores successful update losses (like improved_kronzo)
            
        # Add tracking for verification
        self.total_decisions = 0
        self.accepted_decisions = 0
        self.rejected_due_to_history = 0
        self.accepted_due_to_improvement = 0
        self.initialization_phase_decisions = 0
    
    def add_baseline_loss(self, baseline_loss_value: float):
        """Add a baseline loss value to the history (called after each iteration decision)."""
        if self.use_baseline_history:
            self.baseline_history.append(baseline_loss_value)
        # In success mode, this method is not used
    
    def add_successful_loss(self, successful_loss_value: float):
        """Add a successful update loss to the history (called only when update is accepted)."""
        if not self.use_baseline_history:
            self.loss_history.append(successful_loss_value)
        # In baseline mode, this method is not used
    
    def should_accept_update(self, candidate_loss: float, baseline_loss: float) -> bool:
        """
        Determine if we should accept an update based on candidate loss.
        
        DUAL MODE LOGIC:
        - BASELINE MODE: Compare against recent baseline losses (less restrictive)
        - SUCCESS MODE: Compare against successful update losses (more restrictive, like improved_kronzo)
        
        Args:
            candidate_loss: Loss value for the candidate update
            baseline_loss: Current baseline loss f(θ) (for initialization and escape hatch)
            
        Returns:
            True if update should be accepted
        """
        self.total_decisions += 1
        
        if self.use_baseline_history:
            # BASELINE HISTORY MODE (new approach)
            # During initialization phase (not enough baseline history yet)
            if len(self.baseline_history) < self.history_size:
                # Accept if candidate improves over baseline OR is close to baseline
                improvement_threshold = 0.01  # Accept if within 1% of baseline
                should_accept = candidate_loss <= baseline_loss * (1 + improvement_threshold)
                if should_accept:
                    self.accepted_decisions += 1
                    self.accepted_due_to_improvement += 1
                return should_accept
                
            # Conservative phase: candidate must beat the worst of recent baseline performance
            max_baseline_loss = max(self.baseline_history)
            
            # Primary criterion: beat recent baseline performance
            should_accept = candidate_loss <= max_baseline_loss
            
            # Escape hatch: accept significant improvements over current baseline
            if not should_accept:
                relative_improvement = (baseline_loss - candidate_loss) / baseline_loss
                if relative_improvement > 0.05:  # 5% improvement over current baseline
                    should_accept = True
                    if self.total_decisions % 50 == 0:  # Log occasionally
                        print(f"  New Improved KronZO escape hatch: {relative_improvement:.2%} improvement")
        
        else:
            # SUCCESS HISTORY MODE (like improved_kronzo)
            # During initialization phase (not enough success history yet)
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
                        print(f"  New Improved KronZO escape hatch: {relative_improvement:.2%} improvement")
        
        if should_accept:
            self.accepted_decisions += 1
        else:
            self.rejected_due_to_history += 1
            
        return should_accept
    
    def get_best_historical_baseline(self) -> float:
        """Get the best (minimum) baseline from history."""
        if self.use_baseline_history:
            if len(self.baseline_history) == 0:
                return float('inf')
            return min(self.baseline_history)
        else:
            if len(self.loss_history) == 0:
                return float('inf')
            return min(self.loss_history)
    
    def get_worst_historical_baseline(self) -> float:
        """Get the worst (maximum) baseline from history."""
        if self.use_baseline_history:
            if len(self.baseline_history) == 0:
                return float('inf')
            return max(self.baseline_history)
        else:
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
        if self.use_baseline_history:
            if len(self.baseline_history) == 0:
                return {
                    'count': 0,
                    'best_baseline': float('inf'),
                    'worst_baseline': float('inf'),
                    'threshold': float('inf'),
                    'phase': 'empty',
                    'mode': 'baseline_history',
                    'acceptance_rate': self.get_acceptance_rate(),
                    'total_decisions': self.total_decisions,
                    'accepted': self.accepted_decisions,
                    'rejected_due_to_history': self.rejected_due_to_history,
                    'accepted_due_to_improvement': self.accepted_due_to_improvement,
                    'initialization_decisions': self.initialization_phase_decisions
                }
            
            return {
                'count': len(self.baseline_history),
                'best_baseline': min(self.baseline_history),
                'worst_baseline': max(self.baseline_history),
                'threshold': max(self.baseline_history),
                'phase': 'initialization' if len(self.baseline_history) < self.history_size else 'conservative',
                'mode': 'baseline_history',
                'acceptance_rate': self.get_acceptance_rate(),
                'total_decisions': self.total_decisions,
                'accepted': self.accepted_decisions,
                'rejected_due_to_history': self.rejected_due_to_history,
                'accepted_due_to_improvement': self.accepted_due_to_improvement,
                'initialization_decisions': self.initialization_phase_decisions
            }
        else:
            if len(self.loss_history) == 0:
                return {
                    'count': 0,
                    'best': float('inf'),
                    'worst': float('inf'),
                    'threshold': float('inf'),
                    'phase': 'empty',
                    'mode': 'success_history',
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
                'mode': 'success_history',
                'acceptance_rate': self.get_acceptance_rate(),
                'total_decisions': self.total_decisions,
                'accepted': self.accepted_decisions,
                'rejected_due_to_history': self.rejected_due_to_history,
                'accepted_due_to_improvement': self.accepted_due_to_improvement
            }

def new_improved_kronzo_step(model, 
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
                            rank_kronzo: int = 4,
                            va_dict: dict | None = None,
                            vb_dict: dict | None = None,
                            b_dict: dict | None = None,
                            step_interval: int = 50,
                            loss_history: NewImprovedDirectionalHistory | None = None,
                            use_baseline_history: bool = True,
                            use_lowrank_factorization: bool = True,
                            get_batch_fn=None):
    """
    New improved directional KronZO step with DUAL MATRIX SAMPLING support.
    
    DUAL MODE DESIGN:
    - Low-rank mode: A = U_A @ V_A^T, B = U_B @ V_B^T (memory efficient)
    - Traditional mode: A, B sampled directly (like improved_kronzo)
    
    ALGORITHM: 
    - Get ONE fresh batch per iteration: (X_eval, Y_eval) = get_batch_fn()
    - Evaluate ALL candidates f(θ + s_i) on the SAME batch (X_eval, Y_eval) for fair comparison
    - Evaluate baseline f(θ) on the SAME batch (X_eval, Y_eval)
    - Accept update based on history strategy (baseline vs success history)
    
    Algorithm:
    1. Get ONE fresh evaluation batch for this entire iteration
    2. For each direction i = 1, ..., directional_q:
       a. Sample Z_i: Low-rank (U_A @ V_A^T ⊗ U_B @ V_B^T) OR Traditional (A ⊗ B)
       b. Compute projected gradient: c_i = [f(θ + ε*Z_i) - f(θ - ε*Z_i)] / (2ε) on batch (X, Y)
       c. Compute candidate step: s_i = -lr*c_i*Z_i
       d. Evaluate actual update: f(θ + s_i) on the SAME evaluation batch (X_eval, Y_eval)
    3. Evaluate baseline: f(θ) on the SAME evaluation batch (X_eval, Y_eval)
    4. Find best candidate: i_best = argmin_i f(θ + s_i) 
    5. Accept update based on history strategy (baseline or success history)
    
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
        rank_kronzo: Rank for low-rank factorization (only used if use_lowrank_factorization=True)
        va_dict: Dictionary to store V_A matrices (only used if use_lowrank_factorization=True)
        vb_dict: Dictionary to store V_B matrices (only used if use_lowrank_factorization=True)
        b_dict: Dictionary to store B matrices (only used if use_lowrank_factorization=False)
        step_interval: Interval for updating V matrices (only used if use_lowrank_factorization=True)
        loss_history: History tracker for conservative updates
        use_baseline_history: Whether to use baseline history mode vs success history mode
        use_lowrank_factorization: Whether to use low-rank factorization vs traditional matrix sampling
        get_batch_fn: Function to get fresh batches (X_new, Y_new) = get_batch_fn()
        
    Returns:
        baseline_loss: Original loss f(θ) on evaluation batch
        best_candidate_loss: Loss of best candidate f(θ + s_best) on evaluation batch
        best_direction_info: Dict with best direction details
        should_update: Whether to accept the update
        candidate_step_data: Data needed for the actual update
    """
    if va_dict is None:
        va_dict = {}
    if vb_dict is None:
        vb_dict = {}
    if b_dict is None:
        b_dict = {}
    
    if loss_history is None:
        loss_history = NewImprovedDirectionalHistory(use_baseline_history=use_baseline_history)
    
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
    
    # Step 1: Compute projected gradients for all directions using CONDITIONAL perturbations
    for i in range(directional_q):
        direction_seed = zo_random_seed + i
        direction_seeds.append(direction_seed)
        
        # Phase 1a: Compute projected gradient c_i = [f(θ + ε*Z_i) - f(θ - ε*Z_i)] / (2ε)
        # Use ORIGINAL batch (X, Y) for gradient estimation
        
        if use_lowrank_factorization:
            # LOW-RANK MODE: f(θ + ε*Z_i) where Z_i = (U_A @ V_A^T) ⊗ (U_B @ V_B^T)
            lowrank_kronzo_perturb_parameters(
                model, direction_seed, step, zo_eps_global=zo_eps, 
                scaling_factor=1.0, eps=zo_eps,
                strategy=strategy, max_factor=max_factor,
                kronzo_sampling_number=kronzo_sampling_number,
                rank_kronzo=rank_kronzo,
                va_dict=va_dict, vb_dict=vb_dict, step_interval=step_interval
            )
        else:
            # TRADITIONAL MODE: f(θ + ε*Z_i) where Z_i = A ⊗ B (like improved_kronzo)
            kronzo_perturb_parameters(
                model, direction_seed, step, zo_eps_global=zo_eps, 
                scaling_factor=1.0, eps=zo_eps,
                strategy=strategy, max_factor=max_factor,
                kronzo_sampling_number=kronzo_sampling_number,
                b_dict=b_dict, step_interval=step_interval
            )
        loss_plus = zo_forward(model, X, Y, ctx_obj)
        
        # f(θ - ε*Z_i) 
        if use_lowrank_factorization:
            lowrank_kronzo_perturb_parameters(
                model, direction_seed, step, zo_eps_global=zo_eps,
                scaling_factor=-2.0, eps=zo_eps,  # -2 to go from +ε to -ε
                strategy=strategy, max_factor=max_factor,
                kronzo_sampling_number=kronzo_sampling_number,
                rank_kronzo=rank_kronzo,
                va_dict=va_dict, vb_dict=vb_dict, step_interval=step_interval
            )
        else:
            kronzo_perturb_parameters(
                model, direction_seed, step, zo_eps_global=zo_eps,
                scaling_factor=-2.0, eps=zo_eps,  # -2 to go from +ε to -ε
                strategy=strategy, max_factor=max_factor,
                kronzo_sampling_number=kronzo_sampling_number,
                b_dict=b_dict, step_interval=step_interval
            )
        loss_minus = zo_forward(model, X, Y, ctx_obj)
        
        # Reset to θ
        if use_lowrank_factorization:
            lowrank_kronzo_perturb_parameters(
                model, direction_seed, step, zo_eps_global=zo_eps,
                scaling_factor=1.0, eps=zo_eps,  # +1 to go from -ε back to θ
                strategy=strategy, max_factor=max_factor,
                kronzo_sampling_number=kronzo_sampling_number,
                rank_kronzo=rank_kronzo,
                va_dict=va_dict, vb_dict=vb_dict, step_interval=step_interval
            )
        else:
            kronzo_perturb_parameters(
                model, direction_seed, step, zo_eps_global=zo_eps,
                scaling_factor=1.0, eps=zo_eps,  # +1 to go from -ε back to θ
                strategy=strategy, max_factor=max_factor,
                kronzo_sampling_number=kronzo_sampling_number,
                b_dict=b_dict, step_interval=step_interval
            )
        
        # Compute projected gradient coefficient
        c_i = ((loss_plus - loss_minus) / (2 * zo_eps)).item()
        c_i *= (rank_kronzo**2)
        projected_grads.append(c_i)
    
    # Step 2: Evaluate baseline f(θ) on the SAME evaluation batch
    baseline_loss = zo_forward(model, X_eval, Y_eval, ctx_obj)
    
    # Step 3: Evaluate ALL candidate updates on the SAME evaluation batch using CONDITIONAL perturbations
    for i in range(directional_q):
        direction_seed = direction_seeds[i]
        c_i = projected_grads[i]
        
        # Apply candidate step: θ_candidate = θ - lr*c_i*Z_i
        candidate_scaling = -lr * c_i
        
        if use_lowrank_factorization:
            lowrank_kronzo_perturb_parameters(
                model, direction_seed, step, zo_eps_global=zo_eps,
                scaling_factor=candidate_scaling, eps=1.0,  # Apply full candidate step directly
                strategy=strategy, max_factor=max_factor,
                kronzo_sampling_number=kronzo_sampling_number,
                rank_kronzo=rank_kronzo,
                va_dict=va_dict, vb_dict=vb_dict, step_interval=step_interval
            )
        else:
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
        
        # Reset model back to θ using CONDITIONAL perturbations
        if use_lowrank_factorization:
            lowrank_kronzo_perturb_parameters(
                model, direction_seed, step, zo_eps_global=zo_eps,
                scaling_factor=-candidate_scaling, eps=1.0,  # Undo the candidate step exactly
                strategy=strategy, max_factor=max_factor,
                kronzo_sampling_number=kronzo_sampling_number,
                rank_kronzo=rank_kronzo,
                va_dict=va_dict, vb_dict=vb_dict, step_interval=step_interval
            )
        else:
            kronzo_perturb_parameters(
                model, direction_seed, step, zo_eps_global=zo_eps,
                scaling_factor=-candidate_scaling, eps=1.0,  # Undo the candidate step exactly
                strategy=strategy, max_factor=max_factor,
                kronzo_sampling_number=kronzo_sampling_number,
                b_dict=b_dict, step_interval=step_interval
            )
    
    # Step 4: Decide whether to accept best candidate
    # Compare candidate loss with historical performance (baseline or success history)
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
        'num_candidates': len(candidate_losses),
        'use_lowrank_factorization': use_lowrank_factorization,  # Include for debugging
        'rank_kronzo': rank_kronzo if use_lowrank_factorization else None  # Include rank info for debugging
    }
    
    candidate_step_data = {
        'best_seed': direction_seeds[best_direction_idx] if best_direction_idx >= 0 else None,
        'best_projected_grad': projected_grads[best_direction_idx] if best_direction_idx >= 0 else 0.0,
        'lr': lr,
        'zo_eps': zo_eps,
        'strategy': strategy,
        'max_factor': max_factor,
        'kronzo_sampling_number': kronzo_sampling_number,
        'use_lowrank_factorization': use_lowrank_factorization,
        'rank_kronzo': rank_kronzo,
        'va_dict': va_dict,
        'vb_dict': vb_dict,
        'b_dict': b_dict,
        'step_interval': step_interval,
        'step': step
    }
    
    return (baseline_loss.item(), best_loss, best_direction_info, 
            should_update, candidate_step_data)

def new_improved_kronzo_update(model,
                              optimizer,  # kept for API compatibility
                              candidate_step_data: dict,
                              master_process: bool = False,
                              weight_decay: float = 0.1,
                              named_parameters_to_optim: list[tuple[str, torch.Tensor]] | None = None):
    """
    Apply the new improved KronZO update using pre-computed candidate step data with DUAL MODE support.
    
    DUAL MODE DESIGN:
    - Low-rank mode: Uses low-rank factorization A = U_A @ V_A^T, B = U_B @ V_B^T
    - Traditional mode: Uses direct matrix sampling A, B (like improved_kronzo)
    
    CRITICAL: This applies exactly the same step that was evaluated in new_improved_kronzo_step.
    The perturbation method (low-rank vs traditional) matches the evaluation phase.
    
    Args:
        model: The model to update
        optimizer: Optimizer (kept for API compatibility)
        candidate_step_data: Data from new_improved_kronzo_step containing update info
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
    use_lowrank_factorization = candidate_step_data['use_lowrank_factorization']
    rank_kronzo = candidate_step_data['rank_kronzo']
    va_dict = candidate_step_data['va_dict']
    vb_dict = candidate_step_data['vb_dict']
    b_dict = candidate_step_data['b_dict']
    step_interval = candidate_step_data['step_interval']
    step = candidate_step_data['step']
    
    if best_seed is None:
        # No valid update to apply
        return
    
    # Apply exactly the same step that was evaluated
    # Use the same scaling factor: candidate_scaling = -lr * c_best
    candidate_scaling = -lr * best_projected_grad
    
    # Use zo_eps to match the evaluation phase exactly
    zo_eps = candidate_step_data.get('zo_eps', lr)  # Fallback for compatibility
    
    # Apply the evaluated step using the SAME method as evaluation
    if use_lowrank_factorization:
        # LOW-RANK MODE: Apply using lowrank_kronzo_perturb_parameters (same as evaluation)
        lowrank_kronzo_perturb_parameters(
            model, best_seed, step, zo_eps_global=zo_eps,
            scaling_factor=candidate_scaling, eps=1.0,  # Exact same call as in evaluation
            strategy=strategy, max_factor=max_factor,
            kronzo_sampling_number=kronzo_sampling_number,
            rank_kronzo=rank_kronzo,
            va_dict=va_dict, vb_dict=vb_dict, step_interval=step_interval
        )
    else:
        # TRADITIONAL MODE: Apply using kronzo_perturb_parameters (same as evaluation)
        kronzo_perturb_parameters(
            model, best_seed, step, zo_eps_global=zo_eps,
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

def new_improved_kronzo_update_momentum(model,
                                       optimizer,  # kept for API compatibility  
                                       candidate_step_data: dict,
                                       exp_avg_m: dict[str, torch.Tensor],
                                       beta1: float = 0.9,
                                       master_process: bool = False,
                                       weight_decay: float = 0.1,
                                       named_parameters_to_optim: list[tuple[str, torch.Tensor]] | None = None):
    """
    Apply new improved KronZO update with CORRECT momentum implementation using DUAL MODE support.
    
    DUAL MODE DESIGN:
    - Low-rank mode: Uses low-rank factorization A = U_A @ V_A^T, B = U_B @ V_B^T
    - Traditional mode: Uses direct matrix sampling A, B (like improved_kronzo)
    
    FIXED: Apply momentum to the gradient COEFFICIENT, not to the matrices.
    This ensures consistency with the evaluation phase and proper convergence.
    
    CORRECT ALGORITHM:
    1. Apply momentum to the projected gradient coefficient: c_momentum = β*c_prev + (1-β)*c_current
    2. Use the SAME perturbation function as evaluation (low-rank or traditional) with momentum coefficient
    3. This maintains exact consistency between evaluation and update phases
    
    Args:
        model: The model to update
        optimizer: Optimizer (kept for API compatibility)
        candidate_step_data: Data from new_improved_kronzo_step
        exp_avg_m: Dictionary storing momentum states (now stores scalar coefficients)
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
    use_lowrank_factorization = candidate_step_data['use_lowrank_factorization']
    rank_kronzo = candidate_step_data['rank_kronzo']
    va_dict = candidate_step_data['va_dict']
    vb_dict = candidate_step_data['vb_dict']
    b_dict = candidate_step_data['b_dict']
    step_interval = candidate_step_data['step_interval']
    step = candidate_step_data['step']
    
    if best_seed is None:
        # No valid update to apply
        return

    # CRITICAL FIX: Apply momentum to the gradient COEFFICIENT, not the matrices
    # This maintains consistency with the evaluation phase
    
    # Use a global momentum key for the gradient coefficient (persistent across steps)
    momentum_key = "global_grad_coeff"
    
    # Initialize momentum if first time
    if momentum_key not in exp_avg_m:
        exp_avg_m[momentum_key] = 0.0
    
    # Apply momentum to the gradient coefficient: m_t = β*m_{t-1} + (1-β)*c_t
    exp_avg_m[momentum_key] = beta1 * exp_avg_m[momentum_key] + (1 - beta1) * best_projected_grad
    momentum_grad_coeff = exp_avg_m[momentum_key]
    
    # Debug logging
    if step % 100 == 0 and master_process:
        mode_text = "Low-Rank" if use_lowrank_factorization else "Traditional"
        rank_text = f", rank={rank_kronzo}" if use_lowrank_factorization else ""
        print(f"New Improved {mode_text} KronZO-M step {step}: original_coeff={best_projected_grad:.6f}, "
              f"momentum_coeff={momentum_grad_coeff:.6f}, beta1={beta1}{rank_text}")
    
    # CRITICAL FIX: Use the SAME perturbation function as evaluation
    # but with the momentum-adjusted gradient coefficient
    candidate_scaling = -lr * momentum_grad_coeff  # Use momentum coefficient instead of original
    
    # Get zo_eps for consistency
    zo_eps = candidate_step_data.get('zo_eps', lr)
    
    # Apply the update using the EXACT same function as evaluation, just with momentum scaling
    if use_lowrank_factorization:
        # LOW-RANK MODE: Apply using lowrank_kronzo_perturb_parameters
        lowrank_kronzo_perturb_parameters(
            model, best_seed, step, zo_eps_global=zo_eps,
            scaling_factor=candidate_scaling, eps=1.0,  # Use momentum-adjusted scaling
            strategy=strategy, max_factor=max_factor,
            kronzo_sampling_number=kronzo_sampling_number,
            rank_kronzo=rank_kronzo,
            va_dict=va_dict, vb_dict=vb_dict, step_interval=step_interval
        )
    else:
        # TRADITIONAL MODE: Apply using kronzo_perturb_parameters
        kronzo_perturb_parameters(
            model, best_seed, step, zo_eps_global=zo_eps,
            scaling_factor=candidate_scaling, eps=1.0,  # Use momentum-adjusted scaling
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

def lowrank_kronzo_perturb_parameters(model, 
                                     zo_random_seed: int,
                                     step: int, 
                                     zo_eps_global: float,
                                     scaling_factor: float = 1.0,
                                     eps: float = None,
                                     strategy: str = "approx_square",
                                     max_factor: int = 32,
                                     kronzo_sampling_number: int = 1,
                                     rank_kronzo: int = 4,
                                     va_dict: dict = None,
                                     vb_dict: dict = None,
                                     step_interval: int = 50):
    """
    Low-rank KronZO parameter perturbation using A = U_A V_A^T, B = U_B V_B^T factorization.
    
    MEMORY EFFICIENT DESIGN:
    - Store only V_A (m1×rank_kronzo) and V_B (m2×rank_kronzo) matrices  
    - Generate U_A (m1×rank_kronzo) and U_B (m2×rank_kronzo) fresh each time
    - Total storage: O((m1+n1+m2+n2)*rank_kronzo) vs O(m1*n1+m2*n2) for full matrices
    - Rank control: rank(A⊗B) ≤ rank_kronzo^2
    
    Algorithm:
    1. For each parameter matrix P (d_out × d_in):
       a. Choose factorization: (m1×m2) ⊗ (n1×n2) where m1*m2=d_out, n1*n2=d_in
       b. Generate U_A (m1×rank_kronzo), U_B (m2×rank_kronzo) from seed
       c. Use stored V_A (n1×rank_kronzo), V_B (n2×rank_kronzo) [updated every step_interval]
       d. Compute: A = U_A @ V_A^T, B = U_B @ V_B^T  
       e. Perturb: P ← P + scaling_factor * eps * (A ⊗ B)
    
    Args:
        model: Model to perturb
        zo_random_seed: Random seed for U matrix generation (CRITICAL: use the direction_seed passed by caller)
        step: Current training step
        zo_eps_global: Global perturbation size
        scaling_factor: Scaling factor for perturbation
        eps: Override perturbation size (defaults to zo_eps_global)
        strategy: Kronecker factorization strategy
        max_factor: Maximum factor size
        kronzo_sampling_number: Number of Kronecker products to sample and sum
        rank_kronzo: Rank for low-rank factorization
        va_dict: Dictionary storing V_A matrices
        vb_dict: Dictionary storing V_B matrices  
        step_interval: Interval for updating V matrices
        
    Returns:
        named_parameters_to_optim: List of (name, parameter) tuples
    """
    if eps is None:
        eps = zo_eps_global
    if va_dict is None:
        va_dict = {}
    if vb_dict is None:
        vb_dict = {}
        
    # CRITICAL FIX: DO NOT reset the seed here - use the direction_seed passed by caller
    # The zo_random_seed parameter now contains the direction-specific seed (zo_random_seed + i)
    # torch.manual_seed(zo_random_seed)  # REMOVED: This was overwriting direction_seed!
    
    # Create list of named parameters to optimize
    named_parameters_to_optim = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            clean_name = name
            if name.startswith('_orig_mod.'):
                clean_name = name[len('_orig_mod.'):]
            named_parameters_to_optim.append((clean_name, param))
    
    for clean_name, param in named_parameters_to_optim:
        if param.ndim >= 2:  # Matrix parameters
            d_out, d_in = param.shape
            
            if kronzo_sampling_number == 1:
                # Standard single Kronecker product
                m1, m2, n1, n2 = choose_kron_dims(d_out, d_in, strategy, max_factor)
                
                # FIXED: Use direction-specific seed with temporal diversity
                u_seed = zo_random_seed + hash(clean_name) % 100000 + step * 13
                torch.manual_seed(u_seed)
                
                U_A = torch.randn(m1, rank_kronzo, device=param.device, dtype=param.dtype)
                U_B = torch.randn(m2, rank_kronzo, device=param.device, dtype=param.dtype)
                
                # Simple normalization (revert from QR orthogonalization)
                U_A = U_A / (rank_kronzo ** 0.5)  # Each matrix has variance 1/r
                U_B = U_B / (rank_kronzo ** 0.5)  # Each matrix has variance 1/r
                
                # Get or create V matrices (stored and updated every step_interval)
                va_key = f"{clean_name}_VA"
                vb_key = f"{clean_name}_VB"
                
                need_new_v = (step % step_interval == 0 or 
                             va_key not in va_dict or 
                             vb_key not in vb_dict or
                             va_dict[va_key].size(1) != rank_kronzo or
                             vb_dict[vb_key].size(1) != rank_kronzo)
                
                if need_new_v:
                    # FIXED: Use stable seeding for V matrices with temporal diversity (single sampling, no factorization index)
                    v_seed = 1000000 + (step // step_interval) * 1000 + hash(clean_name) % 1000
                    torch.manual_seed(v_seed)
                    V_A = torch.randn(n1, rank_kronzo, device=param.device, dtype=param.dtype)
                    V_B = torch.randn(n2, rank_kronzo, device=param.device, dtype=param.dtype)
                    
                    # Simple normalization (revert from QR orthogonalization)
                    V_A = V_A / (rank_kronzo ** 0.5)  # Each matrix has variance 1/r
                    V_B = V_B / (rank_kronzo ** 0.5)  # Each matrix has variance 1/r
                    
                    va_dict[va_key] = V_A
                    vb_dict[vb_key] = V_B
                else:
                    V_A = va_dict[va_key]
                    V_B = vb_dict[vb_key]
                
                # Compute A = U_A @ V_A^T and B = U_B @ V_B^T
                A = U_A @ V_A.t()  # (m1 × n1)
                B = U_B @ V_B.t()  # (m2 × n2)
                
                # SAFETY CHECK: Ensure matrices have reasonable norms
                A_norm = A.norm()
                B_norm = B.norm()
                if A_norm > 10.0 or B_norm > 10.0:
                    # Renormalize if too large
                    A = A / max(A_norm / 2.0, 1.0)
                    B = B / max(B_norm / 2.0, 1.0)
                
                # Apply Kronecker product perturbation: A ⊗ B
                perturbation = torch.kron(A, B)  # (d_out × d_in)
                
                # SAFETY CHECK: Clip perturbation if too large
                threshold = 10 * (d_out*d_in)**0.5
                perturbation_norm = perturbation.norm()
                if perturbation_norm > threshold:
                    perturbation.mul_(threshold / (perturbation_norm + 1e-8))
                    if step % 100 == 0:  # Log occasionally
                        print(f"WARNING: Clipped large perturbation for {clean_name}: norm={perturbation_norm:.2f}")
                
                param.data.add_(perturbation, alpha=scaling_factor * eps)
                
            else:
                # Multi-sampling: sum multiple low-rank Kronecker products
                factorizations = choose_diverse_kron_dims(d_out, d_in, kronzo_sampling_number, strategy, max_factor)
                total_perturbation = torch.zeros(d_out, d_in, device=param.device, dtype=param.dtype)
                
                for i, (m1, m2, n1, n2) in enumerate(factorizations):
                    # FIXED: Use direction and factorization-specific seed with temporal diversity
                    u_seed = zo_random_seed + hash(clean_name) % 100000 + step * 13 + i * 997
                    torch.manual_seed(u_seed)
                    
                    U_A = torch.randn(m1, rank_kronzo, device=param.device, dtype=param.dtype)
                    U_B = torch.randn(m2, rank_kronzo, device=param.device, dtype=param.dtype)
                    
                    # Simple normalization (revert from QR orthogonalization)
                    U_A = U_A / (rank_kronzo ** 0.5)  # Each matrix has variance 1/r
                    U_B = U_B / (rank_kronzo ** 0.5)  # Each matrix has variance 1/r
                    
                    # Get or create V matrices for this factorization
                    va_key = f"{clean_name}_VA_{i}"
                    vb_key = f"{clean_name}_VB_{i}"
                    
                    need_new_v = (step % step_interval == 0 or 
                                 va_key not in va_dict or 
                                 vb_key not in vb_dict or
                                 va_dict[va_key].size(1) != rank_kronzo or
                                 vb_dict[vb_key].size(1) != rank_kronzo)
                    
                    if need_new_v:
                        # FIXED: Use stable seeding for V matrices with temporal and factorization diversity
                        v_seed = 1000000 + (step // step_interval) * 1000 + hash(clean_name) % 1000 + i * 100
                        torch.manual_seed(v_seed)
                        V_A = torch.randn(n1, rank_kronzo, device=param.device, dtype=param.dtype)
                        V_B = torch.randn(n2, rank_kronzo, device=param.device, dtype=param.dtype)
                        
                        # Simple normalization (revert from QR orthogonalization)
                        V_A = V_A / (rank_kronzo ** 0.5)  # Each matrix has variance 1/r
                        V_B = V_B / (rank_kronzo ** 0.5)  # Each matrix has variance 1/r
                        
                        va_dict[va_key] = V_A
                        vb_dict[vb_key] = V_B
                    else:
                        V_A = va_dict[va_key]
                        V_B = vb_dict[vb_key]
                    
                    # Compute A = U_A @ V_A^T and B = U_B @ V_B^T
                    A = U_A @ V_A.t()  # (m1 × n1)
                    B = U_B @ V_B.t()  # (m2 × n2)
                    
                    # Accumulate Kronecker product: A ⊗ B
                    kron_perturbation = torch.kron(A, B)  # (d_out × d_in)
                    total_perturbation.add_(kron_perturbation)
                
                # Scale by 1/kronzo_sampling_number and apply
                total_perturbation.mul_(1.0 / kronzo_sampling_number)
                
                # SAFETY CHECK: Clip total perturbation if too large
                total_norm = total_perturbation.norm()
                threshold = 10 * (d_out*d_in)**0.5
                if total_norm > threshold:
                    total_perturbation.mul_(threshold / (total_norm + 1e-8))
                    if step % 100 == 0:  # Log occasionally  
                        print(f"WARNING: Clipped large multi-sample perturbation for {clean_name}: norm={total_norm:.2f}")
                
                param.data.add_(total_perturbation, alpha=scaling_factor * eps)
                
        else:  # Vector parameters (biases)
            # For vectors, use standard Gaussian perturbation (no Kronecker structure)
            # FIXED: Use direction-specific seed with temporal diversity for biases too
            bias_seed = zo_random_seed + hash(clean_name) % 100000 + step * 17
            torch.manual_seed(bias_seed)
            z = torch.normal(mean=0, std=1, size=param.size(), device=param.device, dtype=param.dtype)
            param.data.add_(z, alpha=scaling_factor * eps)
    
    return named_parameters_to_optim 