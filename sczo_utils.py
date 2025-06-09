import torch
import numpy as np
from collections import deque
from common_utils import zo_forward
from kronzo_utils_improved_directional import improved_kronzo_step
from typing import Dict, List, Tuple, Optional
import math

class SCZOState:
    """
    State management for Secant-Constrained Zero-Order optimization.
    
    Maintains gradient estimates per layer with minimal memory footprint.
    Includes multi-secant memory for temporal constraints.
    """
    def __init__(self, model, sczo_alpha: float = 1e-3, sczo_beta: float = 0.9, 
                 sczo_eps: float = 1e-8, sczo_bootstrap_interval: int = 20,
                 sczo_multi_batch: int = 3, sczo_multi_secant_m: int = 3,
                 sczo_multi_secant_rho: float = 0.8, sczo_multi_secant_eps_reg: float = 1e-6,
                 sczo_multi_secant_max_cond: float = 1e3, sczo_multi_secant_min_signal: float = 1e-8,
                 # New preconditioning parameters
                 sczo_precond_type: str = 'identity', 
                 sczo_precond_layerwise_ema: float = 0.99,
                 sczo_precond_diagonal_ema: float = 0.99, 
                 sczo_precond_diagonal_eps: float = 1e-8,
                 sczo_precond_unit_wise: bool = True):
        self.sczo_alpha = sczo_alpha
        self.sczo_beta = sczo_beta  
        self.sczo_eps = sczo_eps
        self.sczo_bootstrap_interval = sczo_bootstrap_interval
        self.sczo_multi_batch = sczo_multi_batch
        
        # Multi-secant memory parameters
        self.multi_secant_m = sczo_multi_secant_m
        self.multi_secant_rho = sczo_multi_secant_rho
        self.multi_secant_eps_reg = sczo_multi_secant_eps_reg
        self.multi_secant_max_cond = sczo_multi_secant_max_cond
        self.multi_secant_min_signal = sczo_multi_secant_min_signal
        
        # Preconditioning parameters
        self.precond_type = sczo_precond_type  # 'identity', 'layerwise', 'diagonal', 'layerwise_diagonal'
        self.precond_layerwise_ema = sczo_precond_layerwise_ema  # EMA for layer-wise scaling
        self.precond_diagonal_ema = sczo_precond_diagonal_ema    # EMA for diagonal preconditioning
        self.precond_diagonal_eps = sczo_precond_diagonal_eps    # Epsilon for diagonal stability
        self.precond_unit_wise = sczo_precond_unit_wise          # Use unit-wise diagonal (memory efficient)
        
        # Initialize gradient estimates per layer
        self.gradient_estimates = {}
        
        # Initialize multi-secant history buffers per layer: stores (a_i, y_i) pairs
        # where a_i = g^T * s_i (scalar) and y_i = loss_change (scalar)
        self.multi_secant_history = {}
        
        # Initialize preconditioning states
        self.precond_layerwise_scales = {}  # τ(l) per layer for M(l) = τ(l) * I
        self.precond_diagonal_vars = {}     # v(l) per layer for M(l) = diag(v(l) + ε)
        
        self._initialize_gradients(model)
        self._initialize_preconditioning(model)
        
        # Tracking
        self.step_count = 0
        self.last_bootstrap_step = -1
        
    def _initialize_gradients(self, model):
        """Initialize gradient estimates and multi-secant history buffers."""
        # Initialize gradient estimates (will be populated by bootstrap)
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Use consistent clean_name logic
                clean_name = name.lstrip("_orig_mod.")
                self.gradient_estimates[clean_name] = torch.zeros_like(param.data, dtype=torch.float16)
                
                # Initialize multi-secant history buffer for this layer
                self.multi_secant_history[clean_name] = deque(maxlen=self.multi_secant_m)
    
    def _initialize_preconditioning(self, model):
        """Initialize preconditioning matrices and state."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                clean_name = name.lstrip("_orig_mod.")
                
                # Initialize layer-wise scaling (τ(l) = 1.0 initially)
                if self.precond_type in ['layerwise', 'layerwise_diagonal']:
                    self.precond_layerwise_scales[clean_name] = 1.0
                
                # Initialize diagonal preconditioning
                if self.precond_type in ['diagonal', 'layerwise_diagonal']:
                    if self.precond_unit_wise:
                        # Unit-wise: one variance per neuron/filter/channel
                        if param.ndim >= 2:  # Weight matrices
                            # For FC layers: [out_features, in_features] -> track per output unit
                            # For Conv layers: [out_channels, in_channels, ...] -> track per output channel
                            unit_shape = (param.shape[0],)
                        else:  # Bias vectors
                            # For bias: [features] -> track per feature
                            unit_shape = param.shape
                        
                        # Initialize with small positive values
                        self.precond_diagonal_vars[clean_name] = torch.ones(
                            unit_shape, dtype=torch.float32, device=param.device
                        ) * self.precond_diagonal_eps
                    else:
                        # Element-wise: full diagonal matrix
                        self.precond_diagonal_vars[clean_name] = torch.ones_like(
                            param.data, dtype=torch.float32
                        ) * self.precond_diagonal_eps
    
    def should_bootstrap(self) -> bool:
        """Check if we should perform ZO gradient bootstrap."""
        return ((self.step_count % self.sczo_bootstrap_interval == 0) or 
                (self.step_count == 0))
    
    def update_step_count(self):
        """Update internal step counter."""
        self.step_count += 1
    
    def get_state(self):
        """Get state dictionary for checkpointing."""
        return {
            'sczo_alpha': self.sczo_alpha,
            'sczo_beta': self.sczo_beta,
            'sczo_eps': self.sczo_eps,
            'sczo_bootstrap_interval': self.sczo_bootstrap_interval,
            'sczo_multi_batch': self.sczo_multi_batch,
            'multi_secant_m': self.multi_secant_m,
            'multi_secant_rho': self.multi_secant_rho,
            'multi_secant_eps_reg': self.multi_secant_eps_reg,
            'multi_secant_max_cond': self.multi_secant_max_cond,
            'multi_secant_min_signal': self.multi_secant_min_signal,
            # Preconditioning parameters
            'precond_type': self.precond_type,
            'precond_layerwise_ema': self.precond_layerwise_ema,
            'precond_diagonal_ema': self.precond_diagonal_ema,
            'precond_diagonal_eps': self.precond_diagonal_eps,
            'precond_unit_wise': self.precond_unit_wise,
            'gradient_estimates': {name: grad.clone() for name, grad in self.gradient_estimates.items()},
            'multi_secant_history': {name: list(hist) for name, hist in self.multi_secant_history.items()},
            # Preconditioning state
            'precond_layerwise_scales': {name: scale for name, scale in self.precond_layerwise_scales.items()},
            'precond_diagonal_vars': {name: var.clone() for name, var in self.precond_diagonal_vars.items()},
            'step_count': self.step_count,
            'last_bootstrap_step': self.last_bootstrap_step
        }
    
    def load_state(self, state_dict):
        """Load state from checkpoint."""
        self.sczo_alpha = state_dict['sczo_alpha']
        self.sczo_beta = state_dict['sczo_beta']
        self.sczo_eps = state_dict['sczo_eps']
        self.sczo_bootstrap_interval = state_dict['sczo_bootstrap_interval']
        self.sczo_multi_batch = state_dict['sczo_multi_batch']
        
        # Multi-secant parameters (with backward compatibility)
        self.multi_secant_m = state_dict.get('multi_secant_m', 3)
        self.multi_secant_rho = state_dict.get('multi_secant_rho', 0.8)
        self.multi_secant_eps_reg = state_dict.get('multi_secant_eps_reg', 1e-6)
        self.multi_secant_max_cond = state_dict.get('multi_secant_max_cond', 1e3)
        self.multi_secant_min_signal = state_dict.get('multi_secant_min_signal', 1e-8)
        
        # Preconditioning parameters (with backward compatibility)
        self.precond_type = state_dict.get('precond_type', 'identity')
        self.precond_layerwise_ema = state_dict.get('precond_layerwise_ema', 0.99)
        self.precond_diagonal_ema = state_dict.get('precond_diagonal_ema', 0.99)
        self.precond_diagonal_eps = state_dict.get('precond_diagonal_eps', 1e-8)
        self.precond_unit_wise = state_dict.get('precond_unit_wise', True)
        
        # Load gradient estimates
        for name, grad in state_dict['gradient_estimates'].items():
            self.gradient_estimates[name] = grad.clone()
            
        # Load multi-secant history (with backward compatibility)
        if 'multi_secant_history' in state_dict:
            for name, hist_list in state_dict['multi_secant_history'].items():
                self.multi_secant_history[name] = deque(hist_list, maxlen=self.multi_secant_m)
        else:
            # Initialize empty history for backward compatibility
            for name in self.gradient_estimates.keys():
                self.multi_secant_history[name] = deque(maxlen=self.multi_secant_m)
        
        # Load preconditioning state (with backward compatibility)
        if 'precond_layerwise_scales' in state_dict:
            self.precond_layerwise_scales = state_dict['precond_layerwise_scales'].copy()
        else:
            self.precond_layerwise_scales = {name: 1.0 for name in self.gradient_estimates.keys()}
            
        if 'precond_diagonal_vars' in state_dict:
            for name, var in state_dict['precond_diagonal_vars'].items():
                self.precond_diagonal_vars[name] = var.clone()
        else:
            self.precond_diagonal_vars = {}
        
        self.step_count = state_dict['step_count']
        self.last_bootstrap_step = state_dict['last_bootstrap_step']

def sczo_initial_gradient_estimation(model, X, Y, step: int, zo_random_seed: int, 
                                   ctx_obj, zo_eps: float, directional_q: int = 20,
                                   strategy: str = "approx_square", max_factor: int = 32,
                                   kronzo_sampling_number: int = 1, b_dict: dict = None,
                                   step_interval: int = 50):
    """
    Estimate initial gradients using improved KronZO methodology.
    
    This gives us per-layer gradient estimates by extracting the directional information
    from the improved KronZO step and reconstructing the gradient estimates.
    
    Returns:
        gradient_estimates: Dict mapping layer names to gradient tensors
    """
    if b_dict is None:
        b_dict = {}
    
    gradient_estimates = {}
    
    # Get list of parameters to optimize
    named_parameters_to_optim = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Use consistent clean_name logic
            clean_name = name.lstrip("_orig_mod.")
            named_parameters_to_optim.append((clean_name, param))
    
    # For each layer, estimate gradient using finite differences
    for clean_name, param in named_parameters_to_optim:
        # Estimate gradient for this layer using finite differences
        gradient_estimates[clean_name] = sczo_estimate_layer_gradient(
            model, param, clean_name, X, Y, ctx_obj, zo_eps, zo_random_seed
        )
    
    return gradient_estimates

def sczo_estimate_layer_gradient(model, param, param_name: str, X, Y, ctx_obj, 
                                zo_eps: float, zo_random_seed: int):
    """
    Estimate gradient for a single layer using finite differences.
    
    For memory efficiency, we use a subsampling approach for large matrices.
    """
    torch.manual_seed(zo_random_seed + hash(param_name) % 1000)
    
    if param.ndim >= 2:  # Matrix parameters
        # For large matrices, use subsampled finite differences
        # This is more memory efficient than full perturbation
        
        original_param = param.data.clone()
        
        # Create random perturbation
        if param.numel() > 1000000:  # For very large layers, use sparse sampling
            # Sample a fraction of elements for gradient estimation
            sample_ratio = min(0.01, 1000.0 / param.numel())  # At most 1% or 1000 elements
            mask = torch.rand_like(param.data) < sample_ratio
            perturbation = torch.randn_like(param.data) * mask
        else:
            # For smaller layers, use full perturbation
            perturbation = torch.randn_like(param.data)
        
        # Forward pass: f(θ + ε*z)
        param.data.add_(perturbation, alpha=zo_eps)
        loss_plus = zo_forward(model, X, Y, ctx_obj)
        
        # Forward pass: f(θ - ε*z)  
        param.data.add_(perturbation, alpha=-2*zo_eps)
        loss_minus = zo_forward(model, X, Y, ctx_obj)
        
        # Restore original parameters
        param.data.copy_(original_param)
        
        # Compute gradient estimate: g = [f(θ+εz) - f(θ-εz)] / (2ε) * z
        grad_coeff = (loss_plus - loss_minus) / (2 * zo_eps)
        gradient_estimate = grad_coeff.item() * perturbation
        
    else:  # Vector parameters (bias, etc.)
        original_param = param.data.clone()
        
        # Create random perturbation
        perturbation = torch.randn_like(param.data)
        
        # Forward pass: f(θ + ε*z)
        param.data.add_(perturbation, alpha=zo_eps)
        loss_plus = zo_forward(model, X, Y, ctx_obj)
        
        # Forward pass: f(θ - ε*z)
        param.data.add_(perturbation, alpha=-2*zo_eps)
        loss_minus = zo_forward(model, X, Y, ctx_obj)
        
        # Restore original parameters
        param.data.copy_(original_param)
        
        # Compute gradient estimate
        grad_coeff = (loss_plus - loss_minus) / (2 * zo_eps)
        gradient_estimate = grad_coeff.item() * perturbation
    
    # Convert to FP16 for memory efficiency
    return gradient_estimate.to(torch.float16)

def sczo_step(model, X, Y, sczo_state, step, zo_random_seed, ctx_obj,
              zo_eps, directional_q=20, strategy="approx_square",
              max_factor=32, kronzo_sampling_number=1, b_dict=None,
              step_interval=50, get_batch_fn=None, λ_max=10.0,
              weight_decay=0.0):
    """
    Enhanced SCZO step with multi-batch spatial constraints + multi-secant temporal memory.
    
    Algorithm:
    1. Bootstrap ZO gradients if needed
    2. For each of M batches (spatial constraints):
       - Compute loss before/after step
       - Get y_k^(i) and accumulate Δg_i
    3. Apply multi-secant temporal constraints using history buffer
    4. Update gradient estimates with combined spatial + temporal information
    """

    if b_dict is None:
        b_dict = {}

    # ------------------------------------------------------------------
    # A. BOOTSTRAP ZO GRADIENT IF NEEDED
    # ------------------------------------------------------------------
    if sczo_state.should_bootstrap():
        bootstrap_grads = sczo_initial_gradient_estimation(
            model, X, Y, step, zo_random_seed, ctx_obj, zo_eps,
            directional_q, strategy, max_factor, kronzo_sampling_number,
            b_dict, step_interval
        )
        for k, v in bootstrap_grads.items():
            sczo_state.gradient_estimates[k] = v
        sczo_state.last_bootstrap_step = step

    # Liste des paramètres manipulés
    named_parameters_to_optim = [
        (name.lstrip("_orig_mod."), p)          # clean_name, param
        for name, p in model.named_parameters() if p.requires_grad
    ]

    α   = sczo_state.sczo_alpha
    ε   = sczo_state.sczo_eps
    β   = sczo_state.sczo_beta
    mb  = sczo_state.sczo_multi_batch

    # ------------------------------------------------------------------
    # B. SPATIAL CONSTRAINTS: Multi-batch accumulation  
    # ------------------------------------------------------------------
    delta_g_sums = {n: torch.zeros_like(g, dtype=torch.float32)
                    for n, g in sczo_state.gradient_estimates.items()}

    loss_changes = []
    first_before, first_after = None, None

    for i in range(mb):
        # Mini‑batch
        X_b, Y_b = get_batch_fn() if get_batch_fn else (X, Y)

        # 1. Perte AVANT (removed weight decay from here)
        loss_before = zo_forward(model, X_b, Y_b, ctx_obj)

        # 3. Pas θ ← θ − α g (take optimization step)
        for clean_name, p in named_parameters_to_optim:
            if clean_name in sczo_state.gradient_estimates:
                p.data.add_(-α * sczo_state.gradient_estimates[clean_name].to(p.dtype))

        # 4. Perte APRÈS
        loss_after = zo_forward(model, X_b, Y_b, ctx_obj)
        y_i = (loss_after - loss_before).item()
        loss_changes.append(y_i)

        # 5. Compute spatial secant constraint Δg_i per layer
        for clean_name, p in named_parameters_to_optim:
            if clean_name not in sczo_state.gradient_estimates: 
                continue
            g_fp32   = sczo_state.gradient_estimates[clean_name].to(torch.float32)
            g_dot_g  = torch.sum(g_fp32 * g_fp32).item()

            num   = y_i + α * g_dot_g
            denom = α * α * g_dot_g + ε
            λ     = max(min(num / denom,  λ_max), -λ_max)   # clamp

            delta_g_i = (-α * λ) * g_fp32                   # Δg_i FP32
            delta_g_sums[clean_name].add_(delta_g_i)        # accumulate

        # Log première paire
        if i == 0:
            first_before, first_after = loss_before.item(), loss_after.item()

        # 6. Restore θ pour batch suivant (undo the step)
        for clean_name, p in named_parameters_to_optim:
            if clean_name in sczo_state.gradient_estimates:
                p.data.add_(α * sczo_state.gradient_estimates[clean_name].to(p.dtype))

    # ------------------------------------------------------------------
    # C. TEMPORAL CONSTRAINTS: Multi-secant memory + EMA update
    # ------------------------------------------------------------------
    temporal_stats = sczo_apply_multi_batch_and_temporal_update(
        sczo_state, named_parameters_to_optim, delta_g_sums, mb, 
        loss_changes, first_before, first_after
    )

    # ------------------------------------------------------------------
    # D. FINAL OPTIMIZATION STEP ON WEIGHTS
    # ------------------------------------------------------------------
    for clean_name, param in named_parameters_to_optim:
        if clean_name in sczo_state.gradient_estimates:
            param.data.add_(-α * sczo_state.gradient_estimates[clean_name].to(param.dtype))

    sczo_state.update_step_count()

    # ------------------------------------------------------------------
    # E. STEP INFO FOR LOGGING
    # ------------------------------------------------------------------
    step_info = {
        "loss_change_mean": float(np.mean(loss_changes)),
        "loss_change_std":  float(np.std(loss_changes)),
        "raw_loss_changes": loss_changes.copy(),  # For debugging
        "used_batches":     mb,
        "grad_norm":        sum(g.norm().item() for g in sczo_state.gradient_estimates.values()),
        "temporal_constraints_used": temporal_stats.get('used_temporal_constraints', 0),
        "temporal_restarts": temporal_stats.get('restarted_layers', 0),
        "avg_condition_number": temporal_stats.get('avg_condition_number', 0.0),
        "lambda_clipped_fraction": temporal_stats.get('lambda_clipped_count', 0) / max(len(temporal_stats.get('lambda_values', [])), 1)
    }

    return first_before, first_after, step_info

def sczo_apply_multi_batch_and_temporal_update(sczo_state: SCZOState, named_parameters_to_optim,
                                             delta_g_sums: dict, batch_count: int,
                                             loss_changes: List[float], first_before: float, 
                                             first_after: float):
    """
    Apply combined spatial (multi-batch) and temporal (multi-secant) constraints.
    
    Algorithm:
    1. Compute spatial gradient update: Δg_spatial = (1/M) * Σ Δg_i
    2. For each layer, add current (s_k, y_k) to history buffer
    3. Solve multi-secant temporal constraints using history
    4. Combine spatial + temporal updates with EMA
    
    Args:
        sczo_state: SCZO state with gradient estimates and history buffers
        named_parameters_to_optim: List of (clean_name, param) pairs
        delta_g_sums: Dictionary of accumulated spatial gradient updates
        batch_count: Number of batches used for spatial constraints
        loss_changes: List of loss changes y_i from each batch
        first_before: Loss before first step (for logging)
        first_after: Loss after first step (for logging)
    """
    β = sczo_state.sczo_beta
    α = sczo_state.sczo_alpha
    
    # Compute average loss change for temporal constraint
    y_k_avg = sum(loss_changes) / len(loss_changes)
    
    # Statistics for logging
    temporal_stats = {
        'used_temporal_constraints': 0,
        'restarted_layers': 0,
        'avg_condition_number': 0.0,
        'avg_temporal_delta': 0.0,
        'lambda_clipped_count': 0,
        'lambda_values': []
    }
    
    for clean_name, param in named_parameters_to_optim:
        if clean_name not in sczo_state.gradient_estimates:
            continue
            
        # --- SPATIAL UPDATE ---
        # Average spatial gradient update: Δg_spatial = (1/M) * Σ Δg_i
        delta_g_spatial = delta_g_sums[clean_name] / float(batch_count)
        
        # --- TEMPORAL UPDATE (only if multi-secant memory is enabled) ---
        if sczo_state.multi_secant_m > 0:
            g_current = sczo_state.gradient_estimates[clean_name].to(torch.float32)
            
            # Reuse g_dot_g from spatial update if available, otherwise compute
            if clean_name in delta_g_sums:
                # We already computed some form of g interaction, get fresh g_dot_g
                g_dot_g = torch.sum(g_current * g_current).item()
            else:
                g_dot_g = torch.sum(g_current * g_current).item()
            
            # Compute a_k = g^T * s_k where s_k = -α * g, so a_k = -α * ||g||^2
            a_k = -α * g_dot_g  # More efficient: reuse g_dot_g
            
            # Add (a_k, y_k_avg) to history buffer
            history_buffer = sczo_state.multi_secant_history[clean_name]
            history_buffer.append((a_k, y_k_avg))
            
            # Solve multi-secant temporal constraints
            delta, restart_needed, solver_info = sczo_multi_secant_solve(
                g_current, history_buffer,
                sczo_state.multi_secant_rho,
                sczo_state.multi_secant_eps_reg, 
                sczo_state.multi_secant_max_cond,
                sczo_state.multi_secant_min_signal
            )
            
            # Handle restart if needed
            if restart_needed:
                history_buffer.clear()
                temporal_stats['restarted_layers'] += 1
                delta = 0.0  # No temporal update
            
            # Compute temporal gradient update: Δg_temporal = δ * g
            delta_g_temporal = delta * g_current
            
            # --- COMBINED UPDATE ---
            # Total update: Δg_total = Δg_spatial + λ_temporal * Δg_temporal
            # For now, equal weighting. Could be made adaptive.
            temporal_weight = 0.3 if len(history_buffer) >= 1 else 0.0  # Use temporal as soon as we have 1 pair
            
            delta_g_total = delta_g_spatial + temporal_weight * delta_g_temporal
            
            # Update statistics
            if solver_info['num_constraints'] > 0:
                temporal_stats['used_temporal_constraints'] += solver_info['num_constraints']
                temporal_stats['avg_condition_number'] += solver_info['condition_number']
                temporal_stats['avg_temporal_delta'] += abs(delta)
                temporal_stats['lambda_values'].append(delta)
                if abs(delta) > 1.0:
                    temporal_stats['lambda_clipped_count'] += 1
        else:
            # Spatial-only SCZO: no temporal constraints
            delta_g_total = delta_g_spatial
        
        # Apply global gradient clipping for safety
        g_clip = 0.5  # Tighter clipping for stability
        torch.clamp_(delta_g_total, -g_clip, g_clip)
        torch.nan_to_num_(delta_g_total, nan=0.0, posinf=g_clip, neginf=-g_clip)
        
        # Apply preconditioning to the gradient update: Δg_preconditioned = M^(-1) * Δg_total
        delta_g_preconditioned = sczo_apply_preconditioning(
            sczo_state, clean_name, param, delta_g_total
        )
        
        # EMA update: g ← g + β * Δg_preconditioned
        g_current_for_update = sczo_state.gradient_estimates[clean_name].to(torch.float32)
        g_updated = g_current_for_update + β * delta_g_preconditioned
        
        # Store back as FP16
        sczo_state.gradient_estimates[clean_name] = g_updated.to(torch.float16)
    
    # Update preconditioning matrices after all gradient estimates are updated
    # (but only if not using identity preconditioning)
    if sczo_state.precond_type != 'identity':
        sczo_update_preconditioning(
            sczo_state, sczo_state.gradient_estimates, 
            named_parameters_to_optim, loss_changes
        )
    
    # Normalize statistics
    num_layers = len([name for name, _ in named_parameters_to_optim 
                     if name in sczo_state.gradient_estimates])
    if num_layers > 0:
        temporal_stats['avg_condition_number'] /= num_layers
        temporal_stats['avg_temporal_delta'] /= num_layers

    return temporal_stats

def sczo_update_gradient_estimates(sczo_state: SCZOState, named_parameters_to_optim, y_k: float):
    """
    Update gradient estimates using secant constraint with Identity preconditioning.
    
    For each layer l:
    1. s_k^(l) = -α * g^(l)  (step taken)
    2. numerator = y_k + α * ⟨g^(l), g^(l)⟩  (since s_k = -α * g)
    3. denominator = α² * ⟨g^(l), g^(l)⟩ + ε
    4. λ = numerator / denominator
    5. Δg = λ * (-α) * g^(l)  (since M^(-1) * s_k = M^(-1) * (-α * g) = -α * g for Identity M)
    6. g^(l) ← g^(l) + β * Δg  (EMA update)
    """
    alpha = sczo_state.sczo_alpha
    beta = sczo_state.sczo_beta
    eps = sczo_state.sczo_eps
    
    for clean_name, param in named_parameters_to_optim:
        if clean_name in sczo_state.gradient_estimates:
            g_current = sczo_state.gradient_estimates[clean_name]
            
            # Convert to computation dtype for numerical stability
            g_float = g_current.to(torch.float32)
            
            # Compute dot product: ⟨g, g⟩
            g_dot_g = torch.sum(g_float * g_float).item()
            
            # Numerator: y_k + α * ⟨g, g⟩
            numerator = y_k + alpha * g_dot_g
            
            # Denominator: α² * ⟨g, g⟩ + ε  
            denominator = alpha * alpha * g_dot_g + eps
            
            # Avoid division by zero
            if abs(denominator) > eps:
                lambda_val = numerator / denominator
                
                # Δg = λ * (-α) * g  (for Identity preconditioning)
                delta_g = lambda_val * (-alpha) * g_float
                
                # EMA update: g ← g + β * Δg
                g_updated = g_float + beta * delta_g
                
                # Convert back to FP16 and store
                sczo_state.gradient_estimates[clean_name] = g_updated.to(torch.float16)

def sczo_update_with_weight_decay(model, sczo_state: SCZOState, weight_decay: float):
    """
    Apply weight decay separately after SCZO update.
    
    This is applied after the main SCZO step to maintain the secant constraint.
    """
    if weight_decay > 0:
        for name, param in model.named_parameters():
            if param.requires_grad:
                clean_name = name[len("_orig_mod."):] if name.startswith("_orig_mod.") else name
                
                # Check if this is a weight (not bias/layernorm)
                is_weight = ("bias" not in clean_name and 
                            "layer_norm" not in clean_name and 
                            "layernorm" not in clean_name)
                
                if is_weight:
                    param.data.mul_(1 - sczo_state.sczo_alpha * weight_decay)

def sczo_update_preconditioning(sczo_state: SCZOState, gradient_estimates: dict, 
                               named_parameters_to_optim: list, loss_changes: list):
    """
    Update preconditioning matrices based on recent gradient estimates and loss changes.
    
    Args:
        sczo_state: SCZO state containing preconditioning parameters
        gradient_estimates: Current gradient estimates per layer
        named_parameters_to_optim: List of (clean_name, param) pairs
        loss_changes: List of loss changes from recent steps
    """
    if sczo_state.precond_type == 'identity':
        return  # No preconditioning to update
    
    # Update layer-wise scaling τ(l)
    if sczo_state.precond_type in ['layerwise', 'layerwise_diagonal']:
        for clean_name, param in named_parameters_to_optim:
            if clean_name not in gradient_estimates:
                continue
                
            g_current = gradient_estimates[clean_name].to(torch.float32)
            
            # Compute current gradient norm for this layer
            g_norm = torch.norm(g_current).item()
            
            # Estimate step size ratio: τ = ||g|| / ||Δθ|| where Δθ = α * g
            # So τ = ||g|| / (α * ||g||) = 1/α, but we use empirical estimation
            # For now, use RMSprop-style: τ = sqrt(E[g²]) updated by EMA
            if g_norm > 1e-12:  # Avoid division by zero
                g_squared_norm = g_norm ** 2
                
                # EMA update: τ ← ρ * τ + (1-ρ) * sqrt(||g||²)
                ema_factor = sczo_state.precond_layerwise_ema
                current_scale = sczo_state.precond_layerwise_scales.get(clean_name, 1.0)
                
                # Update with square root for stability (like RMSprop)
                new_scale = ema_factor * current_scale + (1 - ema_factor) * math.sqrt(g_squared_norm)
                sczo_state.precond_layerwise_scales[clean_name] = max(new_scale, 1e-8)  # Prevent zero scaling
    
    # Update diagonal preconditioning v(l)
    if sczo_state.precond_type in ['diagonal', 'layerwise_diagonal']:
        for clean_name, param in named_parameters_to_optim:
            if clean_name not in gradient_estimates or clean_name not in sczo_state.precond_diagonal_vars:
                continue
                
            g_current = gradient_estimates[clean_name].to(torch.float32)
            
            # Compute element-wise or unit-wise gradient squares
            if sczo_state.precond_unit_wise:
                # Unit-wise: aggregate gradients per output unit/channel
                if param.ndim >= 2:  # Weight matrices
                    # Sum over input dimensions, keep output dimension
                    # For [out, in] -> [out], for [out, in, h, w] -> [out]
                    reduce_dims = tuple(range(1, param.ndim))
                    g_squared_per_unit = torch.sum(g_current ** 2, dim=reduce_dims)
                else:  # Bias vectors
                    g_squared_per_unit = g_current ** 2
            else:
                # Element-wise: full gradient squares
                g_squared_per_unit = g_current ** 2
            
            # EMA update: v ← ρ * v + (1-ρ) * g²
            ema_factor = sczo_state.precond_diagonal_ema
            current_var = sczo_state.precond_diagonal_vars[clean_name]
            
            new_var = ema_factor * current_var + (1 - ema_factor) * g_squared_per_unit.to(current_var.device)
            sczo_state.precond_diagonal_vars[clean_name] = new_var

def sczo_apply_preconditioning(sczo_state: SCZOState, clean_name: str, param: torch.Tensor, 
                              gradient_update: torch.Tensor) -> torch.Tensor:
    """
    Apply preconditioning to a gradient update: Δg_preconditioned = M^(-1) * Δg
    
    Args:
        sczo_state: SCZO state containing preconditioning matrices
        clean_name: Layer name
        param: Parameter tensor (for shape information)
        gradient_update: Raw gradient update to precondition
        
    Returns:
        Preconditioned gradient update
    """
    if sczo_state.precond_type == 'identity':
        return gradient_update
    
    preconditioned_update = gradient_update.clone()
    
    # Apply layer-wise scaling: Δg ← Δg / τ(l)
    if sczo_state.precond_type in ['layerwise', 'layerwise_diagonal']:
        if clean_name in sczo_state.precond_layerwise_scales:
            scale = sczo_state.precond_layerwise_scales[clean_name]
            preconditioned_update = preconditioned_update / max(scale, 1e-8)
    
    # Apply diagonal preconditioning: Δg ← Δg / sqrt(v(l) + ε)
    if sczo_state.precond_type in ['diagonal', 'layerwise_diagonal']:
        if clean_name in sczo_state.precond_diagonal_vars:
            diagonal_precond = sczo_state.precond_diagonal_vars[clean_name]
            
            # Compute sqrt(v + ε) for numerical stability
            precond_denom = torch.sqrt(diagonal_precond + sczo_state.precond_diagonal_eps)
            
            if sczo_state.precond_unit_wise:
                # Unit-wise: broadcast to parameter shape
                if param.ndim >= 2:  # Weight matrices
                    # Reshape to [out, 1, 1, ...] for broadcasting
                    shape = [diagonal_precond.shape[0]] + [1] * (param.ndim - 1)
                    precond_denom = precond_denom.view(shape)
                # For biases, precond_denom already has the right shape
                
                preconditioned_update = preconditioned_update / precond_denom.to(preconditioned_update.device)
            else:
                # Element-wise: direct division
                preconditioned_update = preconditioned_update / precond_denom.to(preconditioned_update.device)
    
    return preconditioned_update

def sczo_multi_secant_solve(gradient_estimate: torch.Tensor, history_buffer: deque, 
                           rho: float, eps_reg: float, max_cond: float, 
                           min_signal: float) -> Tuple[float, bool, dict]:
    """
    Solve multi-secant scalar least-squares problem for temporal constraints.
    
    Minimizes: Σ_i w_i * (g^T s_i - y_i)² where w_i = rho^(len-1-i)
    
    Args:
        gradient_estimate: Current gradient estimate g
        history_buffer: Deque of (a_i, y_i) pairs where a_i = g^T s_i
        rho: Exponential decay weight for older constraints
        eps_reg: Tikhonov regularization parameter
        max_cond: Maximum condition number before restart
        min_signal: Minimum signal strength to keep constraint
        
    Returns:
        delta: Scalar multiplier for gradient update (δ in theory)
        restart_needed: Whether to restart the history buffer
        info: Dictionary with solver statistics
    """
    if len(history_buffer) == 0:
        return 0.0, False, {'num_constraints': 0, 'condition_number': 0.0}
    
    # Extract (a_i, y_i) pairs and compute exponential weights
    pairs = list(history_buffer)
    n = len(pairs)
    weights = [rho**(n-1-i) for i in range(n)]
    
    # Filter out weak signals and sign-inconsistent pairs
    filtered_pairs = []
    filtered_weights = []
    
    for i, ((a_i, y_i), w) in enumerate(zip(pairs, weights)):
        # Check signal strength
        signal_strong_enough = abs(y_i) >= min_signal
        
        # RELAXED: Accept all strong signals (removed strict sign consistency check)
        sign_consistent = True  # Accept all strong signals
        
        if signal_strong_enough and sign_consistent:
            filtered_pairs.append((a_i, y_i))
            filtered_weights.append(w)
    
    if len(filtered_pairs) == 0:
        return 0.0, True, {'num_constraints': 0, 'condition_number': float('inf'), 'restart': 'no_valid_constraints'}
    
    # Compute weighted least-squares solution
    # min Σ w_i * (δ * a_i - y_i)² => δ = (Σ w_i * a_i * y_i) / (Σ w_i * a_i² + eps_reg)
    A_sum = sum(w * a for w, (a, _) in zip(filtered_weights, filtered_pairs))
    AA_sum = sum(w * a * a for w, (a, _) in zip(filtered_weights, filtered_pairs))
    y_sum = sum(w * y for w, (_, y) in zip(filtered_weights, filtered_pairs))
    Ay_sum = sum(w * a * y for w, (a, y) in zip(filtered_weights, filtered_pairs))
    
    # Add adaptive Tikhonov regularization
    adaptive_eps_reg = eps_reg * max(AA_sum, 1e-12)  # Scale with problem size
    AA_sum_reg = AA_sum + adaptive_eps_reg
    
    # Compute condition number estimate: ratio of max to min eigenvalue
    if AA_sum > 1e-12:
        cond_est = AA_sum_reg / adaptive_eps_reg
    else:
        cond_est = float('inf')
    
    # Check for ill-conditioning
    restart_needed = cond_est > max_cond or AA_sum_reg < 1e-12
    
    if restart_needed:
        return 0.0, True, {
            'num_constraints': len(filtered_pairs),
            'condition_number': cond_est,
            'restart': 'ill_conditioned'
        }
    
    # Compute the scalar multiplier
    delta = Ay_sum / AA_sum_reg
    
    return delta, False, {
        'num_constraints': len(filtered_pairs),
        'condition_number': cond_est,
        'delta': delta,
        'A_sum': A_sum,
        'AA_sum': AA_sum,
        'y_sum': y_sum
    } 