import torch
import math
import numpy as np
from common_utils import zo_forward # Import from common_utils

# -----------------------------------------------------------------------------
# KronZO (Kronecker Zero-Order) specific functions
# -----------------------------------------------------------------------------

def find_closest_divisor(n, target):
    """Find divisor of n closest to target"""
    divisors = [i for i in range(1, n+1) if n % i == 0]
    return min(divisors, key=lambda x: abs(x - target))

def choose_kron_dims_approx_square(d_out: int, d_in: int):
    """
    Returns (m1, m2, n1, n2) with m1 * m2 = d_out and n1 * n2 = d_in
    by favoring factors close to square roots.
    """

    def best_pair(n: int):
        root = int(math.sqrt(n))
        for off in range(root + 1):
            for cand in (root - off, root + off):
                if cand >= 1 and n % cand == 0:
                    return cand, n // cand
        return 1, n  # n is prime: trivial decomposition

    m1, m2 = best_pair(d_out)   # rows
    n1, n2 = best_pair(d_in)    # columns
    return m1, m2, n1, n2

def choose_kron_dims_fixed_factor(d_out: int, d_in: int, max_factor: int = 32):
    """
    Variant with "bounded factor": each first factor ≤ max_factor.
    """
    out_f = [i for i in range(1, min(max_factor + 1, d_out + 1)) if d_out % i == 0]
    m1 = max(out_f) if out_f else 1
    m2 = d_out // m1

    in_f = [i for i in range(1, min(max_factor + 1, d_in + 1)) if d_in % i == 0]
    n1 = max(in_f) if in_f else 1
    n2 = d_in // n1

    return m1, m2, n1, n2

def choose_kron_dims_power2(d_out: int, d_in: int):
    """
    Product of maximal powers of 2.
    """
    def largest_pow2_divisor(n: int):
        if n == 0:
            return 1
        p = 1
        while n % (p << 1) == 0:
            p <<= 1
        return p

    m1 = largest_pow2_divisor(d_out)
    m2 = d_out // m1
    n1 = largest_pow2_divisor(d_in)
    n2 = d_in // n1
    return m1, m2, n1, n2

def choose_kron_dims(d_out: int, d_in: int, strategy: str = "approx_square", max_factor: int = 32):
    if strategy == "approx_square":
        return choose_kron_dims_approx_square(d_out, d_in)
    if strategy == "fixed_factor":
        return choose_kron_dims_fixed_factor(d_out, d_in, max_factor)
    if strategy == "power2":
        return choose_kron_dims_power2(d_out, d_in)
    raise ValueError(f"Unknown Kronecker strategy: {strategy}")

def kronzo_perturb_parameters(model,
                              zo_random_seed: int,
                              step: int,
                              zo_eps_global: float, # Added to pass global config
                              scaling_factor: float = 1.0,
                              eps: float | None = None,
                              strategy: str = "approx_square",
                              max_factor: int = 32):
    """
    Apply Kronecker-type perturbations (or Gaussian for vectors) 
    to all trainable parameters of the model.
    Returns the list of (name, param) tuples of perturbed parameters.
    
    Args:
        model: The model to perturb
        zo_random_seed: Random seed for reproducibility
        step: Current step (unused, kept for compatibility)
        zo_eps_global: Global perturbation size
        scaling_factor: Scaling factor for perturbations (+1, -1, -2, etc.)
        eps: Override perturbation size (uses zo_eps_global if None)
        strategy: Kronecker factorization strategy
        max_factor: Maximum factor size for factorization
    """
    if eps is None:
        eps = zo_eps_global

    torch.manual_seed(zo_random_seed)

    # Prepare list of parameters to optimize
    named_parameters_to_optim: list[tuple[str, torch.Tensor]] = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            clean = name[len("_orig_mod."):] if name.startswith("_orig_mod.") else name
            named_parameters_to_optim.append((clean, param))

    # Apply perturbations
    for clean_name, param in named_parameters_to_optim:
        if param.ndim >= 2:                          # matrices
            d_out, d_in = param.shape
            m1, m2, n1, n2 = choose_kron_dims(d_out, d_in, strategy, max_factor)

            A = torch.randn(m1, n1, device=param.device, dtype=param.dtype)
            B = torch.randn(m2, n2, device=param.device, dtype=param.dtype)
            perturbation = torch.kron(A, B)          # (d_out, d_in)

            param.data.add_(scaling_factor * perturbation * eps)
        else:                                        # vectors / biases
            z = torch.normal(0.0, 1.0,
                             size=param.size(),
                             device=param.device,
                             dtype=param.dtype)
            param.data.add_(scaling_factor * z * eps)

    return named_parameters_to_optim

def kronzo_step(model, X, Y, step, zo_random_seed, zo_eps_global, ctx_obj, strategy='approx_square', max_factor=32, eps=None):
    """
    Estimate gradient using KronZO (Kronecker Zero-Order optimization).
    
    Algorithm:
    1. For each matrix parameter W ∈ ℝ^(d_out × d_in):
       - Choose factorization: m1×n1 = d_out, m2×n2 = d_in  
       - Sample A ∈ ℝ^(m1×n1), B ∈ ℝ^(m2×n2)
       - Perturbation: ΔW = A ⊗ B (Kronecker product)
    2. Compute finite difference: c = [f(θ + ε*ΔW) - f(θ - ε*ΔW)]/(2ε)
    3. This gives projected gradient ∇f(θ) · ΔW along Kronecker direction
    
    Args:
        model: The model to optimize
        X, Y: Input batch
        step: Current step (for debugging)
        zo_random_seed: Random seed for reproducibility
        strategy: Kronecker factorization strategy ('approx_square', 'fixed_factor', 'power2')
        max_factor: Maximum factor size for 'fixed_factor' strategy
        eps: Perturbation size (uses global zo_eps if None)
    
    Returns:
        loss: Loss from f(θ + ε*ΔW)
        projected_grad: Projected gradient coefficient c
        named_params: List of (name, parameter) tuples
    """
    if eps is None:
        eps = zo_eps_global  # Use global zo_eps if not provided
    
    # First function evaluation: f(θ + ε*ΔW)
    named_params = kronzo_perturb_parameters(model, zo_random_seed, step, zo_eps_global, scaling_factor=1, eps=eps, strategy=strategy, max_factor=max_factor)
    loss1 = zo_forward(model, X, Y, ctx_obj)

    # Second function evaluation: f(θ - ε*ΔW)
    kronzo_perturb_parameters(model, zo_random_seed, step, zo_eps_global, scaling_factor=-2, eps=eps, strategy=strategy, max_factor=max_factor)
    loss2 = zo_forward(model, X, Y, ctx_obj)

    # Calculate projected gradient: c = [f(θ + ε*ΔW) - f(θ - ε*ΔW)]/(2ε)
    projected_grad = ((loss1 - loss2) / (2 * eps)).item()

    # Reset model back to original parameters: θ
    kronzo_perturb_parameters(model, zo_random_seed, step, zo_eps_global, scaling_factor=1, eps=eps, strategy=strategy, max_factor=max_factor)
    
    return loss1, projected_grad, named_params

def kronzo_update(model,
                  optimizer,                      # kept for API compatibility
                  projected_grad: float,
                  zo_random_seed: int,
                  step: int,
                  lr: float,
                  weight_decay: float,
                  master_process: bool,
                  named_parameters_to_optim: list[tuple[str, torch.Tensor]] | None = None,
                  strategy: str = "approx_square",
                  max_factor: int = 32):
    """
    Update model parameters using KronZO: θ ← θ - lr * c * ΔW
    where ΔW is Kronecker-structured perturbation A ⊗ B
    """
    torch.manual_seed(zo_random_seed)

    if named_parameters_to_optim is None:
        named_parameters_to_optim = [(n if not n.startswith("_orig_mod.") else n[len("_orig_mod."):], p)
                                     for n, p in model.named_parameters()
                                     if p.requires_grad]

    # Debug logging only on first step
    if step == 0 and master_process:
        print(f"KronZO: Total parameters to optimize: {len(named_parameters_to_optim)}")
        print(f"KronZO: Using lr = {lr:.6f}, weight_decay = {weight_decay:.6f}")
        print(f"KronZO: Strategy = {strategy}, max_factor = {max_factor}")
        print(f"KronZO: projected_grad = {projected_grad:.6f}")

    for clean_name, param in named_parameters_to_optim:
        if param.ndim >= 2:                          # matrices
            d_out, d_in = param.shape
            m1, m2, n1, n2 = choose_kron_dims(d_out, d_in, strategy, max_factor)

            A = torch.randn(m1, n1, device=param.device, dtype=param.dtype)
            B = torch.randn(m2, n2, device=param.device, dtype=param.dtype)
            perturbation = torch.kron(A, B)

            is_weight = ("bias" not in clean_name
                         and "layer_norm" not in clean_name
                         and "layernorm" not in clean_name)

            if is_weight:
                param.data.sub_(lr * (projected_grad * perturbation +
                                      weight_decay * param.data))
            else:
                param.data.sub_(lr * projected_grad * perturbation)
        else:                                        # vectors
            z = torch.normal(0.0, 1.0,
                             size=param.size(),
                             device=param.device,
                             dtype=param.dtype)

            is_weight = ("bias" not in clean_name
                         and "layer_norm" not in clean_name
                         and "layernorm" not in clean_name)

            if is_weight:
                param.data.sub_(lr * (projected_grad * z +
                                      weight_decay * param.data))
            else:
                param.data.sub_(lr * projected_grad * z)

def kronzo_update_momentum(model,
                            optimizer,              # kept for API compatibility
                            projected_grad: float,
                            zo_random_seed: int,
                            exp_avg_m: dict[str, torch.Tensor],
                            step: int,
                            lr: float,
                            weight_decay: float,
                            master_process: bool,
                            beta1: float = 0.9,
                            named_parameters_to_optim: list[tuple[str, torch.Tensor]] | None = None,
                            strategy: str = "approx_square",
                            max_factor: int = 32):
    """
    Update model parameters using KronZO with momentum:
    m_t = β₁ * m_{t-1} + (1-β₁) * g_t
    θ = θ - lr * m_t
    """
    torch.manual_seed(zo_random_seed)

    if named_parameters_to_optim is None:
        named_parameters_to_optim = [(n if not n.startswith("_orig_mod.") else n[len("_orig_mod."):], p)
                                     for n, p in model.named_parameters()
                                     if p.requires_grad]

    # Debug logging only on first step
    if step == 0 and master_process:
        print(f"KronZO-M: Total parameters to optimize: {len(named_parameters_to_optim)}")
        print(f"KronZO-M: Using lr = {lr:.6f}, weight_decay = {weight_decay:.6f}")
        print(f"KronZO-M: Strategy = {strategy}, max_factor = {max_factor}, beta1 = {beta1:.2f}")
        print(f"KronZO-M: projected_grad = {projected_grad:.6f}")

    for clean_name, param in named_parameters_to_optim:
        if param.ndim >= 2:                          # matrices
            d_out, d_in = param.shape
            m1, m2, n1, n2 = choose_kron_dims(d_out, d_in, strategy, max_factor)

            A = torch.randn(m1, n1, device=param.device, dtype=param.dtype)
            B = torch.randn(m2, n2, device=param.device, dtype=param.dtype)
            perturbation = torch.kron(A, B)

            # momentum
            if clean_name not in exp_avg_m:
                exp_avg_m[clean_name] = torch.zeros_like(perturbation)
            exp_avg_m[clean_name] = (beta1 * exp_avg_m[clean_name] +
                                     (1 - beta1) * projected_grad * perturbation)

            is_weight = ("bias" not in clean_name
                         and "layer_norm" not in clean_name
                         and "layernorm" not in clean_name)

            if is_weight:
                param.data.sub_(lr * (exp_avg_m[clean_name] +
                                      weight_decay * param.data))
            else:
                param.data.sub_(lr * exp_avg_m[clean_name])
        else:                                        # vectors
            z = torch.normal(0.0, 1.0,
                             size=param.size(),
                             device=param.device,
                             dtype=param.dtype)

            if clean_name not in exp_avg_m:
                exp_avg_m[clean_name] = projected_grad * z
            else:
                exp_avg_m[clean_name] = (beta1 * exp_avg_m[clean_name] +
                                         (1 - beta1) * projected_grad * z)

            is_weight = ("bias" not in clean_name
                         and "layer_norm" not in clean_name
                         and "layernorm" not in clean_name)

            if is_weight:
                param.data.sub_(lr * (exp_avg_m[clean_name] +
                                      weight_decay * param.data))
            else:
                param.data.sub_(lr * exp_avg_m[clean_name])

def dikronzo_perturb_parameters(model,
                                zo_random_seed: int,
                                zo_eps_global: float, # Added global config
                                scaling_factor: float = 1.0,
                                eps: float | None = None,
                                strategy: str = "approx_square",
                                max_factor: int = 32):
    """
    Perturb model parameters with Kronecker product A ⊗ B
    (Gaussian for vectors). Used for both +eps and -eps perturbations.
    """
    if eps is None:
        eps = zo_eps_global

    torch.manual_seed(zo_random_seed)

    named_params = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            clean = name[len("_orig_mod."):] if name.startswith("_orig_mod.") else name
            named_params.append((clean, p))

    for _, p in named_params:
        if p.ndim >= 2:                      # matrices
            d_out, d_in = p.shape
            m1, m2, n1, n2 = choose_kron_dims(d_out, d_in, strategy, max_factor)
            A = torch.randn(m1, n1, device=p.device, dtype=p.dtype)
            B = torch.randn(m2, n2, device=p.device, dtype=p.dtype)
            perturb = torch.kron(A, B)
            p.data.add_(scaling_factor * perturb * eps)
        else:                                # vectors
            z = torch.normal(0.0, 1.0, size=p.size(), device=p.device, dtype=p.dtype)
            p.data.add_(scaling_factor * z * eps)

    return named_params


def dikronzo_step(model,
                  X, Y,
                  step: int,
                  zo_random_seed: int,
                  directional_q_global: int, # Added global config
                  zo_eps_global: float, # Added global config
                  ctx_obj, # Added context for zo_forward
                  directional_q: int | None = None,
                  eps: float | None = None,
                  strategy: str = "approx_square",
                  max_factor: int = 32,
                  direct_movement: bool = False):
    """
    Directional selection DiMeZO-style but with Kronecker perturbations.
    Returns:
        best_loss, grad_coeff OR direction_seed, seed, success
    """
    q = directional_q or directional_q_global
    eps_val = eps or zo_eps_global # Renamed to avoid conflict with outer scope

    baseline = zo_forward(model, X, Y, ctx_obj)
    best_loss, best_seed = float('inf'), None

    # Phase 1: Search for the best direction
    for i in range(q):
        seed = zo_random_seed + i
        # θ + ε ΔW
        dikronzo_perturb_parameters(model, seed, zo_eps_global, 1, eps_val, strategy, max_factor)
        loss_plus = zo_forward(model, X, Y, ctx_obj)
        # keep best
        if loss_plus < best_loss:
            best_loss, best_seed = loss_plus, seed
        # reset
        dikronzo_perturb_parameters(model, seed, zo_eps_global, -1, eps_val, strategy, max_factor)

    success = best_loss < baseline

    # Phase 2: Gradient estimation or direct movement
    if direct_movement:
        # Only return the seed of the best direction
        return best_loss, None, best_seed, success

    # θ + ε ΔW  (already known: best_loss)
    dikronzo_perturb_parameters(model, best_seed, zo_eps_global, 1, eps_val, strategy, max_factor)
    f_plus = best_loss
    # θ - ε ΔW
    dikronzo_perturb_parameters(model, best_seed, zo_eps_global, -2, eps_val, strategy, max_factor)
    f_minus = zo_forward(model, X, Y, ctx_obj)
    # coeff = (f+ - f-) / (2ε)
    grad_coeff = ((f_plus - f_minus) / (2 * eps_val)).item()
    # reset θ
    dikronzo_perturb_parameters(model, best_seed, zo_eps_global, 1, eps_val, strategy, max_factor)

    return best_loss, grad_coeff, best_seed, success


def dikronzo_update(model,
                    optimizer,                # for API compatibility
                    grad_or_seed,
                    best_seed: int,
                    step: int,
                    lr: float,
                    weight_decay: float,
                    master_process: bool,
                    named_parameters_to_optim: list[tuple[str, torch.Tensor]] | None = None,
                    strategy: str = "approx_square",
                    max_factor: int = 32,
                    direct_movement: bool = False):
    """
    Update parameters:
      • direct_movement mode: θ ← θ + α * ΔW_best
      • gradient mode: θ ← θ - α * c * ΔW_best
    """
    torch.manual_seed(best_seed)

    if named_parameters_to_optim is None:
        named_parameters_to_optim = [(n if not n.startswith("_orig_mod.") else n[len("_orig_mod."):], p)
                                     for n, p in model.named_parameters()
                                     if p.requires_grad]

    # Debug logging only on first step
    if step == 0 and master_process:
        mode_str = "DiKronZO-Direct" if direct_movement else "DiKronZO-Grad"
        print(f"{mode_str}: Total parameters to optimize: {len(named_parameters_to_optim)}")
        print(f"{mode_str}: Using lr = {lr:.6f}, weight_decay = {weight_decay:.6f}")
        print(f"{mode_str}: Strategy = {strategy}, max_factor = {max_factor}")
        print(f"{mode_str}: direct_movement = {direct_movement}")
        if not direct_movement:
            print(f"{mode_str}: grad_coeff = {grad_or_seed:.6f}")

    coeff = grad_or_seed            # either None (direct) or grad_coeff (gradient)

    for name, p in named_parameters_to_optim:
        is_weight = ("bias" not in name and
                     "layer_norm" not in name and
                     "layernorm" not in name)

        if p.ndim >= 2:              # matrices
            d_out, d_in = p.shape
            m1, m2, n1, n2 = choose_kron_dims(d_out, d_in, strategy, max_factor)
            A = torch.randn(m1, n1, device=p.device, dtype=p.dtype)
            B = torch.randn(m2, n2, device=p.device, dtype=p.dtype)
            perturb = torch.kron(A, B)
        else:                        # vectors
            perturb = torch.normal(0.0, 1.0, size=p.size(), device=p.device, dtype=p.dtype)

        if direct_movement:
            update = lr * perturb
        else:
            update = -lr * coeff * perturb

        if is_weight and not direct_movement:
            # weight decay only in gradient mode (optional in direct)
            p.data.add_(update - lr * weight_decay * p.data)
        else:
            p.data.add_(update) 