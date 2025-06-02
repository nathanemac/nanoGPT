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

def get_prime_factors(n):
    """Get prime factorization of n as a list of factors."""
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors

def generate_overlapping_factorizations(d_out: int, d_in: int, num_samples: int, strategy: str = "approx_square", max_factor: int = 32):
    """
    Generate overlapping Kronecker factorizations using different prime factor combinations.
    
    The idea is to create different "tiling patterns" that overlap maximally.
    For example, for 24×24:
    - A₁(3×4) ⊗ B₁(8×6): uses factors [3,4] and [8,6]
    - A₂(2×12) ⊗ B₂(12×2): uses factors [2,12] and [12,2]
    
    Args:
        d_out: Output dimension
        d_in: Input dimension
        num_samples: Number of overlapping factorizations to generate
        strategy: Base strategy (for fallback)
        max_factor: Maximum factor size (for filtering)
        
    Returns:
        List of (m1, m2, n1, n2) tuples with overlapping patterns
    """
    if num_samples == 1:
        return [choose_kron_dims(d_out, d_in, strategy, max_factor)]
    
    # Get prime factorizations
    out_primes = get_prime_factors(d_out)
    in_primes = get_prime_factors(d_in)
    
    # Generate all possible ways to group the prime factors
    def get_factor_groupings(primes):
        """Generate different ways to group prime factors into two numbers."""
        if len(primes) <= 1:
            return [(1, primes[0] if primes else 1)]
        
        groupings = []
        # Try all possible ways to split the prime factors
        from itertools import combinations
        
        for i in range(len(primes) + 1):
            for subset in combinations(range(len(primes)), i):
                group1_factors = [primes[j] for j in subset]
                group2_factors = [primes[j] for j in range(len(primes)) if j not in subset]
                
                group1 = 1
                for f in group1_factors:
                    group1 *= f
                    
                group2 = 1
                for f in group2_factors:
                    group2 *= f
                
                # Apply max_factor constraint if using fixed_factor strategy
                if strategy == "fixed_factor":
                    if group1 <= max_factor and group2 <= max_factor:
                        groupings.append((group1, group2))
                else:
                    groupings.append((group1, group2))
        
        # Remove duplicates and sort for deterministic ordering
        groupings = list(set(groupings))
        groupings.sort(key=lambda x: abs(x[0] - x[1]))  # Prefer more balanced splits first
        return groupings
    
    out_groupings = get_factor_groupings(out_primes)
    in_groupings = get_factor_groupings(in_primes)
    
    factorizations = []
    seen_factorizations = set()
    
    # Generate overlapping factorizations by combining different groupings
    for i in range(min(num_samples, len(out_groupings) * len(in_groupings))):
        out_idx = i % len(out_groupings)
        in_idx = (i // len(out_groupings)) % len(in_groupings)
        
        m1, m2 = out_groupings[out_idx]
        n1, n2 = in_groupings[in_idx]
        
        candidate = (m1, m2, n1, n2)
        
        if candidate not in seen_factorizations:
            factorizations.append(candidate)
            seen_factorizations.add(candidate)
            
            if len(factorizations) >= num_samples:
                break
    
    # If we need more factorizations, create variations by swapping
    while len(factorizations) < num_samples and len(factorizations) > 0:
        # Take an existing factorization and create a variation
        base_idx = (len(factorizations) - 1) % len(factorizations)
        m1, m2, n1, n2 = factorizations[base_idx]
        
        # Try swapping m1 ↔ n1 (creates different overlap pattern)
        if m1 <= d_in and n1 <= d_out and d_out % n1 == 0 and d_in % m1 == 0:
            swapped = (n1, d_out // n1, m1, d_in // m1)
            if swapped not in seen_factorizations:
                factorizations.append(swapped)
                seen_factorizations.add(swapped)
                continue
        
        # Try swapping m2 ↔ n2
        if m2 <= d_in and n2 <= d_out and d_out % n2 == 0 and d_in % m2 == 0:
            swapped = (d_out // n2, n2, d_in // m2, m2)
            if swapped not in seen_factorizations:
                factorizations.append(swapped)
                seen_factorizations.add(swapped)
                continue
        
        # If no more variations possible, break to avoid infinite loop
        break
    
    # Fallback: if we still don't have enough, use the base strategy
    while len(factorizations) < num_samples:
        base_factorization = choose_kron_dims(d_out, d_in, strategy, max_factor)
        if base_factorization not in seen_factorizations:
            factorizations.append(base_factorization)
            seen_factorizations.add(base_factorization)
        else:
            # If even base factorization is duplicate, just stop
            break
    
    return factorizations[:num_samples]

def choose_diverse_kron_dims(d_out: int, d_in: int, num_samples: int, strategy: str = "approx_square", max_factor: int = 32):
    """
    Generate diverse Kronecker factorizations for multi-sampling.
    Uses overlapping strategy based on prime factorizations.
    
    NOTE: The 'strategy' parameter behavior depends on num_samples:
    - If num_samples == 1: Uses original strategy ('approx_square', 'fixed_factor', 'power2')
    - If num_samples > 1: Uses prime-factor overlapping strategy, but:
      * 'fixed_factor' strategy still applies max_factor constraints
      * Other strategies use max_factor as a general constraint
      * Original strategy is used as fallback if needed
    
    Args:
        d_out: Output dimension
        d_in: Input dimension  
        num_samples: Number of overlapping factorizations to generate
        strategy: Base strategy - controls constraints and fallback behavior
        max_factor: Maximum factor size for constraints
        
    Returns:
        List of (m1, m2, n1, n2) tuples with overlapping patterns
    """
    return generate_overlapping_factorizations(d_out, d_in, num_samples, strategy, max_factor)

def kronzo_perturb_parameters(model,
                              zo_random_seed: int,
                              step: int,
                              zo_eps_global: float, # Added to pass global config
                              scaling_factor: float = 1.0,
                              eps: float | None = None,
                              strategy: str = "approx_square",
                              max_factor: int = 32,
                              kronzo_sampling_number: int = 1,
                              b_dict: dict | None = None,
                              step_interval: int = 50):
    """
    Apply Kronecker-type perturbations (or Gaussian for vectors) 
    to all trainable parameters of the model.
    
    Implements ν-step B matrix updates: B matrices are stored and updated every step_interval steps,
    while A matrices are sampled fresh each iteration.
    
    Returns the list of (name, param) tuples of perturbed parameters.
    
    Args:
        model: The model to perturb
        zo_random_seed: Random seed for reproducibility
        step: Current step (used for B matrix update timing)
        zo_eps_global: Global perturbation size
        scaling_factor: Scaling factor for perturbations (+1, -1, -2, etc.)
        eps: Override perturbation size (uses zo_eps_global if None)
        strategy: Kronecker factorization strategy
        max_factor: Maximum factor size for factorization
        kronzo_sampling_number: Number of Kronecker products to sample and sum
        b_dict: Dictionary to store B matrices (updated every step_interval steps)
        step_interval: Interval for updating B matrices (every ν steps)
    """
    if eps is None:
        eps = zo_eps_global

    if b_dict is None:
        b_dict = {}

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
            
            if kronzo_sampling_number == 1:
                # Standard single Kronecker product with B matrix storage
                m1, m2, n1, n2 = choose_kron_dims(d_out, d_in, strategy, max_factor)
                
                # Always sample fresh A matrix
                A = torch.randn(m1, n1, device=param.device, dtype=param.dtype)
                
                # B matrix: stored and updated every step_interval steps
                b_key = f"{clean_name}_B"
                need_new_b = (step % step_interval == 0 or b_key not in b_dict)
                
                if need_new_b:
                    # Update B matrix - use step-based seed for consistency across A seed changes
                    torch.manual_seed(1000000 + step * 1000 + hash(clean_name) % 1000)
                    B = torch.randn(m2, n2, device=param.device, dtype=param.dtype)
                    b_dict[b_key] = B
                else:
                    # Use stored B matrix
                    B = b_dict[b_key]
                
                perturbation = torch.kron(A, B)          # (d_out, d_in)
            else:
                # Multi-sampling: sum multiple diverse Kronecker products
                factorizations = choose_diverse_kron_dims(d_out, d_in, kronzo_sampling_number, strategy, max_factor)
                perturbation = torch.zeros(d_out, d_in, device=param.device, dtype=param.dtype)
                
                for i, (m1, m2, n1, n2) in enumerate(factorizations):
                    # Always sample fresh A matrix (with different seed for each factorization)
                    torch.manual_seed(zo_random_seed + i + 1)
                    A = torch.randn(m1, n1, device=param.device, dtype=param.dtype)
                    
                    # B matrix: stored and updated every step_interval steps
                    b_key = f"{clean_name}_B_{i}"
                    need_new_b = (step % step_interval == 0 or b_key not in b_dict)
                    
                    if need_new_b:
                        # Update B matrix - use step-based seed with factorization index for uniqueness
                        torch.manual_seed(1000000 + step * 1000 + hash(clean_name) % 1000 + i)
                        B = torch.randn(m2, n2, device=param.device, dtype=param.dtype)
                        b_dict[b_key] = B
                    else:
                        # Use stored B matrix
                        B = b_dict[b_key]
                    
                    kron_product = torch.kron(A, B)
                    perturbation.add_(kron_product)
                
                # Scale by 1/kronzo_sampling_number to maintain similar magnitude
                perturbation.mul_(1.0 / kronzo_sampling_number)

            param.data.add_(scaling_factor * perturbation * eps)
        else:                                        # vectors / biases
            z = torch.normal(0.0, 1.0,
                             size=param.size(),
                             device=param.device,
                             dtype=param.dtype)
            param.data.add_(scaling_factor * z * eps)

    return named_parameters_to_optim

def kronzo_step(model, X, Y, step, zo_random_seed, zo_eps_global, ctx_obj, strategy='approx_square', max_factor=32, eps=None, kronzo_sampling_number=1, b_dict=None, step_interval=50):
    """
    Estimate gradient using KronZO (Kronecker Zero-Order optimization).
    
    Algorithm:
    1. For each matrix parameter W ∈ ℝ^(d_out × d_in):
       - Choose factorization: m1×n1 = d_out, m2×n2 = d_in  
       - Sample A ∈ ℝ^(m1×n1) fresh each iteration
       - Use stored B ∈ ℝ^(m2×n2) (updated every ν steps)
       - Perturbation: ΔW = A ⊗ B (Kronecker product)
    2. Compute finite difference: c = [f(θ + ε*ΔW) - f(θ - ε*ΔW)]/(2ε)
    3. This gives projected gradient ∇f(θ) · ΔW along Kronecker direction
    
    With multi-sampling (kronzo_sampling_number > 1):
    - Sample multiple overlapping Kronecker products and sum them
    - Uses different factorizations for better space coverage
    - B matrices are stored separately for each factorization
    
    Args:
        model: The model to optimize
        X, Y: Input batch
        step: Current step (for B matrix update timing)
        zo_random_seed: Random seed for reproducibility
        strategy: Kronecker factorization strategy ('approx_square', 'fixed_factor', 'power2')
        max_factor: Maximum factor size for 'fixed_factor' strategy
        eps: Perturbation size (uses global zo_eps if None)
        kronzo_sampling_number: Number of Kronecker products to sample and sum
        b_dict: Dictionary to store B matrices (will be created if None)
        step_interval: Interval for updating B matrices (every ν steps)
    
    Returns:
        loss: Loss from f(θ + ε*ΔW)
        projected_grad: Projected gradient coefficient c
        named_params: List of (name, parameter) tuples
    """
    if eps is None:
        eps = zo_eps_global  # Use global zo_eps if not provided
    
    if b_dict is None:
        b_dict = {}
    
    # First function evaluation: f(θ + ε*ΔW)
    named_params = kronzo_perturb_parameters(model, zo_random_seed, step, zo_eps_global, scaling_factor=1, eps=eps, strategy=strategy, max_factor=max_factor, kronzo_sampling_number=kronzo_sampling_number, b_dict=b_dict, step_interval=step_interval)
    loss1 = zo_forward(model, X, Y, ctx_obj)

    # Second function evaluation: f(θ - ε*ΔW)
    kronzo_perturb_parameters(model, zo_random_seed, step, zo_eps_global, scaling_factor=-2, eps=eps, strategy=strategy, max_factor=max_factor, kronzo_sampling_number=kronzo_sampling_number, b_dict=b_dict, step_interval=step_interval)
    loss2 = zo_forward(model, X, Y, ctx_obj)

    # Calculate projected gradient: c = [f(θ + ε*ΔW) - f(θ - ε*ΔW)]/(2ε)
    projected_grad = ((loss1 - loss2) / (2 * eps)).item()

    # Reset model back to original parameters: θ
    kronzo_perturb_parameters(model, zo_random_seed, step, zo_eps_global, scaling_factor=1, eps=eps, strategy=strategy, max_factor=max_factor, kronzo_sampling_number=kronzo_sampling_number, b_dict=b_dict, step_interval=step_interval)
    
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
                  max_factor: int = 32,
                  kronzo_sampling_number: int = 1,
                  b_dict: dict | None = None,
                  step_interval: int = 50):
    """
    Update model parameters using KronZO: θ ← θ - lr * c * ΔW
    where ΔW is Kronecker-structured perturbation A ⊗ B
    
    With multi-sampling: ΔW = Σᵢ (Aᵢ ⊗ Bᵢ) using overlapping factorizations
    B matrices are stored and updated every step_interval steps.
    """
    torch.manual_seed(zo_random_seed)

    if named_parameters_to_optim is None:
        named_parameters_to_optim = [(n if not n.startswith("_orig_mod.") else n[len("_orig_mod."):], p)
                                     for n, p in model.named_parameters()
                                     if p.requires_grad]

    if b_dict is None:
        b_dict = {}

    # Debug logging only on first step
    if step == 0 and master_process:
        print(f"KronZO: Total parameters to optimize: {len(named_parameters_to_optim)}")
        print(f"KronZO: Using lr = {lr:.6f}, weight_decay = {weight_decay:.6f}")
        print(f"KronZO: Strategy = {strategy}, max_factor = {max_factor}")
        print(f"KronZO: kronzo_sampling_number = {kronzo_sampling_number}")
        print(f"KronZO: step_interval = {step_interval} (B matrix update frequency)")
        print(f"KronZO: projected_grad = {projected_grad:.6f}")

    for clean_name, param in named_parameters_to_optim:
        if param.ndim >= 2:                          # matrices
            d_out, d_in = param.shape
            
            if kronzo_sampling_number == 1:
                # Standard single Kronecker product with B matrix storage
                m1, m2, n1, n2 = choose_kron_dims(d_out, d_in, strategy, max_factor)
                
                # Always sample fresh A matrix
                A = torch.randn(m1, n1, device=param.device, dtype=param.dtype)
                
                # B matrix: stored and updated every step_interval steps
                b_key = f"{clean_name}_B"
                need_new_b = (step % step_interval == 0 or b_key not in b_dict)
                
                if need_new_b:
                    # Update B matrix - use step-based seed for consistency across A seed changes
                    torch.manual_seed(1000000 + step * 1000 + hash(clean_name) % 1000)
                    B = torch.randn(m2, n2, device=param.device, dtype=param.dtype)
                    b_dict[b_key] = B
                else:
                    # Use stored B matrix
                    B = b_dict[b_key]
                
                perturbation = torch.kron(A, B)
            else:
                # Multi-sampling: sum multiple overlapping Kronecker products
                factorizations = choose_diverse_kron_dims(d_out, d_in, kronzo_sampling_number, strategy, max_factor)
                perturbation = torch.zeros(d_out, d_in, device=param.device, dtype=param.dtype)
                
                for i, (m1, m2, n1, n2) in enumerate(factorizations):
                    # Always sample fresh A matrix (with different seed for each factorization)
                    torch.manual_seed(zo_random_seed + i + 1)
                    A = torch.randn(m1, n1, device=param.device, dtype=param.dtype)
                    
                    # B matrix: stored and updated every step_interval steps
                    b_key = f"{clean_name}_B_{i}"
                    need_new_b = (step % step_interval == 0 or b_key not in b_dict)
                    
                    if need_new_b:
                        # Update B matrix - use step-based seed with factorization index for uniqueness
                        torch.manual_seed(1000000 + step * 1000 + hash(clean_name) % 1000 + i)
                        B = torch.randn(m2, n2, device=param.device, dtype=param.dtype)
                        b_dict[b_key] = B
                    else:
                        # Use stored B matrix
                        B = b_dict[b_key]
                    
                    kron_product = torch.kron(A, B)
                    perturbation.add_(kron_product)
                
                # Scale by 1/kronzo_sampling_number to maintain similar magnitude
                perturbation.mul_(1.0 / kronzo_sampling_number)

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
                            max_factor: int = 32,
                            kronzo_sampling_number: int = 1,
                            b_dict: dict | None = None,
                            step_interval: int = 50):
    """
    Update model parameters using KronZO with momentum:
    m_t = β₁ * m_{t-1} + (1-β₁) * g_t
    θ = θ - lr * m_t
    
    EFFICIENT MOMENTUM STORAGE:
    - For matrices: Store momentum for A matrices only (size m1×n1), not full Kronecker products  
    - Apply momentum with stored B: momentum_A ⊗ B
    - This saves massive memory compared to storing full A⊗B tensors
    
    With multi-sampling: g_t uses Σᵢ (Aᵢ ⊗ Bᵢ) with overlapping factorizations
    B matrices are stored and updated every step_interval steps.
    """
    torch.manual_seed(zo_random_seed)

    if named_parameters_to_optim is None:
        named_parameters_to_optim = [(n if not n.startswith("_orig_mod.") else n[len("_orig_mod."):], p)
                                     for n, p in model.named_parameters()
                                     if p.requires_grad]

    if b_dict is None:
        b_dict = {}

    # Debug logging only on first step
    if step == 0 and master_process:
        print(f"KronZO-M: Total parameters to optimize: {len(named_parameters_to_optim)}")
        print(f"KronZO-M: Using lr = {lr:.6f}, weight_decay = {weight_decay:.6f}")
        print(f"KronZO-M: Strategy = {strategy}, max_factor = {max_factor}, beta1 = {beta1:.2f}")
        print(f"KronZO-M: kronzo_sampling_number = {kronzo_sampling_number}")
        print(f"KronZO-M: step_interval = {step_interval} (B matrix update frequency)")
        print(f"KronZO-M: projected_grad = {projected_grad:.6f}")
        print(f"KronZO-M: Efficient storage - momentum for A matrices only")

    for clean_name, param in named_parameters_to_optim:
        if param.ndim >= 2:                          # matrices
            d_out, d_in = param.shape
            
            if kronzo_sampling_number == 1:
                # Standard single Kronecker product with efficient momentum storage
                m1, m2, n1, n2 = choose_kron_dims(d_out, d_in, strategy, max_factor)
                
                # Always sample fresh A matrix
                A = torch.randn(m1, n1, device=param.device, dtype=param.dtype)
                
                # B matrix: stored and updated every step_interval steps
                b_key = f"{clean_name}_B"
                need_new_b = (step % step_interval == 0 or b_key not in b_dict)
                
                if need_new_b:
                    # Update B matrix - use step-based seed for consistency across A seed changes
                    torch.manual_seed(1000000 + step * 1000 + hash(clean_name) % 1000)
                    B = torch.randn(m2, n2, device=param.device, dtype=param.dtype)
                    b_dict[b_key] = B
                else:
                    # Use stored B matrix
                    B = b_dict[b_key]
                
                # EFFICIENT MOMENTUM: Store only A matrix scaled by gradient coefficient
                momentum_key = f"{clean_name}_A"
                if momentum_key not in exp_avg_m:
                    exp_avg_m[momentum_key] = torch.zeros_like(A)
                
                # Update momentum: m_t = β * m_{t-1} + (1-β) * (c * A)
                grad_scaled_A = projected_grad * A
                exp_avg_m[momentum_key] = beta1 * exp_avg_m[momentum_key] + (1 - beta1) * grad_scaled_A
                
                # Apply momentum: A_momentum ⊗ B
                momentum_perturbation = torch.kron(exp_avg_m[momentum_key], B)
            else:
                # Multi-sampling: efficient momentum for each factorization
                factorizations = choose_diverse_kron_dims(d_out, d_in, kronzo_sampling_number, strategy, max_factor)
                momentum_perturbation = torch.zeros(d_out, d_in, device=param.device, dtype=param.dtype)
                
                for i, (m1, m2, n1, n2) in enumerate(factorizations):
                    # Always sample fresh A matrix (with different seed for each factorization)
                    torch.manual_seed(zo_random_seed + i + 1)
                    A = torch.randn(m1, n1, device=param.device, dtype=param.dtype)
                    
                    # B matrix: stored and updated every step_interval steps
                    b_key = f"{clean_name}_B_{i}"
                    need_new_b = (step % step_interval == 0 or b_key not in b_dict)
                    
                    if need_new_b:
                        # Update B matrix - use step-based seed with factorization index for uniqueness
                        torch.manual_seed(1000000 + step * 1000 + hash(clean_name) % 1000 + i)
                        B = torch.randn(m2, n2, device=param.device, dtype=param.dtype)
                        b_dict[b_key] = B
                    else:
                        # Use stored B matrix
                        B = b_dict[b_key]
                    
                    # EFFICIENT MOMENTUM: Store only A matrix scaled by gradient coefficient
                    momentum_key = f"{clean_name}_A_{i}"
                    if momentum_key not in exp_avg_m:
                        exp_avg_m[momentum_key] = torch.zeros_like(A)
                    
                    # Update momentum: m_t = β * m_{t-1} + (1-β) * (c * A)
                    grad_scaled_A = projected_grad * A
                    exp_avg_m[momentum_key] = beta1 * exp_avg_m[momentum_key] + (1 - beta1) * grad_scaled_A
                    
                    # Apply momentum: A_momentum ⊗ B
                    kron_momentum = torch.kron(exp_avg_m[momentum_key], B)
                    momentum_perturbation.add_(kron_momentum)
                
                # Scale by 1/kronzo_sampling_number to maintain similar magnitude
                momentum_perturbation.mul_(1.0 / kronzo_sampling_number)

            is_weight = ("bias" not in clean_name
                         and "layer_norm" not in clean_name
                         and "layernorm" not in clean_name)

            if is_weight:
                param.data.sub_(lr * (momentum_perturbation +
                                      weight_decay * param.data))
            else:
                param.data.sub_(lr * momentum_perturbation)
        else:                                        # vectors
            z = torch.normal(0.0, 1.0,
                             size=param.size(),
                             device=param.device,
                             dtype=param.dtype)

            # Momentum for vectors (standard approach)
            momentum_key = clean_name
            if momentum_key not in exp_avg_m:
                exp_avg_m[momentum_key] = torch.zeros_like(z)
            exp_avg_m[momentum_key] = (beta1 * exp_avg_m[momentum_key] +
                                       (1 - beta1) * projected_grad * z)

            is_weight = ("bias" not in clean_name
                         and "layer_norm" not in clean_name
                         and "layernorm" not in clean_name)

            if is_weight:
                param.data.sub_(lr * (exp_avg_m[momentum_key] +
                                      weight_decay * param.data))
            else:
                param.data.sub_(lr * exp_avg_m[momentum_key])

def dikronzo_perturb_parameters(model,
                                zo_random_seed: int,
                                zo_eps_global: float, # Added global config
                                scaling_factor: float = 1.0,
                                eps: float | None = None,
                                strategy: str = "approx_square",
                                max_factor: int = 32,
                                kronzo_sampling_number: int = 1,
                                b_dict: dict | None = None,
                                step_interval: int = 50,
                                step: int = 0):
    """
    Perturb model parameters with Kronecker product A ⊗ B
    (Gaussian for vectors). Used for both +eps and -eps perturbations.
    
    With multi-sampling: Uses Σᵢ (Aᵢ ⊗ Bᵢ) with overlapping factorizations.
    """
    if eps is None:
        eps = zo_eps_global

    if b_dict is None:
        b_dict = {}

    torch.manual_seed(zo_random_seed)

    named_params = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            clean = name[len("_orig_mod."):] if name.startswith("_orig_mod.") else name
            named_params.append((clean, p))

    for clean_name, p in named_params:
        if p.ndim >= 2:                      # matrices
            d_out, d_in = p.shape
            
            if kronzo_sampling_number == 1:
                # Standard single Kronecker product with B matrix storage
                m1, m2, n1, n2 = choose_kron_dims(d_out, d_in, strategy, max_factor)
                
                # Always sample fresh A matrix
                A = torch.randn(m1, n1, device=p.device, dtype=p.dtype)
                
                # B matrix: stored and updated every step_interval steps
                b_key = f"{clean_name}_B"
                need_new_b = (step % step_interval == 0 or b_key not in b_dict)
                
                if need_new_b:
                    # Update B matrix - use step-based seed for consistency across A seed changes
                    torch.manual_seed(1000000 + step * 1000 + hash(clean_name) % 1000)
                    B = torch.randn(m2, n2, device=p.device, dtype=p.dtype)
                    b_dict[b_key] = B
                else:
                    # Use stored B matrix
                    B = b_dict[b_key]
                
                perturb = torch.kron(A, B)
            else:
                # Multi-sampling: sum multiple overlapping Kronecker products
                factorizations = choose_diverse_kron_dims(d_out, d_in, kronzo_sampling_number, strategy, max_factor)
                perturb = torch.zeros(d_out, d_in, device=p.device, dtype=p.dtype)
                
                for i, (m1, m2, n1, n2) in enumerate(factorizations):
                    # Always sample fresh A matrix (with different seed for each factorization)
                    torch.manual_seed(zo_random_seed + i + 1)
                    A = torch.randn(m1, n1, device=p.device, dtype=p.dtype)
                    
                    # B matrix: stored and updated every step_interval steps
                    b_key = f"{clean_name}_B_{i}"
                    need_new_b = (step % step_interval == 0 or b_key not in b_dict)
                    
                    if need_new_b:
                        # Update B matrix - use step-based seed with factorization index for uniqueness
                        torch.manual_seed(1000000 + step * 1000 + hash(clean_name) % 1000 + i)
                        B = torch.randn(m2, n2, device=p.device, dtype=p.dtype)
                        b_dict[b_key] = B
                    else:
                        # Use stored B matrix
                        B = b_dict[b_key]
                    
                    kron_product = torch.kron(A, B)
                    perturb.add_(kron_product)
                
                # Scale by 1/kronzo_sampling_number to maintain similar magnitude
                perturb.mul_(1.0 / kronzo_sampling_number)
            
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
                  direct_movement: bool = False,
                  kronzo_sampling_number: int = 1,
                  b_dict: dict | None = None,
                  step_interval: int = 50):
    """
    Directional selection DiMeZO-style but with Kronecker perturbations.
    
    With multi-sampling: Each direction uses Σᵢ (Aᵢ ⊗ Bᵢ) with overlapping factorizations.
    B matrices are stored and updated every step_interval steps.
    
    Returns:
        best_loss, grad_coeff OR direction_seed, seed, success
    """
    q = directional_q or directional_q_global
    eps_val = eps or zo_eps_global # Renamed to avoid conflict with outer scope

    if b_dict is None:
        b_dict = {}

    baseline = zo_forward(model, X, Y, ctx_obj)
    best_loss, best_seed = float('inf'), None

    # Phase 1: Search for the best direction
    for i in range(q):
        seed = zo_random_seed + i
        # θ + ε ΔW
        dikronzo_perturb_parameters(model, seed, zo_eps_global, 1, eps_val, strategy, max_factor, kronzo_sampling_number, b_dict, step_interval, step)
        loss_plus = zo_forward(model, X, Y, ctx_obj)
        # keep best
        if loss_plus < best_loss:
            best_loss, best_seed = loss_plus, seed
        # reset
        dikronzo_perturb_parameters(model, seed, zo_eps_global, -1, eps_val, strategy, max_factor, kronzo_sampling_number, b_dict, step_interval, step)

    success = best_loss < baseline

    # Phase 2: Gradient estimation or direct movement
    if direct_movement:
        # Only return the seed of the best direction
        return best_loss, None, best_seed, success

    # θ + ε ΔW  (already known: best_loss)
    dikronzo_perturb_parameters(model, best_seed, zo_eps_global, 1, eps_val, strategy, max_factor, kronzo_sampling_number, b_dict, step_interval, step)
    f_plus = best_loss
    # θ - ε ΔW
    dikronzo_perturb_parameters(model, best_seed, zo_eps_global, -2, eps_val, strategy, max_factor, kronzo_sampling_number, b_dict, step_interval, step)
    f_minus = zo_forward(model, X, Y, ctx_obj)
    # coeff = (f+ - f-) / (2ε)
    grad_coeff = ((f_plus - f_minus) / (2 * eps_val)).item()
    # reset θ
    dikronzo_perturb_parameters(model, best_seed, zo_eps_global, 1, eps_val, strategy, max_factor, kronzo_sampling_number, b_dict, step_interval, step)

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
                    direct_movement: bool = False,
                    kronzo_sampling_number: int = 1,
                    b_dict: dict | None = None,
                    step_interval: int = 50):
    """
    Update model parameters using DiKronZO.
    
    Two update modes:
    1. Direct Movement: Move directly toward the best direction
    2. Gradient Descent: Use estimated gradient coefficient
    """
    if named_parameters_to_optim is None:
        named_parameters_to_optim = [(n if not n.startswith("_orig_mod.") else n[len("_orig_mod."):], p)
                                     for n, p in model.named_parameters()
                                     if p.requires_grad]

    if b_dict is None:
        b_dict = {}

    if direct_movement:
        # Direct movement mode - move toward the best direction without gradient
        torch.manual_seed(best_seed)
        for clean_name, p in named_parameters_to_optim:
            if p.ndim >= 2:                      # matrices
                d_out, d_in = p.shape
                
                if kronzo_sampling_number == 1:
                    # Standard single Kronecker product with B matrix storage
                    m1, m2, n1, n2 = choose_kron_dims(d_out, d_in, strategy, max_factor)
                    
                    # Always sample fresh A matrix
                    A = torch.randn(m1, n1, device=p.device, dtype=p.dtype)
                    
                    # B matrix: stored and updated every step_interval steps
                    b_key = f"{clean_name}_B"
                    need_new_b = (step % step_interval == 0 or b_key not in b_dict)
                    
                    if need_new_b:
                        # Update B matrix - use step-based seed for consistency across A seed changes
                        torch.manual_seed(1000000 + step * 1000 + hash(clean_name) % 1000)
                        B = torch.randn(m2, n2, device=p.device, dtype=p.dtype)
                        b_dict[b_key] = B
                    else:
                        # Use stored B matrix
                        B = b_dict[b_key]
                    
                    direction = torch.kron(A, B)
                else:
                    # Multi-sampling: sum multiple overlapping Kronecker products
                    factorizations = choose_diverse_kron_dims(d_out, d_in, kronzo_sampling_number, strategy, max_factor)
                    direction = torch.zeros(d_out, d_in, device=p.device, dtype=p.dtype)
                    
                    for i, (m1, m2, n1, n2) in enumerate(factorizations):
                        # Always sample fresh A matrix (with different seed for each factorization)
                        torch.manual_seed(best_seed + i + 1)
                        A = torch.randn(m1, n1, device=p.device, dtype=p.dtype)
                        
                        # B matrix: stored and updated every step_interval steps
                        b_key = f"{clean_name}_B_{i}"
                        need_new_b = (step % step_interval == 0 or b_key not in b_dict)
                        
                        if need_new_b:
                            # Update B matrix - use step-based seed with factorization index for uniqueness
                            torch.manual_seed(1000000 + step * 1000 + hash(clean_name) % 1000 + i)
                            B = torch.randn(m2, n2, device=p.device, dtype=p.dtype)
                            b_dict[b_key] = B
                        else:
                            # Use stored B matrix
                            B = b_dict[b_key]
                        
                        kron_product = torch.kron(A, B)
                        direction.add_(kron_product)
                    
                    # Scale by 1/kronzo_sampling_number to maintain similar magnitude
                    direction.mul_(1.0 / kronzo_sampling_number)

                # Direct movement update: θ ← θ + α * direction
                is_weight = ("bias" not in clean_name
                             and "layer_norm" not in clean_name
                             and "layernorm" not in clean_name)
                if is_weight:
                    # With weight decay: θ ← θ * (1 - α * λ) + α * direction
                    p.data = p.data * (1 - lr * weight_decay) + lr * direction
                else:
                    p.data.add_(direction, alpha=lr)
            else:                                # vectors
                z = torch.normal(0.0, 1.0,
                                 size=p.size(),
                                 device=p.device,
                                 dtype=p.dtype)

                is_weight = ("bias" not in clean_name
                             and "layer_norm" not in clean_name
                             and "layernorm" not in clean_name)
                if is_weight:
                    # With weight decay: θ ← θ * (1 - α * λ) + α * z
                    p.data = p.data * (1 - lr * weight_decay) + lr * z
                else:
                    p.data.add_(z, alpha=lr)
    else:
        # Gradient descent mode - use gradient coefficient
        torch.manual_seed(best_seed)
        projected_grad = grad_or_seed

        for clean_name, p in named_parameters_to_optim:
            if p.ndim >= 2:                      # matrices
                d_out, d_in = p.shape
                
                if kronzo_sampling_number == 1:
                    # Standard single Kronecker product with B matrix storage
                    m1, m2, n1, n2 = choose_kron_dims(d_out, d_in, strategy, max_factor)
                    
                    # Always sample fresh A matrix
                    A = torch.randn(m1, n1, device=p.device, dtype=p.dtype)
                    
                    # B matrix: stored and updated every step_interval steps
                    b_key = f"{clean_name}_B"
                    need_new_b = (step % step_interval == 0 or b_key not in b_dict)
                    
                    if need_new_b:
                        # Update B matrix - use step-based seed for consistency across A seed changes
                        torch.manual_seed(1000000 + step * 1000 + hash(clean_name) % 1000)
                        B = torch.randn(m2, n2, device=p.device, dtype=p.dtype)
                        b_dict[b_key] = B
                    else:
                        # Use stored B matrix
                        B = b_dict[b_key]
                    
                    perturbation = torch.kron(A, B)
                else:
                    # Multi-sampling: sum multiple overlapping Kronecker products
                    factorizations = choose_diverse_kron_dims(d_out, d_in, kronzo_sampling_number, strategy, max_factor)
                    perturbation = torch.zeros(d_out, d_in, device=p.device, dtype=p.dtype)
                    
                    for i, (m1, m2, n1, n2) in enumerate(factorizations):
                        # Always sample fresh A matrix (with different seed for each factorization)
                        torch.manual_seed(best_seed + i + 1)
                        A = torch.randn(m1, n1, device=p.device, dtype=p.dtype)
                        
                        # B matrix: stored and updated every step_interval steps
                        b_key = f"{clean_name}_B_{i}"
                        need_new_b = (step % step_interval == 0 or b_key not in b_dict)
                        
                        if need_new_b:
                            # Update B matrix - use step-based seed with factorization index for uniqueness
                            torch.manual_seed(1000000 + step * 1000 + hash(clean_name) % 1000 + i)
                            B = torch.randn(m2, n2, device=p.device, dtype=p.dtype)
                            b_dict[b_key] = B
                        else:
                            # Use stored B matrix
                            B = b_dict[b_key]
                        
                        kron_product = torch.kron(A, B)
                        perturbation.add_(kron_product)
                    
                    # Scale by 1/kronzo_sampling_number to maintain similar magnitude
                    perturbation.mul_(1.0 / kronzo_sampling_number)

                # Gradient descent update
                is_weight = ("bias" not in clean_name
                             and "layer_norm" not in clean_name
                             and "layernorm" not in clean_name)
                if is_weight:
                    p.data.sub_(lr * (projected_grad * perturbation + weight_decay * p.data))
                else:
                    p.data.sub_(lr * projected_grad * perturbation)
            else:                                # vectors
                z = torch.normal(0.0, 1.0,
                                 size=p.size(),
                                 device=p.device,
                                 dtype=p.dtype)

                is_weight = ("bias" not in clean_name
                             and "layer_norm" not in clean_name
                             and "layernorm" not in clean_name)
                if is_weight:
                    p.data.sub_(lr * (projected_grad * z + weight_decay * p.data))
                else:
                    p.data.sub_(lr * projected_grad * z)

def dikronzo_update_momentum(model,
                            optimizer,              # for API compatibility
                            grad_or_seed,
                            best_seed: int,
                            exp_avg_m: dict[str, torch.Tensor],
                            step: int,
                            lr: float,
                            weight_decay: float,
                            master_process: bool,
                            beta1: float = 0.9,
                            named_parameters_to_optim: list[tuple[str, torch.Tensor]] | None = None,
                            strategy: str = "approx_square",
                            max_factor: int = 32,
                            direct_movement: bool = False,
                            kronzo_sampling_number: int = 1,
                            b_dict: dict | None = None,
                            step_interval: int = 50):
    """
    Update model parameters using DiKronZO with momentum.
    
    EFFICIENT MOMENTUM STORAGE:
    - For matrices: Store momentum for A matrices only (size m1×n1), not full Kronecker products
    - Apply momentum with stored B: momentum_A ⊗ B
    - This saves massive memory: A⊗B storage vs (A⊗B) storage 
    
    Two update modes:
    1. Direct Movement: g_t = A⊗B direction, m_t accumulates A matrices
    2. Gradient Descent: g_t = c*(A⊗B), m_t accumulates c*A matrices  
    """
    if named_parameters_to_optim is None:
        named_parameters_to_optim = [(n if not n.startswith("_orig_mod.") else n[len("_orig_mod."):], p)
                                     for n, p in model.named_parameters()
                                     if p.requires_grad]

    if b_dict is None:
        b_dict = {}

    # Debug logging only on first step
    if step == 0 and master_process:
        print(f"DiKronZO-M: Total parameters to optimize: {len(named_parameters_to_optim)}")
        print(f"DiKronZO-M: Using lr = {lr:.6f}, weight_decay = {weight_decay:.6f}")
        print(f"DiKronZO-M: Strategy = {strategy}, max_factor = {max_factor}, beta1 = {beta1:.2f}")
        print(f"DiKronZO-M: kronzo_sampling_number = {kronzo_sampling_number}")
        print(f"DiKronZO-M: step_interval = {step_interval}, direct_movement = {direct_movement}")
        print(f"DiKronZO-M: Efficient storage - momentum for A matrices only")

    if direct_movement:
        # Direct movement mode with momentum
        torch.manual_seed(best_seed)
        for clean_name, p in named_parameters_to_optim:
            if p.ndim >= 2:                      # matrices
                d_out, d_in = p.shape
                
                if kronzo_sampling_number == 1:
                    # Standard single Kronecker product with efficient momentum storage
                    m1, m2, n1, n2 = choose_kron_dims(d_out, d_in, strategy, max_factor)
                    
                    # Always sample fresh A matrix
                    A = torch.randn(m1, n1, device=p.device, dtype=p.dtype)
                    
                    # B matrix: stored and updated every step_interval steps
                    b_key = f"{clean_name}_B"
                    need_new_b = (step % step_interval == 0 or b_key not in b_dict)
                    
                    if need_new_b:
                        torch.manual_seed(1000000 + step * 1000 + hash(clean_name) % 1000)
                        B = torch.randn(m2, n2, device=p.device, dtype=p.dtype)
                        b_dict[b_key] = B
                    else:
                        B = b_dict[b_key]
                    
                    # EFFICIENT MOMENTUM: Store only A matrix, not full Kronecker product
                    momentum_key = f"{clean_name}_A"
                    if momentum_key not in exp_avg_m:
                        exp_avg_m[momentum_key] = torch.zeros_like(A)
                    
                    # Update momentum: m_t = β * m_{t-1} + (1-β) * A
                    exp_avg_m[momentum_key] = beta1 * exp_avg_m[momentum_key] + (1 - beta1) * A
                    
                    # Apply momentum: A_momentum ⊗ B 
                    momentum_direction = torch.kron(exp_avg_m[momentum_key], B)
                else:
                    # Multi-sampling: efficient momentum for each factorization
                    factorizations = choose_diverse_kron_dims(d_out, d_in, kronzo_sampling_number, strategy, max_factor)
                    momentum_direction = torch.zeros(d_out, d_in, device=p.device, dtype=p.dtype)
                    
                    for i, (m1, m2, n1, n2) in enumerate(factorizations):
                        # Always sample fresh A matrix (with different seed for each factorization)
                        torch.manual_seed(best_seed + i + 1)
                        A = torch.randn(m1, n1, device=p.device, dtype=p.dtype)
                        
                        # B matrix: stored and updated every step_interval steps
                        b_key = f"{clean_name}_B_{i}"
                        need_new_b = (step % step_interval == 0 or b_key not in b_dict)
                        
                        if need_new_b:
                            torch.manual_seed(1000000 + step * 1000 + hash(clean_name) % 1000 + i)
                            B = torch.randn(m2, n2, device=p.device, dtype=p.dtype)
                            b_dict[b_key] = B
                        else:
                            B = b_dict[b_key]
                        
                        # EFFICIENT MOMENTUM: Store only A matrix for each factorization
                        momentum_key = f"{clean_name}_A_{i}"
                        if momentum_key not in exp_avg_m:
                            exp_avg_m[momentum_key] = torch.zeros_like(A)
                        
                        # Update momentum: m_t = β * m_{t-1} + (1-β) * A
                        exp_avg_m[momentum_key] = beta1 * exp_avg_m[momentum_key] + (1 - beta1) * A
                        
                        # Apply momentum: A_momentum ⊗ B
                        kron_momentum = torch.kron(exp_avg_m[momentum_key], B)
                        momentum_direction.add_(kron_momentum)
                    
                    # Scale by 1/kronzo_sampling_number to maintain similar magnitude
                    momentum_direction.mul_(1.0 / kronzo_sampling_number)

                # Direct movement update with momentum: θ ← θ + α * momentum_direction
                is_weight = ("bias" not in clean_name
                             and "layer_norm" not in clean_name
                             and "layernorm" not in clean_name)
                if is_weight:
                    # With weight decay: θ ← θ * (1 - α * λ) + α * momentum_direction
                    p.data = p.data * (1 - lr * weight_decay) + lr * momentum_direction
                else:
                    p.data.add_(momentum_direction, alpha=lr)
            else:                                # vectors
                z = torch.normal(0.0, 1.0,
                                 size=p.size(),
                                 device=p.device,
                                 dtype=p.dtype)

                # Momentum for vectors (standard approach)
                momentum_key = clean_name
                if momentum_key not in exp_avg_m:
                    exp_avg_m[momentum_key] = torch.zeros_like(z)
                exp_avg_m[momentum_key] = (beta1 * exp_avg_m[momentum_key] +
                                         (1 - beta1) * z)

                is_weight = ("bias" not in clean_name
                             and "layer_norm" not in clean_name
                             and "layernorm" not in clean_name)
                if is_weight:
                    p.data = p.data * (1 - lr * weight_decay) + lr * exp_avg_m[momentum_key]
                else:
                    p.data.add_(exp_avg_m[momentum_key], alpha=lr)
    else:
        # Gradient descent mode with momentum
        torch.manual_seed(best_seed)
        projected_grad = grad_or_seed

        for clean_name, p in named_parameters_to_optim:
            if p.ndim >= 2:                      # matrices
                d_out, d_in = p.shape
                
                if kronzo_sampling_number == 1:
                    # Standard single Kronecker product with efficient momentum storage
                    m1, m2, n1, n2 = choose_kron_dims(d_out, d_in, strategy, max_factor)
                    
                    # Always sample fresh A matrix
                    A = torch.randn(m1, n1, device=p.device, dtype=p.dtype)
                    
                    # B matrix: stored and updated every step_interval steps
                    b_key = f"{clean_name}_B"
                    need_new_b = (step % step_interval == 0 or b_key not in b_dict)
                    
                    if need_new_b:
                        torch.manual_seed(1000000 + step * 1000 + hash(clean_name) % 1000)
                        B = torch.randn(m2, n2, device=p.device, dtype=p.dtype)
                        b_dict[b_key] = B
                    else:
                        B = b_dict[b_key]
                    
                    # EFFICIENT MOMENTUM: Store only A matrix scaled by gradient coefficient
                    momentum_key = f"{clean_name}_A"
                    if momentum_key not in exp_avg_m:
                        exp_avg_m[momentum_key] = torch.zeros_like(A)
                    
                    # Update momentum: m_t = β * m_{t-1} + (1-β) * (c * A)
                    grad_scaled_A = projected_grad * A
                    exp_avg_m[momentum_key] = beta1 * exp_avg_m[momentum_key] + (1 - beta1) * grad_scaled_A
                    
                    # Apply momentum: A_momentum ⊗ B
                    momentum_perturbation = torch.kron(exp_avg_m[momentum_key], B)
                else:
                    # Multi-sampling: efficient momentum for each factorization
                    factorizations = choose_diverse_kron_dims(d_out, d_in, kronzo_sampling_number, strategy, max_factor)
                    momentum_perturbation = torch.zeros(d_out, d_in, device=p.device, dtype=p.dtype)
                    
                    for i, (m1, m2, n1, n2) in enumerate(factorizations):
                        # Always sample fresh A matrix (with different seed for each factorization)
                        torch.manual_seed(best_seed + i + 1)
                        A = torch.randn(m1, n1, device=p.device, dtype=p.dtype)
                        
                        # B matrix: stored and updated every step_interval steps
                        b_key = f"{clean_name}_B_{i}"
                        need_new_b = (step % step_interval == 0 or b_key not in b_dict)
                        
                        if need_new_b:
                            torch.manual_seed(1000000 + step * 1000 + hash(clean_name) % 1000 + i)
                            B = torch.randn(m2, n2, device=p.device, dtype=p.dtype)
                            b_dict[b_key] = B
                        else:
                            B = b_dict[b_key]
                        
                        # EFFICIENT MOMENTUM: Store only A matrix scaled by gradient coefficient
                        momentum_key = f"{clean_name}_A_{i}"
                        if momentum_key not in exp_avg_m:
                            exp_avg_m[momentum_key] = torch.zeros_like(A)
                        
                        # Update momentum: m_t = β * m_{t-1} + (1-β) * (c * A)
                        grad_scaled_A = projected_grad * A
                        exp_avg_m[momentum_key] = beta1 * exp_avg_m[momentum_key] + (1 - beta1) * grad_scaled_A
                        
                        # Apply momentum: A_momentum ⊗ B
                        kron_momentum = torch.kron(exp_avg_m[momentum_key], B)
                        momentum_perturbation.add_(kron_momentum)
                    
                    # Scale by 1/kronzo_sampling_number to maintain similar magnitude
                    momentum_perturbation.mul_(1.0 / kronzo_sampling_number)

                # Gradient descent update with momentum
                is_weight = ("bias" not in clean_name
                             and "layer_norm" not in clean_name
                             and "layernorm" not in clean_name)
                if is_weight:
                    p.data.sub_(lr * (momentum_perturbation + weight_decay * p.data))
                else:
                    p.data.sub_(lr * momentum_perturbation)
            else:                                # vectors
                z = torch.normal(0.0, 1.0,
                                 size=p.size(),
                                 device=p.device,
                                 dtype=p.dtype)

                # Momentum for vectors (standard approach)
                momentum_key = clean_name
                if momentum_key not in exp_avg_m:
                    exp_avg_m[momentum_key] = torch.zeros_like(z)
                exp_avg_m[momentum_key] = (beta1 * exp_avg_m[momentum_key] +
                                         (1 - beta1) * projected_grad * z)

                is_weight = ("bias" not in clean_name
                             and "layer_norm" not in clean_name
                             and "layernorm" not in clean_name)
                if is_weight:
                    p.data.sub_(lr * (exp_avg_m[momentum_key] + weight_decay * p.data))
                else:
                    p.data.sub_(lr * exp_avg_m[momentum_key]) 