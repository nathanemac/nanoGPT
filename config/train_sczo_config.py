# =============================================================================
# CONFIG FOR SCZO (SECANT-CONSTRAINED ZERO-ORDER) OPTIMIZATION
# =============================================================================
# This config file contains all parameters for the novel SCZO training method.
# SCZO uses secant constraints to update gradient estimates without backward passes.

import torch

# =============================================================================
# TRAINING METHOD SELECTION
# =============================================================================
train_method = 'sczo'  # Secant-Constrained Zero-Order optimization

# =============================================================================
# DATASET AND MODEL SELECTION
# =============================================================================
dataset = 'shakespeare'  # OPTIONS: 'shakespeare', 'openwebtext', 'gpt2'
init_from = 'scratch'    # OPTIONS: 'scratch', 'resume', 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'

# =============================================================================
# OUTPUT AND LOGGING SETTINGS
# =============================================================================
out_dir = 'out-sczo'            # Output directory for checkpoints and logs
eval_interval = 500             # How often to evaluate on validation set
log_interval = 1                # How often to log training progress
eval_iters = 50                 # Number of iterations for evaluation
eval_only = False               # If True, only run evaluation and exit
always_save_checkpoint = True   # If True, always save checkpoint after eval

# Weights & Biases logging
wandb_log = False               # Enable wandb logging
wandb_project = 'nanogpt'       # Wandb project name
wandb_run_name = 'sczo-run'     # Wandb run name

# =============================================================================
# DATA CONFIGURATION
# =============================================================================
gradient_accumulation_steps = 1 # Simulate larger batch sizes (SCZO works best with fresh batches)
batch_size = 64                 # Micro-batch size per GPU
block_size = 256               # Context length (sequence length)

# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================
# Small model (for testing/shakespeare)
n_layer = 6      # Number of transformer layers
n_head = 6       # Number of attention heads
n_embd = 384      # Embedding dimension

dropout = 0.0     # Dropout rate
bias = False      # Use bias in LayerNorm and Linear layers

# =============================================================================
# SCZO HYPERPARAMETERS
# =============================================================================
# NOTE: When decay_lr=True, sczo_alpha will automatically follow the learning_rate schedule
# The value below is used as the base/initial value, just like learning_rate
sczo_alpha = 2e-4            # Step size (will follow learning rate if decay_lr=True)
sczo_beta = 0.9              # EMA factor for gradient estimate updates
sczo_eps = 1e-8              # Damping parameter for numerical stability  
sczo_bootstrap_interval = 20 # Frequency of ZO gradient bootstrap (every T steps)
sczo_multi_batch = 8         # Number of batches for multi-signal gradient estimation

# Multi-secant memory parameters (temporal constraints)
sczo_multi_secant_m = 0          # DISABLED: Set to 0 to test spatial-only SCZO first
sczo_multi_secant_rho = 0.9      # Exponential decay weight for historical constraints  
sczo_multi_secant_eps_reg = 1e-6 # Tikhonov regularization for LS solver
sczo_multi_secant_max_cond = 2e4  # Max condition number before restart
sczo_multi_secant_min_signal = 1e-10  # Min signal strength to keep constraint

# =============================================================================
# PRECONDITIONING PARAMETERS  
# =============================================================================
# Preconditioning type: 'identity', 'layerwise', 'diagonal', 'layerwise_diagonal'
sczo_precond_type = 'diagonal'           # Start with diagonal (option 2 from analysis)
sczo_precond_layerwise_ema = 0.99        # EMA factor for layer-wise scaling τ(l)
sczo_precond_diagonal_ema = 0.99         # EMA factor for diagonal preconditioning v(l)  
sczo_precond_diagonal_eps = 1e-8         # Epsilon for diagonal stability
sczo_precond_unit_wise = True            # Use unit-wise diagonal (memory efficient)

# =============================================================================
# ZERO-ORDER OPTIMIZATION PARAMETERS (for bootstrap)
# =============================================================================
zo_eps = 1e-3               # Perturbation size for ZO gradient estimation during bootstrap
directional_q = 20               # Number of directions for ZO bootstrap

# =============================================================================
# KRONECKER PRODUCT PARAMETERS (for bootstrap)
# =============================================================================
kron_strategy = 'approx_square'  # Kronecker factorization strategy for bootstrap
kron_max_factor = 32             # Maximum factor size for Kronecker factorization
kronzo_sampling_number = 1       # Number of Kronecker products to sample
step_interval = 10               # Interval for updating B matrices during bootstrap

# =============================================================================
# OPTIMIZER SETTINGS
# =============================================================================
learning_rate = 1e-3      # Base learning rate - when decay_lr=True, sczo_alpha will follow this schedule
max_iters = 5000         # Total number of training iterations
weight_decay = 1e-1       # L2 regularization strength

# Standard optimizer parameters (kept for compatibility)
beta1 = 0.9               # Maps to sczo_beta (kept for compatibility)
beta2 = 0.95              # Not used by SCZO
grad_clip = 1.0           # Gradient clipping (applied to gradient estimates)

# =============================================================================
# MOMENTUM SETTINGS (SCZO doesn't use traditional momentum)
# =============================================================================
use_momentum = False      # SCZO uses secant updates instead of momentum

# =============================================================================
# LEARNING RATE SCHEDULE
# =============================================================================
decay_lr = True           # Whether to decay learning rate (applies to sczo_alpha)
warmup_iters = 500        # Number of warmup iterations
lr_decay_iters = 5000     # Should be ~= max_iters for cosine decay
min_lr = 1e-4             # Minimum learning rate

# =============================================================================
# SYSTEM SETTINGS
# =============================================================================
device = 'cuda'           # OPTIONS: 'cuda', 'cpu', 'mps'
dtype = 'bfloat16'        # OPTIONS: 'float32', 'bfloat16', 'float16'
compile = True            # Use PyTorch 2.0 compilation
backend = 'nccl'          # DDP backend

# =============================================================================
# COMPATIBILITY PARAMETERS (not used by SCZO but needed for config compatibility)
# =============================================================================
# These parameters are not used by SCZO but are needed for config compatibility
rank_r = 4                       # Not used by SCZO
rank_adaptive = False            # Not used by SCZO
min_rank = 1                     # Not used by SCZO
max_rank = 16                    # Not used by SCZO
svd_tau = 0.6                    # Not used by SCZO
svd_max_rank = 16                # Not used by SCZO
use_full_svd = False             # Not used by SCZO
momentum_beta = 0.9              # Not used by SCZO (uses sczo_beta instead)
dimezo_direct_movement = False   # Not used by SCZO
dikronzo_direct_movement = False # Not used by SCZO
use_adaptive_eps = False         # Not used by SCZO (could be added later)
adaptive_eps_window = 20         # Not used by SCZO
adaptive_eps_lr_coupling = 0.5   # Not used by SCZO
adaptive_eps_success_high = 0.7  # Not used by SCZO
adaptive_eps_success_low = 0.3   # Not used by SCZO
loss_history_size = 10           # Not used by SCZO
use_baseline_history = False     # Not used by SCZO
use_lowrank_factorization = False # Not used by SCZO
rank_kronzo = 4                  # Not used by SCZO

# =============================================================================
# SCZO METHOD CONFIGURATION VARIANTS
# =============================================================================
# Uncomment to test different SCZO configurations

# # SHAKESPEARE SMALL SCALE TESTING
# if dataset == 'shakespeare':
#     n_layer = 6
#     n_head = 6  
#     n_embd = 384
#     max_iters = 5000
#     sczo_alpha = 3e-3
#     sczo_bootstrap_interval = 10
#     batch_size = 64
#     block_size = 256

# # LARGE SCALE OPENWEBTEXT  
# if dataset == 'openwebtext':
#     max_iters = 100000
#     sczo_alpha = 5e-4  # Smaller step size for large scale
#     sczo_bootstrap_interval = 50  # Less frequent bootstrap for efficiency
#     gradient_accumulation_steps = 5  # Larger effective batch size

# =============================================================================
# SCZO ALGORITHM NOTES
# =============================================================================
"""
SCZO Algorithm Summary:
1. Maintain gradient estimates g^(l) per layer l
2. Each iteration:
   a. Sample fresh mini-batch B_k
   b. Compute loss_before = f(θ, B_k)
   c. Apply step: θ ← θ - α * g
   d. Compute loss_after = f(θ, B_k)  
   e. Update gradients via secant constraint: g^T s_k = y_k
3. Periodic ZO bootstrap every T_bootstrap steps

Memory Usage: ~2-2.5 floats per parameter (model + gradients + optional preconditioning)
Computation: Only 2 forward passes per iteration (vs 3*directional_q for KronZO)

Key Benefits:
- Theoretical foundation via quasi-Newton secant conditions
- Memory efficient gradient estimates  
- Fresh batch every iteration prevents overfitting
- Only forward passes required
""" 