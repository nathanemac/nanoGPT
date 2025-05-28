# =============================================================================
# COMPREHENSIVE CONFIG FOR KRONZO-BASED OPTIMIZATION METHODS
# =============================================================================
# This config file contains all parameters for KronZO and DiKronZO training.
# Modify the parameters below to customize your training setup.

import torch

# =============================================================================
# TRAINING METHOD SELECTION
# =============================================================================
train_method = 'dikronzo'  # OPTIONS: 'kronzo', 'dikronzo'

# =============================================================================
# DATASET AND MODEL SELECTION
# =============================================================================
dataset = 'shakespeare'  # OPTIONS: 'shakespeare', 'openwebtext', 'gpt2'
init_from = 'scratch'    # OPTIONS: 'scratch', 'resume', 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'

# =============================================================================
# OUTPUT AND LOGGING SETTINGS
# =============================================================================
out_dir = 'out-kronzo'          # Output directory for checkpoints and logs
eval_interval = 1000            # How often to evaluate on validation set
log_interval = 1                # How often to log training progress
eval_iters = 50                 # Number of iterations for evaluation
eval_only = False               # If True, only run evaluation and exit
always_save_checkpoint = True   # If True, always save checkpoint after eval

# Weights & Biases logging
wandb_log = False               # Enable wandb logging
wandb_project = 'nanogpt'       # Wandb project name
wandb_run_name = 'kronzo-run'   # Wandb run name

# =============================================================================
# DATA CONFIGURATION
# =============================================================================
gradient_accumulation_steps = 1  # Simulate larger batch sizes (must be divisible by number of GPUs)
batch_size = 64                  # Micro-batch size per GPU
block_size = 256                 # Context length (sequence length)

# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================
# Small model (for shakespeare/testing)
n_layer = 6      # Number of transformer layers
n_head = 6       # Number of attention heads
n_embd = 384     # Embedding dimension

# Medium model (uncomment for larger experiments)
# n_layer = 12
# n_head = 12
# n_embd = 768

# Large model (uncomment for full-scale experiments)
# n_layer = 24
# n_head = 16
# n_embd = 1024

dropout = 0.1    # Dropout rate (0.0 for pretraining, 0.1+ for finetuning)
bias = False     # Use bias in LayerNorm and Linear layers

# =============================================================================
# OPTIMIZER SETTINGS
# =============================================================================
learning_rate = 1e-3  # Maximum learning rate
max_iters = 5000      # Total number of training iterations
weight_decay = 1e-1   # L2 regularization strength

# Standard optimizer parameters (used as fallback)
beta1 = 0.9          # Adam beta1 (momentum coefficient)
beta2 = 0.95         # Adam beta2 (RMSprop coefficient)
grad_clip = 1.0      # Clip gradients at this value (0.0 to disable)

# =============================================================================
# ZERO-ORDER OPTIMIZATION PARAMETERS
# =============================================================================
zo_eps = 1e-3        # Perturbation size for gradient estimation
                     # OPTIONS: 1e-4 (small), 1e-3 (default), 1e-2 (large)

# =============================================================================
# KRONECKER PRODUCT PARAMETERS
# =============================================================================
kron_strategy = 'approx_square'  # Kronecker factorization strategy
                                 # OPTIONS: 'approx_square' (factors close to sqrt),
                                 #          'fixed_factor' (factors ≤ max_factor),
                                 #          'power2' (largest power-of-2 divisors)

kron_max_factor = 32             # Maximum factor size for 'fixed_factor' strategy
                                 # OPTIONS: 16 (small), 32 (default), 64 (large)

step_interval = 50               # Interval for updating B matrices (every ν steps)
                                 # OPTIONS: 20 (frequent updates), 50 (default), 100 (infrequent)

# =============================================================================
# MOMENTUM SETTINGS (for KronZO with momentum)
# =============================================================================
use_momentum = True # Enable momentum for KronZO
                     # OPTIONS: True (KronZO-M), False (standard KronZO)
momentum_beta = 0.9  # Momentum coefficient
                     # OPTIONS: 0.9 (default), 0.95 (stronger momentum), 0.8 (weaker)

# =============================================================================
# DIRECTIONAL SELECTION PARAMETERS (for DiKronZO)
# =============================================================================
directional_q = 50           # Number of directions to try in DiKronZO
                             

# =============================================================================
# ADAPTIVE ZO_EPS SETTINGS
# =============================================================================
use_adaptive_eps = False        # Enable adaptive perturbation size
                                # OPTIONS: True (adaptive), False (fixed)

adaptive_eps_window = 20        # Window size for tracking success rate
                                # OPTIONS: 10 (responsive), 20 (default), 50 (stable)

adaptive_eps_lr_coupling = 0.5  # Coupling strength between eps and learning rate
                                 # OPTIONS: 0.0 (no coupling), 0.5 (default), 1.0 (full coupling)

adaptive_eps_success_high = 0.7  # Success rate threshold for increasing eps
                                 # OPTIONS: 0.6-0.8 (typical range)

adaptive_eps_success_low = 0.3   # Success rate threshold for decreasing eps
                                 # OPTIONS: 0.2-0.4 (typical range)

# =============================================================================
# ZERO-ORDER QUERY BUDGET PARAMETERS
# =============================================================================
zo_q = 1             # Number of gradient estimates to average (for averaging methods)
                     # OPTIONS: 1 (standard), 2-5 (averaging multiple estimates)
                     # NOTE: Not used by KronZO directional selection, but needed for compatibility

# =============================================================================
# DIMEZO SPECIFIC PARAMETERS (for compatibility)
# =============================================================================
dimezo_direct_movement = False  # Movement strategy for DiMeZO (not used by KronZO)
                                # OPTIONS: True (direct movement), False (gradient estimation)

# =============================================================================
# LOZO-SPECIFIC PARAMETERS (for compatibility)
# =============================================================================
# These parameters are not used by KronZO methods but are needed for config compatibility
rank_r = 4           # Fixed rank for U and V matrices (not used by KronZO)
rank_adaptive = False        # Enable adaptive rank scheduling (not used by KronZO)
min_rank = 1                 # Minimum rank for adaptive scheduling (not used by KronZO)
max_rank = 16                # Maximum rank for adaptive scheduling (not used by KronZO)
rank_strategy = 'linear'     # Rank scheduling strategy (not used by KronZO)

# =============================================================================
# SVD-LOZO SPECIFIC PARAMETERS (for compatibility)
# =============================================================================
# These parameters are not used by KronZO methods but are needed for config compatibility
svd_tau = 0.6               # Threshold for adaptive rank selection (not used by KronZO)
svd_max_rank = 16           # Maximum rank for randomized SVD (not used by KronZO)
use_full_svd = False        # Use full SVD vs randomized SVD (not used by KronZO)

# =============================================================================
# LEARNING RATE SCHEDULE
# =============================================================================
decay_lr = True      # Whether to decay learning rate
warmup_iters = 500   # Number of warmup iterations
lr_decay_iters = 5000  # Should be ~= max_iters for cosine decay
min_lr = 1e-4        # Minimum learning rate (should be ~= learning_rate/10)

# =============================================================================
# SYSTEM SETTINGS
# =============================================================================
device = 'cuda'      # OPTIONS: 'cuda', 'cpu', 'mps' (for Apple Silicon)
dtype = 'bfloat16'   # OPTIONS: 'float32', 'bfloat16', 'float16'
compile = True       # Use PyTorch 2.0 compilation (set False for CPU)

# DDP settings (for multi-GPU training)
backend = 'nccl'     # OPTIONS: 'nccl', 'gloo'

# =============================================================================
# DATASET-SPECIFIC CONFIGURATIONS
# =============================================================================
# Uncomment the appropriate section based on your dataset choice

# # SHAKESPEARE DATASET (small, fast training)
# if dataset == 'shakespeare':
#     batch_size = 64
#     block_size = 256
#     max_iters = 5000
#     eval_interval = 1000
#     n_layer, n_head, n_embd = 6, 6, 384
#     learning_rate = 1e-3
#     zo_eps = 1e-3
#     kron_max_factor = 32

# # OPENWEBTEXT DATASET (medium scale)
# if dataset == 'openwebtext':
#     batch_size = 12
#     block_size = 1024
#     max_iters = 600000
#     eval_interval = 2000
#     gradient_accumulation_steps = 40
#     n_layer, n_head, n_embd = 12, 12, 768
#     learning_rate = 6e-4
#     zo_eps = 1e-3
#     kron_max_factor = 64

# # GPT2 DATASET (large scale)
# if dataset == 'gpt2':
#     batch_size = 12
#     block_size = 1024
#     max_iters = 600000
#     eval_interval = 2000
#     gradient_accumulation_steps = 40
#     n_layer, n_head, n_embd = 12, 12, 768
#     learning_rate = 6e-4
#     zo_eps = 1e-3
#     kron_max_factor = 64

# =============================================================================
# KRONECKER FACTORIZATION EXAMPLES
# =============================================================================
# For a matrix W ∈ ℝ^(d_out × d_in), KronZO uses perturbations A ⊗ B where:
# - A ∈ ℝ^(m1 × n1), B ∈ ℝ^(m2 × n2)
# - m1 * m2 = d_out, n1 * n2 = d_in
# - Storage: m1*n1 + m2*n2 instead of d_out*d_in
#
# Strategy examples for W ∈ ℝ^(768 × 768):
# - 'approx_square': A(28×28) ⊗ B(27×27) ≈ sqrt factorization
# - 'fixed_factor': A(32×32) ⊗ B(24×24) with max_factor=32
# - 'power2': A(32×32) ⊗ B(24×24) using largest power-of-2 divisors 