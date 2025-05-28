# =============================================================================
# COMPREHENSIVE CONFIG FOR LOZO-BASED OPTIMIZATION METHODS
# =============================================================================
# This config file contains all parameters for LoZO, LoZO-M, SVD-LoZO, and DiLoZO training.
# Modify the parameters below to customize your training setup.

import torch

# =============================================================================
# TRAINING METHOD SELECTION
# =============================================================================
train_method = 'dilozo'  # OPTIONS: 'lozo', 'lozom', 'svdlozo', 'dilozo'

# =============================================================================
# DATASET AND MODEL SELECTION
# =============================================================================
dataset = 'shakespeare'  # OPTIONS: 'shakespeare', 'openwebtext'
init_from = 'scratch'    # OPTIONS: 'scratch', 'resume', 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'

# =============================================================================
# OUTPUT AND LOGGING SETTINGS
# =============================================================================
out_dir = 'out-lozo'            # Output directory for checkpoints and logs
eval_interval = 500            # How often to evaluate on validation set
log_interval = 1                # How often to log training progress
eval_iters = 50                 # Number of iterations for evaluation
eval_only = False               # If True, only run evaluation and exit
always_save_checkpoint = True   # If True, always save checkpoint after eval

# Weights & Biases logging
wandb_log = False               # Enable wandb logging
wandb_project = 'nanogpt'       # Wandb project name
wandb_run_name = 'lozo-run'     # Wandb run name

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

dropout = 0.0    # Dropout rate (0.0 for pretraining, 0.1+ for finetuning)
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
# grad_clip = 1.0      # Clip gradients at this value (0.0 to disable)

# =============================================================================
# ZERO-ORDER OPTIMIZATION PARAMETERS
# =============================================================================
zo_eps = 1e-3        # Perturbation size for gradient estimation
                     # OPTIONS: 1e-4 (small), 1e-3 (default), 1e-2 (large)

zo_q = 1             # Number of gradient estimates to average (for LoZO/SVD-LoZO)
                     # OPTIONS: 1 (standard), 2-5 (averaging multiple estimates)
                     # NOTE: Higher values = more accurate but slower

# =============================================================================
# LOW-RANK PARAMETERS
# =============================================================================
rank_r = 4           # Fixed rank for U and V matrices
                     # OPTIONS: 1-4 (small), 8-16 (medium), 32+ (large)

step_interval = 50   # Interval for updating V matrices (every ν steps)
                     # OPTIONS: 20 (frequent updates), 50 (default), 100 (infrequent)

# =============================================================================
# RANK-ADAPTIVE SETTINGS
# =============================================================================
rank_adaptive = True        # Enable adaptive rank scheduling
                             # OPTIONS: True (adaptive), False (fixed rank_r)

min_rank = 2                 # Minimum rank for adaptive scheduling
                             # OPTIONS: 1 (start small), 2-4 (conservative start)

max_rank = 16                # Maximum rank for adaptive scheduling
                             # OPTIONS: 8-16 (medium), 32-64 (large)

rank_strategy = 'linear'     # Rank scheduling strategy
                             # OPTIONS: 'linear' (steady increase), 
                             #          'exponential' (slow start, fast later),
                             #          'cosine' (smooth S-curve)

# =============================================================================
# MOMENTUM SETTINGS (for LoZO-M)
# =============================================================================
use_momentum = True # Enable momentum for LoZO-M
                     # OPTIONS: True (LoZO-M), False (standard LoZO)
momentum_beta = 0.9  # Momentum coefficient
                     # OPTIONS: 0.9 (default), 0.95 (stronger momentum), 0.8 (weaker)

# =============================================================================
# DIRECTIONAL SELECTION PARAMETERS (for DiLoZO)
# =============================================================================
directional_q = 50           # Number of directions to try in DiLoZO
                             # OPTIONS: 5 (fast), 10 (default), 20 (thorough)

# =============================================================================
# DIMEZO SPECIFIC PARAMETERS
# =============================================================================
dimezo_direct_movement = False  # Movement strategy for DiMeZO
                                # OPTIONS: True (direct movement), False (gradient estimation)

# =============================================================================
# SVD-LOZO SPECIFIC PARAMETERS
# =============================================================================
svd_tau = 0.6               # Threshold for adaptive rank selection (keep σ > τ * σ_max)
                            # OPTIONS: 0.1-0.3 (aggressive), 0.6 (default), 0.8-0.9 (conservative)

svd_max_rank = 16           # Maximum rank for randomized SVD
                            # OPTIONS: 8-16 (medium), 32-64 (large)

use_full_svd = False        # Use full SVD vs randomized SVD
                            # OPTIONS: False (randomized, faster), True (full, more accurate)

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
# KRONZO SPECIFIC PARAMETERS
# =============================================================================
kron_strategy = 'approx_square'  # Kronecker factorization strategy
                                 # OPTIONS: 'approx_square' (factors close to sqrt),
                                 #          'fixed_factor' (factors ≤ max_factor),
                                 #          'power2' (largest power-of-2 divisors)

kron_max_factor = 32             # Maximum factor size for 'fixed_factor' strategy
                                 # OPTIONS: 16 (small), 32 (default), 64 (large)

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
# NUMERICAL STABILITY SETTINGS
# =============================================================================
# New parameters for numerical stability
use_zo_clipping = False           # Enable/disable ZO-specific gradient and parameter clipping
zo_grad_clip = 1.0              # Gradient clipping for ZO methods (applied to projected gradients)
zo_param_clip = 10.0            # Parameter clipping for ZO methods (applied to parameter updates)
zo_loss_threshold = 100.0       # Loss threshold for detecting instability
zo_recovery_lr_factor = 0.1     # Learning rate reduction factor when recovering from instability
zo_max_recovery_attempts = 3    # Maximum number of recovery attempts before stopping

# Debug and logging control
zo_verbose_debug = False        # Enable/disable verbose ZO debug prints (f+, f-, grad_coeff)
zo_verbose_clipping = False     # Enable/disable clipping event logging

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
#     rank_r = 4

# OPENWEBTEXT DATASET (medium scale) - Updated for stability
# if dataset == 'openwebtext':
#     batch_size = 12
#     block_size = 1024
#     max_iters = 50000
#     eval_interval = 2000
#     gradient_accumulation_steps = 2
#     n_layer, n_head, n_embd = 12, 12, 768
#     learning_rate = 6e-4  # Reduced from 1e-3
#     zo_eps = 1e-3         # Increased from 1e-4
#     rank_r = 8            # Increased from 1
#     min_rank = 4          # Increased from 1
#     max_rank = 16
#     directional_q = 50    
#     use_zo_clipping = False
#     use_adaptive_eps = False
#     rank_adaptive = True

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
#     rank_r = 16 