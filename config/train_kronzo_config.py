# =============================================================================
# COMPREHENSIVE CONFIG FOR KRONZO-BASED OPTIMIZATION METHODS
# =============================================================================
# This config file contains all parameters for KronZO, DiKronZO, and Improved Directional KronZO training.
# Modify the parameters below to customize your training setup.

import torch

# =============================================================================
# TRAINING METHOD SELECTION
# =============================================================================
train_method = 'improved_kronzo'  # OPTIONS: 'kronzo', 'dikronzo', 'improved_kronzo', 'new_improved_kronzo'
                                  # 'kronzo': Standard KronZO with single direction
                                  # 'dikronzo': Directional KronZO with best direction selection
                                  # 'improved_kronzo': Advanced directional KronZO with conservative updates
                                  # 'new_improved_kronzo': Baseline history management (less restrictive)

# =============================================================================
# DATASET AND MODEL SELECTION
# =============================================================================
dataset = 'openwebtext'  # OPTIONS: 'shakespeare', 'openwebtext', 'gpt2'
init_from = 'scratch'    # OPTIONS: 'scratch', 'resume', 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'

# =============================================================================
# OUTPUT AND LOGGING SETTINGS
# =============================================================================
out_dir = 'out-improved-kronzo'   # Output directory for checkpoints and logs
eval_interval = 500            # How often to evaluate on validation set
log_interval = 1                 # How often to log training progress
eval_iters = 50                  # Number of iterations for evaluation
eval_only = False                # If True, only run evaluation and exit
always_save_checkpoint = True    # If True, always save checkpoint after eval

# Weights & Biases logging
wandb_log = False                # Enable wandb logging
wandb_project = 'nanogpt'        # Wandb project name
wandb_run_name = 'improved-kronzo-run'  # Wandb run name

# =============================================================================
# DATA CONFIGURATION
# =============================================================================
gradient_accumulation_steps = 1  # Simulate larger batch sizes
batch_size = 32                 # Micro-batch size per GPU
block_size = 1024                # Context length (sequence length)

# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================
# Small model (for shakespeare/testing)
n_layer = 12      # Number of transformer layers
n_head = 12       # Number of attention heads
n_embd = 768     # Embedding dimension

dropout = 0.0    # Dropout rate
bias = False     # Use bias in LayerNorm and Linear layers

# =============================================================================
# OPTIMIZER SETTINGS
# =============================================================================
learning_rate = 3e-4  # 
max_iters = 50000      # Total number of training iterations (LOW FOR BENCHMARKING)
weight_decay = 1e-1   # L2 regularization strength

# Standard optimizer parameters (kept for compatibility)
beta1 = 0.9          # Adam beta1 (momentum coefficient)
beta2 = 0.95         # Adam beta2 (RMSprop coefficient)
grad_clip = 1.0      # Clip gradients at this value (0.0 to disable)

# =============================================================================
# ZERO-ORDER OPTIMIZATION PARAMETERS
# =============================================================================
zo_eps = 3e-4        # Perturbation size for gradient estimation (increased for better signal)

# =============================================================================
# KRONECKER PRODUCT PARAMETERS
# =============================================================================
kron_strategy = 'approx_square'  # Kronecker factorization strategy
                                 # OPTIONS: 'approx_square' (factors close to sqrt),
                                 #          'fixed_factor' (factors ≤ max_factor),
                                 #          'power2' (largest power-of-2 divisors)

kron_max_factor = 64             # Maximum factor size for 'fixed_factor' strategy
                                # Also used as constraint in multi-sampling when kron_strategy='fixed_factor'

kronzo_sampling_number = 1       # Number of Kronecker products to sample and sum
                                # OPTIONS: 1 (standard KronZO using kron_strategy), 
                                #          2-5 (multi-sampling with overlapping prime-factor strategy)

rank_kronzo = 4                # Rank for low-rank factorization of A and B matrices
                                # A = U_A V_A^T, B = U_B V_B^T where U,V have rank_kronzo columns
                                # Controls rank(A⊗B) ≤ rank_kronzo^2, enabling memory-efficient KronZO
                                # MEMORY SAVINGS: O((m1+n1+m2+n2)*rank_kronzo) vs O(m1*n1+m2*n2)
                                

step_interval = 5               # Interval for updating B matrices (every ν steps)
                                

# =============================================================================
# MOMENTUM SETTINGS
# =============================================================================
use_momentum = False              # Enable momentum for improved KronZO
momentum_beta = 0.9             # Momentum coefficient

# =============================================================================
# DIRECTIONAL SELECTION PARAMETERS
# =============================================================================
directional_q = 33                # Number of directions to evaluate
                                # Standard KronZO: Not used (single direction)
                                # DiKronZO: 5-15 directions typical
                                # Improved KronZO: 20-50 directions for thorough evaluation
                                # NOTE: Higher values = more expensive but potentially better updates
                                # Cost: 3*directional_q function evaluations per step for improved_kronzo

# CONSERVATIVE UPDATE PARAMETERS (for improved_kronzo and new_improved_kronzo)  
loss_history_size = 10            # Number of history entries to track
                                # improved_kronzo: tracks successful update losses
                                # new_improved_kronzo: tracks baseline losses from recent iterations
                                # Controls conservative update acceptance

# DEBUGGING FLAG FOR NEW_IMPROVED_KRONZO CONSERVATIVE UPDATE STRATEGY
use_baseline_history = False       # For new_improved_kronzo only:
                                # True: Track baseline losses regardless of acceptance (new approach)
                                #       - History: f(θ_k; batch_k) from recent iterations
                                #       - Accept if: candidate_loss ≤ max{recent baseline losses}
                                #       - Updates history ALWAYS after each decision
                                # False: Track successful update losses only (like improved_kronzo)
                                #        - History: Only losses from accepted updates
                                #        - Accept if: candidate_loss ≤ max{successful update losses}
                                #        - Updates history ONLY when update is accepted
                                
# DEBUGGING FLAG FOR LOW-RANK FACTORIZATION vs TRADITIONAL MATRIX SAMPLING
use_lowrank_factorization = False   # For new_improved_kronzo only:
                                # True: Use low-rank factorization A = U_A V_A^T, B = U_B V_B^T (new approach)
                                #       - Memory efficient: O((m1+n1+m2+n2)*rank_kronzo) storage
                                #       - Rank control: rank(A⊗B) ≤ rank_kronzo^2
                                # False: Use traditional full matrix sampling A, B (like improved_kronzo)  
                                #        - Standard storage: O(m1*n1+m2*n2) for full matrices
                                #        - Direct matrix generation without factorization

# =============================================================================
# ADAPTIVE ZO_EPS SETTINGS
# =============================================================================
use_adaptive_eps = False        # Disable for improved directional (can be added later)
adaptive_eps_window = 20
adaptive_eps_lr_coupling = 0.5
adaptive_eps_success_high = 0.7
adaptive_eps_success_low = 0.3

# =============================================================================
# LOZO-SPECIFIC PARAMETERS (for compatibility)
# =============================================================================
# These parameters are not used by KronZO methods but are needed for config compatibility
rank_r = 4                      # Not used by KronZO
rank_adaptive = False           # Not used by KronZO
min_rank = 1                    # Not used by KronZO
max_rank = 16                   # Not used by KronZO
svd_tau = 0.6                   # Not used by KronZO
svd_max_rank = 16               # Not used by KronZO
use_full_svd = False            # Not used by KronZO

# =============================================================================
# LEARNING RATE SCHEDULE
# =============================================================================
decay_lr = True      # Whether to decay learning rate
warmup_iters = 30    # 
lr_decay_iters = 300  # Should be ~= max_iters for cosine decay
min_lr = 1e-5        # Higher minimum learning rate (was 1e-4)


# =============================================================================
# SYSTEM SETTINGS
# =============================================================================
device = 'cuda'      # OPTIONS: 'cuda', 'cpu', 'mps'
dtype = 'bfloat16'   # OPTIONS: 'float32', 'bfloat16', 'float16'
compile = True       # Use PyTorch 2.0 compilation
backend = 'nccl'     # DDP backend

# =============================================================================
# TRAINING METHOD SPECIFIC CONFIGURATIONS
# =============================================================================
# Uncomment and modify the appropriate section based on your chosen train_method

# # STANDARD KRONZO CONFIGURATION
# if train_method == 'kronzo':
#     directional_q = 1  # Not used, included for compatibility
#     out_dir = 'out-kronzo'
#     wandb_run_name = 'kronzo-run'
#     learning_rate = 1e-3
#     zo_eps = 1e-3
#     directional_q = 10  # Not used by kronzo
#     loss_history_size = 10  # Not used by kronzo

# # DIRECTIONAL KRONZO (DIKRONZO) CONFIGURATION  
# if train_method == 'dikronzo':
#     directional_q = 10  # Number of directions to try
#     out_dir = 'out-dikronzo'
#     wandb_run_name = 'dikronzo-run'
#     learning_rate = 1e-3
#     zo_eps = 1e-3
#     loss_history_size = 10  # Not used by dikronzo

# # IMPROVED DIRECTIONAL KRONZO CONFIGURATION (DEFAULT)
# if train_method == 'improved_kronzo':
#     directional_q = 33  # More thorough direction evaluation
#     out_dir = 'out-improved-kronzo'
#     wandb_run_name = 'improved-kronzo-run'
#     learning_rate = 3e-4  # Slightly higher for meaningful steps
#     zo_eps = 5e-4  # Increased for better signal
#     loss_history_size = 10  # Conservative update history

# # NEW IMPROVED DIRECTIONAL KRONZO CONFIGURATION (BASELINE HISTORY)
# if train_method == 'new_improved_kronzo':
#     directional_q = 33  # More thorough direction evaluation
#     out_dir = 'out-new-improved-kronzo'
#     wandb_run_name = 'new-improved-kronzo-run'
#     learning_rate = 3e-4  # Slightly higher for meaningful steps
#     zo_eps = 5e-4  # Increased for better signal
#     loss_history_size = 10  # Baseline history size (less restrictive than improved_kronzo)

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
#     directional_q = 20 if train_method == 'improved_kronzo' else 10

# # OPENWEBTEXT DATASET (medium scale)
# if dataset == 'openwebtext':
#     batch_size = 32
#     block_size = 1024
#     max_iters = 50000
#     eval_interval = 500
#     gradient_accumulation_steps = 1
#     n_layer, n_head, n_embd = 12, 12, 768
#     learning_rate = 3e-4 if train_method == 'improved_kronzo' else 6e-4
#     zo_eps = 5e-4 if train_method == 'improved_kronzo' else 1e-3
#     kron_max_factor = 64
#     directional_q = 33 if train_method == 'improved_kronzo' else 10

# =============================================================================
# ALGORITHM NOTES
# =============================================================================
# 
# STANDARD KRONZO ('kronzo'):
# - Uses single Kronecker perturbation A ⊗ B
# - Estimates gradient: c = [f(θ + ε*Z) - f(θ - ε*Z)] / (2ε)
# - Updates: θ ← θ - lr*c*Z
# - Fast, memory efficient
#
# DIRECTIONAL KRONZO ('dikronzo'):
# - Evaluates multiple directions, picks best
# - For each direction: evaluate f(θ + ε*Z_i)
# - Choose best direction, then estimate gradient on it
# - More expensive but potentially better directions
#
# IMPROVED DIRECTIONAL KRONZO ('improved_kronzo'):
# - Evaluates actual update candidates, not just directions
# - For each direction: compute candidate update θ - lr*c_i*Z_i
# - Evaluate f(θ - lr*c_i*Z_i) for each candidate
# - Conservative acceptance: only accept if candidate beats historical performance
# - Most expensive but most sophisticated update selection
# - Cost: 3*directional_q function evaluations per step
#
# NEW IMPROVED DIRECTIONAL KRONZO ('new_improved_kronzo'):
# - Same as improved_kronzo but with baseline history management
# - History: stores baseline losses f(θ_k; batch_k) from recent iterations
# - Acceptance: candidate_loss ≤ max{recent baseline losses} (less restrictive)
# - Updates history after each decision, regardless of acceptance
# - LOW-RANK FACTORIZATION: A = U_A V_A^T, B = U_B V_B^T for memory efficiency
# - Storage: O((m1+n1+m2+n2)*rank_kronzo) vs O(m1*n1+m2*n2) for full matrices
# - Rank control: rank(A⊗B) ≤ rank_kronzo^2
# - Cost: 3*directional_q function evaluations per step + 1 baseline evaluation (overhead)
#
# Memory efficiency comparison:
# - Full perturbation: O(d_out * d_in) storage per parameter
# - Standard Kronecker: O(m1*n1 + m2*n2) storage, where m1*m2=d_out, n1*n2=d_in
# - Low-rank Kronecker: O((m1+n1+m2+n2)*rank_kronzo) storage
# - Typical savings: ~10x-100x memory reduction vs standard Kronecker
# - Ultra savings: ~1000x-10000x memory reduction vs full perturbation 