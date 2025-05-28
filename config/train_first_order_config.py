# =============================================================================
# COMPREHENSIVE CONFIG FOR FIRST-ORDER OPTIMIZATION METHODS (Adam, SGD)
# =============================================================================
# This config file contains all parameters for training with first-order methods.
# Modify the parameters below to customize your training setup.

import torch

# =============================================================================
# TRAINING METHOD SELECTION
# =============================================================================
train_method = 'adam'  # OPTIONS: 'adam', 'sgd'

# =============================================================================
# DATASET AND MODEL SELECTION
# =============================================================================
dataset = 'shakespeare'  # OPTIONS: 'shakespeare', 'openwebtext', 'gpt2'
init_from = 'scratch'    # OPTIONS: 'scratch', 'resume', 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'

# =============================================================================
# OUTPUT AND LOGGING SETTINGS
# =============================================================================
out_dir = 'out-first-order'  # Output directory for checkpoints and logs
eval_interval = 1000         # How often to evaluate on validation set
log_interval = 1             # How often to log training progress
eval_iters = 50              # Number of iterations for evaluation
eval_only = False            # If True, only run evaluation and exit
always_save_checkpoint = True # If True, always save checkpoint after eval

# Weights & Biases logging
wandb_log = False            # Enable wandb logging
wandb_project = 'nanogpt'    # Wandb project name
wandb_run_name = 'first-order-run'  # Wandb run name

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

# Adam-specific parameters
beta1 = 0.9          # Adam beta1 (momentum coefficient)
beta2 = 0.95         # Adam beta2 (RMSprop coefficient)

# Gradient clipping
grad_clip = 1.0      # Clip gradients at this value (0.0 to disable)

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

# # OPENWEBTEXT DATASET (medium scale)
# if dataset == 'openwebtext':
#     batch_size = 12
#     block_size = 1024
#     max_iters = 600000
#     eval_interval = 2000
#     gradient_accumulation_steps = 40
#     n_layer, n_head, n_embd = 12, 12, 768
#     learning_rate = 6e-4

# # GPT2 DATASET (large scale)
# if dataset == 'gpt2':
#     batch_size = 12
#     block_size = 1024
#     max_iters = 600000
#     eval_interval = 2000
#     gradient_accumulation_steps = 40
#     n_layer, n_head, n_embd = 12, 12, 768
#     learning_rate = 6e-4 