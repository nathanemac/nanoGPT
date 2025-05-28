# Config for DiKronZO training on Shakespeare dataset

import torch

# Output directory
out_dir = 'out-dikronzo-shakespeare'

# Logging settings
eval_interval = 500
log_interval = 1
eval_iters = 50
eval_only = False
always_save_checkpoint = True
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*' (e.g., 'gpt2-medium')

# Wandb logging
wandb_log = False # Set to True for wandb logging
wandb_project = 'shakespeare' # Or your project name
wandb_run_name = 'dikronzo-shakespeare'

# Data
dataset = 'shakespeare'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256

# Model (small GPT for Shakespeare)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

# Training settings
max_iters = 8000
lr_decay_iters = 8000
min_lr = 1e-4
learning_rate = 1e-3
warmup_iters = 500
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
weight_decay = 1e-1

# Training method
train_method = 'dikronzo'

# DiKronZO specific parameters
zo_eps = 1e-3                    # Perturbation size
kron_strategy = 'approx_square'  # Kronecker factorization strategy ('approx_square', 'fixed_factor', 'power2')
kron_max_factor = 32             # Maximum factor size for 'fixed_factor' strategy
step_interval = 100               # Interval for updating B matrices

# Directional search parameters
directional_q = 100               # Number of directions to try (DiKronZO specific)

# DiKronZO Momentum settings (optional)
use_momentum = True             # Set to True to enable momentum for DiKronZO
momentum_beta = 0.9              # Momentum coefficient (if use_momentum is True)

# Adaptive zo_eps settings (optional)
use_adaptive_eps = False         # Enable adaptive zo_eps for DiKronZO
adaptive_eps_window = 20
adaptive_eps_lr_coupling = 0.3
adaptive_eps_success_high = 0.8
adaptive_eps_success_low = 0.2

# System settings
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True # use PyTorch 2.0 to compile the model to be faster 