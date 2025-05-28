# Config for KronZO training on GPT-2 pretraining dataset (e.g., OpenWebText)

import torch

# Output directory
out_dir = 'out-kronzo-gpt2'

# Logging settings
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*' (e.g., 'gpt2-medium')

# Wandb logging
wandb_log = False # Set to True for wandb logging
wandb_project = 'owt' # Or your project name
wandb_run_name = 'kronzo-gpt2'

# Data
dataset = 'openwebtext'
gradient_accumulation_steps = 40  # Corresponds to a total batch size of 40*12 = 480
batch_size = 12                 # Micro-batch size
block_size = 1024               # Context length

# Model (GPT-2 124M)
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # For pretraining, 0 is good
bias = False

# Optimizer
learning_rate = 6e-4  # Max learning rate
max_iters = 600000    # Total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Learning rate schedule
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600000 # Should be ~= max_iters
min_lr = 6e-5         # Minimum learning rate

# Training method
train_method = 'kronzo'

# KronZO specific parameters
zo_eps = 1e-3                    # Perturbation size
kron_strategy = 'approx_square'  # Kronecker factorization strategy ('approx_square', 'fixed_factor', 'power2')
kron_max_factor = 32             # Maximum factor size for 'fixed_factor' strategy

# KronZO Momentum settings
use_momentum = False             # Set to True to enable momentum for KronZO
momentum_beta = 0.9              # Momentum coefficient (if use_momentum is True)

# Adaptive zo_eps settings
use_adaptive_eps = False         # Enable adaptive zo_eps for KronZO
adaptive_eps_window = 50         # Larger window for larger scale experiments
adaptive_eps_lr_coupling = 0.5
adaptive_eps_success_high = 0.7
adaptive_eps_success_low = 0.3

# System settings
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True 