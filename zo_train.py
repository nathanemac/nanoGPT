import os
import time
import math
import pickle
from contextlib import nullcontext
import psutil
import torch.cuda as cuda
from tqdm import tqdm

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT
from common_utils import zo_forward, AdaptiveZoEps
from lozo_utils import (
    get_current_rank, lowrank_zo_perturb_parameters, lowrank_zo_step, 
    lowrank_zo_update, lowrank_zo_update_direct, lowrank_zo_update_momentum,
    svdlozo_perturb_parameters, svdlozo_step, svdlozo_update, svdlozo_update_direct,
    dilozo_perturb_parameters, dilozo_step, dilozo_update, dilozo_update_momentum
)
from mezo_utils import (
    mezo_perturb_parameters, mezo_step, mezo_update, mezo_update_momentum,
    dimezo_perturb_parameters, dimezo_step, dimezo_update, dimezo_update_momentum
)
from kronzo_utils import (
    kronzo_perturb_parameters, kronzo_step, kronzo_update, kronzo_update_momentum,
    dikronzo_perturb_parameters, dikronzo_step, dikronzo_update, dikronzo_update_momentum
)
from kronzo_utils_improved_directional import (
    ImprovedDirectionalHistory, improved_kronzo_step, 
    improved_kronzo_update, improved_kronzo_update_momentum
)
from first_order_utils import first_order_update

# -----------------------------------------------------------------------------
# Default config values
out_dir = 'out' # Default output directory, method-specific configs might change this
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'
wandb_log = False
wandb_project = 'owt'
wandb_run_name = 'gpt2' # Will be overridden by method-specific or configurator
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8
batch_size = 12
block_size = 1024
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False
learning_rate = 6e-4
max_iters = 600000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 6e-5
backend = 'nccl'
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True

# Zero-Order specific parameters (defaults, can be overridden by method configs)
zo_eps = 1e-3
rank_r = 4
step_interval = 10
use_momentum = False # General momentum flag, method-specific configs will set this for lozom, mezom etc.
momentum_beta = 0.9
zo_q = 1
rank_adaptive = False
min_rank = 1
max_rank = 16
svd_tau = 0.6
svd_max_rank = 16
use_full_svd = False
train_method = 'adam' # Default training method, WILL be overridden by method-specific config
directional_q = 10
dimezo_direct_movement = False
kron_max_factor = 32
kron_strategy = 'approx_square'
kronzo_sampling_number = 1  # Number of Kronecker products to sample and sum
dikronzo_direct_movement = False
use_adaptive_eps = False
adaptive_eps_window = 20
adaptive_eps_lr_coupling = 0.5
adaptive_eps_success_high = 0.7
adaptive_eps_success_low = 0.3
loss_history_size = 10  # For improved directional KronZO conservative updates

# This will be set by train.sh via command line argument
method_config_file = None 

# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]

# Layered configuration loading:
# 1. Defaults are set above.
# 2. Method-specific config file (e.g., config/lozo.py) is loaded if provided.
#    This is passed via --method_config_file by train.sh
# 3. configurator.py (general overrides, e.g., for a specific run) is loaded.
# 4. Command-line arguments (parsed by configurator.py) can override anything.

# The actual loading of method_config_file and configurator.py happens in a specific order:
# We need to parse CLI args first to get method_config_file if it's passed that way.
# For now, assume method_config_file is primarily set by an environment variable or a direct argument
# that configurator.py can pick up early.

# Let configurator.py handle the loading of method_config_file if specified
# It will also parse other CLI arguments.
# We need to ensure 'method_config_file' is a known key for configurator.py if we want it to load it.
# A simple way is to exec it if the path is valid. 

# --- Configuration Loading Order --- 
# 1. Defaults (already set)
# 2. configurator.py will parse all CLI args. If 'method_config_file' is an arg, it gets set.
#    If not, we hope it's set by train.sh in a way that configurator.py can see (e.g. if it directly modifies globals).
#    A more robust way is to read it from sys.argv here BEFORE configurator.py, if it has a specific flag like --method_config.

# Simplification: We expect configurator.py to allow overriding `method_config_file`
# and then we conditionally load it. This requires configurator.py to be aware of this new var.
# For now, let's assume method_config_file will be passed and loaded by configurator.py
# or explicitly set via shell script into the python call itself.

# The most robust approach is to parse for a dedicated --method_config_path argument here first.
# Example: python zo_train.py --method_config_path config/lozo.py --other_arg ...
import sys
parsed_method_config_file = None
for i, arg in enumerate(sys.argv):
    if arg == '--method_config_file' and i + 1 < len(sys.argv):
        parsed_method_config_file = sys.argv[i+1]
        # Remove these args so configurator.py doesn't see them or complain
        sys.argv.pop(i)
        sys.argv.pop(i) # Popped the value
        break
if parsed_method_config_file and os.path.exists(parsed_method_config_file):
    print(f"Loading method-specific config: {parsed_method_config_file}")
    print("=" * 80)
    print(f"📋 METHOD CONFIG FILE CONTENTS ({parsed_method_config_file}):")
    print("=" * 80)
    with open(parsed_method_config_file, 'r') as f:
        config_content = f.read()
        print(config_content)
    print("=" * 80)
    print("🔧 EXECUTING CONFIG...")
    print("=" * 80)
    with open(parsed_method_config_file) as f:
        exec(f.read())
    print(f"✅ Method config loaded successfully: {parsed_method_config_file}")
else:
    if parsed_method_config_file:
        print(f"Warning: Method config file specified but not found: {parsed_method_config_file}")
    # If not parsed_method_config_file, it means --method_config_file was not in sys.argv
    # configurator.py might still load a global `method_config_file` variable if set by other means

# Now load general overrides from configurator.py and command line
# Ensure `configurator.py` does NOT re-parse `method_config_file` if we handled it.
if os.path.exists('configurator.py'):
    print("Loading general config from configurator.py and CLI overrides...")
    exec(open('configurator.py').read())

# Rebuild config_keys and config AFTER all files have been exec'd and CLI args parsed
# This ensures that any new variables introduced by config files are captured.
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
config = {k: globals()[k] for k in config_keys if k in globals()} # Build the final config dictionary

# Update wandb_run_name and out_dir based on the final train_method from the config
if 'train_method' in config:
    # Ensure train_method is a global variable as well, if it came from a config file
    if 'train_method' not in globals() or globals()['train_method'] != config['train_method']:
        globals()['train_method'] = config['train_method']

    config['wandb_run_name'] = f"gpt2-{config['train_method']}"
    wandb_run_name = config['wandb_run_name'] # Update global for wandb.init
    
    new_out_dir = f"out-{config['train_method']}"
    config['out_dir'] = new_out_dir # Update config dict entry
    out_dir = new_out_dir # Update global out_dir for os.makedirs and checkpointing
    globals()['out_dir'] = new_out_dir # Ensure global scope is also updated

# Make sure critical variables used later are correctly set in globals() if they came from config
for key_to_sync in ['learning_rate', 'zo_eps', 'adaptive_eps_window', 
                    'adaptive_eps_lr_coupling', 'adaptive_eps_success_high', 
                    'adaptive_eps_success_low']:
    if key_to_sync in config and (key_to_sync not in globals() or globals()[key_to_sync] != config[key_to_sync]):
        globals()[key_to_sync] = config[key_to_sync]

# -----------------------------------------------------------------------------

# The rest of the script is identical to lozo_train.py from this point onwards
# (DDP setup, data loader, model init, optimizer, training loop, etc.)

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

data_dir = os.path.join('data', dataset)
if not os.path.exists(os.path.join(data_dir, 'train.bin')):
    data_dir = '/nobackups/allanath/data-nanogpt'

def get_batch(split):
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

iter_num = 0
best_val_loss = 1e9

meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout)
if init_from == 'scratch':
    print("Initializing a new model from scratch")
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
    v_dict = checkpoint.get('v_dict', {})
    step = checkpoint.get('step', 0)
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size
model.to(device)

if not init_from == 'resume':
    v_dict = {}
    step = 0
    exp_avg_m = {}
    v_old_dict = {}
    b_dict = {}  # B matrix storage for KronZO methods
    improved_directional_history = ImprovedDirectionalHistory(loss_history_size)  # For improved KronZO
elif 'exp_avg_m' in checkpoint and (train_method in ['lozom', 'mezom', 'dilozo', 'kronzo', 'dikronzo'] and use_momentum):
    exp_avg_m = checkpoint['exp_avg_m']
    v_old_dict = checkpoint.get('v_old_dict', {})
    b_dict = checkpoint.get('b_dict', {})  # Load B matrices for KronZO methods
    improved_directional_history = ImprovedDirectionalHistory(loss_history_size)  # For improved KronZO (not saved in checkpoint)
else:
    exp_avg_m = {}
    v_old_dict = {}
    b_dict = {}  # Initialize B matrix storage
    improved_directional_history = ImprovedDirectionalHistory(loss_history_size)  # For improved KronZO

if use_adaptive_eps:
    adaptive_eps_manager = AdaptiveZoEps(
        base_eps=zo_eps,
        base_lr=learning_rate, 
        window_size=adaptive_eps_window,
        eps_increase_rate=config.get('adaptive_eps_increase_rate', 0.05), 
        eps_decrease_rate=config.get('adaptive_eps_decrease_rate', 0.1),  
        lr_coupling_strength=adaptive_eps_lr_coupling,
        min_eps=config.get('adaptive_min_eps', 1e-6), 
        max_eps=config.get('adaptive_max_eps', 1e-3), 
        adaptive_eps_success_high=adaptive_eps_success_high,
        adaptive_eps_success_low=adaptive_eps_success_low
    )
    if master_process:
        print(f"Using adaptive zo_eps with base_eps={zo_eps}, base_lr={learning_rate}, range=[{adaptive_eps_manager.min_eps:.1e}, {adaptive_eps_manager.max_eps:.1e}]")
    if init_from == 'resume' and checkpoint is not None and 'adaptive_eps_manager_state' in checkpoint:
        # Ensure adaptive_eps_manager is initialized before loading state
        if adaptive_eps_manager is None and use_adaptive_eps: # Check use_adaptive_eps from config
            # This block might be redundant if adaptive_eps_manager is always init based on use_adaptive_eps
            # but it's a safeguard.
            adaptive_eps_manager = AdaptiveZoEps(
                base_eps=config.get('zo_eps', zo_eps), # Use config value or global default
                base_lr=config.get('learning_rate', learning_rate),
                window_size=config.get('adaptive_eps_window', adaptive_eps_window),
                eps_increase_rate=config.get('adaptive_eps_increase_rate', 0.05),
                eps_decrease_rate=config.get('adaptive_eps_decrease_rate', 0.1),
                lr_coupling_strength=config.get('adaptive_eps_lr_coupling', adaptive_eps_lr_coupling),
                min_eps=config.get('adaptive_min_eps', 1e-6),
                max_eps=config.get('adaptive_max_eps', 1e-3),
                adaptive_eps_success_high=config.get('adaptive_eps_success_high', adaptive_eps_success_high),
                adaptive_eps_success_low=config.get('adaptive_eps_success_low', adaptive_eps_success_low)
            )
            if master_process:
                print(f"Re-initialized adaptive_eps_manager prior to loading state.")
        
        if adaptive_eps_manager is not None: # Now it should be initialized if use_adaptive_eps is true
            adaptive_eps_manager.load_state(checkpoint['adaptive_eps_manager_state'])
            if master_process:
                print(f"Loaded adaptive_eps_manager state after init. Current eps: {adaptive_eps_manager.get_eps():.2e}")
        elif master_process and use_adaptive_eps: # Log if still None despite use_adaptive_eps being true
             print("Warning: use_adaptive_eps is true, adaptive_eps_manager_state in checkpoint, but manager is still None before loading.")
else:
    adaptive_eps_manager = None

optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume' and 'optimizer' in checkpoint:
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None 

if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X_batch, Y_batch = get_batch(split)
            with ctx:
                logits, loss_val = model(X_batch, Y_batch)
            losses[k] = loss_val.item()
        out[split] = losses.mean()
    model.train()
    return out

def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

if wandb_log and master_process:
    import wandb
    # Ensure config sent to wandb is the final, complete config
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

X, Y = get_batch('train')
t0 = time.time()
local_iter_num = 0
raw_model = model.module if ddp else model
running_mfu = -1.0

pbar = tqdm(total=max_iters, desc='Training', disable=not master_process)
loss = torch.tensor(0.0)

while True:
    lr = get_lr(iter_num) if decay_lr else learning_rate
    effective_zo_eps_to_use = zo_eps
    if use_adaptive_eps and adaptive_eps_manager is not None:
        current_lr_for_eps = get_lr(iter_num) if decay_lr else learning_rate 
        effective_zo_eps_to_use = adaptive_eps_manager.get_eps()

    if iter_num % eval_interval == 0 and master_process:
        losses_est = estimate_loss()
        print(f"\nstep {iter_num}: train loss {losses_est['train']:.4f}, val loss {losses_est['val']:.4f}")
        if wandb_log:
            wandb_dict = {
                "iter": iter_num,
                "train/loss": losses_est['train'],
                "val/loss": losses_est['val'],
                "lr": lr,
                "mfu": running_mfu*100,
            }
            if rank_adaptive and (train_method == 'lozo' or train_method == 'lozom' or train_method == 'dilozo'): 
                current_rank = get_current_rank(iter_num, max_iters, min_rank, max_rank, rank_adaptive, rank_r)
                wandb_dict["rank"] = current_rank
                wandb_dict["rank_progress"] = min(iter_num / max_iters, 1.0)
            wandb.log(wandb_dict)
        if losses_est['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses_est['val']
            if iter_num > 0:
                checkpoint_data = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                    'v_dict': v_dict,
                    'step': step,
                }
                if use_adaptive_eps and adaptive_eps_manager is not None:
                    checkpoint_data['adaptive_eps_manager_state'] = adaptive_eps_manager.get_state()
                if train_method in ['lozom', 'mezom', 'dilozo', 'kronzo', 'dikronzo', 'improved_kronzo'] and use_momentum:
                    checkpoint_data['exp_avg_m'] = exp_avg_m
                    if train_method == 'lozom':
                        checkpoint_data['v_old_dict'] = v_old_dict
                if train_method in ['kronzo', 'dikronzo', 'improved_kronzo']:
                    checkpoint_data['b_dict'] = b_dict  # Save B matrices for KronZO methods
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint_data, os.path.join(out_dir, 'ckpt.pt')) # Use a different name for the dict
    if iter_num == 0 and eval_only:
        break

    if train_method == 'lozo' or train_method == 'lozom':
        current_rank_val = get_current_rank(iter_num, max_iters, min_rank, max_rank, rank_adaptive, rank_r)
        accumulated_projected_grad = 0.0
        accumulated_grad_dict = {}
        first_zo_seed = None  # Store first seed for updates
        accumulated_loss = 0.0  # Properly accumulate loss
        for micro_step in range(gradient_accumulation_steps):
            current_zo_seed = np.random.randint(1000000000)
            if micro_step == 0:
                first_zo_seed = current_zo_seed  # Save first seed for consistent updates
            loss_val, projected_or_dict_grad, _ = lowrank_zo_step(
                raw_model, X, Y, v_dict, step, current_zo_seed, 
                effective_zo_eps_to_use, step_interval, rank_r, zo_q, ctx, current_rank=current_rank_val
            )
            accumulated_loss += loss_val / gradient_accumulation_steps  # Proper loss accumulation
            if zo_q == 1:
                projected_grad = projected_or_dict_grad
                if micro_step == 0:
                    accumulated_projected_grad = projected_grad / gradient_accumulation_steps
                else:
                    accumulated_projected_grad += projected_grad / gradient_accumulation_steps
            else:
                grad_dict = projected_or_dict_grad
                if micro_step == 0:
                    accumulated_grad_dict = {name: g / gradient_accumulation_steps for name, g in grad_dict.items()}
                else:
                    for name, g in grad_dict.items():
                        accumulated_grad_dict[name] += g / gradient_accumulation_steps
            # Get next batch for next iteration (but not after the last micro-step)
            if micro_step < gradient_accumulation_steps - 1:
                X, Y = get_batch('train')
        loss = accumulated_loss  # Use properly accumulated loss
        if zo_q == 1:
            if train_method == 'lozom':
                lowrank_zo_update_momentum(raw_model, optimizer, accumulated_projected_grad, first_zo_seed, v_dict, exp_avg_m, v_old_dict, step, lr, rank_r, step_interval, weight_decay, master_process, momentum_beta, current_rank=current_rank_val)
            else:
                lowrank_zo_update(raw_model, optimizer, accumulated_projected_grad, first_zo_seed, v_dict, step, lr, rank_r, weight_decay, master_process, current_rank=current_rank_val)
        else:
            lowrank_zo_update_direct(raw_model, optimizer, accumulated_grad_dict, lr, weight_decay)
        # Add adaptive epsilon support for LOZO 
        if use_adaptive_eps and adaptive_eps_manager is not None:
            current_lr_for_eps_record = get_lr(iter_num) if decay_lr else learning_rate
            adaptive_eps_manager.record_step(
                loss=accumulated_loss, 
                projected_grad=accumulated_projected_grad if zo_q == 1 else 0.0, 
                learning_rate=current_lr_for_eps_record,
                method_info={'success': True}  # LOZO doesn't have explicit success tracking
            )
        step += 1
    elif train_method == 'svdlozo':
        accumulated_projected_grad = 0.0
        accumulated_grad_dict = {}
        named_params_for_update = None
        first_zo_seed = None  # Store first seed for updates
        accumulated_loss = 0.0  # Properly accumulate loss
        for micro_step in range(gradient_accumulation_steps):
            current_zo_seed = np.random.randint(1000000000)
            if micro_step == 0:
                first_zo_seed = current_zo_seed  # Save first seed for consistent updates
            debug_svd = True if micro_step == 0 and iter_num <= 2 else False 
            loss_val, projected_or_dict_grad, current_named_params = svdlozo_step(
                raw_model, X, Y, step, current_zo_seed, 
                effective_zo_eps_to_use, svd_tau, svd_max_rank, use_full_svd, zo_q, ctx
            )
            if micro_step == 0:
                named_params_for_update = current_named_params
            accumulated_loss += loss_val / gradient_accumulation_steps  # Proper loss accumulation
            if zo_q == 1:
                projected_grad = projected_or_dict_grad
                if micro_step == 0:
                    accumulated_projected_grad = projected_grad / gradient_accumulation_steps
                else:
                    accumulated_projected_grad += projected_grad / gradient_accumulation_steps
            else:
                grad_dict = projected_or_dict_grad
                if micro_step == 0:
                    accumulated_grad_dict = {name: g / gradient_accumulation_steps for name, g in grad_dict.items()}
                else:
                    for name, g in grad_dict.items():
                        accumulated_grad_dict[name] += g / gradient_accumulation_steps
            # Get next batch for next iteration (but not after the last micro-step)
            if micro_step < gradient_accumulation_steps - 1:
                X, Y = get_batch('train')
        loss = accumulated_loss  # Use properly accumulated loss
        if zo_q == 1:
            svdlozo_update(raw_model, optimizer, accumulated_projected_grad, first_zo_seed, step, lr, 
                           effective_zo_eps_to_use, svd_tau, svd_max_rank, use_full_svd, weight_decay, master_process, 
                           named_parameters_to_optim=named_params_for_update)
        else:
            svdlozo_update_direct(raw_model, optimizer, accumulated_grad_dict, lr, weight_decay, 
                                  named_parameters_to_optim=named_params_for_update)
        # Add adaptive epsilon support for SVD-LOZO
        if use_adaptive_eps and adaptive_eps_manager is not None:
            current_lr_for_eps_record = get_lr(iter_num) if decay_lr else learning_rate
            adaptive_eps_manager.record_step(
                loss=accumulated_loss, 
                projected_grad=accumulated_projected_grad if zo_q == 1 else 0.0, 
                learning_rate=current_lr_for_eps_record,
                method_info={'success': True}  # SVD-LOZO doesn't have explicit success tracking
            )
        step += 1
    elif train_method == 'mezo' or train_method == 'mezom':
        accumulated_projected_grad = 0.0
        named_params_for_update = None
        first_zo_seed = None  # Store first seed for updates
        accumulated_loss = 0.0  # Properly accumulate loss
        for micro_step in range(gradient_accumulation_steps):
            current_zo_seed = np.random.randint(1000000000)
            if micro_step == 0:
                first_zo_seed = current_zo_seed  # Save first seed for consistent updates
            loss_val, projected_grad, current_named_params = mezo_step(
                raw_model, X, Y, step, current_zo_seed, effective_zo_eps_to_use, ctx
            )
            if micro_step == 0:
                named_params_for_update = current_named_params
            # Get next batch for next iteration (but not after the last micro-step)
            if micro_step < gradient_accumulation_steps - 1:
                X, Y = get_batch('train')
            accumulated_loss += loss_val / gradient_accumulation_steps  # Proper loss accumulation
            if micro_step == 0:
                accumulated_projected_grad = projected_grad / gradient_accumulation_steps
            else:
                accumulated_projected_grad += projected_grad / gradient_accumulation_steps
        loss = accumulated_loss  # Use properly accumulated loss
        if train_method == 'mezom':
            mezo_update_momentum(raw_model, optimizer, accumulated_projected_grad, first_zo_seed, exp_avg_m, step, lr, weight_decay, master_process, momentum_beta, named_parameters_to_optim=named_params_for_update)
        else:
            mezo_update(raw_model, optimizer, accumulated_projected_grad, first_zo_seed, step, lr, weight_decay, master_process, named_parameters_to_optim=named_params_for_update)
        # Add adaptive epsilon support for MeZO
        if use_adaptive_eps and adaptive_eps_manager is not None:
            current_lr_for_eps_record = get_lr(iter_num) if decay_lr else learning_rate
            adaptive_eps_manager.record_step(
                loss=accumulated_loss, 
                projected_grad=accumulated_projected_grad, 
                learning_rate=current_lr_for_eps_record,
                method_info={'success': True}  # MeZO doesn't have explicit success tracking
            )
        step += 1
    elif train_method == 'dimezo':
        accumulated_projected_grad = 0.0
        best_direction_seed_for_update = None
        first_microbatch_loss_val = 0.0
        first_microbatch_success_val = False
        accumulated_loss = 0.0  # Properly accumulate loss
        for micro_step in range(gradient_accumulation_steps):
            current_zo_seed = np.random.randint(1000000000)
            loss_val, grad_or_direction, current_best_seed, success = dimezo_step(
                raw_model, X, Y, step, current_zo_seed, 
                directional_q_global=directional_q, 
                zo_eps=effective_zo_eps_to_use, 
                master_process=master_process, 
                ctx_obj=ctx,
                eps=effective_zo_eps_to_use, 
                direct_movement=dimezo_direct_movement
            )
            # Get next batch for next iteration (but not after the last micro-step)
            if micro_step < gradient_accumulation_steps - 1:
                X, Y = get_batch('train')
            accumulated_loss += loss_val / gradient_accumulation_steps  # Proper loss accumulation
            if micro_step == 0:
                best_direction_seed_for_update = current_best_seed
                if dimezo_direct_movement:
                     best_direction_for_update_dict = grad_or_direction
                else:
                    accumulated_projected_grad = grad_or_direction / gradient_accumulation_steps
                first_microbatch_loss_val = loss_val
                first_microbatch_success_val = success
            elif not dimezo_direct_movement :
                accumulated_projected_grad += grad_or_direction / gradient_accumulation_steps
        loss = accumulated_loss  # Use properly accumulated loss
        if dimezo_direct_movement:
            if use_momentum:
                dimezo_update_momentum(raw_model, optimizer, best_direction_for_update_dict, best_direction_seed_for_update, exp_avg_m, step, lr, weight_decay, master_process, beta1=momentum_beta, direct_movement=True)
            else:
                dimezo_update(raw_model, optimizer, best_direction_for_update_dict, best_direction_seed_for_update, step, lr, weight_decay, master_process, direct_movement=True)
        else:
            if use_momentum:
                dimezo_update_momentum(raw_model, optimizer, accumulated_projected_grad, best_direction_seed_for_update, exp_avg_m, step, lr, weight_decay, master_process, beta1=momentum_beta, direct_movement=False)
            else:
                dimezo_update(raw_model, optimizer, accumulated_projected_grad, best_direction_seed_for_update, step, lr, weight_decay, master_process, direct_movement=False)
        if use_adaptive_eps and adaptive_eps_manager is not None:
            proj_grad_for_adaptive = accumulated_projected_grad if not dimezo_direct_movement else 0.0 
            current_lr_for_eps_record = get_lr(iter_num) if decay_lr else learning_rate
            adaptive_eps_manager.record_step(
                loss=first_microbatch_loss_val, 
                projected_grad=proj_grad_for_adaptive, 
                learning_rate=current_lr_for_eps_record, 
                method_info={'success': first_microbatch_success_val} 
            )
        step += 1
    elif train_method == 'dilozo':
        current_rank_val = get_current_rank(iter_num, max_iters, min_rank, max_rank, rank_adaptive, rank_r)
        accumulated_grad_coeff_val = 0.0
        best_u_seed_for_update = None
        first_microbatch_loss_val = 0.0
        first_microbatch_success_val = False
        accumulated_loss = 0.0  # Properly accumulate loss
        for micro_step in range(gradient_accumulation_steps):
            current_zo_seed = np.random.randint(1000000000)
            iter_loss_val, grad_coeff_val, current_best_u_seed, success_val = dilozo_step(
                raw_model, X, Y, v_dict, step, current_zo_seed,
                directional_q_val=directional_q, 
                zo_eps=effective_zo_eps_to_use, 
                step_interval=step_interval, rank_r=rank_r, 
                master_process=master_process, ctx_obj=ctx,
                eps=effective_zo_eps_to_use, 
                current_rank=current_rank_val
            )
            # Get next batch for next iteration (but not after the last micro-step)
            if micro_step < gradient_accumulation_steps - 1:
                X, Y = get_batch('train')
            accumulated_loss += iter_loss_val / gradient_accumulation_steps  # Proper loss accumulation
            if micro_step == 0:
                accumulated_grad_coeff_val = grad_coeff_val / gradient_accumulation_steps
                best_u_seed_for_update = current_best_u_seed
                first_microbatch_loss_val = iter_loss_val 
                first_microbatch_success_val = success_val
            else:
                accumulated_grad_coeff_val += grad_coeff_val / gradient_accumulation_steps
        loss = accumulated_loss  # Use properly accumulated loss
        if use_momentum:
            dilozo_update_momentum(raw_model, optimizer, accumulated_grad_coeff_val, best_u_seed_for_update, 
                                   v_dict, exp_avg_m, step, lr, rank_r, 
                                   weight_decay=weight_decay, master_process=master_process, beta1=momentum_beta, 
                                   current_rank=current_rank_val)
        else:
            dilozo_update(raw_model, optimizer, accumulated_grad_coeff_val, best_u_seed_for_update, 
                          v_dict, step, lr, rank_r, 
                          weight_decay=weight_decay, master_process=master_process, 
                          current_rank=current_rank_val)
        if use_adaptive_eps and adaptive_eps_manager is not None:
            current_lr_for_eps_record = get_lr(iter_num) if decay_lr else learning_rate
            adaptive_eps_manager.record_step(
                loss=first_microbatch_loss_val, 
                projected_grad=accumulated_grad_coeff_val, 
                learning_rate=current_lr_for_eps_record,
                method_info={'success': first_microbatch_success_val}
            )
        step += 1
    elif train_method == 'kronzo':
        accumulated_projected_grad = 0.0
        named_params_for_update = None
        first_zo_seed = None  # Store first seed for updates
        accumulated_loss = 0.0  # Properly accumulate loss
        for micro_step in range(gradient_accumulation_steps):
            current_zo_seed = np.random.randint(1000000000)
            if micro_step == 0:
                first_zo_seed = current_zo_seed  # Save first seed for consistent updates
            loss_val, projected_grad, current_named_params = kronzo_step(
                raw_model, X, Y, step, current_zo_seed, 
                zo_eps_global=effective_zo_eps_to_use, 
                ctx_obj=ctx,
                strategy=kron_strategy, max_factor=kron_max_factor,
                eps=effective_zo_eps_to_use, kronzo_sampling_number=kronzo_sampling_number,
                b_dict=b_dict, step_interval=step_interval
            )
            if micro_step == 0:
                named_params_for_update = current_named_params
            # Get next batch for next iteration (but not after the last micro-step)
            if micro_step < gradient_accumulation_steps - 1:
                X, Y = get_batch('train')
            accumulated_loss += loss_val / gradient_accumulation_steps  # Proper loss accumulation
            if micro_step == 0:
                accumulated_projected_grad = projected_grad / gradient_accumulation_steps
            else:
                accumulated_projected_grad += projected_grad / gradient_accumulation_steps
        loss = accumulated_loss  # Use properly accumulated loss
        if use_momentum:
            kronzo_update_momentum(raw_model, optimizer, accumulated_projected_grad, first_zo_seed, 
                                   exp_avg_m, step, lr, weight_decay, master_process,
                                   beta1=momentum_beta,
                                   named_parameters_to_optim=named_params_for_update, 
                                   strategy=kron_strategy, max_factor=kron_max_factor,
                                   kronzo_sampling_number=kronzo_sampling_number,
                                   b_dict=b_dict, step_interval=step_interval)
        else:
            kronzo_update(raw_model, optimizer, accumulated_projected_grad, first_zo_seed, step, lr, 
                          weight_decay, master_process,
                          named_parameters_to_optim=named_params_for_update,
                          strategy=kron_strategy, max_factor=kron_max_factor,
                          kronzo_sampling_number=kronzo_sampling_number,
                          b_dict=b_dict, step_interval=step_interval)
        # Add adaptive epsilon support for KronZO
        if use_adaptive_eps and adaptive_eps_manager is not None:
            current_lr_for_eps_record = get_lr(iter_num) if decay_lr else learning_rate
            adaptive_eps_manager.record_step(
                loss=accumulated_loss, 
                projected_grad=accumulated_projected_grad, 
                learning_rate=current_lr_for_eps_record,
                method_info={'success': True}  # KronZO doesn't have explicit success tracking
            )
        step += 1
    elif train_method == 'dikronzo':
        accumulated_grad_val = 0.0
        best_direction_seed_for_update = None
        first_microbatch_loss_val = 0.0
        first_microbatch_success_val = False
        accumulated_loss = 0.0  # Properly accumulate loss
        for micro_step in range(gradient_accumulation_steps):
            current_zo_seed = np.random.randint(1_000_000_000)
            loss_step_val, grad_or_coeff_val, current_best_seed, success_val = dikronzo_step(
                raw_model, X, Y, 
                step               = step,
                zo_random_seed     = current_zo_seed,
                directional_q_global=directional_q, 
                zo_eps_global      = effective_zo_eps_to_use, 
                ctx_obj            = ctx, 
                eps                = effective_zo_eps_to_use, 
                strategy           = kron_strategy,
                max_factor         = kron_max_factor,
                direct_movement    = dikronzo_direct_movement,
                kronzo_sampling_number = kronzo_sampling_number,
                b_dict             = b_dict,
                step_interval      = step_interval
            )
            # Get next batch for next iteration (but not after the last micro-step)
            if micro_step < gradient_accumulation_steps - 1:
                X, Y = get_batch('train')
            accumulated_loss += loss_step_val / gradient_accumulation_steps  # Proper loss accumulation
            if micro_step == 0:
                first_microbatch_loss_val    = loss_step_val
                first_microbatch_success_val = success_val
                best_direction_seed_for_update = current_best_seed
                if dikronzo_direct_movement:
                    pass
                else:
                    accumulated_grad_val = grad_or_coeff_val / gradient_accumulation_steps
            elif not dikronzo_direct_movement:
                 accumulated_grad_val += grad_or_coeff_val / gradient_accumulation_steps
        loss = accumulated_loss  # Use properly accumulated loss
        if dikronzo_direct_movement:
            if use_momentum:
                dikronzo_update_momentum(
                    raw_model, optimizer,
                    grad_or_seed       = None, 
                    best_seed          = best_direction_seed_for_update,
                    exp_avg_m          = exp_avg_m,
                    step               = step,
                    lr                 = lr,
                    weight_decay       = weight_decay,
                    master_process     = master_process,
                    beta1              = momentum_beta,
                    direct_movement    = True,
                    strategy           = kron_strategy,
                    max_factor         = kron_max_factor,
                    kronzo_sampling_number = kronzo_sampling_number,
                    b_dict             = b_dict,
                    step_interval      = step_interval
                )
            else:
                dikronzo_update(
                    raw_model, optimizer,
                    grad_or_seed       = None, 
                    best_seed          = best_direction_seed_for_update,
                    step               = step,
                    lr                 = lr,
                    weight_decay       = weight_decay,
                    master_process     = master_process,
                    direct_movement    = True,
                    strategy           = kron_strategy,
                    max_factor         = kron_max_factor,
                    kronzo_sampling_number = kronzo_sampling_number,
                    b_dict             = b_dict,
                    step_interval      = step_interval
                )
        else:
            if use_momentum:
                dikronzo_update_momentum(
                    raw_model, optimizer,
                    grad_or_seed       = accumulated_grad_val, 
                    best_seed          = best_direction_seed_for_update,
                    exp_avg_m          = exp_avg_m,
                    step               = step,
                    lr                 = lr,
                    weight_decay       = weight_decay,
                    master_process     = master_process,
                    beta1              = momentum_beta,
                    direct_movement    = False,
                    strategy           = kron_strategy,
                    max_factor         = kron_max_factor,
                    kronzo_sampling_number = kronzo_sampling_number,
                    b_dict             = b_dict,
                    step_interval      = step_interval
                )
            else:
                dikronzo_update(
                    raw_model, optimizer,
                    grad_or_seed       = accumulated_grad_val, 
                    best_seed          = best_direction_seed_for_update,
                    step               = step,
                    lr                 = lr,
                    weight_decay       = weight_decay,
                    master_process     = master_process,
                    direct_movement    = False,
                    strategy           = kron_strategy,
                    max_factor         = kron_max_factor,
                    kronzo_sampling_number = kronzo_sampling_number,
                    b_dict             = b_dict,
                    step_interval      = step_interval
                )
        if use_adaptive_eps and adaptive_eps_manager is not None:
            proj_grad_for_adaptive = accumulated_grad_val if not dikronzo_direct_movement else 0.0
            current_lr_for_eps_record = get_lr(iter_num) if decay_lr else learning_rate
            adaptive_eps_manager.record_step(
                loss           = first_microbatch_loss_val,
                projected_grad = proj_grad_for_adaptive,
                learning_rate  = current_lr_for_eps_record,
                method_info    = {'success': first_microbatch_success_val}
            )
        step += 1
    elif train_method == 'improved_kronzo':
        # Improved directional KronZO with conservative updates
        accumulated_loss = 0.0
        accumulated_should_update = False
        accumulated_candidate_step_data = None
        accumulated_best_candidate_loss = 0.0
        
        for micro_step in range(gradient_accumulation_steps):
            current_zo_seed = np.random.randint(1_000_000_000)
            
            # Evaluate current directional candidates on SAME batch (X, Y)
            (baseline_loss_val, best_candidate_loss_val, best_direction_info, 
             should_update_val, candidate_step_data_val) = improved_kronzo_step(
                raw_model, X, Y,
                step=step,
                zo_random_seed=current_zo_seed,
                directional_q=directional_q,
                zo_eps=effective_zo_eps_to_use,
                lr=lr,  # Use current learning rate for candidate evaluation
                ctx_obj=ctx,
                strategy=kron_strategy,
                max_factor=kron_max_factor,
                kronzo_sampling_number=kronzo_sampling_number,
                b_dict=b_dict,
                step_interval=step_interval,
                loss_history=improved_directional_history,
                get_batch_fn=lambda: get_batch('train')  # CRITICAL: Use fresh batches for evaluation
            )
            
            accumulated_loss += baseline_loss_val / gradient_accumulation_steps
            
            # Use first micro-step's decision for the overall update
            if micro_step == 0:
                accumulated_should_update = should_update_val
                accumulated_candidate_step_data = candidate_step_data_val
                accumulated_best_candidate_loss = best_candidate_loss_val
                
                # Enhanced logging with history information
                if master_process and step % 50 == 0:  # Log every 50 steps
                    history_info = improved_directional_history.get_history_info()
                    improvement = best_direction_info.get('improvement', 0.0)
                    relative_improvement = best_direction_info.get('relative_improvement', 0.0)
                    used_fresh_batch = best_direction_info.get('used_fresh_batch', False)
                    candidate_range = best_direction_info.get('candidate_loss_range', 0.0)
                    num_candidates = best_direction_info.get('num_candidates', 0)
                    
                    print(f"  Improved KronZO step {step}: baseline={baseline_loss_val:.4f}, "
                          f"best_candidate={best_candidate_loss_val:.4f}, "
                          f"improvement={improvement:.4f} ({relative_improvement:.2%}), "
                          f"history_phase={history_info['phase']}, threshold={history_info['threshold']:.4f}, "
                          f"fresh_batch={used_fresh_batch}, range={candidate_range:.4f}, "
                          f"candidates={num_candidates}, accept={should_update_val}")
            
            # Get next batch for next iteration (but not after the last micro-step)
            if micro_step < gradient_accumulation_steps - 1:
                X, Y = get_batch('train')
        
        # Apply update if accepted and update loss history
        if accumulated_should_update:
            if use_momentum:
                improved_kronzo_update_momentum(
                    raw_model, optimizer, accumulated_candidate_step_data, exp_avg_m,
                    beta1=momentum_beta, master_process=master_process, weight_decay=weight_decay
                )
            else:
                improved_kronzo_update(
                    raw_model, optimizer, accumulated_candidate_step_data,
                    master_process=master_process, weight_decay=weight_decay
                )
            
            # CORRECTED: Update loss history with successful candidate loss ONLY on successful updates
            improved_directional_history.add_loss(accumulated_best_candidate_loss)
            
        elif master_process and step % 50 == 0:
            print(f"  Improved KronZO step {step}: Update rejected - candidate not better than history")
        
        # Set loss for logging (use baseline loss for consistency with evaluation)
        loss = torch.tensor(accumulated_loss, device=device, dtype=torch.float32)
        
        step += 1
    else:
        loss, X, Y = first_order_update(
            raw_model, optimizer, X, Y, 
            grad_clip, gradient_accumulation_steps, ctx,
            get_batch_fn=get_batch
        )

    # Fetch next batch for next iteration
    X, Y = get_batch('train')

    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        if device_type == 'cuda':
            memory_allocated = cuda.memory_allocated() / 1e9
            memory_reserved = cuda.memory_reserved() / 1e9
        else:
            process = psutil.Process()
            memory_allocated = process.memory_info().rss / 1e9
            memory_reserved = 0
        eps_info = {}
        if use_adaptive_eps and adaptive_eps_manager is not None:
            eps_stats = adaptive_eps_manager.get_stats()
            eps_info = {
                'eps': f"{eps_stats['eps']:.2e}", 
                'success': f"{eps_stats['success_rate']:.2f}"
            }
        pbar_info = {
            'loss': f'{lossf:.4f}',
            'lr': f'{lr:.2e}',
            'mem': f'{memory_allocated:.1f}GB',
            'mfu': f'{running_mfu*100:.1f}%',
            'step': step
        }
        if rank_adaptive and (train_method == 'lozo' or train_method == 'lozom' or train_method == 'dilozo'):
            current_rank = get_current_rank(iter_num, max_iters, min_rank, max_rank, rank_adaptive, rank_r)
            pbar_info['rank'] = f'{current_rank}/{max_rank}'
        if eps_info:
            pbar_info.update(eps_info)
        pbar.set_postfix(pbar_info)
        pbar.update(1)
    iter_num += 1
    local_iter_num += 1

    if iter_num > max_iters:
        break

pbar.close()

if ddp:
    destroy_process_group() 