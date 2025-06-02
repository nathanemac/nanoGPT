#!/usr/bin/env python3
"""
Overnight DiKronZO Testing Script

Tests DiKronZO with various parameter combinations:
- kronzo_sampling_number: [1, 2, 5, 10]
- step_interval: [1, 5, 10, 20]
- directional_q: [1, 5, 10]

Runs each combination and extracts performance metrics.
All files are organized in a dedicated experiment directory.

Use with: nohup python overnight_dikronzo_test.py > overnight_results.log 2>&1 &

Output structure:
dikronzo_experiment_YYYYMMDD_HHMMSS/
├── experiment_info.txt           # Experiment configuration and info
├── summary.txt                   # Final results table
├── dikronzo_exp_*.log            # Individual experiment logs
└── results/
    ├── all_results.json          # Complete results in JSON format
    └── result_*.json             # Individual experiment results
"""

import os
import sys
import time
import subprocess
import re
import json
from datetime import datetime
from itertools import product
import shutil

# Test parameter combinations
KRONZO_SAMPLING_NUMBERS = [1, 2, 5, 10]
STEP_INTERVALS = [1, 5, 10, 20] 
DIRECTIONAL_QS = [1, 5, 10]

# Base config (will be modified for each test)
BASE_CONFIG = {
    'train_method': 'dikronzo',
    'dataset': 'shakespeare',
    'n_layer': 6,
    'n_head': 6, 
    'n_embd': 384,
    'learning_rate': 1e-3,
    'max_iters': 5000,
    'zo_eps': 1e-3,
    'kron_max_factor': 32,
    'use_momentum': True,
    'momentum_beta': 0.9,
    'warmup_iters': 500,
    'lr_decay_iters': 5000,
    'min_lr': 1e-4,
    'eval_interval': 500,
    'dikronzo_direct_movement': False  # Use gradient estimation mode
}

def create_config_file(config_params, config_filename):
    """Create a temporary config file with specified parameters."""
    config_content = f'''# =============================================================================
# TEMPORARY CONFIG FOR OVERNIGHT DIKRONZO TESTING
# =============================================================================
# Generated automatically by overnight_dikronzo_test.py
# Test parameters: {config_params}

import torch

# =============================================================================
# TRAINING METHOD SELECTION
# =============================================================================
train_method = '{config_params['train_method']}'

# =============================================================================
# DATASET AND MODEL SELECTION
# =============================================================================
dataset = '{config_params['dataset']}'
init_from = 'scratch'

# =============================================================================
# OUTPUT AND LOGGING SETTINGS
# =============================================================================
out_dir = 'out-{config_params['train_method']}'
eval_interval = {config_params['eval_interval']}
log_interval = 1
eval_iters = 50
eval_only = False
always_save_checkpoint = True
wandb_log = False
wandb_project = 'nanogpt'
wandb_run_name = 'dikronzo-test'

# =============================================================================
# DATA CONFIGURATION
# =============================================================================
gradient_accumulation_steps = 1
batch_size = 32
block_size = 256

# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================
n_layer = {config_params['n_layer']}
n_head = {config_params['n_head']}
n_embd = {config_params['n_embd']}
dropout = 0.0
bias = False

# =============================================================================
# OPTIMIZER SETTINGS
# =============================================================================
learning_rate = {config_params['learning_rate']}
max_iters = {config_params['max_iters']}
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# =============================================================================
# ZERO-ORDER OPTIMIZATION PARAMETERS
# =============================================================================
zo_eps = {config_params['zo_eps']}

# =============================================================================
# KRONECKER PRODUCT PARAMETERS
# =============================================================================
kron_strategy = 'approx_square'
kron_max_factor = {config_params['kron_max_factor']}
kronzo_sampling_number = {config_params['kronzo_sampling_number']}
step_interval = {config_params['step_interval']}

# =============================================================================
# MOMENTUM SETTINGS
# =============================================================================
use_momentum = {config_params['use_momentum']}
momentum_beta = {config_params['momentum_beta']}

# =============================================================================
# DIRECTIONAL SELECTION PARAMETERS (for DiKronZO)
# =============================================================================
directional_q = {config_params['directional_q']}

# =============================================================================
# ADAPTIVE ZO_EPS SETTINGS
# =============================================================================
use_adaptive_eps = False

# =============================================================================
# ZERO-ORDER QUERY BUDGET PARAMETERS
# =============================================================================
zo_q = 1

# =============================================================================
# DIMEZO SPECIFIC PARAMETERS (for compatibility)
# =============================================================================
dimezo_direct_movement = False

# =============================================================================
# LOZO-SPECIFIC PARAMETERS (for compatibility)
# =============================================================================
rank_r = 4
rank_adaptive = False
min_rank = 1
max_rank = 16
rank_strategy = 'linear'

# =============================================================================
# SVD-LOZO SPECIFIC PARAMETERS (for compatibility)
# =============================================================================
svd_tau = 0.6
svd_max_rank = 16
use_full_svd = False

# =============================================================================
# LEARNING RATE SCHEDULE
# =============================================================================
decay_lr = True
warmup_iters = {config_params['warmup_iters']}
lr_decay_iters = {config_params['lr_decay_iters']}
min_lr = {config_params['min_lr']}

# =============================================================================
# SYSTEM SETTINGS
# =============================================================================
device = 'cuda'
dtype = 'bfloat16'
compile = True
backend = 'nccl'

# =============================================================================
# DIKRONZO SPECIFIC PARAMETERS
# =============================================================================
dikronzo_direct_movement = {config_params.get('dikronzo_direct_movement', False)}
'''
    
    with open(config_filename, 'w') as f:
        f.write(config_content)

def run_experiment(config_params, experiment_id, experiment_dir):
    """Run a single experiment with given parameters."""
    print(f"\n{'='*80}")
    print(f"EXPERIMENT {experiment_id}")
    print(f"{'='*80}")
    print(f"Parameters:")
    print(f"  kronzo_sampling_number: {config_params['kronzo_sampling_number']}")
    print(f"  step_interval: {config_params['step_interval']}")
    print(f"  directional_q: {config_params['directional_q']}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create temporary config file in experiment directory
    config_filename = os.path.join(experiment_dir, f"temp_config_{experiment_id}.py")
    create_config_file(config_params, config_filename)
    
    # Create log filename in experiment directory
    log_filename = os.path.join(experiment_dir, f"dikronzo_exp_{experiment_id}_sampling{config_params['kronzo_sampling_number']}_interval{config_params['step_interval']}_dirq{config_params['directional_q']}.log")
    
    try:
        # Run the training
        cmd = [
            'python', 'nanoGPT/zo_train.py', 
            '--method_config_file', config_filename
        ]
        
        start_time = time.time()
        
        with open(log_filename, 'w') as log_file:
            # Write experiment info to log
            log_file.write(f"EXPERIMENT {experiment_id}\n")
            log_file.write(f"Parameters: {config_params}\n")
            log_file.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write(f"Command: {' '.join(cmd)}\n")
            log_file.write("-" * 80 + "\n\n")
            log_file.flush()
            
            # Run training
            result = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, text=True)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Extract results from log
        results = extract_results_from_log(log_filename)
        results['duration_minutes'] = duration / 60
        results['exit_code'] = result.returncode
        results['experiment_id'] = experiment_id
        results['config'] = config_params.copy()
        
        print(f"Completed in {duration/60:.1f} minutes")
        print(f"Exit code: {result.returncode}")
        if results['best_val_loss'] is not None:
            print(f"Best validation loss: {results['best_val_loss']:.4f}")
        if results['final_train_loss'] is not None:
            print(f"Final training loss: {results['final_train_loss']:.4f}")
        
        return results
        
    except Exception as e:
        print(f"Error running experiment {experiment_id}: {e}")
        return {
            'experiment_id': experiment_id,
            'config': config_params.copy(),
            'error': str(e),
            'exit_code': -1
        }
    finally:
        # Clean up temporary config file
        if os.path.exists(config_filename):
            os.remove(config_filename)

def extract_results_from_log(log_filename):
    """Extract key metrics from a training log file."""
    results = {
        'best_val_loss': None,
        'final_train_loss': None,
        'final_val_loss': None,
        'avg_iter_per_sec': None,
        'peak_memory_gb': None,
        'final_step': None
    }
    
    if not os.path.exists(log_filename):
        return results
    
    try:
        with open(log_filename, 'r') as f:
            content = f.read()
        
        # Extract validation losses (find the best one)
        val_losses = []
        val_loss_pattern = r'step \d+: train loss ([\d.]+), val loss ([\d.]+)'
        for match in re.finditer(val_loss_pattern, content):
            train_loss = float(match.group(1))
            val_loss = float(match.group(2))
            val_losses.append((train_loss, val_loss))
        
        if val_losses:
            # Best validation loss
            results['best_val_loss'] = min(val_losses, key=lambda x: x[1])[1]
            # Final losses
            results['final_train_loss'] = val_losses[-1][0]
            results['final_val_loss'] = val_losses[-1][1]
        
        # Extract iteration speed from progress bars
        iter_speeds = []
        speed_pattern = r'(\d+\.\d+)s/it'
        for match in re.finditer(speed_pattern, content):
            seconds_per_iter = float(match.group(1))
            if seconds_per_iter > 0:
                iter_speeds.append(1.0 / seconds_per_iter)
        
        if iter_speeds:
            # Average of last 10 measurements for stability
            results['avg_iter_per_sec'] = sum(iter_speeds[-10:]) / len(iter_speeds[-10:])
        
        # Extract memory usage
        memory_pattern = r'mem=(\d+\.\d+)GB'
        memory_values = []
        for match in re.finditer(memory_pattern, content):
            memory_gb = float(match.group(1))
            memory_values.append(memory_gb)
        
        if memory_values:
            results['peak_memory_gb'] = max(memory_values)
        
        # Extract final step
        step_pattern = r'step=(\d+)'
        steps = []
        for match in re.finditer(step_pattern, content):
            steps.append(int(match.group(1)))
        
        if steps:
            results['final_step'] = max(steps)
            
    except Exception as e:
        print(f"Error extracting results from {log_filename}: {e}")
    
    return results

def format_results_table(all_results):
    """Format results in a nice table."""
    # Filter successful runs
    successful_runs = [r for r in all_results if r.get('exit_code') == 0 and r.get('best_val_loss') is not None]
    
    if not successful_runs:
        return "No successful runs to display."
    
    # Create table header
    table = []
    table.append("DiKronZO Overnight Test Results")
    table.append("=" * 120)
    table.append(f"{'Exp':<4} {'Sampling':<8} {'Interval':<8} {'Dir_Q':<6} {'Best_Val':<9} {'Final_Train':<11} {'It/s':<6} {'Mem(GB)':<8} {'Steps':<6} {'Time(min)':<9}")
    table.append("-" * 120)
    
    # Sort by best validation loss
    successful_runs.sort(key=lambda x: x['best_val_loss'])
    
    for result in successful_runs:
        config = result['config']
        exp_id = result['experiment_id']
        sampling = config['kronzo_sampling_number']
        interval = config['step_interval']
        dir_q = config['directional_q']
        best_val = result['best_val_loss']
        final_train = result['final_train_loss']
        iter_speed = result['avg_iter_per_sec']
        memory = result['peak_memory_gb']
        steps = result['final_step']
        duration = result.get('duration_minutes', 0)
        
        # Format values
        best_val_str = f"{best_val:.4f}" if best_val else "N/A"
        final_train_str = f"{final_train:.4f}" if final_train else "N/A"
        iter_speed_str = f"{iter_speed:.2f}" if iter_speed else "N/A"
        memory_str = f"{memory:.1f}" if memory else "N/A"
        steps_str = str(steps) if steps else "N/A"
        duration_str = f"{duration:.1f}" if duration > 0 else "N/A"
        
        table.append(f"{exp_id:<4} {sampling:<8} {interval:<8} {dir_q:<6} {best_val_str:<9} {final_train_str:<11} {iter_speed_str:<6} {memory_str:<8} {steps_str:<6} {duration_str:<9}")
    
    table.append("-" * 120)
    table.append(f"Total successful runs: {len(successful_runs)} / {len(all_results)}")
    
    # Add analysis section
    if len(successful_runs) > 1:
        table.append("\nAnalysis:")
        table.append("-" * 40)
        
        # Best configuration
        best_run = successful_runs[0]  # Already sorted by best val loss
        best_config = best_run['config']
        table.append(f"Best validation loss: {best_run['best_val_loss']:.4f}")
        table.append(f"Best configuration:")
        table.append(f"  kronzo_sampling_number: {best_config['kronzo_sampling_number']}")
        table.append(f"  step_interval: {best_config['step_interval']}")
        table.append(f"  directional_q: {best_config['directional_q']}")
        
        # Performance analysis
        fastest_run = max(successful_runs, key=lambda x: x['avg_iter_per_sec'] or 0)
        table.append(f"\nFastest training: {fastest_run['avg_iter_per_sec']:.2f} it/s (Exp {fastest_run['experiment_id']})")
        
        lowest_memory = min(successful_runs, key=lambda x: x['peak_memory_gb'] or float('inf'))
        table.append(f"Lowest memory usage: {lowest_memory['peak_memory_gb']:.1f} GB (Exp {lowest_memory['experiment_id']})")
    
    return "\n".join(table)

def main():
    """Main function to run all experiments."""
    print("DiKronZO Overnight Testing Script")
    print("=" * 50)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total experiments: {len(KRONZO_SAMPLING_NUMBERS) * len(STEP_INTERVALS) * len(DIRECTIONAL_QS)}")
    
    # Verify environment
    if not os.path.exists('nanoGPT/zo_train.py'):
        print("ERROR: nanoGPT/zo_train.py not found. Please run from the correct directory.")
        sys.exit(1)
    
    # Create main experiment directory
    experiment_dir = f"dikronzo_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Create subdirectories for organization
    results_dir = os.path.join(experiment_dir, "results")
    logs_dir = os.path.join(experiment_dir, "logs")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    print(f"Experiment directory: {experiment_dir}")
    print(f"All files will be saved in: {os.path.abspath(experiment_dir)}")
    
    all_results = []
    experiment_id = 1
    
    # Run all combinations
    for sampling, interval, dir_q in product(KRONZO_SAMPLING_NUMBERS, STEP_INTERVALS, DIRECTIONAL_QS):
        # Create config for this experiment
        config = BASE_CONFIG.copy()
        config.update({
            'kronzo_sampling_number': sampling,
            'step_interval': interval,
            'directional_q': dir_q
        })
        
        # Run experiment (pass experiment_dir for file organization)
        result = run_experiment(config, experiment_id, experiment_dir)
        all_results.append(result)
        
        # Save individual result in results subdirectory
        result_file = os.path.join(results_dir, f"result_{experiment_id}.json")
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        experiment_id += 1
        
        # Brief pause between experiments
        time.sleep(2)
    
    # Save all results in results subdirectory
    all_results_file = os.path.join(results_dir, "all_results.json")
    with open(all_results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Generate and save summary table in main experiment directory
    summary_table = format_results_table(all_results)
    summary_file = os.path.join(experiment_dir, "summary.txt")
    with open(summary_file, 'w') as f:
        f.write(summary_table)
    
    # Create experiment info file
    info_file = os.path.join(experiment_dir, "experiment_info.txt")
    with open(info_file, 'w') as f:
        f.write(f"DiKronZO Overnight Testing Experiment\n")
        f.write(f"=====================================\n\n")
        f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total experiments: {len(KRONZO_SAMPLING_NUMBERS) * len(STEP_INTERVALS) * len(DIRECTIONAL_QS)}\n\n")
        f.write(f"Parameter combinations:\n")
        f.write(f"- kronzo_sampling_number: {KRONZO_SAMPLING_NUMBERS}\n")
        f.write(f"- step_interval: {STEP_INTERVALS}\n")
        f.write(f"- directional_q: {DIRECTIONAL_QS}\n\n")
        f.write(f"Base configuration:\n")
        for key, value in BASE_CONFIG.items():
            f.write(f"- {key}: {value}\n")
        f.write(f"\nDirectory structure:\n")
        f.write(f"- {experiment_dir}/\n")
        f.write(f"  ├── experiment_info.txt (this file)\n")
        f.write(f"  ├── summary.txt (final results table)\n")
        f.write(f"  ├── dikronzo_exp_*.log (individual experiment logs)\n")
        f.write(f"  └── results/\n")
        f.write(f"      ├── all_results.json (complete results)\n")
        f.write(f"      └── result_*.json (individual results)\n")
    
    # Print final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(summary_table)
    print(f"\nAll results saved to: {experiment_dir}/")
    print(f"Summary: {summary_file}")
    print(f"Detailed results: {all_results_file}")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main() 