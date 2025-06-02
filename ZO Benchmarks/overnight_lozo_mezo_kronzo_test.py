#!/usr/bin/env python3
"""
Overnight Directional ZO Methods Testing Script

Tests DiLoZO, DiMeZO, and DiKronZO with various parameter combinations:
- methods: ['dilozo', 'dimezo', 'dikronzo'] 
- momentum: [False, True] for all methods
- learning_rate: [6e-4, 1e-3, 3e-3, 6e-3]
- zo_eps: [5e-4, 1e-3, 2e-3, 5e-3]

Total: 96 experiments (3 methods × 2 momentum settings × 4 learning_rates × 4 zo_eps)

Runs each combination and extracts performance metrics.
All files are organized in a dedicated experiment directory.

Use with: nohup python overnight_lozo_mezo_kronzo_test.py > overnight_directional_results.log 2>&1 &

Output structure:
directional_zo_experiment_YYYYMMDD_HHMMSS/
├── experiment_info.txt           # Experiment configuration and info
├── summary.txt                   # Final results table
├── directional_exp_*.log         # Individual experiment logs
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
METHODS = ['dilozo', 'dimezo', 'dikronzo']  # All directional methods support momentum
MOMENTUM_SETTINGS = [False, True]  # Test with and without momentum
LEARNING_RATES = [6e-4, 1e-3, 3e-3, 6e-3]  # Appropriate for ZO pretraining
ZO_EPS_VALUES = [5e-4, 1e-3, 2e-3, 5e-3]   # Fine to coarse perturbations

# Base config (will be modified for each test)
BASE_CONFIG = {
    'dataset': 'shakespeare',
    'n_layer': 6,
    'n_head': 6, 
    'n_embd': 384,
    'max_iters': 3000,
    'warmup_iters': 300,
    'lr_decay_iters': 3000,
    'eval_interval': 500,
    'weight_decay': 1e-1,
    'momentum_beta': 0.9,
    'rank_r': 2,
    'step_interval': 50,
    'directional_q': 10,
    'kron_max_factor': 32,
    'kron_strategy': 'approx_square',
    'kronzo_sampling_number': 2,
    'dimezo_direct_movement': False,
    'dikronzo_direct_movement': False
}

def create_config_file(config_params, config_filename):
    """Create a temporary config file with specified parameters."""
    config_content = f'''# =============================================================================
# TEMPORARY CONFIG FOR OVERNIGHT DIRECTIONAL ZO TESTING
# =============================================================================
# Generated automatically by overnight_lozo_mezo_kronzo_test.py
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
wandb_run_name = '{config_params['train_method']}-test'

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
weight_decay = {config_params['weight_decay']}
beta1 = 0.9
beta2 = 0.95

# =============================================================================
# ZERO-ORDER OPTIMIZATION PARAMETERS
# =============================================================================
zo_eps = {config_params['zo_eps']}
zo_q = 1

# =============================================================================
# LOW-RANK PARAMETERS (for DiLoZO)
# =============================================================================
rank_r = {config_params['rank_r']}
step_interval = {config_params['step_interval']}
rank_adaptive = False
min_rank = 1
max_rank = 16
rank_strategy = 'linear'

# =============================================================================
# MOMENTUM SETTINGS
# =============================================================================
use_momentum = {config_params['use_momentum']}
momentum_beta = {config_params['momentum_beta']}

# =============================================================================
# DIRECTIONAL SELECTION PARAMETERS
# =============================================================================
directional_q = {config_params['directional_q']}

# =============================================================================
# KRONECKER PRODUCT PARAMETERS (for DiKronZO)
# =============================================================================
kron_strategy = '{config_params['kron_strategy']}'
kron_max_factor = {config_params['kron_max_factor']}
kronzo_sampling_number = {config_params['kronzo_sampling_number']}

# =============================================================================
# DIMEZO SPECIFIC PARAMETERS
# =============================================================================
dimezo_direct_movement = {config_params['dimezo_direct_movement']}

# =============================================================================
# DIKRONZO SPECIFIC PARAMETERS
# =============================================================================
dikronzo_direct_movement = {config_params['dikronzo_direct_movement']}

# =============================================================================
# SVD-LOZO SPECIFIC PARAMETERS (for compatibility)
# =============================================================================
svd_tau = 0.6
svd_max_rank = 16
use_full_svd = False

# =============================================================================
# ADAPTIVE ZO_EPS SETTINGS
# =============================================================================
use_adaptive_eps = False

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
# NUMERICAL STABILITY SETTINGS
# =============================================================================
use_zo_clipping = False
zo_grad_clip = 1.0
zo_param_clip = 10.0
zo_loss_threshold = 100.0
zo_recovery_lr_factor = 0.1
zo_max_recovery_attempts = 3
zo_verbose_debug = False
zo_verbose_clipping = False
'''
    
    with open(config_filename, 'w') as f:
        f.write(config_content)

def run_experiment(config_params, experiment_id, experiment_dir):
    """Run a single experiment with given parameters."""
    print(f"\n{'='*80}")
    print(f"EXPERIMENT {experiment_id}")
    print(f"{'='*80}")
    print(f"Parameters:")
    print(f"  method: {config_params['train_method']}")
    print(f"  learning_rate: {config_params['learning_rate']}")
    print(f"  zo_eps: {config_params['zo_eps']}")
    print(f"  use_momentum: {config_params['use_momentum']}")
    print(f"  min_lr: {config_params['min_lr']:.0e}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create temporary config file in experiment directory
    config_filename = os.path.join(experiment_dir, f"temp_config_{experiment_id}.py")
    create_config_file(config_params, config_filename)
    
    # Create log filename in experiment directory
    method_name = config_params['train_method']
    lr_str = f"{config_params['learning_rate']:.0e}".replace('-', '')
    eps_str = f"{config_params['zo_eps']:.0e}".replace('-', '')
    log_filename = os.path.join(experiment_dir, f"directional_exp_{experiment_id}_{method_name}_lr{lr_str}_eps{eps_str}.log")
    
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
    table.append("Directional ZO Methods Overnight Test Results")
    table.append("=" * 140)
    table.append(f"{'Exp':<4} {'Method':<8} {'LR':<8} {'zo_eps':<8} {'Momentum':<8} {'Best_Val':<9} {'Final_Train':<11} {'It/s':<6} {'Mem(GB)':<8} {'Steps':<6} {'Time(min)':<9}")
    table.append("-" * 140)
    
    # Sort by best validation loss
    successful_runs.sort(key=lambda x: x['best_val_loss'])
    
    for result in successful_runs:
        config = result['config']
        exp_id = result['experiment_id']
        method = config['train_method']
        lr = config['learning_rate']
        zo_eps = config['zo_eps']
        momentum = "Yes" if config['use_momentum'] else "No"
        best_val = result['best_val_loss']
        final_train = result['final_train_loss']
        iter_speed = result['avg_iter_per_sec']
        memory = result['peak_memory_gb']
        steps = result['final_step']
        duration = result.get('duration_minutes', 0)
        
        # Format values
        lr_str = f"{lr:.0e}"
        eps_str = f"{zo_eps:.0e}"
        best_val_str = f"{best_val:.4f}" if best_val else "N/A"
        final_train_str = f"{final_train:.4f}" if final_train else "N/A"
        iter_speed_str = f"{iter_speed:.2f}" if iter_speed else "N/A"
        memory_str = f"{memory:.1f}" if memory else "N/A"
        steps_str = str(steps) if steps else "N/A"
        duration_str = f"{duration:.1f}" if duration > 0 else "N/A"
        
        table.append(f"{exp_id:<4} {method:<8} {lr_str:<8} {eps_str:<8} {momentum:<8} {best_val_str:<9} {final_train_str:<11} {iter_speed_str:<6} {memory_str:<8} {steps_str:<6} {duration_str:<9}")
    
    table.append("-" * 140)
    table.append(f"Total successful runs: {len(successful_runs)} / {len(all_results)}")
    
    # Add analysis section
    if len(successful_runs) > 1:
        table.append("\nAnalysis:")
        table.append("-" * 50)
        
        # Best configuration
        best_run = successful_runs[0]  # Already sorted by best val loss
        best_config = best_run['config']
        table.append(f"Best validation loss: {best_run['best_val_loss']:.4f}")
        table.append(f"Best configuration:")
        table.append(f"  method: {best_config['train_method']}")
        table.append(f"  learning_rate: {best_config['learning_rate']}")
        table.append(f"  zo_eps: {best_config['zo_eps']}")
        
        # Method-wise analysis
        table.append(f"\nMethod-wise Best Performance:")
        for method in METHODS:
            method_runs = [r for r in successful_runs if r['config']['train_method'] == method]
            if method_runs:
                best_method_run = method_runs[0]  # Already sorted by best val loss
                momentum_status = "with momentum" if best_method_run['config']['use_momentum'] else "without momentum"
                table.append(f"  {method} ({momentum_status}): {best_method_run['best_val_loss']:.4f} (lr={best_method_run['config']['learning_rate']:.0e}, eps={best_method_run['config']['zo_eps']:.0e})")
        
        # Performance analysis
        fastest_run = max(successful_runs, key=lambda x: x['avg_iter_per_sec'] or 0)
        table.append(f"\nFastest training: {fastest_run['avg_iter_per_sec']:.2f} it/s ({fastest_run['config']['train_method']} - Exp {fastest_run['experiment_id']})")
        
        lowest_memory = min(successful_runs, key=lambda x: x['peak_memory_gb'] or float('inf'))
        table.append(f"Lowest memory usage: {lowest_memory['peak_memory_gb']:.1f} GB ({lowest_memory['config']['train_method']} - Exp {lowest_memory['experiment_id']})")
        
        # Momentum analysis for all methods
        table.append(f"\nMomentum Analysis:")
        for method in METHODS:
            no_momentum_runs = [r for r in successful_runs if r['config']['train_method'] == method and not r['config']['use_momentum']]
            momentum_runs = [r for r in successful_runs if r['config']['train_method'] == method and r['config']['use_momentum']]
            
            if no_momentum_runs and momentum_runs:
                best_no_momentum = min(r['best_val_loss'] for r in no_momentum_runs)
                best_momentum = min(r['best_val_loss'] for r in momentum_runs)
                improvement = ((best_no_momentum - best_momentum) / best_no_momentum) * 100
                table.append(f"  {method}: {best_no_momentum:.4f} vs {method}-M: {best_momentum:.4f} (improvement: {improvement:.2f}%)")
            elif no_momentum_runs:
                best_no_momentum = min(r['best_val_loss'] for r in no_momentum_runs)
                table.append(f"  {method}: {best_no_momentum:.4f} (no momentum runs completed)")
            elif momentum_runs:
                best_momentum = min(r['best_val_loss'] for r in momentum_runs)
                table.append(f"  {method}-M: {best_momentum:.4f} (no non-momentum runs completed)")
    
    return "\n".join(table)

def main():
    """Main function to run all experiments."""
    print("Directional ZO Methods Overnight Testing Script")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Calculate total experiments
    total_experiments = len(METHODS) * len(MOMENTUM_SETTINGS) * len(LEARNING_RATES) * len(ZO_EPS_VALUES)
    
    print(f"Total experiments: {total_experiments}")
    print(f"Methods: {METHODS}")
    print(f"Momentum settings: {MOMENTUM_SETTINGS}")
    print(f"Learning rates: {LEARNING_RATES}")
    print(f"zo_eps values: {ZO_EPS_VALUES}")
    print(f"\nMethod details:")
    print(f"- dilozo: DiLoZO (with and without momentum)")
    print(f"- dimezo: DiMeZO (with and without momentum)")
    print(f"- dikronzo: DiKronZO (with and without momentum)")
    print(f"\nParameter rationale:")
    print(f"- Learning rates: Appropriate for ZO methods in pretraining")
    print(f"- zo_eps: Fine to coarse perturbations")
    print(f"- min_lr: Set dynamically to lr/10 for proper learning rate decay")
    print(f"- momentum: All methods now support momentum (beta={BASE_CONFIG['momentum_beta']})")
    
    # Verify environment
    if not os.path.exists('nanoGPT/zo_train.py'):
        print("ERROR: nanoGPT/zo_train.py not found. Please run from the correct directory.")
        sys.exit(1)
    
    # Create main experiment directory
    experiment_dir = f"directional_zo_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Create subdirectories for organization
    results_dir = os.path.join(experiment_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Experiment directory: {experiment_dir}")
    print(f"All files will be saved in: {os.path.abspath(experiment_dir)}")
    
    all_results = []
    experiment_id = 1
    
    # Run all combinations
    for method, use_momentum, lr, zo_eps in product(METHODS, MOMENTUM_SETTINGS, LEARNING_RATES, ZO_EPS_VALUES):
        # Create config for this experiment
        config = BASE_CONFIG.copy()
        config.update({
            'train_method': method,
            'learning_rate': lr,
            'zo_eps': zo_eps,
            'min_lr': lr / 10,  # Set min_lr to lr/10
            'use_momentum': use_momentum
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
        f.write(f"Directional ZO Methods Overnight Testing Experiment\n")
        f.write(f"==================================================\n\n")
        f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total experiments: {total_experiments}\n\n")
        f.write(f"Methods tested:\n")
        f.write(f"- dilozo: DiLoZO (with and without momentum)\n")
        f.write(f"- dimezo: DiMeZO (with and without momentum)\n")
        f.write(f"- dikronzo: DiKronZO (with and without momentum)\n\n")
        f.write(f"Parameter combinations:\n")
        f.write(f"- methods: {METHODS}\n")
        f.write(f"- momentum: {MOMENTUM_SETTINGS}\n")
        f.write(f"- learning_rate: {LEARNING_RATES}\n")
        f.write(f"- zo_eps: {ZO_EPS_VALUES}\n\n")
        f.write(f"Base configuration:\n")
        for key, value in BASE_CONFIG.items():
            f.write(f"- {key}: {value}\n")
        f.write(f"\nDirectory structure:\n")
        f.write(f"- {experiment_dir}/\n")
        f.write(f"  ├── experiment_info.txt (this file)\n")
        f.write(f"  ├── summary.txt (final results table)\n")
        f.write(f"  ├── directional_exp_*.log (individual experiment logs)\n")
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