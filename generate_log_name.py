#!/usr/bin/env python3
"""
Generate intelligent log file names based on training configuration.
This script extracts relevant parameters from config files and creates descriptive log names.
"""

import sys
import os
from datetime import datetime

def extract_config_params(config_file):
    """Extract relevant parameters from config file"""
    # Create a clean namespace to execute the config
    config_namespace = {}
    
    # Execute the config file to get all parameters
    with open(config_file, 'r') as f:
        config_code = f.read()
    
    exec(config_code, config_namespace)
    
    # Extract relevant parameters
    params = {}
    
    # Core parameters
    params['train_method'] = config_namespace.get('train_method', 'unknown')
    params['dataset'] = config_namespace.get('dataset', 'unknown')
    
    # Momentum parameters
    params['use_momentum'] = config_namespace.get('use_momentum', False)
    
    # Query budget parameters
    params['zo_q'] = config_namespace.get('zo_q', 1)
    params['directional_q'] = config_namespace.get('directional_q', 1)
    
    # Rank parameters (for LoZO/DiLoZO)
    params['rank_r'] = config_namespace.get('rank_r', 1)
    params['rank_adaptive'] = config_namespace.get('rank_adaptive', False)
    params['rank_strategy'] = config_namespace.get('rank_strategy', 'linear')
    
    # Kronecker parameters (for KronZO/DiKronZO)
    params['kron_strategy'] = config_namespace.get('kron_strategy', 'approx_square')
    
    # SVD parameters (for SVD-LoZO)
    params['svd_tau'] = config_namespace.get('svd_tau', 0.6)
    params['use_full_svd'] = config_namespace.get('use_full_svd', False)
    
    # DiMeZO parameters
    params['dimezo_direct_movement'] = config_namespace.get('dimezo_direct_movement', False)
    
    return params

def generate_log_name(params):
    """Generate intelligent log file name based on parameters"""
    
    # Start with method and dataset
    method = params.get('train_method', 'unknown')
    dataset = params.get('dataset', 'unknown')
    
    # Create base name
    log_parts = [method, dataset]
    
    # Add momentum suffix if enabled
    if params.get('use_momentum', False):
        log_parts.append('m')
    
    # Determine if method uses directional selection or averaging
    directional_methods = ['dimezo', 'dilozo', 'dikronzo']
    is_directional = method in directional_methods
    
    # Add query budget information
    if is_directional:
        # Directional methods use directional_q
        q_value = params.get('directional_q', 1)
        log_parts.append(f'dir{q_value}')
    else:
        # Averaging methods use zo_q
        q_value = params.get('zo_q', 1)
        if q_value > 1:  # Only add if not default
            log_parts.append(f'avg{q_value}')
    
    # Add rank information for LoZO-based methods
    lozo_methods = ['lozo', 'lozom', 'svdlozo', 'dilozo']
    if method in lozo_methods:
        if params.get('rank_adaptive', False):
            # For adaptive rank, show strategy
            rank_strategy = params.get('rank_strategy', 'linear')
            log_parts.append(f'rank-{rank_strategy}')
        else:
            # For fixed rank, show value
            rank_value = params.get('rank_r', 1)
            log_parts.append(f'rank{rank_value}')
    
    # Add Kronecker strategy for KronZO methods
    kronzo_methods = ['kronzo', 'dikronzo']
    if method in kronzo_methods:
        kron_strategy = params.get('kron_strategy', 'approx_square')
        log_parts.append(f'kron-{kron_strategy}')
    
    # Add SVD information for SVD-LoZO
    if method == 'svdlozo':
        svd_type = 'full' if params.get('use_full_svd', False) else 'rand'
        svd_tau = params.get('svd_tau', 0.6)
        log_parts.append(f'svd-{svd_type}-tau{svd_tau}')
    
    # Add DiMeZO movement strategy
    if method == 'dimezo':
        movement = 'direct' if params.get('dimezo_direct_movement', False) else 'grad'
        log_parts.append(movement)
    
    # Join all parts with hyphens
    log_name = '-'.join(log_parts)
    
    return log_name

def main():
    if len(sys.argv) != 2:
        print("Usage: python generate_log_name.py <config_file>")
        sys.exit(1)
    
    config_file = sys.argv[1]
    
    if not os.path.exists(config_file):
        print(f"Error: Config file '{config_file}' not found")
        sys.exit(1)
    
    try:
        # Extract parameters from config
        params = extract_config_params(config_file)
        
        # Generate log name
        log_name = generate_log_name(params)
        
        # Print the log name (this will be captured by the shell script)
        print(log_name)
        
    except Exception as e:
        print(f"Error generating log name: {e}", file=sys.stderr)
        # Fallback to simple name
        config_basename = os.path.basename(config_file).replace('.py', '')
        print(config_basename)

if __name__ == "__main__":
    main() 