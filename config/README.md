# NanoGPT Configuration System

This directory contains comprehensive configuration files for training nanoGPT with various optimization methods.

## Overview

The configuration system has been simplified to use **4 main config files** that contain all parameters for different optimization families:

1. **`train_first_order_config.py`** - First-order methods (Adam, SGD)
2. **`train_mezo_config.py`** - MeZO-based methods (MeZO, MeZO-M, DiMeZO)
3. **`train_lozo_config.py`** - LoZO-based methods (LoZO, LoZO-M, SVD-LoZO, DiLoZO)
4. **`train_kronzo_config.py`** - KronZO-based methods (KronZO, DiKronZO)

## Usage

### Quick Start

```bash
# Train with LoZO on single GPU
bash lozo_train.sh lozo single

# Train with MeZO on multiple GPUs
bash lozo_train.sh mezo multi

# Train with first-order methods (Adam/SGD)
bash lozo_train.sh first-order single

# Train with KronZO
bash lozo_train.sh kronzo single
```

### Customizing Training

To customize your training, edit the appropriate config file:

1. **Choose your method**: Set `train_method` parameter
2. **Choose your dataset**: Set `dataset` parameter  
3. **Adjust parameters**: Modify any other parameters as needed

## Configuration Parameters

### Common Parameters (All Methods)

- **`train_method`**: Optimization method to use
- **`dataset`**: Dataset ('shakespeare', 'openwebtext', 'gpt2')
- **`init_from`**: Model initialization ('scratch', 'resume', 'gpt2*')
- **`out_dir`**: Output directory for checkpoints
- **`learning_rate`**: Maximum learning rate
- **`max_iters`**: Total training iterations
- **`batch_size`**: Micro-batch size per GPU
- **`block_size`**: Context length (sequence length)

### Model Architecture

- **`n_layer`**: Number of transformer layers
- **`n_head`**: Number of attention heads  
- **`n_embd`**: Embedding dimension
- **`dropout`**: Dropout rate
- **`bias`**: Use bias in layers

### Zero-Order Specific Parameters

#### Basic ZO Parameters
- **`zo_eps`**: Perturbation size (1e-4 to 1e-2)
- **`zo_q`**: Number of gradient estimates to average (LoZO/MeZO)

#### Low-Rank Parameters (LoZO family)
- **`rank_r`**: Fixed rank for U and V matrices
- **`step_interval`**: Interval for updating V matrices

#### Rank-Adaptive Settings
- **`rank_adaptive`**: Enable adaptive rank scheduling
- **`min_rank`**: Starting rank
- **`max_rank`**: Ending rank
- **`rank_strategy`**: Scheduling strategy ('linear', 'exponential', 'cosine')

#### Momentum Settings
- **`use_momentum`**: Enable momentum variants
- **`momentum_beta`**: Momentum coefficient

#### Directional Selection (DiMeZO/DiLoZO)
- **`directional_q`**: Number of directions to try
- **`dimezo_direct_movement`**: Movement strategy (DiMeZO only)

#### Kronecker Parameters (KronZO family)
- **`kron_strategy`**: Factorization strategy ('approx_square', 'fixed_factor', 'power2')
- **`kron_max_factor`**: Maximum factor size

#### SVD Parameters (SVD-LoZO)
- **`svd_tau`**: Threshold for adaptive rank selection
- **`svd_max_rank`**: Maximum rank for SVD
- **`use_full_svd`**: Use full vs randomized SVD

#### Adaptive Perturbation
- **`use_adaptive_eps`**: Enable adaptive perturbation size
- **`adaptive_eps_window`**: Window size for tracking
- **`adaptive_eps_lr_coupling`**: Coupling with learning rate

## Method Selection Guide

### First-Order Methods (`train_first_order_config.py`)
- **Adam** (`train_method = 'adam'`): Standard adaptive gradient method
- **SGD** (`train_method = 'sgd'`): Stochastic gradient descent

### MeZO Family (`train_mezo_config.py`)
- **MeZO** (`train_method = 'mezo'`): Memory-efficient zero-order optimization
- **MeZO-M** (`train_method = 'mezom'`): MeZO with momentum
- **DiMeZO** (`train_method = 'dimezo'`): Directional MeZO with direction selection

### LoZO Family (`train_lozo_config.py`)
- **LoZO** (`train_method = 'lozo'`): Low-rank zero-order optimization
- **LoZO-M** (`train_method = 'lozom'`): LoZO with momentum
- **SVD-LoZO** (`train_method = 'svdlozo'`): SVD-guided low-rank optimization
- **DiLoZO** (`train_method = 'dilozo'`): Directional LoZO with direction selection

### KronZO Family (`train_kronzo_config.py`)
- **KronZO** (`train_method = 'kronzo'`): Kronecker product zero-order optimization
- **DiKronZO** (`train_method = 'dikronzo'`): Directional KronZO with direction selection

## Dataset Configurations

Each config file includes commented dataset-specific configurations:

### Shakespeare (Small, Fast)
- Small model (6 layers, 384 dim)
- Short context (256 tokens)
- Fast training (5K iterations)

### OpenWebText (Medium Scale)
- Medium model (12 layers, 768 dim)
- Long context (1024 tokens)
- Extended training (600K iterations)

### GPT-2 (Large Scale)
- GPT-2 124M architecture
- Long context (1024 tokens)
- Full-scale training (600K iterations)

## Examples

### Basic Training
```bash
# Quick test with LoZO on Shakespeare
bash lozo_train.sh lozo single

# Production training with MeZO on OpenWebText
bash lozo_train.sh mezo multi
```

### Advanced Configurations

Edit the config file to enable advanced features:

```python
# Enable rank-adaptive LoZO
rank_adaptive = True
min_rank = 1
max_rank = 16
rank_strategy = 'exponential'

# Enable momentum
use_momentum = True
momentum_beta = 0.9

# Enable adaptive perturbation
use_adaptive_eps = True
adaptive_eps_window = 20
```

## Migration from Old Configs

Old config files have been moved to `old_configs_backup/` directory. To migrate:

1. Identify your old config's method and dataset
2. Use the appropriate new config file
3. Set `train_method` and `dataset` parameters
4. Copy any custom parameters you had

## Troubleshooting

### Common Issues

1. **Config not found**: Ensure you're using the correct config type name
2. **Method not recognized**: Check `train_method` spelling in config file
3. **GPU memory issues**: Reduce `batch_size` or `block_size`
4. **Convergence issues**: Adjust `learning_rate` or `zo_eps`

### Getting Help

```bash
# Show usage
bash lozo_train.sh invalid_name

# Check config file syntax
python -c "exec(open('config/train_lozo_config.py').read())"
```

## File Structure

```
config/
├── README.md                      # This file
├── train_first_order_config.py    # Adam, SGD
├── train_mezo_config.py           # MeZO, MeZO-M, DiMeZO
├── train_lozo_config.py           # LoZO, LoZO-M, SVD-LoZO, DiLoZO
├── train_kronzo_config.py         # KronZO, DiKronZO
├── old_configs_backup/            # Backup of old config files
└── [other files...]               # Evaluation and utility configs
``` 