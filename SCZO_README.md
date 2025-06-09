# SCZO: Secant-Constrained Zero-Order Optimization

## Overview

**SCZO (Secant-Constrained Zero-Order)** is a novel optimization method for training large language models without backward passes. It uses secant constraints to update gradient estimates while maintaining memory efficiency and requiring only 2 forward passes per iteration.

## Key Innovation

SCZO enforces the **secant constraint**:
```
g_{k+1}^T s_k = y_k
```

Where:
- `g_{k+1}`: Updated gradient estimate
- `s_k = θ_{k+1} - θ_k`: Parameter step
- `y_k = f(θ_{k+1}, B_k) - f(θ_k, B_k)`: Loss difference on the same batch

This constraint ensures that the gradient estimate is consistent with the observed loss change.

## Algorithm

```
For each iteration k:
1. Sample fresh mini-batch B_k
2. Compute loss_before = f(θ_k, B_k)
3. Apply step: θ_{k+1} = θ_k - α * g_k
4. Compute loss_after = f(θ_{k+1}, B_k)
5. Update gradient estimates using secant constraint:
   g_{k+1} = g_k + λ * s_k
   where λ = (y_k - g_k^T s_k) / ||s_k||²
6. Every T_bootstrap steps: Bootstrap g_k using ZO estimation
```

## Key Features

### ✅ **Memory Efficient**
- Stores only gradient estimates (FP16)
- No need for full Hessian or momentum buffers
- ~2-2.5 floats per parameter total memory

### ✅ **Minimal Forward Passes**
- Only 2 forward passes per iteration
- Much more efficient than multi-candidate methods
- Fresh mini-batch every iteration prevents overfitting

### ✅ **Robust Gradient Estimation**
- Periodic ZO gradient bootstrapping using improved KronZO
- Secant constraint maintains gradient consistency
- EMA smoothing for stability

### ✅ **Non-Destructive Integration**
- Completely separate from existing methods
- Own config file and utilities
- Can be easily enabled/disabled

## Files Structure

```
nanoGPT/
├── sczo_utils.py                    # Core SCZO implementation
├── config/train_sczo_config.py      # SCZO configuration
├── test_sczo.py                     # Unit tests
├── train_sczo_example.py            # Example training script
├── SCZO_README.md                   # This documentation
└── zo_train.py                      # Modified to include SCZO
```

## Usage

### 1. Basic Training

```bash
# Using the SCZO config
python zo_train.py --config config/train_sczo_config.py

# Or using the example script
python train_sczo_example.py
```

### 2. Custom Configuration

Create your own config file based on `config/train_sczo_config.py`:

```python
# Custom SCZO config
train_method = 'sczo'

# SCZO hyperparameters
sczo_alpha = 1e-3           # Step size
sczo_beta = 0.9             # EMA factor for gradient updates
sczo_eps = 1e-8             # Numerical damping
sczo_bootstrap_interval = 20 # ZO gradient bootstrap frequency

# Model and dataset settings
dataset = 'shakespeare'
n_layer = 6
n_head = 6
n_embd = 384
```

### 3. Integration with Existing Configs

Add SCZO to any existing config:

```python
# Change training method
train_method = 'sczo'

# Add SCZO parameters
sczo_alpha = 1e-3
sczo_beta = 0.9
sczo_eps = 1e-8
sczo_bootstrap_interval = 20
```

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sczo_alpha` | 1e-3 | Step size for parameter updates |
| `sczo_beta` | 0.9 | EMA factor for gradient estimate updates |
| `sczo_eps` | 1e-8 | Damping parameter for numerical stability |
| `sczo_bootstrap_interval` | 20 | Frequency of ZO gradient bootstrapping |

### Hyperparameter Tuning Guidelines

- **`sczo_alpha`**: Start with 1e-3, reduce if training is unstable
- **`sczo_beta`**: Higher values (0.9-0.99) for smoother gradients
- **`sczo_bootstrap_interval`**: 10-50 steps, more frequent for noisy problems

## Performance Characteristics

### Memory Usage
- **Gradient estimates**: 0.5 floats/param (FP16)
- **Model weights**: 1.0 floats/param
- **Total**: ~2-2.5 floats/param (vs 3-4 for Adam)

### Computational Cost
- **2 forward passes per iteration**
- **No backward passes**
- **Minimal overhead for gradient updates**

### Convergence Properties
- **Fresh batches prevent overfitting**
- **Secant constraint ensures consistency**
- **Periodic bootstrapping maintains accuracy**

## Comparison with Other Methods

| Method | Forward Passes | Memory | Backward Passes |
|--------|---------------|---------|-----------------|
| **SCZO** | **2** | **~2.5x** | **0** |
| MeZO | 2 | ~2x | 0 |
| KronZO | 10-50 | ~3x | 0 |
| Adam | 1 | ~3x | 1 |

## Testing

Run the test suite to verify the implementation:

```bash
python test_sczo.py
```

Expected output:
```
============================================================
SCZO (Secant-Constrained Zero-Order) Implementation Test
============================================================
Testing SCZO basic functionality...
✓ SCZO state initialized with 4 layers
...
🎉 All SCZO tests passed successfully!
============================================================
```

## Implementation Details

### Core Components

1. **`SCZOState`**: Manages gradient estimates and hyperparameters
2. **`sczo_step`**: Performs one SCZO optimization step
3. **`sczo_initial_gradient_estimation`**: Bootstrap ZO gradients
4. **`sczo_update_gradient_estimates`**: Apply secant constraint
5. **`sczo_update_with_weight_decay`**: Handle weight decay separately

### Mathematical Foundation

The secant constraint update is derived from:

```
min_g (1/2)||g - g_k||² subject to g^T s_k = y_k
```

Solution:
```
λ = (y_k - g_k^T s_k) / ||s_k||²
g_{k+1} = g_k + λ * s_k
```

This ensures the new gradient estimate `g_{k+1}` satisfies the secant condition while changing minimally from the previous estimate.

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure all dependencies are installed
2. **Memory issues**: Reduce model size or batch size
3. **Slow convergence**: Increase `sczo_alpha` or decrease `sczo_bootstrap_interval`
4. **Unstable training**: Decrease `sczo_alpha` or increase `sczo_beta`

### Debug Mode

Enable verbose logging by modifying the config:

```python
# Add to config for debugging
log_interval = 10  # Log every 10 steps
eval_interval = 100  # Evaluate every 100 steps
```

## Future Extensions

### Planned Features
- [ ] Diagonal preconditioning (M = diag)
- [ ] Scalar per-layer preconditioning
- [ ] Adaptive step size scheduling
- [ ] Multi-batch secant constraints
- [ ] Distributed training support

### Research Directions
- [ ] Theoretical convergence analysis
- [ ] Comparison with quasi-Newton methods
- [ ] Extension to other model architectures
- [ ] Hybrid SCZO-Adam methods

## Citation

If you use SCZO in your research, please cite:

```bibtex
@misc{sczo2024,
  title={SCZO: Secant-Constrained Zero-Order Optimization for Large Language Models},
  author={[Your Name]},
  year={2024},
  note={Implementation in nanoGPT framework}
}
```

## License

This implementation follows the same license as the nanoGPT project.

---

**Happy training with SCZO! 🚀** 