# Improved Directional KronZO Implementation

This implementation provides an enhanced version of the directional strategy for KronZO (Kronecker Zero-Order) optimization that evaluates actual update candidates instead of just gradient directions.

## Algorithm Overview

### Standard Directional Approach
The original directional approach:
1. Sample directional_q random directions Z_i
2. Choose best direction: Z_best ∈ argmin f(θ + ε*Z_i)
3. Compute projected gradient: c = [f(θ + ε*Z_best) - f(θ - ε*Z_best)]/(2ε)
4. Update: θ ← θ - lr*c*Z_best

### Improved Directional Approach
The improved approach evaluates the full update step:
1. For each direction i = 1, ..., directional_q:
   - Sample Z_i = A_i ⊗ B_i (Kronecker product)
   - Compute projected gradient: c_i = [f(θ + ε*Z_i) - f(θ - ε*Z_i)]/(2ε)
   - Evaluate candidate step: f(θ - lr*c_i*Z_i)
2. Choose best candidate: i_best = argmin_i f(θ - lr*c_i*Z_i)
3. Conservative update: Accept only if f(θ - lr*c_i_best*Z_i_best) beats any of the last 10 loss values
4. If accepted: θ ← θ - lr*c_i_best*Z_i_best, else: θ remains unchanged

## Key Features

### 1. **Conservative Updates**
- Maintains history of the last `loss_history_size` loss values (default: 10)
- Only accepts updates that improve over historical performance
- Prevents deterioration in stochastic settings

### 2. **Memory Efficient**
- Stores only A and B matrices for Kronecker products
- Efficient momentum storage for A matrices only (not full A⊗B tensors)
- Recomputes Kronecker products when needed

### 3. **Cost Analysis**
- **Function Evaluations**: 3*directional_q per step
  - 2*directional_q for gradient estimation (f(θ±ε*Z_i))
  - directional_q for candidate evaluation (f(θ - lr*c_i*Z_i))
- **Memory**: O(Σ(m_i*n_i + p_i*q_i)) for A and B matrices
- **Storage Overhead**: B matrices updated every ν steps (default: ν=20)

### 4. **Multi-Sampling Support**
- Supports multiple overlapping Kronecker factorizations
- Uses prime-factor based diversity strategy
- Scales perturbations by 1/kronzo_sampling_number

## Configuration

### Method Selection
```python
train_method = 'improved_kronzo'  # Use improved directional strategy
```

### Key Parameters
```python
# Directional Selection
directional_q = 10               # Number of directions to evaluate
loss_history_size = 10           # Conservative update history size

# Kronecker Parameters  
kron_strategy = 'approx_square'  # Factorization strategy
kron_max_factor = 64             # Maximum factor size
kronzo_sampling_number = 2       # Multi-sampling diversity
step_interval = 20               # B matrix update frequency

# Optimization
use_momentum = True              # Enable momentum
momentum_beta = 0.9             # Momentum coefficient
learning_rate = 1e-3            # Learning rate
weight_decay = 1e-1             # L2 regularization
```

## Usage

### Training Command
```bash
./train.sh improved_kronzo single --batch_size=64 --max_iters=2000
```

### Configuration File
The method uses `config/train_improved_kronzo_config.py` which contains:
- Algorithm parameters (directional_q, loss_history_size)
- Kronecker settings (strategy, sampling)
- Conservative update settings
- Model and training hyperparameters

## Implementation Details

### File Structure
```
nanoGPT/
├── kronzo_utils_improved_directional.py    # Core implementation
├── config/train_improved_kronzo_config.py  # Configuration
├── zo_train.py                             # Main training script (modified)
└── train.sh                               # Training launcher (modified)
```

### Key Classes and Functions

#### `ImprovedDirectionalHistory`
- Maintains sliding window of loss values
- Implements conservative update logic
- Methods: `add_loss()`, `should_accept_update()`

#### `improved_kronzo_step()`
- Evaluates directional candidates with full update steps
- Returns baseline loss, best candidate, and update decision
- Memory-efficient implementation with A/B matrix storage

#### `improved_kronzo_update()` / `improved_kronzo_update_momentum()`
- Applies accepted updates to model parameters
- Supports both standard and momentum variants
- Efficient Kronecker product reconstruction

### Memory Efficiency

#### Standard Storage
- Full Kronecker products: O(d_out * d_in) per direction
- Total memory: O(directional_q * d_out * d_in)

#### Efficient Storage (This Implementation)
- A matrices: O(m1 * n1) per direction
- B matrices: O(m2 * n2) per layer (shared across directions)
- Total memory: O(directional_q * m1 * n1 + layers * m2 * n2)
- **Savings**: ~90% memory reduction for typical factorizations

### Momentum Implementation

```python
# Standard momentum: stores full tensors
momentum[layer] = β * momentum[layer] + (1-β) * grad_tensor  # O(d_out * d_in)

# Efficient momentum: stores A matrices only  
momentum_A[layer] = β * momentum_A[layer] + (1-β) * (c * A)  # O(m1 * n1)
# Apply as: momentum_A ⊗ B
```

## Comparison with Standard Methods

| Method | Function Evals | Memory | Conservative Updates |
|--------|---------------|--------|---------------------|
| Standard KronZO | 2 | O(m*n) | No |
| Directional KronZO | 2*q | O(d*d) | No |
| **Improved KronZO** | **3*q** | **O(m*n)** | **Yes** |

Where:
- q = directional_q
- d = layer dimension
- m,n = Kronecker factor dimensions (m*n << d*d)

## Expected Benefits

1. **Better Direction Selection**: Evaluates actual updates rather than just gradient directions
2. **Robustness**: Conservative updates prevent deterioration
3. **Memory Efficiency**: Kronecker structure with efficient storage
4. **Practical Performance**: More stable training in stochastic settings

## Limitations

1. **Computational Cost**: 50% more function evaluations than standard directional
2. **Hyperparameter Sensitivity**: Requires tuning of directional_q and loss_history_size
3. **Conservative Bias**: May reject beneficial updates in noisy settings

## Future Extensions

1. **Adaptive History Size**: Dynamic adjustment based on training stability
2. **Probabilistic Acceptance**: Soft acceptance criteria instead of hard thresholds
3. **Multi-Objective**: Consider both loss improvement and gradient magnitude
4. **Distributed Training**: Efficient implementation for multi-GPU setups

## References

- **KronZO**: Kronecker Zero-Order optimization with structured perturbations
- **Directional Optimization**: Best-direction selection strategies
- **Conservative Updates**: History-based acceptance criteria 