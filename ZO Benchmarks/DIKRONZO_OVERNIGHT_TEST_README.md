# DiKronZO Overnight Testing Script

## Overview

This script tests DiKronZO with various parameter combinations to find optimal settings. It runs **48 experiments** total (4 × 4 × 3 combinations).

## Parameters Tested

- **`kronzo_sampling_number`**: [1, 2, 5, 10] - Number of Kronecker products to sample and sum
- **`step_interval`**: [1, 5, 10, 20] - Interval for updating B matrices (every ν steps)  
- **`directional_q`**: [1, 5, 10] - Number of directions to try in DiKronZO

## Base Configuration

- **Method**: DiKronZO (directional Kronecker Zero-Order)
- **Dataset**: Shakespeare (small, fast)
- **Model**: 6 layers, 6 heads, 384 embedding (small model)
- **Training**: 2000 iterations, lr=1e-3, momentum enabled
- **Evaluation**: Every 500 steps

## Usage

### 1. Run in Foreground (for testing)
```bash
cd /home/allanath/Bureau/ZO_journal
python overnight_dikronzo_test.py
```

### 2. Run with nohup (recommended for overnight)
```bash
cd /home/allanath/Bureau/ZO_journal
nohup python overnight_dikronzo_test.py > overnight_results.log 2>&1 &
```

### 3. Monitor Progress
```bash
# Check overall progress
tail -f overnight_results.log

# Check specific experiment logs (replace YYYYMMDD_HHMMSS with actual timestamp)
ls dikronzo_experiment_*/dikronzo_exp_*.log
tail -f dikronzo_experiment_*/dikronzo_exp_1_sampling1_interval1_dirq1.log
```

## Output Directory Structure

The script creates a organized directory structure:

```
dikronzo_experiment_YYYYMMDD_HHMMSS/
├── experiment_info.txt           # Experiment configuration and info
├── summary.txt                   # Final results table
├── dikronzo_exp_*.log            # Individual experiment logs
└── results/
    ├── all_results.json          # Complete results in JSON format
    └── result_*.json             # Individual experiment results
```

## Results Format

The final summary table shows:

| Column | Description |
|--------|-------------|
| Exp | Experiment ID |
| Sampling | kronzo_sampling_number |
| Interval | step_interval |
| Dir_Q | directional_q |
| Best_Val | Best validation loss achieved |
| Final_Train | Final training loss |
| It/s | Iterations per second |
| Mem(GB) | Peak memory usage |
| Steps | Total steps completed |
| Time(min) | Duration in minutes |

## Expected Runtime

- Each experiment: ~10-15 minutes (2000 iterations)
- Total runtime: ~8-12 hours (48 experiments)
- Memory usage: ~0.5-2 GB per experiment

## Key Insights to Look For

1. **Best validation loss**: Which parameter combination achieves lowest loss?
2. **Training speed**: Which settings provide fastest iterations/second?
3. **Memory efficiency**: Which configurations use least memory?
4. **Sampling effect**: How does increasing `kronzo_sampling_number` affect performance?
5. **B matrix update frequency**: How does `step_interval` impact convergence?
6. **Directional search**: How does `directional_q` affect optimization quality?

## Important Notes

- **Fixed B matrix seeding bug**: All B matrices in multi-sampling are now unique
- **Overlapping factorizations**: Multi-sampling uses prime-factor-based overlapping strategy
- **Memory efficient**: Never stores full Kronecker products, only A and B matrices
- **Gradient estimation mode**: Uses standard gradient estimation (not direct movement)
- **Organized output**: All files are contained in a single experiment directory

## Post-Analysis Commands

```bash
# View best results
cat dikronzo_experiment_*/summary.txt

# Check for failed experiments  
grep "Exit code: [^0]" overnight_results.log

# Analyze specific parameter effects
grep "Best validation loss" dikronzo_experiment_*/summary.txt

# View experiment details
cat dikronzo_experiment_*/experiment_info.txt
``` 