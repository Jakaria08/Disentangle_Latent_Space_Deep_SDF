# Reconstruction Hyperparameter Optimization Guide

## Overview
This guide explains how to use `hparams_reconstruction_optuna.py` for optimizing Deep SDF reconstruction quality using Optuna.

## Features

### Hyperparameters Tuned (Reconstruction-Focused Only)
1. **Latent Space:**
   - `CodeLength`: [32, 64, 128, 200, 256]
   - `CodeRegularizationLambda`: [1e-6 to 1e-2, log scale]

2. **SDF Parameters:**
   - `ClampingDistance`: [0.05, 0.1, 0.15, 0.2]

3. **Learning Rates:**
   - Network learning rate: [1e-5 to 1e-2, log scale]
   - Latent learning rate: [1e-4 to 1e-1, log scale]

4. **Training Stability:**
   - `GradientClipNorm`: [0.1 to 2.0]

5. **Architecture Search:**
   - Network depth: [6 to 10 layers]
   - Hidden dimension: [256, 512, 1024]
   - Dropout probability: [0.0 to 0.3]
   - Latent injection layer: [2 to n_layers//2]

6. **Batch Parameters:**
   - `ScenesPerBatch`: [32, 64, 128]

### Training Configuration
- **Epochs per trial:** 200 (reduced from 2001 for faster exploration)
- **Total trials:** 50 (default)
- **Optimization metric:** Chamfer Distance (lower is better)
- **Search algorithm:** TPE (Tree-structured Parzen Estimator)
- **Pruning:** MedianPruner for early stopping of poor trials

## Installation

### 1. Install Required Packages

```bash
# Install Optuna and dependencies
pip install -r requirements_optuna.txt
```

Or install manually:
```bash
pip install optuna>=3.0.0 tensorboard>=2.8.0 sqlalchemy>=1.4.0 plotly>=5.0.0
```

### 2. Verify Installation

```bash
python -c "import optuna; print(f'Optuna version: {optuna.__version__}')"
```

## Usage

### Basic Usage

```bash
python hparams_reconstruction_optuna.py \
    --experiment /home/jakaria/INR/Deep3DComp/examples/CALSNIC_control_L \
    --n_trials 50
```

### Advanced Options

```bash
python hparams_reconstruction_optuna.py \
    --experiment /home/jakaria/INR/Deep3DComp/examples/CALSNIC_control_L \
    --n_trials 50 \
    --study_name my_reconstruction_study \
    --storage sqlite:///my_custom_optuna.db
```

### Command Line Arguments

- `--experiment` or `-e`: **[REQUIRED]** Path to experiment directory containing `specs.json`
- `--n_trials`: Number of trials to run (default: 50)
- `--study_name`: Name for the Optuna study (default: "reconstruction_optimization")
- `--storage`: Database URL for persistent storage (default: auto-created SQLite in experiment dir)

## Output Structure

After running, you'll find results in:
```
examples/CALSNIC_control_L/optuna_reconstruction_search/
├── optuna_study.db                    # Optuna database (all trial history)
├── best_params.json                   # Best hyperparameters found
├── best_specs.json                    # Complete specs.json with best params
├── optimization_history.png           # Visualization of optimization progress
├── param_importances.png              # Which parameters matter most
└── optuna_trial_0/                    # Individual trial directories
    ├── specs.json
    ├── ModelParameters/
    ├── LatentCodes/
    └── Logs/
```

## Understanding Results

### 1. Best Parameters (`best_params.json`)
```json
{
  "chamfer_distance": 0.001234,
  "params": {
    "CodeLength": 128,
    "ClampingDistance": 0.1,
    "CodeRegularizationLambda": 0.0001,
    "net_lr_initial": 0.0005,
    "lat_lr_initial": 0.001,
    ...
  },
  "trial_number": 23
}
```

### 2. Best Specifications (`best_specs.json`)
This is a complete `specs.json` file with the optimal hyperparameters. You can:
- Copy it to your experiment directory
- Use it for full training runs
- Share it for reproducibility

### 3. Optimization History
Shows how the best Chamfer distance improved over trials.

### 4. Parameter Importance
Shows which hyperparameters had the biggest impact on reconstruction quality.

## Workflow

### Step 1: Run Optimization
```bash
# Start the search (this will take a while!)
python hparams_reconstruction_optuna.py \
    --experiment /home/jakaria/INR/Deep3DComp/examples/CALSNIC_control_L \
    --n_trials 50
```

### Step 2: Monitor Progress
Watch the terminal output for:
- Trial progress
- Current best Chamfer distance
- Which parameters are being tested

### Step 3: Analyze Results
```bash
# Check the best parameters
cat examples/CALSNIC_control_L/optuna_reconstruction_search/best_params.json

# View visualizations
eog examples/CALSNIC_control_L/optuna_reconstruction_search/optimization_history.png
eog examples/CALSNIC_control_L/optuna_reconstruction_search/param_importances.png
```

### Step 4: Use Best Configuration
```bash
# Copy best specs to your experiment
cp examples/CALSNIC_control_L/optuna_reconstruction_search/best_specs.json \
   examples/CALSNIC_control_L/specs_optimized.json

# Train with full epochs using optimized parameters
python train_deep_sdf.py -e examples/CALSNIC_control_L
# (Make sure specs_optimized.json is renamed to specs.json or adjust accordingly)
```

## Resume Interrupted Search

If the search is interrupted, you can resume:

```bash
# The study is stored in SQLite database, so just run again with same parameters
python hparams_reconstruction_optuna.py \
    --experiment /home/jakaria/INR/Deep3DComp/examples/CALSNIC_control_L \
    --n_trials 50
```

Optuna will automatically resume from where it left off!

## Tips for Better Results

### 1. Start Small
For initial testing, run fewer trials:
```bash
python hparams_reconstruction_optuna.py --experiment <path> --n_trials 10
```

### 2. Parallel Execution
If you have multiple GPUs, you can run trials in parallel by starting multiple processes with the same database.

### 3. Adjust Epochs
If you want faster trials, edit line 145 in the script:
```python
trial_specs["NumEpochs"] = 100  # Even shorter for very fast exploration
```

### 4. Focus on Specific Parameters
If you want to tune only certain parameters, comment out others in the `create_trial_specs()` function.

### 5. Monitor Disk Space
Each trial creates a directory with model checkpoints. Consider:
- Enabling cleanup in the script (uncomment line 253)
- Or manually remove old trial directories after finding best params

## Troubleshooting

### Issue: Out of Memory
**Solution:** Reduce batch size range in the script:
```python
trial_specs["ScenesPerBatch"] = trial.suggest_categorical(
    "ScenesPerBatch", [16, 32]  # Smaller batches
)
```

### Issue: Training Too Slow
**Solutions:**
1. Reduce epochs per trial (line 145)
2. Reduce number of trials
3. Use fewer samples per scene

### Issue: No Improvement
**Solutions:**
1. Check if base specs.json is reasonable
2. Expand hyperparameter search ranges
3. Increase number of trials

### Issue: Import Errors
**Solution:** Make sure all dependencies are installed:
```bash
pip install -r requirements_optuna.txt
pip install torch numpy pandas matplotlib
```

## Customization

### Change Search Ranges
Edit `create_trial_specs()` function in the script:

```python
# Example: Expand learning rate range
net_lr_initial = trial.suggest_float(
    "net_lr_initial", 1e-6, 1e-1, log=True  # Wider range
)
```

### Add New Parameters
Add new suggestions in `create_trial_specs()`:

```python
# Example: Add SamplesPerScene tuning
trial_specs["SamplesPerScene"] = trial.suggest_categorical(
    "SamplesPerScene", [8192, 16384, 32768]
)
```

### Change Optimization Metric
Edit `objective()` function to optimize different metrics (e.g., training loss, inference time, etc.)

## Expected Runtime

- **Per trial:** ~30-60 minutes (depends on data size and hardware)
- **50 trials:** ~25-50 hours on single GPU
- **Recommendation:** Run overnight or over weekend

## Comparison with Old Hyperparameter Search

| Feature | Old (`hyperparameter_check.py`) | New (`hparams_reconstruction_optuna.py`) |
|---------|----------------------------------|------------------------------------------|
| Focus | Disentanglement | Reconstruction quality |
| Algorithm | Grid/Random | TPE (smart search) |
| Efficiency | Tests all combinations | Learns from previous trials |
| Trials | ~100-1000 (exhaustive) | ~50 (focused) |
| Pruning | No | Yes (stops bad trials early) |
| Architecture Search | No | Yes |
| Storage | Separate files | Unified database |
| Resumable | No | Yes |

## Next Steps After Optimization

1. **Validate best parameters** with full training (2000 epochs)
2. **Test on validation/test set** to ensure no overfitting
3. **Document** the optimal configuration
4. **Compare** reconstruction quality with baseline
5. **Consider** running multiple studies with different data splits

## Questions or Issues?

If you encounter problems:
1. Check logs in trial directories
2. Review error messages in terminal
3. Verify all dependencies are installed
4. Ensure base `specs.json` is valid

---

**Note:** This script does NOT modify any existing files except the newly created trial directories. Your original configurations remain untouched.
