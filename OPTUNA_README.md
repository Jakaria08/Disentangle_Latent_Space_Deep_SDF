# Optuna Hyperparameter Search for Deep SDF VAE

This directory contains an Optuna-based hyperparameter optimization system for the Deep SDF VAE model used for 3D shape reconstruction and disentangled representation learning.

## Overview

The hyperparameter search optimizes key parameters that affect:
- **Disentanglement quality** (measured by SAP scores)
- **Reconstruction quality** (measured by Chamfer distances) 
- **Loss function balance** (contrastive, KL divergence, Jacobian penalties)

## Key Features

- **Non-invasive**: Works with existing `train_deep_sdf.py` without modifications
- **Comprehensive search space**: Optimizes 15+ critical hyperparameters
- **Efficient**: Uses pruning to terminate unpromising trials early
- **Resumable**: Supports database storage for interrupted/distributed searches
- **Metrics extraction**: Automatically extracts SAP scores, Chamfer distances, and loss values

## Files

- `hparams_optuna_search.py` - Main Optuna search script
- `requirements_optuna.txt` - Additional Python dependencies
- `examples/optuna_search_specs.json` - Example default configuration
- `run_optuna_search.sh` - Example script to run the search

## Hyperparameter Search Space

### Loss Function Parameters
- `beta_final` (1e-5 to 1e-2, log scale) - KL divergence loss weight
- `w_cls` (0.1 to 1.0) - Classification contrastive loss weight  
- `w_code_reg` (0.1 to 2.0) - Code regularization weight
- `w_jacobian` (1e-5 to 1e-2, log scale) - Jacobian penalty weight

### Contrastive Loss Parameters
- `temp` (50 to 300) - Temperature for classification contrastive loss
- `temp_reg` (5 to 50) - Temperature for regression contrastive loss
- `threshold` (0.05 to 0.5) - Threshold for regression contrastive loss

### Architecture Parameters
- `CodeLength` [16, 32, 64, 128, 200] - Latent space dimension
- `latent_dropout` [True, False] - Use latent dropout
- `dropout_prob` (0.0 to 0.3) - Dropout probability

### Training Parameters
- `net_lr_initial` (1e-5 to 1e-2, log scale) - Network learning rate
- `lat_lr_initial` (1e-4 to 1e-1, log scale) - Latent learning rate
- `CodeRegularizationLambda` (1e-6 to 1e-2, log scale) - L2 regularization
- `GradientClipNorm` (0.1 to 2.0) - Gradient clipping threshold
- `annealing_epochs` (1 to 10) - KL annealing period
- `ClampingDistance` (0.05 to 0.2) - SDF clamping distance

### Loss Function Switches
- `guided_contrastive_loss` [True, False] - Enable contrastive loss
- `jacobian_loss` [True, False] - Enable Jacobian penalty

## Installation

1. Install additional dependencies:
```bash
pip install -r requirements_optuna.txt
```

2. Or install manually:
```bash
pip install optuna>=3.0.0 tensorboard>=2.8.0
```

## Usage

### Basic Usage

1. **Prepare your configuration**: Copy and modify `examples/optuna_search_specs.json`:
```bash
cp examples/optuna_search_specs.json my_search_specs.json
# Edit paths and parameters as needed
```

2. **Run the search**:
```bash
python hparams_optuna_search.py \
    --search_dir "./optuna_results" \
    --default_specs "my_search_specs.json" \
    --n_trials 50 \
    --max_epochs 100
```

### Advanced Usage

#### Resumable Search with Database Storage
```bash
python hparams_optuna_search.py \
    --search_dir "./optuna_results" \
    --default_specs "my_search_specs.json" \
    --n_trials 100 \
    --max_epochs 75 \
    --study_name "my_search_study" \
    --storage "sqlite:///optuna_study.db"
```

#### Parallel Execution
```bash
# Run multiple processes in parallel (be careful with resource usage)
python hparams_optuna_search.py \
    --n_trials 200 \
    --n_jobs 2 \
    --storage "sqlite:///optuna_study.db"
```

#### Quick Search for Testing
```bash
python hparams_optuna_search.py \
    --n_trials 10 \
    --max_epochs 25 \
    --search_dir "./quick_test"
```

## Command Line Arguments

- `--search_dir` - Directory to store all trial experiments
- `--default_specs` - Path to base specs.json file  
- `--n_trials` - Number of Optuna trials to run (default: 50)
- `--max_epochs` - Maximum training epochs per trial (default: 100)
- `--study_name` - Name for the Optuna study (default: "deepsdf_vae_hparam_search")
- `--storage` - Database URL for persistent storage (optional)
- `--n_jobs` - Number of parallel jobs (default: 1)

## Output and Results

### Directory Structure
```
search_dir/
├── optuna_trial_0000/     # Individual trial directories
│   ├── specs.json         # Trial-specific configuration
│   ├── ModelParameters/   # Trained model checkpoints
│   ├── TensorBoard/       # Training metrics and logs
│   └── ...
├── optuna_trial_0001/
├── ...
├── optuna_results.json    # Summary of best results
└── optuna_search.log      # Search process logs
```

### Results File Format
```json
{
    "best_trial_number": 23,
    "best_value": 0.0234,
    "best_params": {
        "beta_final": 0.002,
        "temp": 181.5,
        "w_cls": 0.25,
        ...
    },
    "best_user_attrs": {
        "final_train_loss": 0.123,
        "best_train_chamfer": 0.0234,
        "best_test_chamfer": 0.0256,
        "training_time": 1234.5
    },
    "n_trials": 50
}
```

## Objective Function

The search minimizes test Chamfer distance with fallback priorities:
1. **Primary**: Best test Chamfer distance (reconstruction quality on held-out data)
2. **Secondary**: Best train Chamfer distance (if test unavailable)
3. **Tertiary**: Best train loss (if Chamfer unavailable)

This prioritizes models that generalize well to unseen 3D shapes while maintaining good reconstruction quality.

## Tips for Effective Search

### 1. Start Small
- Begin with 10-20 trials and short epochs (25-50) to verify setup
- Gradually increase trials and epochs based on initial results

### 2. Monitor Resource Usage
- Each trial creates a full experiment directory with model checkpoints
- Disk space usage: ~100MB-1GB per trial depending on model size
- RAM usage: Depends on batch size and model architecture

### 3. Customize Search Space
Edit the `get_hyperparameter_search_space()` function to:
- Add new hyperparameters
- Modify search ranges based on preliminary results
- Remove parameters that are well-established

### 4. Use Database Storage
For long searches or parallel execution:
```bash
--storage "sqlite:///my_study.db"
```

### 5. Analyze Results
After completion, use Optuna's visualization tools:
```python
import optuna
study = optuna.load_study(study_name="my_study", storage="sqlite:///my_study.db")
optuna.visualization.plot_optimization_history(study)
optuna.visualization.plot_param_importances(study)
```

## Integration with Evaluation

After finding optimal hyperparameters:

1. **Use best configuration** for full training:
```bash
# Copy best specs.json from best trial directory
cp optuna_results/optuna_trial_XXXX/specs.json final_specs.json

# Train with full epochs
python train_deep_sdf.py -e final_experiment
```

2. **Evaluate SAP scores** on the final model:
```python
# Use the evaluation code from test.ipynb
# The hyperparameters should improve both Chamfer distances and SAP scores
```

## Troubleshooting

### Common Issues

1. **TensorBoard import errors**: Install tensorboard separately
2. **CUDA memory issues**: Reduce batch size in specs.json
3. **Disk space issues**: Monitor search_dir size, clean up failed trials
4. **Import errors**: Ensure you're running from the Deep3DComp root directory

### Performance Tuning

- **Reduce evaluation frequency** for faster trials
- **Use smaller models** for initial search phases  
- **Limit number of evaluation shapes** (`EvalTrainSceneNumber`, `EvalTestSceneNumber`)

## Example Workflow

1. **Setup**:
```bash
# Install dependencies
pip install -r requirements_optuna.txt

# Prepare configuration
cp examples/optuna_search_specs.json my_config.json
# Edit paths and parameters
```

2. **Quick test**:
```bash
python hparams_optuna_search.py \
    --default_specs my_config.json \
    --n_trials 5 \
    --max_epochs 20 \
    --search_dir "./test_search"
```

3. **Full search**:
```bash
python hparams_optuna_search.py \
    --default_specs my_config.json \
    --n_trials 100 \
    --max_epochs 100 \
    --search_dir "./full_search" \
    --storage "sqlite:///full_search.db"
```

4. **Use results**:
```bash
# Check results
cat full_search/optuna_results.json

# Train final model with best hyperparameters
cp full_search/optuna_trial_XXXX/specs.json final_specs.json
python train_deep_sdf.py -e final_model
```

This hyperparameter search should help you find the optimal balance between reconstruction quality (low Chamfer distances) and disentanglement quality (high SAP scores) for your 3D shape VAE model.
