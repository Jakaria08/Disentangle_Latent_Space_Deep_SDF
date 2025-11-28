# Quick Start: Reconstruction Hyperparameter Optimization

## ✅ What Was Created

Three new files for reconstruction-focused hyperparameter tuning:

1. **`hparams_reconstruction_optuna.py`** - Main optimization script
2. **`run_reconstruction_optuna.sh`** - Easy-to-use bash script
3. **`RECONSTRUCTION_OPTUNA_GUIDE.md`** - Comprehensive documentation

## 🚀 How to Run (Step-by-Step)

### Step 1: Install Dependencies (REQUIRED - First Time Only)

```bash
pip install -r requirements_optuna.txt
```

This installs:
- `optuna` - Hyperparameter optimization framework
- `tensorboard` - Visualization
- `sqlalchemy` - Database storage
- `plotly` - Advanced plots

**Verify installation:**
```bash
python -c "import optuna; print(f'Optuna {optuna.__version__} installed!')"
```

### Step 2: Run Optimization (Choose One Method)

#### Option A: Use the Easy Script (Recommended)
```bash
./run_reconstruction_optuna.sh
```

#### Option B: Run Directly with Python
```bash
python hparams_reconstruction_optuna.py \
    --experiment /home/jakaria/INR/Deep3DComp/examples/CALSNIC_control_L \
    --n_trials 50
```

#### Option C: Quick Test (10 trials for testing)
```bash
python hparams_reconstruction_optuna.py \
    --experiment /home/jakaria/INR/Deep3DComp/examples/CALSNIC_control_L \
    --n_trials 10
```

## 📊 What Gets Optimized

### Reconstruction-Focused Hyperparameters:

| Category | Parameters | Range |
|----------|-----------|-------|
| **Latent Space** | CodeLength | [32, 64, 128, 200, 256] |
| | CodeRegularizationLambda | [1e-6 to 1e-2] |
| **SDF** | ClampingDistance | [0.05 to 0.2] |
| **Learning** | Network LR | [1e-5 to 1e-2] |
| | Latent LR | [1e-4 to 1e-1] |
| | GradientClipNorm | [0.1 to 2.0] |
| **Architecture** | Depth | [6 to 10 layers] |
| | Width | [256, 512, 1024] |
| | Dropout | [0.0 to 0.3] |
| | Latent injection | [2 to depth//2] |
| **Batch** | ScenesPerBatch | [32, 64, 128] |

**Optimization Goal:** Minimize Chamfer Distance (best reconstruction quality)

## ⏱️ Expected Runtime

- **Per trial:** 30-60 minutes
- **50 trials:** 25-50 hours (recommended to run overnight/weekend)
- **10 trials (test):** 5-10 hours

## 📁 Output Location

All results saved to:
```
/home/jakaria/INR/Deep3DComp/examples/CALSNIC_control_L/optuna_reconstruction_search/
```

Key files:
- `best_params.json` - Best hyperparameters found
- `best_specs.json` - Complete configuration (ready to use)
- `optimization_history.png` - Progress visualization
- `param_importances.png` - Which params matter most
- `optuna_study.db` - Full trial history (can resume if interrupted)

## 🔍 Monitor Progress

While running, you'll see:
```
[I 2025-11-24 10:30:15,123] Trial 5 finished with value: 0.00234
[I 2025-11-24 10:30:15,124] Best trial so far: Trial 3 with value: 0.00198
```

## 📈 After Completion

### 1. View Best Parameters
```bash
cat examples/CALSNIC_control_L/optuna_reconstruction_search/best_params.json
```

### 2. View Visualizations
```bash
eog examples/CALSNIC_control_L/optuna_reconstruction_search/optimization_history.png
eog examples/CALSNIC_control_L/optuna_reconstruction_search/param_importances.png
```

### 3. Use Best Configuration for Full Training
```bash
# Copy optimized specs
cp examples/CALSNIC_control_L/optuna_reconstruction_search/best_specs.json \
   examples/CALSNIC_control_L/specs_optimized.json

# Edit to set full epochs (2001 instead of 200)
# Then train with optimized parameters
python train_deep_sdf.py -e examples/CALSNIC_control_L
```

## 🛡️ Safety Features

✅ **NO modifications** to existing files (`train_deep_sdf.py`, original `specs.json`, etc.)  
✅ **Separate trial directories** - each trial is isolated  
✅ **Resumable** - can interrupt and continue later  
✅ **Version controlled** - all parameters logged in database  

## 🆘 Troubleshooting

### "ModuleNotFoundError: No module named 'optuna'"
**Solution:** Install dependencies
```bash
pip install -r requirements_optuna.txt
```

### Out of Memory
**Solution:** Reduce batch sizes in the script (line ~142):
```python
trial_specs["ScenesPerBatch"] = trial.suggest_categorical("ScenesPerBatch", [16, 32])
```

### Too Slow
**Solution:** Reduce trials or epochs:
```bash
python hparams_reconstruction_optuna.py --experiment <path> --n_trials 20
```

Or edit line 145 in script:
```python
trial_specs["NumEpochs"] = 100  # Faster trials
```

## 📖 Full Documentation

For detailed information, see: **`RECONSTRUCTION_OPTUNA_GUIDE.md`**

## ❓ Questions to Ask Yourself Before Running

1. ✅ Is Optuna installed? (`pip install -r requirements_optuna.txt`)
2. ✅ Do I have enough disk space? (~10-20GB per 50 trials)
3. ✅ Do I have enough time? (25-50 hours for 50 trials)
4. ✅ Is my base `specs.json` working correctly?
5. ✅ Do I want to test first with 10 trials?

## 🎯 Ready to Start?

```bash
# Install dependencies
pip install -r requirements_optuna.txt

# Run optimization
./run_reconstruction_optuna.sh
```

Good luck! 🚀
