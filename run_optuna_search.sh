#!/bin/bash

# Example script to run Optuna hyperparameter search for Deep SDF VAE
# Make sure to modify paths according to your setup

# Install Optuna if not already installed
pip install -r requirements_optuna.txt

# Run hyperparameter search
python hparams_optuna_search.py \
    --search_dir "/home/jakaria/INR/Deep3DComp/optuna_search_results" \
    --default_specs "examples/optuna_search_specs.json" \
    --n_trials 30 \
    --max_epochs 50 \
    --study_name "deepsdf_vae_sap_optimization" \
    --n_jobs 1

# Optional: Run with database storage for resuming/parallel execution
# python hparams_optuna_search.py \
#     --search_dir "/home/jakaria/INR/Deep3DComp/optuna_search_results" \
#     --default_specs "examples/optuna_search_specs.json" \
#     --n_trials 100 \
#     --max_epochs 75 \
#     --study_name "deepsdf_vae_sap_optimization" \
#     --storage "sqlite:///optuna_study.db" \
#     --n_jobs 2

echo "Hyperparameter search completed!"
echo "Results saved in: /home/jakaria/INR/Deep3DComp/optuna_search_results/optuna_results.json"
