#!/bin/bash
# Quick start script for reconstruction hyperparameter optimization

set -e  # Exit on error

echo "=========================================="
echo "Reconstruction Hyperparameter Optimization"
echo "=========================================="
echo ""

# Configuration
EXPERIMENT_DIR="/home/jakaria/INR/Deep3DComp/examples/CALSNIC_control_L"
N_TRIALS=500
STUDY_NAME="reconstruction_optimization"

# Check if optuna is installed
echo "Checking dependencies..."
if ! python -c "import optuna" 2>/dev/null; then
    echo "ERROR: Optuna is not installed!"
    echo "Installing required packages..."
    pip install -r requirements_optuna.txt
    echo "Done!"
fi

# Verify specs.json exists
if [ ! -f "${EXPERIMENT_DIR}/specs.json" ]; then
    echo "ERROR: specs.json not found in ${EXPERIMENT_DIR}"
    exit 1
fi

echo "Configuration:"
echo "  Experiment: ${EXPERIMENT_DIR}"
echo "  Trials: ${N_TRIALS}"
echo "  Study Name: ${STUDY_NAME}"
echo ""

# Ask for confirmation
read -p "Start optimization? This will take several hours. (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Create log directory
mkdir -p logs

# Run optimization with logging
LOG_FILE="logs/optuna_reconstruction_$(date +%Y%m%d_%H%M%S).log"

echo "Starting optimization..."
echo "Logging to: ${LOG_FILE}"
echo ""

python hparams_reconstruction_optuna.py \
    --experiment "${EXPERIMENT_DIR}" \
    --n_trials ${N_TRIALS} \
    --study_name "${STUDY_NAME}" \
    2>&1 | tee "${LOG_FILE}"

echo ""
echo "=========================================="
echo "Optimization Complete!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  ${EXPERIMENT_DIR}/optuna_reconstruction_search/"
echo ""
echo "View best parameters:"
echo "  cat ${EXPERIMENT_DIR}/optuna_reconstruction_search/best_params.json"
echo ""
echo "View best specs:"
echo "  cat ${EXPERIMENT_DIR}/optuna_reconstruction_search/best_specs.json"
echo ""
echo "View logs:"
echo "  cat ${LOG_FILE}"
echo ""
