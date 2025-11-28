#!/usr/bin/env python3
"""
Hyperparameter tuning for Deep SDF reconstruction quality using Optuna.
This script focuses ONLY on reconstruction metrics (Chamfer distance, SDF loss).
No disentanglement parameters are tuned.

Usage:
    python hparams_reconstruction_optuna.py --experiment examples/CALSNIC_control_L
"""

import argparse
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

try:
    from tensorboard.backend.event_processing import event_accumulator
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    logging.warning("TensorBoard not found. Install with: pip install tensorboard")

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import deep_sdf
import deep_sdf.workspace as ws


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def load_base_specs(experiment_directory):
    """Load the base specifications from the experiment directory."""
    specs_filename = os.path.join(experiment_directory, "specs.json")
    
    if not os.path.isfile(specs_filename):
        raise Exception(
            f"The experiment directory ({experiment_directory}) does not contain a "
            + "specs.json file."
        )
    
    with open(specs_filename) as f:
        specs = json.load(f)
    
    return specs


def create_trial_specs(base_specs, trial):
    """
    Create trial-specific specifications by modifying hyperparameters.
    Focuses on reconstruction-related hyperparameters only.
    
    Trial 0: Uses exact configuration from specs.json (baseline)
    Trial 1+: Varies parameters around the baseline configuration
    
    Parameters to tune:
    - CodeLength: Latent dimension
    - ClampingDistance: SDF clamping distance
    - CodeRegularizationLambda: L2 regularization strength
    - Network learning rate (Initial)
    - Latent learning rate (Initial)
    - GradientClipNorm: Gradient clipping threshold
    - Network architecture: width and depth
    - Dropout probability
    """
    trial_specs = base_specs.copy()
    
    # Get baseline values from specs.json
    base_code_length = base_specs.get("CodeLength", 16)
    base_clamping = base_specs.get("ClampingDistance", 0.1)
    base_reg_lambda = base_specs.get("CodeRegularizationLambda", 1e-4)
    base_net_lr = base_specs["LearningRateSchedule"][0]["Initial"]
    base_lat_lr = base_specs["LearningRateSchedule"][1]["Initial"]
    base_grad_clip = base_specs.get("GradientClipNorm", 1.0)
    base_n_layers = len(base_specs["NetworkSpecs"]["dims"])
    base_hidden_dim = base_specs["NetworkSpecs"]["dims"][0]
    base_dropout = base_specs["NetworkSpecs"].get("dropout_prob", 0.2)
    base_latent_in = base_specs["NetworkSpecs"]["latent_in"][0]
    base_batch_size = base_specs.get("ScenesPerBatch", 64)
    
    # Trial 0: Use exact baseline configuration
    if trial.number == 0:
        logging.info("Trial 0: Using baseline configuration from specs.json")
        trial_specs["NumEpochs"] = 150
        trial_specs["SnapshotFrequency"] = 150  # Only save at the end
        trial_specs["AdditionalSnapshots"] = []  # No intermediate snapshots
        return trial_specs
    
    # Trial 1+: Vary parameters around baseline (constrained ranges)
    # 1. CodeLength - Fixed choices (Optuna requires consistent categorical choices)
    trial_specs["CodeLength"] = trial.suggest_categorical(
        "CodeLength", [16, 32, 64, 128, 256]
    )
    
    # 2. ClampingDistance - ±50% of base value
    min_clamp = max(0.05, base_clamping * 0.5)
    max_clamp = min(0.2, base_clamping * 1.5)
    trial_specs["ClampingDistance"] = trial.suggest_float(
        "ClampingDistance", min_clamp, max_clamp, step=0.05
    )
    
    # 3. CodeRegularizationLambda - ±1 order of magnitude
    min_reg = max(1e-6, base_reg_lambda * 0.1)
    max_reg = min(1e-2, base_reg_lambda * 10)
    trial_specs["CodeRegularizationLambda"] = trial.suggest_float(
        "CodeRegularizationLambda", min_reg, max_reg, log=True
    )
    
    # 4. Learning rates - ±1 order of magnitude from baseline
    min_net_lr = max(1e-5, base_net_lr * 0.1)
    max_net_lr = min(1e-2, base_net_lr * 10)
    net_lr_initial = trial.suggest_float(
        "net_lr_initial", min_net_lr, max_net_lr, log=True
    )
    
    min_lat_lr = max(1e-4, base_lat_lr * 0.1)
    max_lat_lr = min(1e-1, base_lat_lr * 10)
    lat_lr_initial = trial.suggest_float(
        "lat_lr_initial", min_lat_lr, max_lat_lr, log=True
    )
    
    # Update learning rate schedule
    trial_specs["LearningRateSchedule"] = [
        {
            "Type": "Step",
            "Initial": net_lr_initial,
            "Interval": 25,  # Adjusted for 100 epochs
            "Factor": 0.5
        },
        {
            "Type": "Step",
            "Initial": lat_lr_initial,
            "Interval": 25,  # Adjusted for 100 epochs
            "Factor": 0.5
        }
    ]
    
    # 5. GradientClipNorm - ±50% of base
    min_clip = max(0.1, base_grad_clip * 0.5)
    max_clip = min(2.0, base_grad_clip * 1.5)
    trial_specs["GradientClipNorm"] = trial.suggest_float(
        "GradientClipNorm", min_clip, max_clip, step=0.1
    )
    
    # 6. Network Architecture Search - Fixed range (Optuna requires consistent ranges)
    n_layers = trial.suggest_int("n_layers", 6, 10)
    
    # Network width - Fixed choices (Optuna requires consistent categorical choices)
    hidden_dim = trial.suggest_categorical(
        "hidden_dim", [256, 512, 1024]
    )
    
    # Create network architecture
    trial_specs["NetworkSpecs"]["dims"] = [hidden_dim] * n_layers
    
    # 7. Dropout probability - ±0.1 from base
    min_dropout = max(0.0, base_dropout - 0.1)
    max_dropout = min(0.3, base_dropout + 0.1)
    dropout_prob = trial.suggest_float(
        "dropout_prob", min_dropout, max_dropout, step=0.05
    )
    trial_specs["NetworkSpecs"]["dropout_prob"] = dropout_prob
    
    # Update dropout layers (all layers)
    trial_specs["NetworkSpecs"]["dropout"] = list(range(n_layers))
    trial_specs["NetworkSpecs"]["norm_layers"] = list(range(n_layers))
    
    # 8. Latent injection layer - Fixed range (depends on n_layers)
    latent_in_layer = trial.suggest_int("latent_in_layer", 2, max(2, n_layers // 2))
    trial_specs["NetworkSpecs"]["latent_in"] = [latent_in_layer]
    
    # 9. Batch parameters - Fixed choices (Optuna requires consistent categorical choices)
    trial_specs["ScenesPerBatch"] = trial.suggest_categorical(
        "ScenesPerBatch", [32, 64, 128]
    )
    
    # Set epochs for trials
    trial_specs["NumEpochs"] = 150
    trial_specs["SnapshotFrequency"] = 150  # Only save at the end
    trial_specs["AdditionalSnapshots"] = []  # No intermediate snapshots
    
    return trial_specs


def extract_tensorboard_metrics(trial_dir):
    """
    Extract per-epoch metrics from TensorBoard event files.
    Returns dict with epoch-by-epoch data: loss, SDF loss, Chamfer distance.
    """
    import glob
    
    # Look for TensorBoard logs (saved to "TensorBoard" directory by train_deep_sdf.py)
    tb_dir = os.path.join(trial_dir, "TensorBoard")
    
    if not os.path.exists(tb_dir):
        logging.warning(f"TensorBoard directory not found: {tb_dir}")
        return None
    
    event_files = glob.glob(os.path.join(tb_dir, "events.out.tfevents.*"))
    
    if not event_files:
        logging.warning(f"No TensorBoard event files found in {tb_dir}")
        return None
    
    if not HAS_TENSORBOARD:
        logging.error("TensorBoard not installed. Cannot extract metrics.")
        return None
    
    try:
        # Load TensorBoard event data
        ea = event_accumulator.EventAccumulator(tb_dir)
        ea.Reload()
        
        # Get available tags
        available_tags = ea.Tags().get('scalars', [])
        logging.info(f"Available TensorBoard tags: {available_tags}")
        
        metrics = {
            "epochs": [],
            "train_loss": [],
            "train_sdf_loss": [],
            "train_chamfer": [],
            "test_chamfer": []
        }
        
        # Extract Loss/train (total loss)
        if "Loss/train" in available_tags:
            for scalar_event in ea.Scalars("Loss/train"):
                metrics["epochs"].append(int(scalar_event.step))
                metrics["train_loss"].append(float(scalar_event.value))
        
        # Extract Loss/train_sdf (SDF loss only)
        if "Loss/train_sdf" in available_tags:
            for scalar_event in ea.Scalars("Loss/train_sdf"):
                metrics["train_sdf_loss"].append(float(scalar_event.value))
        
        # Extract Mean Chamfer Dist/train
        if "Mean Chamfer Dist/train" in available_tags:
            for scalar_event in ea.Scalars("Mean Chamfer Dist/train"):
                metrics["train_chamfer"].append(float(scalar_event.value))
        
        # Extract Mean Chamfer Dist/test
        if "Mean Chamfer Dist/test" in available_tags:
            for scalar_event in ea.Scalars("Mean Chamfer Dist/test"):
                metrics["test_chamfer"].append(float(scalar_event.value))
        
        # Verify we got data
        if not metrics["epochs"]:
            logging.warning("No epoch data extracted from TensorBoard")
            return None
        
        logging.info(f"Extracted {len(metrics['epochs'])} epochs of data from TensorBoard")
        return metrics
        
    except Exception as e:
        logging.error(f"Error extracting TensorBoard metrics: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return None


def save_trial_results(trial_dir, trial_num, trial_specs, metrics, final_chamfer):
    """
    Save comprehensive trial results with per-epoch metrics.
    Saves 3 files:
    1. trial_results.json - Complete data with all metrics
    2. epoch_metrics.txt - Human-readable epoch data
    3. epoch_metrics.json - Epoch data in JSON format
    """
    # Calculate final values
    final_loss = metrics["train_loss"][-1] if metrics and metrics["train_loss"] else None
    final_sdf_loss = metrics["train_sdf_loss"][-1] if metrics and metrics["train_sdf_loss"] else None
    final_train_chamfer = metrics["train_chamfer"][-1] if metrics and metrics["train_chamfer"] else None
    final_test_chamfer = metrics["test_chamfer"][-1] if metrics and metrics["test_chamfer"] else None
    
    # Main results dictionary
    results = {
        "trial_number": trial_num,
        "final_metrics": {
            "train_loss": final_loss,
            "train_sdf_loss": final_sdf_loss,
            "train_chamfer_distance": final_train_chamfer,
            "test_chamfer_distance": final_test_chamfer,
            "optimization_metric": final_chamfer  # What Optuna optimizes
        },
        "parameters": {
            "CodeLength": trial_specs.get("CodeLength"),
            "ClampingDistance": trial_specs.get("ClampingDistance"),
            "CodeRegularizationLambda": trial_specs.get("CodeRegularizationLambda"),
            "GradientClipNorm": trial_specs.get("GradientClipNorm"),
            "NumEpochs": trial_specs.get("NumEpochs"),
            "ScenesPerBatch": trial_specs.get("ScenesPerBatch"),
            "network_lr_initial": trial_specs["LearningRateSchedule"][0]["Initial"],
            "latent_lr_initial": trial_specs["LearningRateSchedule"][1]["Initial"],
            "n_layers": len(trial_specs["NetworkSpecs"]["dims"]),
            "hidden_dim": trial_specs["NetworkSpecs"]["dims"][0],
            "dropout_prob": trial_specs["NetworkSpecs"]["dropout_prob"],
            "latent_in_layer": trial_specs["NetworkSpecs"]["latent_in"][0]
        },
        "per_epoch_metrics": metrics if metrics else {}
    }
    
    # 1. Save complete results to JSON
    results_file = os.path.join(trial_dir, "trial_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logging.info(f"Trial {trial_num} complete results saved to: {results_file}")
    
    # 2. Save epoch metrics in human-readable text format
    if metrics and metrics["epochs"]:
        metrics_txt_file = os.path.join(trial_dir, "epoch_metrics.txt")
        with open(metrics_txt_file, 'w') as f:
            f.write("# Per-epoch metrics from training\n")
            f.write("# Format: epoch train_loss train_sdf_loss train_chamfer test_chamfer\n")
            f.write("#" + "="*80 + "\n")
            
            num_epochs = len(metrics["epochs"])
            for i in range(num_epochs):
                epoch = metrics["epochs"][i] if i < len(metrics["epochs"]) else "-"
                loss = metrics["train_loss"][i] if i < len(metrics["train_loss"]) else "-"
                sdf = metrics["train_sdf_loss"][i] if i < len(metrics["train_sdf_loss"]) else "-"
                tr_ch = metrics["train_chamfer"][i] if i < len(metrics["train_chamfer"]) else "-"
                te_ch = metrics["test_chamfer"][i] if i < len(metrics["test_chamfer"]) else "-"
                f.write(f"{epoch} {loss} {sdf} {tr_ch} {te_ch}\n")
        
        logging.info(f"Trial {trial_num} epoch metrics saved to: {metrics_txt_file}")
        
        # 3. Save epoch metrics as JSON (easier for plotting)
        metrics_json_file = os.path.join(trial_dir, "epoch_metrics.json")
        with open(metrics_json_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        logging.info(f"Trial {trial_num} epoch metrics JSON saved to: {metrics_json_file}")
    
    return final_chamfer


def evaluate_reconstruction(metrics):
    """
    Evaluate reconstruction quality from extracted metrics.
    Uses test Chamfer distance if available, otherwise training loss.
    """
    try:
        # Priority 1: Use test Chamfer distance (best metric)
        if metrics and metrics.get("test_chamfer") and metrics["test_chamfer"]:
            final_test_chamfer = metrics["test_chamfer"][-1]
            logging.info(f"Using test Chamfer distance: {final_test_chamfer:.6f}")
            return final_test_chamfer
        
        # Priority 2: Use training Chamfer distance
        if metrics and metrics.get("train_chamfer") and metrics["train_chamfer"]:
            final_train_chamfer = metrics["train_chamfer"][-1]
            logging.info(f"Using training Chamfer distance: {final_train_chamfer:.6f}")
            return final_train_chamfer
        
        # Priority 3: Use SDF loss as proxy
        if metrics and metrics.get("train_sdf_loss") and metrics["train_sdf_loss"]:
            final_sdf_loss = metrics["train_sdf_loss"][-1]
            logging.info(f"Using SDF loss as proxy: {final_sdf_loss:.6f}")
            return final_sdf_loss
        
        # Priority 4: Use total training loss
        if metrics and metrics.get("train_loss") and metrics["train_loss"]:
            final_loss = metrics["train_loss"][-1]
            logging.info(f"Using training loss as proxy: {final_loss:.6f}")
            return final_loss
        
        logging.warning("No metrics available for evaluation")
        return float('inf')
        
    except Exception as e:
        logging.error(f"Evaluation failed: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return float('inf')


def train_with_specs(trial_specs, experiment_dir, trial_num):
    """
    Train the model with given specifications.
    Returns the best Chamfer distance achieved.
    """
    import torch
    from train_deep_sdf import main_function as train_main_function
    
    # Create a temporary directory for this trial
    trial_dir = os.path.join(experiment_dir, f"optuna_trial_{trial_num:04d}")
    os.makedirs(trial_dir, exist_ok=True)
    
    # Save trial specs
    trial_specs_file = os.path.join(trial_dir, "specs.json")
    with open(trial_specs_file, 'w') as f:
        json.dump(trial_specs, f, indent=2)
    
    # Train the model
    try:
        # Create necessary subdirectories
        os.makedirs(os.path.join(trial_dir, "ModelParameters"), exist_ok=True)
        os.makedirs(os.path.join(trial_dir, "OptimizerParameters"), exist_ok=True)
        os.makedirs(os.path.join(trial_dir, "LatentCodes"), exist_ok=True)
        os.makedirs(os.path.join(trial_dir, "Logs"), exist_ok=True)
        
        # Train using the train_deep_sdf module
        logging.info(f"Starting training for trial {trial_num}")
        logging.info(f"Trial directory: {trial_dir}")
        logging.info("=" * 60)
        
        # Run training
        logging.info(f"Trial {trial_num}: Starting training...")
        train_main_function(
            experiment_directory=trial_dir,
            continue_from=None,
            batch_split=1
        )
        
        # Extract metrics from TensorBoard logs
        logging.info("=" * 60)
        logging.info(f"Trial {trial_num}: Extracting metrics from TensorBoard...")
        metrics = extract_tensorboard_metrics(trial_dir)
        
        if metrics is None:
            logging.error(f"Trial {trial_num}: Failed to extract metrics!")
            raise optuna.exceptions.TrialPruned()
        
        # Evaluate reconstruction quality
        logging.info(f"Trial {trial_num}: Evaluating reconstruction quality...")
        chamfer_distance = evaluate_reconstruction(metrics)
        
        # Save trial results to JSON
        save_trial_results(trial_dir, trial_num, trial_specs, metrics, chamfer_distance)
        
        logging.info("=" * 60)
        logging.info(f"Trial {trial_num} completed successfully!")
        if metrics:
            logging.info(f"  Epochs trained: {len(metrics['epochs'])}")
            if metrics['train_loss']:
                logging.info(f"  Final training loss: {metrics['train_loss'][-1]:.6f}")
            if metrics['train_sdf_loss']:
                logging.info(f"  Final SDF loss: {metrics['train_sdf_loss'][-1]:.6f}")
            if metrics['train_chamfer']:
                logging.info(f"  Final train Chamfer: {metrics['train_chamfer'][-1]:.6f}")
            if metrics['test_chamfer']:
                logging.info(f"  Final test Chamfer: {metrics['test_chamfer'][-1]:.6f}")
        logging.info(f"  Optimization metric: {chamfer_distance:.6f}")
        logging.info("=" * 60)
        
        # Return Chamfer distance as the metric to optimize
        return chamfer_distance
        
    except Exception as e:
        logging.error(f"Trial {trial_num} failed with error: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        raise optuna.exceptions.TrialPruned()
    
    finally:
        # Clean up model files to save space
        try:
            model_dir = os.path.join(trial_dir, "ModelParameters")
            if os.path.exists(model_dir):
                shutil.rmtree(model_dir)
            latent_dir = os.path.join(trial_dir, "LatentCodes")
            if os.path.exists(latent_dir):
                shutil.rmtree(latent_dir)
            opt_dir = os.path.join(trial_dir, "OptimizerParameters")
            if os.path.exists(opt_dir):
                shutil.rmtree(opt_dir)
        except:
            pass


def create_all_trials_summary(study, optuna_dir):
    """
    Create a comprehensive summary of all trials.
    """
    all_trials = []
    
    for trial in study.trials:
        trial_info = {
            "trial_number": trial.number,
            "state": trial.state.name,
            "value": trial.value if trial.value is not None else None,
            "params": trial.params,
            "datetime_start": str(trial.datetime_start),
            "datetime_complete": str(trial.datetime_complete),
            "duration_seconds": (trial.datetime_complete - trial.datetime_start).total_seconds() if trial.datetime_complete and trial.datetime_start else None,
        }
        all_trials.append(trial_info)
    
    # Sort by value (best first)
    completed_trials = [t for t in all_trials if t["value"] is not None]
    completed_trials.sort(key=lambda x: x["value"])
    
    summary = {
        "study_name": study.study_name,
        "total_trials": len(study.trials),
        "completed_trials": len(completed_trials),
        "pruned_trials": len([t for t in study.trials if t.state.name == "PRUNED"]),
        "failed_trials": len([t for t in study.trials if t.state.name == "FAIL"]),
        "best_trial": {
            "trial_number": study.best_trial.number if study.best_trial else None,
            "value": study.best_value if study.best_trial else None,
            "params": study.best_params if study.best_trial else None,
        },
        "all_trials": all_trials,
        "top_10_trials": completed_trials[:10],
    }
    
    # Save summary
    summary_file = os.path.join(optuna_dir, "all_trials_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logging.info(f"All trials summary saved to: {summary_file}")
    
    # Also create a CSV for easy viewing
    csv_file = os.path.join(optuna_dir, "all_trials_summary.csv")
    with open(csv_file, 'w') as f:
        # Header
        f.write("trial_number,state,chamfer_distance,duration_seconds,")
        if completed_trials:
            param_keys = list(completed_trials[0]["params"].keys())
            f.write(",".join(param_keys))
        f.write("\n")
        
        # Data rows
        for trial in all_trials:
            f.write(f"{trial['trial_number']},{trial['state']},")
            f.write(f"{trial['value'] if trial['value'] is not None else 'N/A'},")
            f.write(f"{trial['duration_seconds'] if trial['duration_seconds'] is not None else 'N/A'}")
            if trial["params"]:
                for key in param_keys:
                    f.write(f",{trial['params'].get(key, 'N/A')}")
            f.write("\n")
    
    logging.info(f"All trials CSV saved to: {csv_file}")
    
    return summary


def objective(trial, base_specs, experiment_dir):
    """
    Optuna objective function.
    Minimizes Chamfer distance.
    """
    # Create trial-specific specifications
    trial_specs = create_trial_specs(base_specs, trial)
    
    # Train and evaluate
    chamfer_distance = train_with_specs(trial_specs, experiment_dir, trial.number)
    
    # Report intermediate value for pruning
    trial.report(chamfer_distance, step=0)
    
    # Check if trial should be pruned
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()
    
    return chamfer_distance


def main():
    """Main function for hyperparameter search."""
    setup_logging()
    
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning for Deep SDF reconstruction using Optuna"
    )
    parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory which includes specifications and data.",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=500,
        help="Number of Optuna trials to run (default: 500)",
    )
    parser.add_argument(
        "--study_name",
        type=str,
        default="reconstruction_optimization",
        help="Name of the Optuna study (default: reconstruction_optimization)",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Database URL for Optuna storage (e.g., sqlite:///optuna.db). If not specified, uses in-memory storage.",
    )
    
    args = parser.parse_args()
    
    # Load base specifications
    logging.info(f"Loading base specifications from {args.experiment_directory}")
    base_specs = load_base_specs(args.experiment_directory)
    
    # Create Optuna study directory
    optuna_dir = os.path.join(args.experiment_directory, "optuna_reconstruction_search")
    os.makedirs(optuna_dir, exist_ok=True)
    
    # Setup storage
    if args.storage is None:
        storage = f"sqlite:///{os.path.join(optuna_dir, 'optuna_study.db')}"
    else:
        storage = args.storage
    
    logging.info(f"Creating Optuna study: {args.study_name}")
    logging.info(f"Storage: {storage}")
    
    # Create Optuna study
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="minimize",  # Minimize Chamfer distance
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        load_if_exists=True,
    )
    
    # Run optimization
    logging.info(f"Starting hyperparameter search with {args.n_trials} trials")
    logging.info("=" * 80)
    
    study.optimize(
        lambda trial: objective(trial, base_specs, optuna_dir),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )
    
    # Create comprehensive summary of all trials
    logging.info("=" * 80)
    logging.info("Creating summary of all trials...")
    create_all_trials_summary(study, optuna_dir)
    
    # Print results
    logging.info("=" * 80)
    logging.info("Hyperparameter search completed!")
    logging.info(f"Number of finished trials: {len(study.trials)}")
    
    logging.info("\n" + "=" * 80)
    logging.info("Best trial:")
    trial = study.best_trial
    
    logging.info(f"  Chamfer Distance: {trial.value:.6f}")
    logging.info("  Best hyperparameters:")
    for key, value in trial.params.items():
        logging.info(f"    {key}: {value}")
    
    # Save best parameters
    best_params_file = os.path.join(optuna_dir, "best_params.json")
    with open(best_params_file, 'w') as f:
        json.dump({
            "chamfer_distance": trial.value,
            "params": trial.params,
            "trial_number": trial.number,
        }, f, indent=2)
    
    logging.info(f"\nBest parameters saved to: {best_params_file}")
    
    # Create best specs.json
    best_specs = create_trial_specs(base_specs, trial)
    best_specs_file = os.path.join(optuna_dir, "best_specs.json")
    with open(best_specs_file, 'w') as f:
        json.dump(best_specs, f, indent=2)
    
    logging.info(f"Best specifications saved to: {best_specs_file}")
    
    # Generate optimization history plot
    try:
        import matplotlib.pyplot as plt
        
        fig = optuna.visualization.matplotlib.plot_optimization_history(study)
        plt.savefig(os.path.join(optuna_dir, "optimization_history.png"))
        logging.info(f"Optimization history plot saved to: {os.path.join(optuna_dir, 'optimization_history.png')}")
        
        fig = optuna.visualization.matplotlib.plot_param_importances(study)
        plt.savefig(os.path.join(optuna_dir, "param_importances.png"))
        logging.info(f"Parameter importance plot saved to: {os.path.join(optuna_dir, 'param_importances.png')}")
        
    except Exception as e:
        logging.warning(f"Could not generate plots: {str(e)}")
    
    logging.info("\n" + "=" * 80)
    logging.info("Top 5 trials:")
    top_trials = sorted(study.trials, key=lambda t: t.value)[:5]
    for i, t in enumerate(top_trials, 1):
        logging.info(f"\n{i}. Trial {t.number}:")
        logging.info(f"   Chamfer Distance: {t.value:.6f}")
        logging.info(f"   Parameters: {t.params}")


if __name__ == "__main__":
    main()
