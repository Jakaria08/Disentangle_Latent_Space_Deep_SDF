#!/usr/bin/env python3
"""
Optuna-based hyperparameter search for Deep SDF VAE training.
This script uses Optuna to optimize hyperparameters without modifying the existing training code.
"""

import json
import logging
import os
import random
import copy
import time
from typing import Dict, Any, Optional
import argparse

import numpy as np
import torch
import optuna
from optuna.trial import Trial
import deep_sdf

# Import the training function
from train_deep_sdf import main_function


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler('optuna_search.log'),
            logging.StreamHandler()
        ]
    )


def get_hyperparameter_search_space(trial: Trial) -> Dict[str, Any]:
    """
    Define the hyperparameter search space for Optuna.
    
    Args:
        trial: Optuna trial object
        
    Returns:
        Dictionary with hyperparameter suggestions
    """
    # Key hyperparameters from train_deep_sdf.py
    hparams = {
        # Beta for KL divergence loss (annealing final value)
        'beta_final': trial.suggest_float('beta_final', 1e-5, 1e-2, log=True),
        
        # Temperature parameters for contrastive loss
        'temp': trial.suggest_float('temp', 50.0, 300.0),
        'temp_reg': trial.suggest_float('temp_reg', 5.0, 50.0),
        
        # Loss weights
        'w_cls': trial.suggest_float('w_cls', 0.1, 1.0),
        'w_code_reg': trial.suggest_float('w_code_reg', 0.1, 2.0),
        'w_jacobian': trial.suggest_float('w_jacobian', 1e-5, 1e-2, log=True),
        
        # Threshold for regression contrastive loss
        'threshold': trial.suggest_float('threshold', 0.05, 0.5),
        
        # Code length (latent dimension)
        'CodeLength': trial.suggest_categorical('CodeLength', [16, 32, 64, 128, 200]),
        
        # Learning rates
        'net_lr_initial': trial.suggest_float('net_lr_initial', 1e-5, 1e-2, log=True),
        'lat_lr_initial': trial.suggest_float('lat_lr_initial', 1e-4, 1e-1, log=True),
        
        # Network architecture
        'latent_dropout': trial.suggest_categorical('latent_dropout', [True, False]),
        'dropout_prob': trial.suggest_float('dropout_prob', 0.0, 0.3),
        
        # Code regularization
        'CodeRegularizationLambda': trial.suggest_float('CodeRegularizationLambda', 1e-6, 1e-2, log=True),
        
        # Gradient clipping
        'GradientClipNorm': trial.suggest_float('GradientClipNorm', 0.1, 2.0),
        
        # Annealing epochs for beta
        'annealing_epochs': trial.suggest_int('annealing_epochs', 1, 10),
        
        # Clamping distance
        'ClampingDistance': trial.suggest_float('ClampingDistance', 0.05, 0.2),
        
        # Loss function switches
        'guided_contrastive_loss': trial.suggest_categorical('guided_contrastive_loss', [True, False]),
        'jacobian_loss': trial.suggest_categorical('jacobian_loss', [True, False]),
    }
    
    return hparams


def create_specs_with_hyperparameters(default_specs: Dict[str, Any], 
                                    hparams: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create experiment specifications with suggested hyperparameters.
    
    Args:
        default_specs: Base experiment specifications
        hparams: Hyperparameters from Optuna trial
        
    Returns:
        Updated specifications dictionary
    """
    specs = copy.deepcopy(default_specs)
    
    # Update network-level hyperparameters
    specs['CodeLength'] = hparams['CodeLength']
    specs['CodeRegularizationLambda'] = hparams['CodeRegularizationLambda']
    specs['ClampingDistance'] = hparams['ClampingDistance']
    specs['GradientClipNorm'] = hparams['GradientClipNorm']
    
    # Update network architecture specs
    specs['NetworkSpecs']['latent_dropout'] = hparams['latent_dropout']
    specs['NetworkSpecs']['dropout_prob'] = hparams['dropout_prob']
    
    # Update learning rate schedules
    specs['LearningRateSchedule'][0]['Initial'] = hparams['net_lr_initial']
    specs['LearningRateSchedule'][1]['Initial'] = hparams['lat_lr_initial']
    
    return specs


def modify_training_globals(hparams: Dict[str, Any]):
    """
    Modify global variables in train_deep_sdf module to use our hyperparameters.
    This is a workaround since we can't modify the main_function signature.
    """
    import train_deep_sdf
    
    # Update global hyperparameters in the training module
    train_deep_sdf.beta_final = hparams['beta_final']
    train_deep_sdf.temp = hparams['temp']
    train_deep_sdf.temp_reg = hparams['temp_reg']
    train_deep_sdf.w_cls = hparams['w_cls']
    train_deep_sdf.w_code_reg = hparams['w_code_reg']
    train_deep_sdf.w_jacobian = hparams['w_jacobian']
    train_deep_sdf.threshold = hparams['threshold']
    train_deep_sdf.annealing_epochs = hparams['annealing_epochs']
    train_deep_sdf.guided_contrastive_loss = hparams['guided_contrastive_loss']
    train_deep_sdf.jacobian_loss = hparams['jacobian_loss']


def extract_metrics_from_experiment(experiment_dir: str) -> Dict[str, float]:
    """
    Extract evaluation metrics from a completed experiment.
    
    Args:
        experiment_dir: Path to experiment directory
        
    Returns:
        Dictionary with extracted metrics
    """
    metrics = {}
    
    try:
        # Try to read TensorBoard logs for the latest metrics
        import tensorboard as tb
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        
        tb_log_dir = os.path.join(experiment_dir, "TensorBoard")
        if os.path.exists(tb_log_dir):
            event_acc = EventAccumulator(tb_log_dir)
            event_acc.Reload()
            
            # Extract key metrics
            scalar_tags = event_acc.Tags()['scalars']
            
            if 'Loss/train' in scalar_tags:
                train_loss_events = event_acc.Scalars('Loss/train')
                if train_loss_events:
                    metrics['final_train_loss'] = train_loss_events[-1].value
                    metrics['best_train_loss'] = min(event.value for event in train_loss_events)
            
            if 'Mean Chamfer Dist/train' in scalar_tags:
                train_cd_events = event_acc.Scalars('Mean Chamfer Dist/train')
                if train_cd_events:
                    metrics['final_train_chamfer'] = train_cd_events[-1].value
                    metrics['best_train_chamfer'] = min(event.value for event in train_cd_events)
            
            if 'Mean Chamfer Dist/test' in scalar_tags:
                test_cd_events = event_acc.Scalars('Mean Chamfer Dist/test')
                if test_cd_events:
                    metrics['final_test_chamfer'] = test_cd_events[-1].value
                    metrics['best_test_chamfer'] = min(event.value for event in test_cd_events)
            
            if 'Loss/train_snnl_reg' in scalar_tags:
                snnl_events = event_acc.Scalars('Loss/train_snnl_reg')
                if snnl_events:
                    metrics['final_snnl_loss'] = snnl_events[-1].value
            
            if 'Loss/train_kl' in scalar_tags:
                kl_events = event_acc.Scalars('Loss/train_kl')
                if kl_events:
                    metrics['final_kl_loss'] = kl_events[-1].value
    
    except Exception as e:
        logging.warning(f"Could not read TensorBoard logs from {experiment_dir}: {e}")
        
        # Fallback: try to read from logs file if available
        logs_file = os.path.join(experiment_dir, "Logs.pth")
        if os.path.exists(logs_file):
            try:
                logs = torch.load(logs_file)
                if 'loss' in logs and len(logs['loss']) > 0:
                    metrics['final_train_loss'] = logs['loss'][-1]
                    metrics['best_train_loss'] = min(logs['loss'])
            except Exception as e2:
                logging.warning(f"Could not read logs from {logs_file}: {e2}")
    
    return metrics


def objective(trial: Trial, 
              base_experiment_dir: str,
              default_specs: Dict[str, Any],
              max_epochs: int = 100) -> float:
    """
    Objective function for Optuna optimization.
    
    Args:
        trial: Optuna trial object
        base_experiment_dir: Base directory for experiments
        default_specs: Default experiment specifications
        max_epochs: Maximum number of training epochs
        
    Returns:
        Objective value to minimize (e.g., test Chamfer distance)
    """
    
    # Get hyperparameters for this trial
    hparams = get_hyperparameter_search_space(trial)
    
    # Create experiment directory for this trial
    exp_name = f"optuna_trial_{trial.number:04d}"
    exp_dir = os.path.join(base_experiment_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    try:
        # Create specs with hyperparameters
        specs = create_specs_with_hyperparameters(default_specs, hparams)
        
        # Reduce number of epochs for faster search
        specs['NumEpochs'] = max_epochs
        specs['SnapshotFrequency'] = max(max_epochs // 4, 25)  # Save fewer checkpoints
        specs['LogFrequency'] = max(max_epochs // 10, 10)  # Log less frequently
        specs['EvalTrainFrequency'] = max(max_epochs // 5, 20)  # Evaluate less frequently
        specs['EvalTestFrequency'] = max(max_epochs // 4, 25)  # Evaluate less frequently
        
        # Save specs to experiment directory
        specs_file = os.path.join(exp_dir, "specs.json")
        with open(specs_file, 'w') as f:
            json.dump(specs, f, indent=4)
        
        # Modify global variables in training module
        modify_training_globals(hparams)
        
        # Log trial information
        logging.info(f"Starting trial {trial.number} with hyperparameters:")
        for key, value in hparams.items():
            logging.info(f"  {key}: {value}")
        
        # Run training
        start_time = time.time()
        main_function(exp_dir, continue_from=None, batch_split=1)
        training_time = time.time() - start_time
        
        # Extract metrics
        metrics = extract_metrics_from_experiment(exp_dir)
        
        # Log metrics
        logging.info(f"Trial {trial.number} completed in {training_time:.2f}s")
        logging.info(f"Metrics: {metrics}")
        
        # Report intermediate values to Optuna for pruning
        if 'best_train_chamfer' in metrics:
            trial.report(metrics['best_train_chamfer'], step=max_epochs)
        
        # Define objective value (what we want to minimize)
        # Priority: test Chamfer distance > train Chamfer distance > train loss
        if 'best_test_chamfer' in metrics:
            objective_value = metrics['best_test_chamfer']
        elif 'best_train_chamfer' in metrics:
            objective_value = metrics['best_train_chamfer']
        elif 'best_train_loss' in metrics:
            objective_value = metrics['best_train_loss']
        else:
            # If no metrics available, return a high penalty value
            logging.warning(f"No metrics available for trial {trial.number}")
            objective_value = 1000.0
        
        # Store all metrics as trial user attributes for later analysis
        for key, value in metrics.items():
            trial.set_user_attr(key, value)
        trial.set_user_attr('training_time', training_time)
        
        return objective_value
        
    except Exception as e:
        logging.error(f"Trial {trial.number} failed with error: {e}")
        # Return a high penalty value for failed trials
        return 1000.0


def run_optuna_search(base_experiment_dir: str,
                     default_specs_file: str,
                     n_trials: int = 50,
                     max_epochs: int = 100,
                     study_name: str = "deepsdf_vae_hparam_search",
                     storage_url: Optional[str] = None,
                     n_jobs: int = 1) -> optuna.Study:
    """
    Run Optuna hyperparameter search.
    
    Args:
        base_experiment_dir: Base directory for all experiments
        default_specs_file: Path to default specifications file
        n_trials: Number of trials to run
        max_epochs: Maximum epochs per trial
        study_name: Name of the Optuna study
        storage_url: Database URL for study storage (optional)
        n_jobs: Number of parallel jobs
        
    Returns:
        Completed Optuna study
    """
    
    # Load default specifications
    with open(default_specs_file, 'r') as f:
        default_specs = json.load(f)
    
    # Create base experiment directory
    os.makedirs(base_experiment_dir, exist_ok=True)
    
    # Create or load study
    if storage_url:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_url,
            load_if_exists=True,
            direction='minimize',
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )
    else:
        study = optuna.create_study(
            direction='minimize',
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )
    
    # Define objective function with fixed arguments
    def objective_wrapper(trial):
        return objective(trial, base_experiment_dir, default_specs, max_epochs)
    
    # Run optimization
    logging.info(f"Starting Optuna search with {n_trials} trials")
    study.optimize(objective_wrapper, n_trials=n_trials, n_jobs=n_jobs)
    
    # Log results
    logging.info("Search completed!")
    logging.info(f"Best trial: {study.best_trial.number}")
    logging.info(f"Best value: {study.best_value}")
    logging.info("Best parameters:")
    for key, value in study.best_params.items():
        logging.info(f"  {key}: {value}")
    
    # Save study results
    results_file = os.path.join(base_experiment_dir, "optuna_results.json")
    results = {
        'best_trial_number': study.best_trial.number,
        'best_value': study.best_value,
        'best_params': study.best_params,
        'best_user_attrs': study.best_trial.user_attrs,
        'n_trials': len(study.trials),
    }
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    logging.info(f"Results saved to {results_file}")
    
    return study


def main():
    """Main function for running Optuna hyperparameter search."""
    
    parser = argparse.ArgumentParser(
        description="Optuna-based hyperparameter search for Deep SDF VAE"
    )
    
    parser.add_argument(
        "--search_dir",
        type=str,
        default="/home/jakaria/INR/Deep3DComp/optuna_search",
        help="Directory to store search experiments"
    )
    
    parser.add_argument(
        "--default_specs",
        type=str,
        default="/home/jakaria/INR/Deep3DComp/examples/torus_bump_rotate/specs.json",
        help="Path to default specs.json file"
    )
    
    parser.add_argument(
        "--n_trials",
        type=int,
        default=50,
        help="Number of Optuna trials to run"
    )
    
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=100,
        help="Maximum epochs per trial (reduced for faster search)"
    )
    
    parser.add_argument(
        "--study_name",
        type=str,
        default="deepsdf_vae_hparam_search",
        help="Name of the Optuna study"
    )
    
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Database URL for study storage (e.g., sqlite:///example.db)"
    )
    
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="Number of parallel jobs"
    )
    
    # Add common deep_sdf arguments
    deep_sdf.add_common_args(parser)
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    deep_sdf.configure_logging(args)
    
    # Check if default specs file exists
    if not os.path.exists(args.default_specs):
        logging.error(f"Default specs file not found: {args.default_specs}")
        return
    
    # Run search
    study = run_optuna_search(
        base_experiment_dir=args.search_dir,
        default_specs_file=args.default_specs,
        n_trials=args.n_trials,
        max_epochs=args.max_epochs,
        study_name=args.study_name,
        storage_url=args.storage,
        n_jobs=args.n_jobs
    )
    
    logging.info("Hyperparameter search completed successfully!")


if __name__ == "__main__":
    main()
