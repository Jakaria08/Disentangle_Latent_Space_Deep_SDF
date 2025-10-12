#!/usr/bin/env python3
"""
Simple hyperparameter tester - tests one config at a time by modifying train_deep_sdf.py
This avoids CUDA memory issues by using subprocess to run each test independently
"""

import subprocess
import json
import os
import itertools
import time

def modify_train_file(w_cls, threshold, w_code_reg, temp_reg):
    """Temporarily modify train_deep_sdf.py with test parameters"""
    with open('train_deep_sdf.py', 'r') as f:
        content = f.read()
    
    # Replace the hyperparameter values
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if line.startswith('temp_reg = '):
            lines[i] = f'temp_reg = {temp_reg} # HYPERPARAM_TEST'
        elif line.startswith('w_cls = '):
            lines[i] = f'w_cls = {w_cls} # HYPERPARAM_TEST'
        elif line.startswith('threshold = '):
            lines[i] = f'threshold = {threshold} # HYPERPARAM_TEST'
        elif line.startswith('w_code_reg = '):
            lines[i] = f'w_code_reg = {w_code_reg} # HYPERPARAM_TEST'
    
    with open('train_deep_sdf.py', 'w') as f:
        f.write('\n'.join(lines))

def restore_train_file():
    """Restore original values"""
    with open('train_deep_sdf.py', 'r') as f:
        content = f.read()
    
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if '# HYPERPARAM_TEST' in line:
            if 'temp_reg' in line:
                lines[i] = 'temp_reg = 2 # change this?'
            elif 'w_cls' in line:
                lines[i] = 'w_cls = 0.01'
            elif 'threshold' in line:
                lines[i] = 'threshold = 0.05'
            elif 'w_code_reg' in line:
                lines[i] = 'w_code_reg = 1.0'
    
    with open('train_deep_sdf.py', 'w') as f:
        f.write('\n'.join(lines))

def parse_loss_from_log(log_file):
    """Extract initial and final SNNL reg loss from training log"""
    if not os.path.exists(log_file):
        return None, None
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    snnl_losses = []
    for line in lines:
        if 'SNNL Reg Loss:' in line:
            try:
                loss_val = float(line.split('SNNL Reg Loss:')[1].strip())
                snnl_losses.append(loss_val)
            except:
                pass
    
    if len(snnl_losses) >= 2:
        return snnl_losses[0], snnl_losses[-1]
    elif len(snnl_losses) == 1:
        return snnl_losses[0], snnl_losses[0]
    else:
        return None, None

def run_hyperparam_sweep():
    """Run hyperparameter sweep using subprocess approach"""
    
    # Define search space (smaller for testing)
    w_cls_values = [0.01, 0.05, 0.1, 0.5]
    threshold_values = [0.05, 0.1, 0.15, 0.2]
    w_code_reg_values = [0.5, 1.0, 2.0]
    temp_reg_values = [0.5, 1.0, 2.0, 5.0, 20.0]
    
    param_combinations = list(itertools.product(
        w_cls_values,
        threshold_values,
        w_code_reg_values,
        temp_reg_values
    ))
    
    print(f"Total combinations to test: {len(param_combinations)}")
    print(f"Each will run for 10 epochs")
    print("=" * 80)
    
    successful_configs = []
    results_file = 'examples/hippocampus_MS/all_unsupervised_hippo_saved_models/hyperparam_results.json'
    
    experiment_dir = 'examples/hippocampus_MS/all_unsupervised_hippo_saved_models_hyperparam_test'
    
    for idx, (w_cls, threshold, w_code_reg, temp_reg) in enumerate(param_combinations):
        print(f"\n{'='*80}")
        print(f"SWEEP {idx + 1}/{len(param_combinations)}")
        print(f"Testing: w_cls={w_cls}, threshold={threshold}, "
              f"w_code_reg={w_code_reg}, temp_reg={temp_reg}")
        print(f"{'='*80}\n")
        
        try:
            # Modify train file with test parameters
            modify_train_file(w_cls, threshold, w_code_reg, temp_reg)
            
            # Create temporary experiment directory
            os.makedirs(experiment_dir, exist_ok=True)
            
            # Copy specs from original experiment
            os.system(f'cp -r examples/hippocampus_MS/all_unsupervised_hippo_saved_models/specs.json {experiment_dir}/')
            
            # Modify specs to run for fewer epochs
            with open(f'{experiment_dir}/specs.json', 'r') as f:
                specs = json.load(f)
            specs['NumEpochs'] = 20  # Only 20 epochs for testing
            with open(f'{experiment_dir}/specs.json', 'w') as f:
                json.dump(specs, f, indent=4)
            
            # Run training as subprocess
            log_file = f'hyperparam_test_{idx}.log'
            cmd = f'python train_deep_sdf.py -e {experiment_dir} > {log_file} 2>&1'
            
            print(f"  Running training...")
            result = subprocess.run(cmd, shell=True, timeout=600)  # 10 minute timeout
            
            # Parse results
            initial_loss, final_loss = parse_loss_from_log(log_file)
            
            if initial_loss is not None and final_loss is not None:
                improvement = (initial_loss - final_loss) / initial_loss if initial_loss > 0 else 0
                
                print(f"\n  Results:")
                print(f"    Initial SNNL Reg Loss: {initial_loss:.6f}")
                print(f"    Final SNNL Reg Loss: {final_loss:.6f}")
                print(f"    Improvement: {improvement*100:.2f}%")
                
                if improvement >= 0.05:  # 5% improvement
                    print(f"    ✓ SUCCESS!")
                    successful_configs.append({
                        'w_cls': w_cls,
                        'threshold': threshold,
                        'w_code_reg': w_code_reg,
                        'temp_reg': temp_reg,
                        'initial_loss': initial_loss,
                        'final_loss': final_loss,
                        'improvement': improvement
                    })
                    
                    # Save incrementally
                    with open(results_file, 'w') as f:
                        json.dump(successful_configs, f, indent=2)
                else:
                    print(f"    ✗ FAILED (< 5% improvement)")
            else:
                print(f"    ✗ FAILED (could not parse losses)")
            
            # Cleanup
            os.system(f'rm -rf {experiment_dir}')
            
        except subprocess.TimeoutExpired:
            print(f"  ✗ TIMEOUT after 10 minutes")
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
        finally:
            # Always restore original file
            restore_train_file()
            time.sleep(2)  # Brief delay between runs
        
        # Stop if we found 5 good configs
        if len(successful_configs) >= 5:
            print(f"\nFound 5 successful configurations. Stopping search.")
            break
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"HYPERPARAMETER SEARCH COMPLETE")
    print(f"{'='*80}")
    print(f"Tested {min(idx + 1, len(param_combinations))} configurations")
    print(f"Found {len(successful_configs)} successful configurations")
    
    if successful_configs:
        sorted_configs = sorted(successful_configs, key=lambda x: x['improvement'], reverse=True)
        print(f"\nTop configurations:")
        for i, config in enumerate(sorted_configs[:3]):
            print(f"\n{i+1}. Improvement: {config['improvement']*100:.2f}%")
            print(f"   w_cls={config['w_cls']}, threshold={config['threshold']}, "
                  f"w_code_reg={config['w_code_reg']}, temp_reg={config['temp_reg']}")
        
        print(f"\nResults saved to: {results_file}")
    else:
        print("\nNo successful configurations found.")

if __name__ == "__main__":
    run_hyperparam_sweep()
