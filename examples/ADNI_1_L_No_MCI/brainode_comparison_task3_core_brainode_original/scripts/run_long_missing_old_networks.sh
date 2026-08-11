#!/usr/bin/env bash
set -euo pipefail

REPO="/home/jakaria/INR/Deep3DComp"
PY="/home/jakaria/anaconda3/envs/inr_sdf/bin/python"
GPU="${1:-0}"

EXPERIMENTS=(
  "examples/ADNI_1_L_No_MCI/longitudinal_direct_full256_smallnet_siren_optimized"
  "examples/ADNI_1_L_No_MCI/longitudinal_direct_full256_smallnet_deepsdf_optimized"
  "examples/ADNI_1_L_No_MCI/longitudinal_direct_pca32_siren_optimized"
  "examples/ADNI_1_L_No_MCI/longitudinal_direct_pca32_deepsdf_optimized"
)

cd "$REPO"

for EXP in "${EXPERIMENTS[@]}"; do
  echo "==> Validating $EXP"
  "$PY" train_deep_sdf_longitudinal_direct_flow.py \
    -e "$EXP" \
    --validate-only

  if [[ -f "$EXP/ModelParameters/best.pth" ]]; then
    echo "==> Found existing best checkpoint for $EXP; skipping training"
  else
    echo "==> Training $EXP on GPU $GPU"
    "$PY" train_deep_sdf_longitudinal_direct_flow.py \
      -e "$EXP" \
      --gpu "$GPU"
  fi

  echo "==> Evaluating $EXP"
  "$PY" evaluate_deep_sdf_longitudinal_direct_flow.py \
    -e "$EXP" \
    --checkpoint best \
    --split all \
    --gpu "$GPU" \
    --composed-step-years 0.5

  echo "==> Building dashboard for $EXP"
  "$PY" visualize_deep_sdf_longitudinal_direct_flow.py \
    -e "$EXP" \
    --checkpoint best \
    --split test
done

echo "==> Refreshing BrainODE old pairwise metrics"
"$PY" examples/ADNI_1_L_No_MCI/brainode_comparison_task3_core_brainode_original/scripts/evaluate_brainode.py \
  --checkpoint best \
  --split all \
  --pairwise \
  --device "cuda:$GPU" \
  --output-dir examples/ADNI_1_L_No_MCI/brainode_comparison_task3_core_brainode_original/training/core_attention_pca150/evaluation/best_pairwise

echo "==> Refreshing comparison tables"
"$PY" examples/ADNI_1_L_No_MCI/brainode_comparison_task3_core_brainode_original/scripts/compare_old_and_qc_longitudinal_results.py

echo "==> Refreshing BrainODE conditional future/OOD volume tables"
"$PY" examples/ADNI_1_L_No_MCI/brainode_comparison_task3_core_brainode_original/scripts/brainode_conditional_future_volume.py \
  --device "cuda:$GPU"

echo "Done."
