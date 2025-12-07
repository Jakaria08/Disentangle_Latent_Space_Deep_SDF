#!/bin/bash
# Quick Start Script for Deep Local Shapes Training

echo "======================================================================"
echo "Deep Local Shapes - Quick Start"
echo "======================================================================"
echo ""

# Configuration
EXPERIMENT_DIR="examples/CALSNIC_control_L_local"
ENV_NAME="pytorch_inr"

echo "Experiment: $EXPERIMENT_DIR"
echo "Environment: $ENV_NAME"
echo ""

# Step 1: Validate implementation
echo "Step 1: Validating implementation..."
echo "----------------------------------------------------------------------"
/home/jakaria/anaconda3/envs/$ENV_NAME/bin/python validate_local_shapes.py

if [ $? -ne 0 ]; then
    echo ""
    echo "✗ Validation failed! Please check the errors above."
    exit 1
fi

echo ""
echo "✓ Validation passed!"
echo ""

# Step 2: Check if data exists
echo "Step 2: Checking data..."
echo "----------------------------------------------------------------------"

DATA_SOURCE=$(grep -o '"DataSource" : "[^"]*"' $EXPERIMENT_DIR/specs.json | cut -d'"' -f4)
TRAIN_SPLIT=$(grep -o '"TrainSplit" : "[^"]*"' $EXPERIMENT_DIR/specs.json | cut -d'"' -f4)

if [ ! -d "$DATA_SOURCE" ]; then
    echo "✗ Data source not found: $DATA_SOURCE"
    echo "  Please update DataSource in $EXPERIMENT_DIR/specs.json"
    exit 1
fi

if [ ! -f "$TRAIN_SPLIT" ]; then
    echo "✗ Train split not found: $TRAIN_SPLIT"
    echo "  Please update TrainSplit in $EXPERIMENT_DIR/specs.json"
    exit 1
fi

echo "✓ Data source found: $DATA_SOURCE"
echo "✓ Train split found: $TRAIN_SPLIT"
echo ""

# Step 3: Show configuration
echo "Step 3: Configuration Summary"
echo "----------------------------------------------------------------------"
echo "Grid Size: 8×8×8 = 512 local codes per shape"
echo "Local Code Dimension: 32"
echo "Network: 8 layers × 512 units (ReLU)"
echo "Training Epochs: 2001"
echo "Regularization: L2 (λ=1e-4)"
echo ""

# Step 4: Offer to start training
echo "Step 4: Ready to Train"
echo "----------------------------------------------------------------------"
echo ""
echo "To start training, run:"
echo ""
echo "  python train_local_shapes.py -e $EXPERIMENT_DIR"
echo ""
echo "Or with more memory efficiency (batch splitting):"
echo ""
echo "  python train_local_shapes.py -e $EXPERIMENT_DIR --batch_split 2"
echo ""
echo "Monitor training with TensorBoard:"
echo ""
echo "  tensorboard --logdir $EXPERIMENT_DIR/tb_logs"
echo ""
echo "======================================================================"
