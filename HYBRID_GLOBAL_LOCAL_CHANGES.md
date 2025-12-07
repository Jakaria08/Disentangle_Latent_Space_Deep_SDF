# Hybrid Global + Local Latent Code Implementation

## Overview
Updated the training system to support **both global and local latent codes** for better disentanglement while maintaining reconstruction quality.

## Architecture Changes

### **Global Codes** (256D per shape)
- **Purpose**: Capture high-level, disentangled features
- **Examples**: Overall brain shape, size, major sulci patterns
- **One code per shape** - same for all query points in that shape

### **Local Codes** (8³ = 512 codes of 16D per shape)  
- **Purpose**: Capture fine-grained local details and corrections
- **Spatial organization**: 8×8×8 grid covering [-1,1]³ space
- **Interpolation**: Trilinear interpolation for smooth transitions
- **Smaller than before**: 16D (was 32D) to encourage sparsity

### **How They Work Together**
```
For each query point (x, y, z):

1. Get GLOBAL code → [256D] (same for entire shape)
2. Interpolate LOCAL code → [16D] (varies by position)
3. Concatenate: [global, local, xyz] → [256+16+3=275D]
4. Feed to decoder MLP → SDF prediction
```

## Files Modified

### 1. `train_local_shapes.py`
**Key Changes:**
- Added `global_lat_vecs` embedding: `[num_scenes, 256]`
- Kept `local_lat_vecs` embedding: `[num_scenes * 512, 16]`
- **3 parameter groups** in optimizer:
  - Decoder parameters: LR = 0.0005
  - Global codes: LR = 0.001 (higher for faster convergence)
  - Local codes: LR = 0.0001 (10x lower to encourage sparsity)
  
**Regularization:**
- Global codes: Standard L2 regularization
- Local codes: **10x stronger** regularization (controlled by `LocalSparsityWeight`)
- Encourages network to rely on global code, use local only when necessary

**Saving/Loading:**
- Global codes saved separately: `{epoch}_global.pth`
- Local codes saved as before: `{epoch}.pth`

### 2. `networks/local_decoder.py`
**Key Changes:**
- Updated `forward()` signature:
  ```python
  # OLD: forward(xyz, all_local_codes, indices)
  # NEW: forward(xyz, global_codes, all_local_codes, indices)
  ```
  
- **Combined latent size**: `global_latent_size + latent_size` fed to decoder
- Global code passed directly (no interpolation needed)
- Local code interpolated based on spatial position

### 3. `examples/CALSNIC_control_L_local/specs.json`
**New Parameters:**
```json
{
  "GlobalCodeSize": 256,        // Global latent dimension
  "LocalCodeSize": 16,          // Local latent dimension (reduced from 32)
  "GridSize": 8,                // Still 8×8×8 grid
  "LocalSparsityWeight": 10.0,  // 10x stronger regularization on local codes
  
  "NetworkSpecs": {
    "global_latent_size": 256,  // Pass to decoder
    ...
  },
  
  "LearningRateSchedule": [
    {...},                       // Decoder: 0.0005
    {..., "Initial": 0.001},     // Global codes: 0.001
    {..., "Initial": 0.0001}     // Local codes: 0.0001
  ]
}
```

## Benefits

### For Disentanglement
✅ **Global code** (256D) is much more interpretable than 512×32D local codes  
✅ Can manipulate high-level features by editing global code  
✅ Fewer parameters to analyze for disentanglement (256 vs 16,384)

### For Reconstruction Quality
✅ **Local codes** compensate for global code limitations  
✅ Fine-grained details captured in complex regions (sulci/gyri)  
✅ Sparse regularization ensures local codes only activate when needed

### Training Efficiency
✅ **Progressive specialization**: Global learns first (higher LR), local refines  
✅ Smaller local codes (16D vs 32D) = faster training, less memory  
✅ Strong regularization prevents overfitting to local noise

## Training Tips

### 1. **Monitor Both Code Types**
TensorBoard logs now track:
- `Mean Latent Magnitude/global` - should stabilize around 1-2
- `Mean Latent Magnitude/local` - should be smaller (~0.5-1.0)
- `Learning Rate/Global` and `Learning Rate/Local`

### 2. **Adjust Sparsity Weight**
If local codes are **too active** (high magnitude):
```json
"LocalSparsityWeight": 20.0  // Increase from 10.0
```

If reconstruction **quality suffers**:
```json
"LocalSparsityWeight": 5.0   // Decrease from 10.0
```

### 3. **Progressive Training (Optional)**
Add to training loop (around line 285):
```python
# Freeze local codes for first 500 epochs
if epoch <= 500:
    for param in local_lat_vecs.parameters():
        param.requires_grad = False
else:
    for param in local_lat_vecs.parameters():
        param.requires_grad = True
```

## Usage

### Training from Scratch
```bash
python train_local_shapes.py \
  --experiment examples/CALSNIC_control_L_local
```

### Continue Training
```bash
python train_local_shapes.py \
  --experiment examples/CALSNIC_control_L_local \
  --continue latest
```

### Reconstruction (needs update)
You'll need to update reconstruction scripts to load and use both global and local codes:
```python
# Load global codes
global_codes = torch.load("LatentCodes/latest_global.pth")
# Load local codes  
local_codes = torch.load("LatentCodes/latest.pth")

# Pass both to decoder
sdf = decoder(xyz, global_codes[shape_idx], local_codes, shape_idx)
```

## Expected Results

### Training Progress
- **Epochs 1-500**: Global code learns overall brain structure
- **Epochs 500-1000**: Local codes activate in complex regions (sulci)
- **Epochs 1000+**: Fine-tuning, local sparsity increases

### Final State
- **Global code magnitude**: ~1.5-2.5
- **Local code magnitude**: ~0.5-1.0 (with many near zero)
- **Reconstruction quality**: Comparable to pure local, but with interpretable global code

### Disentanglement Analysis
After training, analyze global code dimensions:
```python
# Extract global codes
all_globals = global_lat_vecs.weight.detach().cpu().numpy()  # [N, 256]

# PCA for visualization
from sklearn.decomposition import PCA
pca = PCA(n_components=10)
global_pca = pca.fit_transform(all_globals)

# Check which dimensions correlate with brain features
# (size, shape, sulci depth, etc.)
```

## Next Steps

1. **Train and monitor** TensorBoard for both code types
2. **Adjust LocalSparsityWeight** based on results
3. **Visualize global code** to identify disentangled dimensions
4. **Update reconstruction scripts** to use both codes
5. **Perform interpolation** in global space for smooth shape transitions

## Questions?

Check the code comments or reach out if you need clarification on any part of the implementation!
