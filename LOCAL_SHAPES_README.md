# Deep Local Shapes Implementation

This implementation follows the **Deep Local Shapes** paper (ECCV 2020) for learning local SDF priors with spatial decomposition.

## Key Features

- **Local Latent Codes**: Instead of 1 global code per shape, uses a **8×8×8 grid = 512 local codes**
- **Small Code Size**: Each local code is only **32 dimensions** (vs 256 for global)
- **Trilinear Interpolation**: Query points get codes interpolated from 8 neighboring grid cells
- **ReLU Backbone**: Uses standard ReLU-based DeepSDF architecture (as per original paper)

## Architecture Details

### Grid Structure
- Grid size: 8×8×8 = 512 cells
- Each cell covers: 0.25 units in normalized [-1, 1]³ space
- Local code dimension: 32

### Network
- Architecture: DeepSDF decoder (ReLU-based)
- Hidden layers: 8 layers × 512 units
- Activation: ReLU
- Skip connections: at layer 4 (latent code re-injected)
- Total parameters per shape: 512 codes × 32 dims = 16,384 floats

### Comparison with Global Approach

| Aspect | Global (Original) | Local (This Implementation) |
|--------|------------------|----------------------------|
| Codes per shape | 1 × 256 = 256 | 512 × 32 = 16,384 |
| Representation | Holistic shape | Spatially decomposed |
| Detail capacity | Limited by single code | Higher (local details) |
| Memory | 1 KB/shape | 64 KB/shape |

## Files Created

1. **`networks/local_decoder.py`**
   - `LocalShapesDecoder`: Main decoder with trilinear interpolation
   - Uses DeepSDF's ReLU-based architecture (exactly as in the paper)
   - Handles spatial grid and code interpolation

2. **`train_local_shapes.py`**
   - Training script adapted for local codes
   - Manages [num_shapes, 512, 32] latent tensor

3. **`reconstruct_local_shapes.py`**
   - Reconstruction by optimizing 512 local codes per test shape
   - Saves local codes for each reconstructed shape

4. **`examples/CALSNIC_control_L_local/specs.json`**
   - Configuration file for local shapes experiment

## Usage

### Training

```bash
python train_local_shapes.py \
    -e examples/CALSNIC_control_L_local \
    --batch_split 1
```

### Reconstruction

```bash
python reconstruct_local_shapes.py \
    -e examples/CALSNIC_control_L_local \
    -c latest \
    -d /path/to/data/SdfSamples \
    -s examples/splits/split_CALSNIC_control_L/test_split_calsnic.json \
    --iters 800
```

## Configuration Parameters

Key parameters in `specs.json`:

- **`GridSize`**: 8 (creates 8³ grid)
- **`LocalCodeSize`**: 32 (dimension of each local code)
- **`NetworkArch`**: "local_siren_decoder"
- **`CodeRegularization`**: true (L2 penalty on codes)
- **`CodeRegularizationLambda`**: 1e-4

## Implementation Details

### Trilinear Interpolation

For a query point `xyz` in [-1, 1]³:

1. Convert to grid coordinates: `grid_coords = (xyz + 1) * (grid_size - 1) / 2`
2. Get 8 neighboring grid cells (corners of containing cube)
3. Compute trilinear weights based on fractional position
4. Interpolate the 8 corner codes
5. Feed interpolated code to decoder

### Training Process

Each training iteration:

1. Sample points from multiple shapes
2. For each point, determine its shape index
3. Look up the shape's 8³ local code grid
4. Interpolate code at point's location
5. Decode SDF value
6. Backprop through interpolation and decoder

### Memory Requirements

- **Per shape in training**:
  - Local codes: 512 × 32 × 4 bytes = 64 KB
  - For 100 shapes: ~6.4 MB (manageable)

- **Per forward pass**:
  - Need to reshape codes: [num_shapes, 512, 32]
  - Interpolate for each query point
  - Slightly slower than global approach but more expressive

## Expected Benefits

1. **Better Detail Capture**: Local codes can represent fine geometric features
2. **Spatial Disentanglement**: Different brain regions encoded separately
3. **Local Shape Completion**: Missing regions can be inferred from neighbors
4. **Interpretability**: Can visualize which local codes capture what features

## Troubleshooting

### Out of Memory
- Reduce `ScenesPerBatch` in specs.json
- Increase `batch_split` parameter
- Reduce grid size (e.g., 6×6×6 = 216 codes)

### Slow Training
- Normal - each forward pass requires interpolation
- ~1.5-2× slower than global approach
- Consider using smaller grid first (4×4×4 = 64 codes)

### Poor Reconstruction
- Increase optimization iterations (--iters 1000+)
- Adjust learning rate in reconstruction
- Check L2 regularization strength

## Next Steps

After basic implementation works:

1. **Visualization**: Plot which local codes activate where
2. **Global + Local**: Add global code for coarse shape + local for details
3. **Adaptive Grid**: Use finer grid where detail is needed
4. **Overlapping Patches**: Smooth boundaries between cells

## References

- Deep Local Shapes (ECCV 2020): Chabra et al.
- Uses ReLU architecture exactly as described in the paper
- Grid size 8³ chosen as simple baseline (paper uses 32³ or 64³)
- Can be extended to SIREN for smoother surfaces later
