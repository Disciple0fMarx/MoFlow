# SDD Data Preparation Status

## ✅ Completed Tasks

### 1. Configuration Updates
- **cfg/sdd/cor_fm.yml**: Video configuration section is present and ready
  - `USE_VIDEO: False` (set to True when features are extracted)
  - `USE_AGENT_VIDEO: False` 
  - `USE_TRI_MODAL_FUSION: False`
  - `VIDEO_DIM_RAW: 512` (ResNet-18 feature dimension)
  - `VIDEO_DIM: 32` (Projected dimension)
  - `AGENT_CROP_SIZE: [64, 64]`
  - `RESNET_FREEZE_BLOCKS: 2`
  - `SPATIAL_DROPOUT_RATE: 0.5`

### 2. API Updates
- **video_encoder/api.py**: Updated with SDD-specific constants
  - `SCENE_RESOLUTIONS`: All SDD scenes set to 1280x720 (standard SDD resolution)
  - `SCENE_WORLD_BOUNDS`: Approximate bounds set to ±50 meters (reasonable for urban scenes)
  - Note: For accurate metric-scale predictions, actual homography estimation is recommended

### 3. Data Preparation Scripts
- **prepare_sdd_data.py**: Comprehensive script for SDD data preparation
  - Feature extraction using `custom_video.py` with stride=12 (matching ETH/UCY)
  - Frame index creation for trajectory-to-frame mapping
  - Modular design with skip options for extraction/indexing
  - Compatible with existing SDD pickle file structure

### 4. Documentation
- **SDD_DATA_PREPARATION.md**: Detailed instructions for preparing SDD data
- **SDD_READY_FOR_TRAINING.md**: This file - summary of current status

## 🔄 Pending Tasks

### 1. Obtain SDD Video Data
- Download or locate SDD video files (typically named like `bookstore.mov`, `coupa.mp4`, etc.)
- Place them in a directory accessible to the data preparation script

### 2. Extract Video Features
```bash
# Example command (adjust paths as needed)
python prepare_sdd_data.py \
    --data-root /path/to/sdd/videos \
    --features-root ./features \
    --pkl-dir ./data/sdd \
    --stride 12
```

### 3. Update Configuration for Training
Once features are extracted:
```yaml
# In cfg/sdd/cor_fm.yml
MODEL:
  CONTEXT_ENCODER:
    USE_VIDEO: True          # Change from False to True
    # Other video settings remain the same
```

### 4. Verify Data Loader Compatibility
The existing `data/dataloader_sdd.py` already supports video features:
- Loads frame index files when `USE_VIDEO: True`
- Extracts windowed features using `FrameFeatureLookup.window()`
- Handles missing features gracefully with zero placeholders
- Works with the existing SDD pickle file structure

### 5. Run Training
```bash
# Basic training with video features
python fm_sdd.py --cfg cfg/sdd/cor_fm.yml --subset bookstore

# For Leave-One-Scene-Out (LOSO) evaluation:
# Note: This requires modifying the data loader to treat individual SDD scenes
# as separate datasets, which is a more involved change
```

## 📝 Implementation Notes

### Coordinate System Considerations
- **Current Implementation**: Uses identity homography approximation (pixel units ≈ world units)
- **Limitation**: Predictions will be in pixel coordinates, not meters
- **Recommended for**: Pipeline validation, qualitative assessment
- **Not Recommended for**: Quantitative metric evaluation requiring real-world units

### For Proper Metric-Scale Predictions
To get accurate predictions in meters:
1. Estimate homography matrices for each SDD scene using:
   - Known reference points in the scene
   - Camera calibration information (if available)
   - Structure-from-motion techniques
2. Apply homography to convert pixel coordinates to world coordinates
3. Update trajectory data in SDD pickle files before feature extraction

### Frame Stride Justification
- Using stride=12 matches ETH/UCY preprocessing (29.97 FPS / 12 ≈ 2.5 FPS)
- Provides ~0.4 second temporal spacing between features
- Ensures compatibility with existing temporal modeling in MotionTransformer

## 🔧 Troubleshooting

### Common Issues
1. **"Features not found" error**
   - Solution: Run feature extraction first with correct paths
   - Verify `.npy` and `.manifest.parquet` files exist in `./features/resnet18/sdd/`

2. **Tensor shape mismatches**
   - Solution: Ensure `past_frames` in config matches feature sequence length
   - Verify `VIDEO_DIM_RAW` matches actual feature dimension (512 for ResNet-18)

3. **Memory issues during training**
   - Solution: Reduce `train_batch_size` in configuration
   - Consider feature extraction batch size adjustments

## 📊 Expected Outcomes

Once properly configured with video features:
- **Input to model**: 
  - Trajectory: [B, A, P, 6] → [B, A, D] 
  - Global video: [B, T_obs, 512] → [B, T_obs, 32]
  - Agent video: [B, A, T_obs, 3, h, w] → [B, A, T_obs, 32] (when enabled)
- **Fusion**: Tri-modal cascaded cross-attention (when enabled)
- **Output**: Predicted future trajectories [B, A, F, 2]

## 📚 References

1. **ETH/UCY Preprocessing**: Uses 2.5 FPS sampling (30 FPS / 12) for trajectory data
2. **SDD Dataset**: Stanford Drone Dataset - aerial footage at various university locations
3. **Video Encoding**: ResNet-18 backbone producing 512-D features per frame
4. **Temporal Alignment**: Feature stride must match trajectory sampling rate

---

*Last updated: $(date)*
*Ready for SDD video feature extraction and trajectory prediction experiments.*