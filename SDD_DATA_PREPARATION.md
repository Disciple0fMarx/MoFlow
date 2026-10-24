# SDD Data Preparation for Video-Conditioned MoFlow

This document explains how to prepare the Stanford Drone Dataset (SDD) for use with the video-conditioned MoFlow implementation in this repository.

## Overview

The SDD dataset preparation follows a similar pattern to the ETH/UCY datasets but requires additional steps to extract video features and align them with trajectory data.

## Steps to Prepare SDD Data

### 1. Prepare SDD Video Files

First, obtain the SDD dataset videos and organize them in a directory structure like:

```
/path/to/sdd/videos/
├── bookstore.mp4
├── coupa.mp4
├── deathCircle.mp4
├── gates.mp4
├── hyang.mp4
├── little.mp4
├── nexus.mp4
└── quad.mp4
```

### 2. Extract Video Features

Use the provided script to extract ResNet-18 features from SDD videos with a stride of 12 (to match ETH/UCY preprocessing):

```bash
python prepare_sdd_data.py \
    --data-root /path/to/sdd/videos \
    --features-root ./features/resnet18 \
    --pkl-dir ./data/sdd/original \
    --stride 12
```

This will:
- Extract features using `video_encoder.custom_video.py` with stride=12
- Save features as `{scene}.npy` and `{scene}.manifest.parquet` in the features directory
- Create frame index files in `{pkl-dir}/{scene}/` for train/test splits

### 3. Verify Data Preparation

After running the preparation script, you should see:

```
[*] Loading canonical MoFlow pickle: ./data/sdd/original/sdd_train.pkl
[*] Found 8985 samples with sequence length 20 timesteps.
[*] Indexing raw tracking files across all scenes for signature matching...
[*] Hashing X raw windows for fast spatial queries...
[*] Re-aligning pickle indexes to video frame IDs...
[+] Complete. Successfully matched X/Y samples.
[==>] Wrote sidecar to: ./data/sdd/original/sdd/sdd_train_frame_index.pkl (N=Z, stride=12)
```

### 4. Configure Model for Video Features

Update your configuration file (e.g., `cfg/sdd/cor_fm.yml`) to enable video features:

```yaml
MODEL:
  CONTEXT_ENCODER:
    NAME: ETHEncoder
    # ... other settings ...
    # ------------------- Video Configuration -------------------
    USE_VIDEO: True          # Enable global video encoder
    USE_AGENT_VIDEO: False   # Disable agent-centric video encoder (for now)
    USE_TRI_MODAL_FUSION: False # Disable tri-modal fusion (for now)
    VIDEO_DIM_RAW: 512       # Raw video feature dimension (ResNet-18)
    VIDEO_DIM: 32            # Projected video dimension
    AGENT_CROP_SIZE: [64, 64] # [height, width] for crops
    RESNET_FREEZE_BLOCKS: 2  # Frozen ResNet blocks
    SPATIAL_DROPOUT_RATE: 0.5 # Dropout after backbone
    VIDEO_STRIDE: 12         # Frame stride for feature extraction (matches preprocessing)
```

### 5. Training with SDD Video Features

Once data is prepared and configured, you can train the model:

```bash
# Train Flow Matching model on SDD with global video features
python fm_sdd.py --cfg cfg/sdd/cor_fm.yml

# Evaluate on test set
python fm_sdd.py --cfg cfg/sdd/cor_fm.yml --eval
```

## Important Notes

### Coordinate System Considerations

1. **Video Features**: The global video encoder is coordinate-system agnostic - it extracts visual features from raw pixels regardless of coordinate system.

2. **Trajectory Data**: The trajectory prediction model expects coordinates in **world units (meters)**. 

3. **SDD Data Format**: The provided SDD pickle files contain trajectory coordinates in **pixel units**.

4. **Homography Requirement**: For physically meaningful predictions in metric units, you need to:
   - Estimate homography matrices to convert pixel coordinates to world coordinates
   - OR use an identity transformation (pixels ≈ meters) with the understanding that results will be in pixel space

### Current Implementation Limitations

The current implementation uses **zero placeholders** for video features because:
- Proper frame-to-sample alignment requires mapping each trajectory sample to its corresponding video frame
- This mapping depends on how the original SDD dataset was synchronized with video frames
- Without access to the original SDD synchronization information, we cannot guarantee correct alignment

To use real video features:
1. Implement proper frame IDentify the frame correspondence between your trajectory samples and video frames
2. Modify `data/dataloader_sdd.py` to map samples to correct frame IDs
3. The video feature loading code will then extract the correct features using the FrameFeatureLookup

### Leave-One-Scene-Out (LOSO) Evaluation

The SDD dataset preparation supports LOSO evaluation consistent with ETH/UCY protocols:

1. The data loader treats the entire SDD dataset as a single "scene" for video features
2. For LOSO, you would need to split the SDD data into its constituent scenes (bookstore, coupa, etc.)
3. Each iteration would hold out one scene for testing and train on the remaining 7 scenes
4. Configuration and training scripts would need modification to support this workflow

## Troubleshooting

### Missing Features Error
If you see: "Expected features at: ./features/resnet18/sdd.npy"
- Run the feature extraction step with correct paths
- Verify the video files exist at your `--data-root` path

### Shape Mismatch Errors
If you encounter tensor shape errors during training:
- Verify that `past_frames` in your config matches the feature sequence length
- Check that `VIDEO_DIM_RAW` matches the feature dimension of your extracted features (512 for ResNet-18)

### Memory Issues
- Reduce `train_batch_size` in your config if you encounter OOM errors
- The feature extraction process is memory-intensive; consider processing videos one at a time if needed