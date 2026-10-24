# MoFlow (Video-Conditioned Extension) - Claude Guidelines

## Project Overview
This project implements an advanced Conditional Flow Matching model for human trajectory prediction, extending the base MoFlow architecture with dual-stream video conditioning:
1. **Global Scene Encoder**: Extracts scene-level features from full frames using ResNet-18
2. **Agent-Centric Video Encoder**: Extracts agent-specific features from crops around each agent's position
3. **Tri-Modal Cascaded Cross-Attention Fusion**: Combines trajectory, agent video, and global video features through sequential cross-attention with strict identity masking

The project supports the ETH/UCY datasets (eth, hotel, univ, zara1, zara2) and provides flexible configuration for ablation studies.

## Key Architectural Components

### Core Modules
- **MotionTransformer**: Main model in `models/backbone_eth_ucy.py`
- **Context Encoder**: Located in `models/context_encoder/` with multiple implementations:
  - `ETHEncoder`: Primary encoder supporting both original variant A and new tri-modal fusion
  - `MTREncoder`: Alternative transformer-based encoder
- **Video Encoders**: In `video_encoder/`:
  - `GlobalVideoEncoder`: Extracts scene-level features
  - `AgentVideoEncoder`: Extracts agent-centric features from crops (newly implemented)
- **Data Loader**: `data/dataloader_eth_ucy.py` handles trajectory data and optional video feature extraction

### Recent Enhancements
This branch (`feature/new-vid-enc`) introduces:
1. **AgentVideoEncoder**: ResNet-18 based encoder for agent-centric crops with configurable freezing and dropout
2. **Tri-Modal Fusion**: Sequential cross-attention in `ETHEncoder`:
   - Step A: Trajectory queries agent video features (with identity mask)
   - Step B: Agent-fused features query global video features
3. **Enhanced Dataloader**: Optional extraction of agent-centric crops using homography-based coordinate transformation
4. **Configuration Extensions**: New hyperparameters in `cfg/eth_ucy/cor_fm.yml` for controlling the new features

## Architectural Non-Negotiables
- **No 3D Convolutions**: Visual extraction uses strictly 2D backbones (ResNet-18) over $T_{obs}$ frames
- **Strict Coordinate Synchronization**: When modifying `data/dataloader_eth_ucy.py`, agent ID arrays and temporal indices must maintain perfect alignment with extracted video crops
- **Mathematical Rigor in Attention**: Custom cross-attention blocks must ensure dimensionality matches $D=128$ (or as defined in cfg) and identity masks must prevent data leakage across agents
- **Tensor Shape Consistency**: All modules must preserve expected tensor shapes for seamless integration:
  - Trajectory features: [B, A, P, 6] → [B, A, D]
  - Global video features: [B, T_obs, D_raw] → [B, T_obs, D]
  - Agent video features: [B, A, T_obs, 3, h, w] → [B, A, T_obs, D]
  - Final context: [B, A, D] (fed to motion decoder)

## Configuration System
The project uses YAML configuration files in `cfg/eth_ucy/`. Key files:
- `cor_fm.yml`: Main configuration for FlowMatching experiments
- `imle.yml`: Configuration for IMLE (implicit model learning) experiments

### Key Configuration Sections
```yaml
MODEL:
  CONTEXT_ENCODER:
    # Original variant A (per-timestep concat)
    USE_VIDEO: True
    VIDEO_DIM_RAW: 512
    VIDEO_DIM: 32
    
    # NEW: Agent-centric video and tri-modal fusion
    USE_AGENT_VIDEO: False     # Enable agent-centric video encoder
    USE_TRI_MODAL_FUSION: False # Enable tri-modal cascaded fusion
    AGENT_CROP_SIZE: [64, 64]  # [height, width] for crops
    RESNET_FREEZE_BLOCKS: 2    # Frozen ResNet blocks
    SPATIAL_DROPOUT_RATE: 0.5  # Dropout after backbone
```

## Essential Commands

### Training & Evaluation
```bash
# Train Flow Matching model (ETH dataset)
python fm_eth.py --cfg cfg/eth_ucy/cor_fm.yml --subset eth

# Train with hotel subset
python fm_eth.py --cfg cfg/eth_ucy/cor_fm.yml --subset hotel

# Evaluate on test set
python fm_eth.py --cfg cfg/eth_ucy/cor_fm.yml --subset eth --eval

# Train IMLE model
python imle_eth.py --cfg cfg/eth_ucy/imle.yml --subset eth
```

### Video Processing Utilities
```bash
# Build frame index (required for video features)
python -m video_encoder.scripts.build_frame_index \
    --data-root <path_to_raw_data> \
    --pkl-dir <path_to_processed_data> \
    --scene eth \
    --split train

# Encode video features (global scene)
python -m video_encoder \
    --backbone resnet18 \
    --data-root <path_to_raw_frames> \
    --out-dir <output_features_dir> \
    --split-parquet <path_to_split_parquet>
```

### Testing
```bash
# Run specific test modules
python -m pytest tests/ -v
python test_video_encoder.py
python test_integrated_encoder.py
```

## Development Guidelines

### Version Control
- **Branch Naming**: Use descriptive names like `feature/agent-video-enc`, `fix/tensor-sync`, etc.
- **Commit Messages**: Follow conventional commits format:
  - `feat(module): description` for new features
  - `fix(module): description` for bug fixes
  - `refactor(module): description` for code refactoring
  - `test(module): description` for test additions/changes
- **Always pull before pushing** to avoid merge conflicts

### Code Quality
- **Type Hints**: Use strict Python type hints, especially for tensor shapes:
  ```python
  tensor: torch.Tensor  # shape: [B, N, T, D]
  ```
- **Tensor Operations**: Prefer `einops.rearrange` over `torch.view` or `torch.transpose`
- **Formatting**: Run `black` and `isort` before committing:
  ```bash
  black .
  isort .
  ```
- **Documentation**: Include docstrings for all public classes and methods
- **Error Handling**: Validate tensor shapes and inputs where appropriate

### Working with Video Features
1. **Frame Index Generation**: Always build frame index before training with video features
2. **Coordinate Systems**: 
   - World coordinates: Used in trajectory data (meters)
   - Pixel coordinates: Used for image cropping (pixels)
   - Conversion: Use `world_to_pixel()` method in dataloader
3. **Crop Extraction**: 
   - Agent-centric crops are extracted around each agent's position
   - Default size: 64×64 pixels (configurable)
   - Edge handling: Zero-padding when crops exceed image boundaries
4. **Feature Consistency**: Ensure all video features (global and agent) use the same backbone and preprocessing

### Debugging Tips
- **Tensor Shape Errors**: Print shapes at key points using `tensor.shape`
- **Coordinate Misalignment**: Visualize a sample crop to verify alignment
- **Attention Issues**: Check that identity masks prevent cross-agent attention
- **NaN Losses**: Check for exploding gradients, consider gradient clipping

## Essential Skills for Claude Code

When working with this project, enable these skills for optimal assistance:

```
/update-config      - For modifying project settings.json
/verify             - To test that changes work as expected
/run                - To launch and drive the application
/code-review        - To review diffs for correctness and quality
/simplify           - To apply simplification and efficiency fixes
/fewer-permission-prompts - To reduce permission prompts for common operations
/loop               - For recurring tasks like checking training progress
/keybindings-help   - To customize keyboard shortcuts
```

### Recommended Skill Combinations
- **Development Cycle**: `/update-config` → `/run` → `/verify` → `/code-review`
- **Refactoring**: `/simplify` → `/code-review` → `/test`
- **Debugging**: `/verify` → `/run` with specific test cases
- **Maintenance**: `/fewer-permission-prompts` → `/update-config`

### Project-Specific Automation
This project includes pre-configured hooks in `.claude/settings.json` that:
- Automatically format Python files with black and isort on save
- Provide confirmation prompts before running training scripts
- Reduce permission prompts for common safe operations
- Offer helpful reminders after video processing and testing

## Project Structure Summary
```
MoFlow/
├── cfg/                    # Configuration files (YAML)
├── data/                   # Dataloaders and data utilities
│   ├── dataloader_eth_ucy.py  # Main ETH/UCY dataloader (enhanced)
│   └── ...                 # Other dataset loaders
├── models/                 # Neural network modules
│   ├── backbone_eth_ucy.py    # Main MotionTransformer model
│   ├── backbone.py            # NBA model variant
│   └── context_encoder/       # Context encoding implementations
│       ├── eth_encoder.py     # ETH-specific encoder (enhanced)
│       ├── mtr_encoder.py     # MotionTransformer encoder
│       └── fusion_module.py   # Cross-modal fusion (original)
├── video_encoder/          # Video processing modules
│   ├── agent_encoder.py     # NEW: Agent-centric video encoder
│   ├── encoder.py           # Global video encoder
│   ├── __init__.py          # Module exports
│   └── scripts/             # Utilities (build_frame_index, etc.)
├── fm_eth.py               # Flow matching training script
├── imle_eth.py             # IMLE training script
└── tests/                  # Test modules
```

## Getting Started
1. **Clone the repository** and ensure you're on the correct branch:
   ```bash
   git checkout feature/new-vid-enc
   ```
2. **Set up the environment** (if not already done):
   ```bash
   # Assuming .venv exists
   source .venv/bin/activate
   ```
3. **Prepare data**: Ensure ETH/UCY data is in the expected format
4. **Build frame indices** for video features (if using video):
   ```bash
   python -m video_encoder.scripts.build_frame_index --data-root /path/to/raw --pkl-dir /path/to/processed --scene eth --split train
   ```
5. **Run experiments**:
   ```bash
   # Baseline (no new features)
   python fm_eth.py --cfg cfg/eth_ucy/cor_fm.yml --subset eth
   
   # With agent-centric video only
   # (Edit cfg/eth_ucy/cor_fm.yml: set USE_AGENT_VIDEO: True)
   
   # Full tri-modal fusion
   # (Edit cfg/eth_ucy/cor_fm.yml: set USE_AGENT_VIDEO: True, USE_TRI_MODAL_FUSION: True)
   ```

## Troubleshooting Common Issues
- **ModuleNotFoundError**: Ensure you're in the project root and have activated the virtual environment
- **Shape Mismatch Errors**: Check that all tensor dimensions match expectations, especially after einops operations
- **Video Feature Errors**: Verify frame indices are built and paths are correct in configuration
- **CUDA Out of Memory**: Reduce batch size or enable gradient checkpointing
- **NaN Losses**: Check learning rate, gradient clipping, and weight initialization

## References
- Conditional Flow Matching: [Lipman et al., 2023](https://arxiv.org/abs/2210.02747)
- Social Trajectory Prediction: ETH/UCY Benchmark
- Video Conditioning Approaches: Insights from Introvert and related works
- Cross-Attention Mechanisms: Vaswani et al., 2017 (Attention is All You Need)

---

*This document serves as the single source of truth for project conventions, architecture, and development practices. For detailed information about Claude Code skills specific to this project, see SKILLS.md. Update this document when significant changes occur to maintain accuracy.*