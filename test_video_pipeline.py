import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2

# Import the modified classes and collation function
from data.dataloader_eth_ucy import ETHDataset, seq_collate_eth

def test_pipeline():
    # 1. Comprehensive Config Mock to satisfy all internal checks
    class ContextEncoder:
        def __init__(self):
            self.AGENTS = 0  
            self.D_MODEL = 128 

    class ModelConfig:
        def __init__(self):
            self.CONTEXT_ENCODER = ContextEncoder()

    class DummyCfg:
        def __init__(self):
            # Dataset parameters
            self.agents = 2 
            self.past_frames = 8
            self.future_frames = 12
            
            # Normalization parameters
            self.data_norm = 'min_max' # Fixes the current AttributeError
            self.past_traj_min = -10.0
            self.past_traj_max = 10.0
            self.fut_traj_min = -10.0
            self.fut_traj_max = 10.0
            
            # Flags for augmentation/rotation
            self.rotate = False       
            self.rotate_aug = False   
            
            # Nested model config required by dataloader init
            self.MODEL = ModelConfig()

    cfg = DummyCfg()
    data_dir = 'data/eth_ucy'
    subset = 'eth'  # You can test 'hotel', 'zara1', etc.
    
    print(f"--- Starting Data Pipeline Test for: {subset} ---")

    try:
        # 2. Initialize the Modified Dataset (Teacher branch)
        dataset = ETHDataset(
            cfg, 
            training=True, 
            data_dir=data_dir, 
            subset=subset, 
            imle=False, 
            type='original' 
        )

        # 3. Setup DataLoader with our new collation logic
        loader = DataLoader(
            dataset, 
            batch_size=4, 
            shuffle=True, 
            collate_fn=seq_collate_eth
        )

        # 4. Fetch the first batch
        batch = next(iter(loader))

        # 5. Validate Shapes
        print("\n[Batch Verification]")
        print(f"Trajectory Shape (Past): {batch['past_traj'].shape}")
        
        if 'video' in batch:
            video_shape = batch['video'].shape
            print(f"Video Tensor Shape: {video_shape}")
            
            # Expected shape: [Batch, T_obs (8), Channels (3), H (224), W (224)]
            if video_shape == (4, 8, 3, 224, 224):
                print("✅ SUCCESS: Video tensor shape is correct.")
            else:
                print(f"❌ ERROR: Video shape mismatch. Got {video_shape}")

            # 6. Visual Check (Overlay ALL pedestrians per scene)
            sample_video = batch['video'][0]      # [8, 3, 224, 224] — first sample's video
            sample_indexes = batch['indexes']     # [4] — dataset indices for each sample in batch

            # Group all samples that share the same start_frame (i.e. same scene)
            frame_ids_in_batch = [int(dataset.frame_ids[int(idx)]) for idx in sample_indexes]
            target_frame = frame_ids_in_batch[0]
            scene_mask = [i for i, fid in enumerate(frame_ids_in_batch) if fid == target_frame]

            print(f"\n[Scene Info] Frame ID: {target_frame} | Pedestrians found: {len(scene_mask)}")

            # Collect trajectory points for every pedestrian in this scene
            # past_traj shape per sample: [1, 8, 6] -> agent 0 -> [8, 6] -> [:, :2] = abs (x,y)
            all_traj_px = []
            for i in scene_mask:
                traj_pts = batch['past_traj'][i, 0, :, :2].cpu().numpy()  # [8, 2]
                traj_px = (traj_pts + 1) / 2 * 224                         # denorm to [0, 224]
                all_traj_px.append(traj_px)

            # Color palette — one color per pedestrian
            colors = plt.cm.get_cmap('tab10', len(all_traj_px))

            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            fig.suptitle(f"Scene @ Frame {target_frame} — {len(all_traj_px)} pedestrian(s)", fontsize=13)

            for ax, (frame_idx, title) in zip(axes, [(0, "Frame 1 (Start)"), (7, "Frame 8 (End)")]):
                img = sample_video[frame_idx].permute(1, 2, 0).cpu().numpy()
                ax.imshow(img)
                ax.set_title(title)

                for p_idx, traj_px in enumerate(all_traj_px):
                    color = colors(p_idx)
                    # Draw the full 8-step trajectory path
                    ax.plot(traj_px[:, 0], traj_px[:, 1],
                            color=color, linewidth=2, alpha=0.8)
                    # Draw all positions as small dots
                    ax.scatter(traj_px[:, 0], traj_px[:, 1],
                               color=color, s=20)
                    # Highlight start (circle) and end (star)
                    ax.scatter(*traj_px[0],  color=color, s=80,  marker='o',
                               edgecolors='white', linewidths=1, label=f'Ped {p_idx}')
                    ax.scatter(*traj_px[-1], color=color, s=120, marker='*',
                               edgecolors='white', linewidths=1)

                ax.legend(loc='upper right', fontsize=7, framealpha=0.6)

            plt.tight_layout()
            output_path = 'debug_pipeline_frames.png'
            plt.savefig(output_path, dpi=150)
            print(f"✅ Visual Check Saved: '{output_path}'")
        else:
            print("❌ ERROR: 'video' key is missing from batch dictionary.")

    except Exception as e:
        print(f"❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pipeline()
