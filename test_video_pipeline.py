import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import numpy as np

# Import the modified classes and collation function
from data.dataloader_eth_ucy import ETHDataset, seq_collate_eth

def test_pipeline():
    # 1. Minimal Config Mock
    # The dataset class expects a cfg object for normalization parameters
    class DummyCfg:
        def __init__(self):
            self.past_traj_min = -10.0
            self.past_traj_max = 10.0
            self.fut_traj_min = -10.0
            self.fut_traj_max = 10.0

    cfg = DummyCfg()
    data_dir = 'data/eth_ucy'
    subset = 'eth'  # You can test 'hotel', 'zara1', etc.
    
    print(f"--- Starting Data Pipeline Test for: {subset} ---")

    # 2. Initialize the Modified Dataset (Teacher branch)
    # Ensure you have your .avi files in data/eth_ucy/videos/
    try:
        dataset = ETHDataset(
            cfg, 
            training=True, 
            data_dir=data_dir, 
            subset=subset, 
            imle=False, 
            type='original' # or 'LED' depending on your setup
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
        print(f"Batch Keys: {list(batch.keys())}")
        print(f"Trajectory Shape (Past): {batch['past_traj'].shape}")
        
        if 'video' in batch:
            video_shape = batch['video'].shape
            print(f"Video Tensor Shape: {video_shape}")
            
            # Expected: [Batch, T_obs (8), Channels (3), H (224), W (224)]
            if video_shape == (4, 8, 3, 224, 224):
                print("✅ SUCCESS: Video tensor shape is correct.")
            else:
                print(f"❌ ERROR: Video shape mismatch. Got {video_shape}")

            # 6. Visual Synchronization Check
            # We will save the 1st and 8th frame of the first item in the batch
            sample_video = batch['video'] # Get first sample in batch
            
            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            
            # Frame 0 (Start of T_obs) - Convert back from Tensor to Image
            img_start = sample_video.permute(1, 2, 0).cpu().numpy()
            ax.imshow(img_start)
            ax.set_title(f"Sample 0: Frame 1 of {subset}")
            
            # Frame 7 (End of T_obs)
            img_end = sample_video[1].permute(1, 2, 0).cpu().numpy()
            ax[2].imshow(img_end)
            ax[2].set_title(f"Sample 0: Frame 8 of {subset}")
            
            output_path = 'debug_pipeline_frames.png'
            plt.savefig(output_path)
            print(f"\n✅ Visual Check Saved: Look at '{output_path}' to confirm alignment.")
        else:
            print("❌ ERROR: 'video' key is missing from the batch dictionary.")

    except Exception as e:
        print(f"❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pipeline()
