import torch
# Assuming your module is saved at models/context_encoder/fusion_module.py
from models.context_encoder.fusion_module import CrossModalFusion

def test_fusion_standalone():
    print("--- Starting Cross-modal Fusion Verification ---")
    
    # 1. Configuration (aligned with MoFlow defaults and technical note)
    batch_size = 4
    t_obs = 8       # Matches T_obs established in Phase 2.1 [143.1]
    d_model = 128   # Standard MoFlow feature dimension [3, 4]
    
    # 2. Initialize the Fusion Module
    # Uses 8 heads as recommended for a 128-dim embedding [4]
    fusion_block = CrossModalFusion(d_model=d_model, nhead=8)
    fusion_block.eval()
    
    # 3. Mock Input Tensors
    # z_traj: Produced by the Trajectory/Social Encoder [5]
    z_traj = torch.randn(batch_size, t_obs, d_model)
    
    # z_video: Produced by the Global Video Encoder (Step 2)
    z_video = torch.randn(batch_size, t_obs, d_model)
    
    print(f"Input z_traj Shape:  {z_traj.shape}")
    print(f"Input z_video Shape: {z_video.shape}")
    
    try:
        # 4. Execute Forward Pass (Cross-Attention)
        with torch.no_grad():
            # Trajectory queries the Video [1]
            z_ctx = fusion_block(z_traj, z_video)
        
        # 5. Verification
        print(f"Output z_ctx Shape:  {z_ctx.shape}")
        
        # Expected shape per Step 2: [Batch, T_obs (8), D (128)]
        expected_shape = (batch_size, t_obs, d_model)
        
        if z_ctx.shape == expected_shape:
            print(f"✅ SUCCESS: Cross-modal Fusion is correctly dimensioned.")
            print(f"   Context produced: {z_ctx.shape} (Ready for Flow Decoder)")
        else:
            print(f"❌ ERROR: Shape mismatch! Expected {expected_shape}, got {z_ctx.shape}")
            
    except Exception as e:
        print(f"❌ TEST CRASHED: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fusion_standalone()
