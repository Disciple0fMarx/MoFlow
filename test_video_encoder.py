import torch
from models.context_encoder.video_encoder import GlobalVideoEncoder

def test_video_encoder_standalone():
    print("--- Starting Global Video Encoder Verification ---")
    
    # 1. Configuration (matches your DummyCfg and Cor_fm.yml)
    batch_size = 4
    t_obs = 8       # Number of observed frames synchronized by dataloader
    d_model = 128   # Embedding dimension for cross-modal fusion
    res = 224       # Resized video resolution
    
    # 2. Initialize Encoder
    # Set pretrained=False for the test to avoid unnecessary downloads
    encoder = GlobalVideoEncoder(d_model=d_model, pretrained=False)
    encoder.eval()
    
    # 3. Mock Multimodal Batch
    # Simulation of the 'video' tensor from seq_collate_eth
    # Shape: [Batch, T_obs, Channels, Height, Width]
    dummy_video = torch.randn(batch_size, t_obs, 3, res, res)
    print(f"Input Batch Shape:  {dummy_video.shape}")
    
    try:
        # 4. Execute Forward Pass
        with torch.no_grad():
            z_video = encoder(dummy_video)
        
        # 5. Dimension Verification
        print(f"Output Token Shape: {z_video.shape}")
        
        # Expected shape per Step 2: [Batch, L_global (8), D (128)]
        expected_shape = (batch_size, t_obs, d_model)
        
        if z_video.shape == expected_shape:
            print(f"✅ SUCCESS: Global Video Encoder is correctly dimensioned.")
            print(f"   Tokens produced: {z_video.shape[1]} (matches T_obs)")
            print(f"   Feature Dim:     {z_video.shape[2]} (matches D_MODEL)")
        else:
            print(f"❌ ERROR: Shape mismatch! Expected {expected_shape}, got {z_video.shape}")
            
    except Exception as e:
        print(f"❌ TEST CRASHED: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_video_encoder_standalone()
