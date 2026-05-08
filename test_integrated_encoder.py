import torch
from models.context_encoder.eth_encoder import ETHEncoder

def test_eth_encoder_integration():
    print("--- Starting Integrated ETHEncoder Verification ---")
    
    # 1. Mock Config (Matches your cor_fm.yml requirements)
    class DummyCfg:
        def __init__(self):
            self.D_MODEL = 128
    
    config = DummyCfg()
    batch_size = 4
    t_obs = 8
    
    # 2. Initialize Integrated Encoder
    # Passing use_pre_norm=True as expected by the original signature
    encoder = ETHEncoder(config, use_pre_norm=True)
    encoder.eval()
    
    # 3. Mock Inputs
    # past_traj = torch.randn(batch_size, t_obs, 2)
    n_agents = 1
    past_traj = torch.randn(batch_size, n_agents, t_obs, 6)  # [B, A, P, D=6]
    
    video_tensor = torch.randn(batch_size, t_obs, 3, 224, 224)
    
    try:
        # Test 1: Multimodal Forward Pass
        print("Testing multimodal pass...")
        z_ctx_multi = encoder(past_traj, video_tensor)
        print(f"✅ Multimodal Output Shape: {z_ctx_multi.shape}") # Should be [5-7]
        
        # Test 2: Baseline Fallback Pass (No video)
        print("Testing baseline fallback...")
        z_ctx_base = encoder(past_traj, video_tensor=None)
        print(f"✅ Baseline Output Shape:   {z_ctx_base.shape}") # Should be [5-7]
        
        if z_ctx_multi.shape == (batch_size, n_agents, 128):
            print("\n🚀 SUCCESS: ETHEncoder is fully integrated and ready for the Backbone.")
        else:
            print("\n❌ ERROR: Dimension mismatch in integrated output.")
            
    except Exception as e:
        print(f"\n❌ INTEGRATION FAILED: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_eth_encoder_integration()
