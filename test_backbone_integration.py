import torch
from models.backbone_eth_ucy import ETHMotionTransformer

def test_backbone_multimodal_forward():
    print("--- Starting Backbone Multimodal Verification ---")
    
    # 1. ROBUST CONFIG MOCK: Supports both .ATTR and .get() access
    class MockSubConfig:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
        def get(self, key, default=None):
            return getattr(self, key, default)

    class DummyCfg:
        def __init__(self):
            # Matches literal class names as per your previous tweak
            self.CONTEXT_ENCODER = MockSubConfig(
                NAME='ETHEncoder', 
                D_MODEL=128
            )
            self.MOTION_DECODER = MockSubConfig(
                NAME='MTRDecoder',
                NUM_DECODER_BLOCKS=2, # Satisfies line 29 of mtr_decoder.py
                D_MODEL=128,
                NUM_ATTN_HEAD=8,
            )

    model_config = DummyCfg()
    
    # 2. Initialize Teacher Backbone
    # Passing None for logger/config as they aren't used in this basic forward check
    model = ETHMotionTransformer(model_config, logger=None, config=None)
    model.eval()

    # 3. Create a Mock Multimodal Batch
    # FIX: Reshape to 4D [Batch, Agents, Time, Dim] to satisfy eth_encoder.py line 28
    batch = {
        # 48 features total (e.g., 8 frames * 6 features per frame)
        'pre_motion_3D': torch.randn(4, 2, 8, 6),     
        'video_frames': torch.randn(4, 8, 3, 224, 224) 
    }

    try:
        print("Executing full backbone forward pass...")
        with torch.no_grad():
            # This tests the extraction logic in ETHMotionTransformer.forward()
            # and the flow into the ETHEncoder and MTRDecoder
            z_ctx = model.context_encoder(batch['pre_motion_3D'], video_tensor=batch['video_frames'])
            
        print(f"✅ Context produced: {z_ctx.shape}")
        
        if z_ctx.shape == (4, 2, 128):
            print("\n🚀 SUCCESS: Backbone is correctly orchestrating the multimodal flow.")
        else:
            print(f"\n❌ ERROR: Unexpected shape {z_ctx.shape}")

    except Exception as e:
        print(f"\n❌ BACKBONE CRASHED: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_backbone_multimodal_forward()
