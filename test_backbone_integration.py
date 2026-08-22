import torch
from models.backbone_eth_ucy import ETHMotionTransformer


class _Log:
    """Minimal logger stub for the backbone's parameter report."""

    def info(self, msg):
        print(msg)


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
            self.NUM_PROPOSED_QUERY = 20
            self.MODEL_OUT_DIM = 24  # A * F * 2
            self.REGRESSION_MLPS = [128, 256, 24]
            self.CLASSIFICATION_MLPS = [128, 128, 1]
            self.CONTEXT_ENCODER = MockSubConfig(
                NAME="ETHEncoder",
                D_MODEL=128,
                AGENTS=2,
                NUM_ATTN_HEAD=8,
                NUM_ATTN_LAYERS=4,
                DROPOUT_OF_ATTN=0.1,
                USE_VIDEO=True,
                VIDEO_DIM_RAW=512,
                VIDEO_DIM=32,
            )
            self.MOTION_DECODER = MockSubConfig(
                NAME="MTRDecoder",
                NUM_DECODER_BLOCKS=2,
                D_MODEL=128,
                NUM_ATTN_HEAD=8,
                DROPOUT_OF_ATTN=0.1,
            )

        def get(self, key, default=None):
            return getattr(self, key, default)

    # 2. Initialize Teacher Backbone
    model = ETHMotionTransformer(DummyCfg(), logger=_Log(), config=None)
    model.eval()

    batch_size, n_agents = 4, 2
    pre_motion_3D = torch.randn(batch_size, n_agents, 8, 6)  # [B, A, P, 6]

    # 3. Multimodal forward via the production call-site signature
    #    (models/backbone_eth_ucy.py: context_encoder(x, z_video=..., agent_crops=...))

    # 3a. Scene-level contract [B, 512] (SDD global pipeline)
    z_scene = torch.randn(batch_size, 512)
    with torch.no_grad():
        z_ctx = model.context_encoder(pre_motion_3D, z_video=z_scene)
    assert z_ctx.shape == (batch_size, n_agents, 128), f"scene-level: {z_ctx.shape}"
    print(f"PASS z_video_global [B, 512]: {tuple(z_ctx.shape)}")

    # 3b. Per-timestep contract [B, P, 512] (ETH/UCY pipeline)
    z_legacy = torch.randn(batch_size, 8, 512)
    with torch.no_grad():
        z_ctx = model.context_encoder(pre_motion_3D, z_video=z_legacy)
    assert z_ctx.shape == (batch_size, n_agents, 128), f"legacy: {z_ctx.shape}"
    print(f"PASS z_video [B, P, 512]:     {tuple(z_ctx.shape)}")

    print("\nSUCCESS: Backbone is correctly orchestrating the multimodal flow.")


if __name__ == "__main__":
    test_backbone_multimodal_forward()
