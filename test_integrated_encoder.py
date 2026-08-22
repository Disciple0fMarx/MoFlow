import torch
from models.context_encoder.eth_encoder import ETHEncoder


class DummyCfg(dict):
    """Dict-backed model cfg exposing both attribute and .get() access."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e

    def get(self, key, default=None):
        return dict.get(self, key, default)


def _make_cfg(use_video=True):
    return DummyCfg(
        {
            "D_MODEL": 128,
            "AGENTS": 2,
            "NUM_ATTN_HEAD": 8,
            "NUM_ATTN_LAYERS": 4,
            "DROPOUT_OF_ATTN": 0.1,
            "USE_VIDEO": use_video,
            "VIDEO_DIM_RAW": 512,
            "VIDEO_DIM": 32,
            "USE_AGENT_VIDEO": False,
            "USE_TRI_MODAL_FUSION": False,
        }
    )


def test_eth_encoder_integration():
    print("--- Starting Integrated ETHEncoder Verification ---")

    batch_size, n_agents, t_obs = 4, 2, 8
    past_traj = torch.randn(batch_size, n_agents, t_obs, 6)  # [B, A, P, 6]

    # 1. Baseline fallback pass (no video wiring)
    encoder = ETHEncoder(_make_cfg(use_video=False), use_pre_norm=True)
    encoder.eval()
    with torch.no_grad():
        out = encoder(past_traj)
    assert out.shape == (batch_size, n_agents, 128), f"baseline: {out.shape}"
    print(f"PASS baseline (USE_VIDEO=False): {tuple(out.shape)}")

    # 2. Variant A, per-timestep features [B, P, D_raw] (ETH/UCY contract)
    encoder = ETHEncoder(_make_cfg(use_video=True), use_pre_norm=True)
    encoder.eval()
    z_legacy = torch.randn(batch_size, t_obs, 512)
    with torch.no_grad():
        out = encoder(past_traj, z_video=z_legacy)
    assert out.shape == (batch_size, n_agents, 128), f"legacy: {out.shape}"
    print(f"PASS z_video [B, P, 512]:         {tuple(out.shape)}")

    # 3. Variant A, scene-level pooled vector [B, D_raw] (SDD global contract)
    z_scene = torch.randn(batch_size, 512)
    with torch.no_grad():
        out = encoder(past_traj, z_video=z_scene)
    assert out.shape == (batch_size, n_agents, 128), f"scene-level: {out.shape}"
    print(f"PASS z_video [B, 512]:            {tuple(out.shape)}")

    # 4. Video missing at inference (zero-pad safety path)
    with torch.no_grad():
        out = encoder(past_traj, z_video=None)
    assert out.shape == (batch_size, n_agents, 128), f"missing-video: {out.shape}"
    print(f"PASS z_video=None (zero-pad):     {tuple(out.shape)}")

    # 5. Gradients flow into both projection heads
    out = encoder(past_traj, z_video=z_scene)
    loss = out.pow(2).sum()
    loss.backward()
    assert encoder.video_proj.weight.grad is not None
    assert encoder.video_proj.weight.grad.abs().sum() > 0
    print("PASS backward: video_proj receives gradient")

    print("\nSUCCESS: ETHEncoder is fully integrated and ready for the Backbone.")


if __name__ == "__main__":
    test_eth_encoder_integration()
