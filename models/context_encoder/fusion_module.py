import torch
import torch.nn as nn

class CrossModalFusion(nn.Module):
    def __init__(self, d_model=128, nhead=8):
        super().__init__()
        # Multi-Head Cross-Attention allows trajectories to "interrogate" the video
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        
        # Layer normalization and residual connection for stability
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, z_traj, z_video):
        """
        z_traj: [Batch, T_obs(8), D(128)]
        z_video: [Batch, T_obs(8), D(128)]
        """
        # Cross-Attention: Q=traj, K=video, V=video
        # The agent's motion path selects relevant visual constraints (e.g., sidewalks)
        z_fused, _ = self.cross_attn(query=z_traj, key=z_video, value=z_video)
        
        # Residual connection + Norm to produce the final Video-conditioned Context
        z_ctx = self.norm(z_traj + z_fused)
        
        return z_ctx
