import numpy as np
import torch
import torch.nn as nn


from models.utils import polyline_encoder
from models.context_encoder.mtr_encoder import SinusoidalPosEmb
from einops import rearrange
import math

from .video_encoder import GlobalVideoEncoder
from .fusion_module import CrossModalFusion


class SocialTransformer(nn.Module):
    def __init__(self, in_dim=48, hidden_dim=256, out_dim=128):
        super(SocialTransformer, self).__init__()
        self.encode_past = nn.Linear(in_dim, hidden_dim, bias=False)
        self.layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=2, dim_feedforward=hidden_dim, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.layer, num_layers=2)
        self.mlp_out = nn.Linear(hidden_dim, out_dim)

    def forward(self, past_traj, mask):
        """
        @param past_traj: [B, A, P, D]
        @param mask:      [B, A] or None
        """
        B, A, P, D = past_traj.shape
        # past_traj = rearrange(past_traj, 'b a p d -> (b a) p d')
        # h_feat = self.encode_past(past_traj.reshape(B * A, -1)).unsqueeze(1)  # [B*A, 1, D]

        past_traj = rearrange(past_traj, 'b a p d -> b a (p d)')
        h_feat = self.encode_past(past_traj)    # [B, A, D]

        h_feat_ = self.transformer_encoder(h_feat, mask=mask)

        h_feat = h_feat + h_feat_
        h_feat = self.mlp_out(h_feat)           # [B, A, D]

        return h_feat
    

class ETHEncoder(nn.Module):
    def __init__(self, config, use_pre_norm): # MAINTAIN original signature
        super().__init__()
        self.model_cfg = config
        self.d_model = config.D_MODEL # Use 128 from your cor_fm.yml [6, 7]
        
        # 1. KINEMATIC BRANCH: Original MoFlow Social Transformer [1]
        # Initializing based on original MoFlow parameters
        self.traj_encoder = SocialTransformer(
            in_dim=48, 
            hidden_dim=256, 
            out_dim=self.d_model
        )

        # 2. VISUAL BRANCH (PHASE 2.2) [8]
        self.video_encoder = GlobalVideoEncoder(d_model=self.d_model)
        
        # 3. CROSS-MODAL FUSION (PHASE 2.2 - Variante B) [3]
        self.fusion_module = CrossModalFusion(d_model=self.d_model)

    def forward(self, past_traj, video_tensor=None):
        """
        past_traj:    [B, A, P, D]
        video_tensor: [B, T_obs, 3, 224, 224] or None
        """
        z_traj = self.traj_encoder(past_traj, mask=None)  # [B, A, D]

        if video_tensor is not None:
            z_video = self.video_encoder(video_tensor)     # [B, T_obs, D]
            z_ctx = self.fusion_module(z_traj, z_video)
        else:
            z_ctx = z_traj

        return z_ctx
