import numpy as np
import torch
import torch.nn as nn


from models.utils import polyline_encoder
from models.context_encoder.mtr_encoder import SinusoidalPosEmb
from einops import rearrange
import math


# ---------------------------------------------------------------------------
# Video-MoFlow variant A (per-timestep concat).
#
# - z_video_global: [B, T_obs, D_VIDEO_RAW=512] cached features from
#   `video_encoder` (ResNet18 by default). Shared across agents in a frame.
# - We project 512 -> d_v (default 32) and concat per-timestep with each
#   agent's per-timestep state vector (which has 6 channels: abs_xy, rel_xy,
#   vel_xy). Time is then flattened into the feature dim, exactly like
#   upstream SocialTransformer does.
#
# Gating: when cfg.USE_VIDEO is False or z_video is None at forward time,
# the module behaves identically to upstream MoFlow.
# ---------------------------------------------------------------------------


class SocialTransformer(nn.Module):
    def __init__(self, in_dim=48, hidden_dim=256, out_dim=128):
        super(SocialTransformer, self).__init__()
        self.encode_past = nn.Linear(in_dim, hidden_dim, bias=False)
        self.layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=2, dim_feedforward=hidden_dim, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.layer, num_layers=2)
        self.mlp_out = nn.Linear(hidden_dim, out_dim)

    def forward(self, past_traj, mask):
        """
        @param past_traj: [B, A, P, D]  (D = 6 baseline, or 6 + d_v with video)
        @param mask:      [B, A] or None
        """
        B, A, P, D = past_traj.shape
        past_traj = rearrange(past_traj, 'b a p d -> b a (p d)')
        h_feat = self.encode_past(past_traj)                     # [B, A, hidden_dim]
        h_feat_ = self.transformer_encoder(h_feat, mask=mask)
        h_feat = h_feat + h_feat_
        h_feat = self.mlp_out(h_feat)                            # [B, A, out_dim]
        return h_feat


class ETHEncoder(nn.Module):
    def __init__(self, config, use_pre_norm):
        super().__init__()
        self.model_cfg = config
        dim = self.model_cfg.D_MODEL

        # --- video wiring (variant A) ---------------------------------------
        self.use_video = bool(self.model_cfg.get('USE_VIDEO', False))
        # past_traj channels per timestep before video concat (abs_xy + rel_xy + vel_xy)
        self.traj_channels = int(self.model_cfg.get('TRAJ_CHANNELS', 6))
        self.past_frames = int(self.model_cfg.get('PAST_FRAMES', 8))
        self.video_dim_raw = int(self.model_cfg.get('VIDEO_DIM_RAW', 512))
        self.video_dim = int(self.model_cfg.get('VIDEO_DIM', 32)) if self.use_video else 0

        if self.use_video:
            self.video_proj = nn.Linear(self.video_dim_raw, self.video_dim)
        else:
            self.video_proj = None

        per_timestep = self.traj_channels + self.video_dim
        social_in_dim = self.past_frames * per_timestep   # baseline: 8*6=48; +video: 8*(6+32)=304

        ### build social encoder
        self.agent_social_encoder = SocialTransformer(in_dim=social_in_dim, hidden_dim=256, out_dim=dim)

        # Positional encoding
        self.pos_encoding = nn.Sequential(
                SinusoidalPosEmb(dim, theta = 10000),
                nn.Linear(dim, dim),
                nn.ReLU(),
                nn.Linear(dim, dim)
            )
        self.agent_query_embedding = nn.Embedding(self.model_cfg.AGENTS, dim)
        self.mlp_pe = nn.Sequential(
            nn.Linear(2*dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        # build transformer encoder layers
        self.layer = nn.TransformerEncoderLayer(d_model=dim,
                                                dropout=self.model_cfg.get('DROPOUT_OF_ATTN', 0.1),
                                                nhead=self.model_cfg.NUM_ATTN_HEAD,
                                                dim_feedforward=dim * 4,
                                                norm_first=use_pre_norm,
                                                batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.layer, num_layers=self.model_cfg.NUM_ATTN_LAYERS)
        self.num_out_channels = dim

    ### polyline encoder MLP PointNet [B, A, D]
    def build_polyline_encoder(self, in_channels, hidden_dim, num_layers, num_pre_layers=1, out_channels=None):
        ret_polyline_encoder = polyline_encoder.PointNetPolylineEncoder(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_pre_layers=num_pre_layers,
            out_channels=out_channels
        )
        return ret_polyline_encoder


    def forward(self, past_traj, z_video=None):
        """
        past_traj: [B, A, P, traj_channels]   (traj_channels=6 baseline)
        z_video:   [B, P, D_VIDEO_RAW] or None
        """
        B, A, P, D = past_traj.shape

        if self.use_video and z_video is not None:
            # Project to small d_v and broadcast across agents.
            z = self.video_proj(z_video)                           # [B, P, d_v]
            z = z.unsqueeze(1).expand(B, A, P, self.video_dim)     # [B, A, P, d_v]
            past_traj = torch.cat([past_traj, z], dim=-1)          # [B, A, P, 6+d_v]
        elif self.use_video and z_video is None:
            # Inference-time safety: pad with zeros so shapes still match.
            zeros = past_traj.new_zeros(B, A, P, self.video_dim)
            past_traj = torch.cat([past_traj, zeros], dim=-1)

        agent_feature = self.agent_social_encoder(past_traj, mask=None)  # [B, A, D]

        ### use positional encoding
        pos_encoding = self.pos_encoding(torch.arange(agent_feature.shape[1]).to(past_traj.device))         # [A, D]

        ### enforce positional encoding earlier here
        agent_query = self.agent_query_embedding(torch.arange(self.model_cfg.AGENTS).to(past_traj.device))  # [A, D]

        pos_encoding = self.mlp_pe(torch.cat([agent_query, pos_encoding], dim=-1)) # [A, D]

        agent_feature = agent_feature + pos_encoding.unsqueeze(0)               # [B, A, D]
        encoder_out = self.transformer_encoder(agent_feature)

        return encoder_out
