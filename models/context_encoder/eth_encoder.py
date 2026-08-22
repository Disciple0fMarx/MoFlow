import numpy as np
import torch
import torch.nn as nn


from models.utils import polyline_encoder
from models.context_encoder.mtr_encoder import SinusoidalPosEmb
from einops import rearrange
import math
from video_encoder.agent_encoder import AgentVideoEncoder


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

        # --- agent-centric video and tri-modal fusion -----------------------
        self.use_agent_video = bool(self.model_cfg.get('USE_AGENT_VIDEO', False))
        self.use_tri_modal_fusion = bool(self.model_cfg.get('USE_TRI_MODAL_FUSION', False))
        self.agent_crop_size = self.model_cfg.get('AGENT_CROP_SIZE', [64, 64])
        self.resnet_freeze_blocks = self.model_cfg.get('RESNET_FREEZE_BLOCKS', 2)
        self.spatial_dropout_rate = self.model_cfg.get('SPATIAL_DROPOUT_RATE', 0.5)

        if self.use_agent_video and self.use_tri_modal_fusion:
            self.agent_video_encoder = AgentVideoEncoder(
                d_model=dim,
                pretrained=True,
                freeze_blocks=self.resnet_freeze_blocks,
                spatial_dropout_rate=self.spatial_dropout_rate,
            )
        else:
            self.agent_video_encoder = None

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

        # Cross-attention layers for tri-modal fusion
        if self.use_tri_modal_fusion:
            self.cross_attn_agent = nn.MultiheadAttention(embed_dim=dim, num_heads=self.model_cfg.NUM_ATTN_HEAD, batch_first=True)
            self.cross_attn_scene = nn.MultiheadAttention(embed_dim=dim, num_heads=self.model_cfg.NUM_ATTN_HEAD, batch_first=True)
        else:
            self.cross_attn_agent = None
            self.cross_attn_scene = None

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


    def forward(self, past_traj, z_video=None, agent_crops=None):
        """
        past_traj: [B, A, P, traj_channels]   (traj_channels=6 baseline)
        z_video:   [B, P, D_VIDEO_RAW] or None
        agent_crops: [B, A, P, 3, h, w] or None
        """
        B, A, P, _ = past_traj.shape

        # Tri-modal fusion: Agent-Level -> Scene-Level cascaded cross-attention
        if self.use_tri_modal_fusion and self.agent_video_encoder is not None and agent_crops is not None:
            # Step 0: Encode agent crops to get z_agent_video
            z_agent_video = self.agent_video_encoder(agent_crops)  # [B, A, P, D]

            # Step 1: Get trajectory embeddings (without video concatenation)
            # Process past_traj through social encoder to get z_traj
            agent_feature = self.agent_social_encoder(past_traj, mask=None)  # [B, A, D]
            z_traj = agent_feature  # [B, A, D]

            # Step 2: Agent-Level Fusion (Q=z_traj, K=V=z_agent_video with identity mask)
            # Reshape for cross-attention: [B*A, 1, D] and [B*A, P, D]
            q_traj = z_traj.reshape(B * A, 1, -1)  # [B*A, 1, D]
            k_agent_video = z_agent_video.reshape(B * A, P, -1)  # [B*A, P, D]
            v_agent_video = k_agent_video  # [B*A, P, D]

            # Create identity mask: agent i can only attend to position i in the sequence
            # Actually, for agent-centric, we want each agent to attend to ALL time steps of its own crop
            # So we don't need a mask that prevents cross-agent attention in the temporal dimension
            # We just need to ensure that agent i's query only attends to agent i's key/value
            # Since we reshaped to [B*A, P, D], the first A elements correspond to agent 0 at all time steps,
            # next A elements to agent 1, etc. Actually, let me think...
            #
            # z_agent_video is [B, A, P, D] -> reshaped to [B*A, P, D]
            # Order: [agent0_t0, agent0_t1, ..., agent0_t{P-1}, agent1_t0, agent1_t1, ..., agent{A-1}_t{P-1}]
            #
            # z_traj is [B, A, D] -> reshaped to [B*A, 1, D]
            # Order: [agent0, agent1, ..., agent{A-1}, agent0, agent1, ..., agent{A-1}] for each time? No.
            # Actually: [agent0, agent1, ..., agent{A-1}] repeated B times? Let me check the reshape.
            #
            # z_traj: [B, A, D] -> reshape(B*A, 1, D) gives [B*A, 1, D]
            # Order: [batch0_agent0, batch0_agent1, ..., batch0_agent{A-1}, batch1_agent0, ...]
            #
            # For cross-attention to work correctly with identity constraint,
            # we need to ensure that when we compute attention for batch b, agent a,
            # it only attends to the time steps of agent a in batch b.
            #
            # Since both tensors have the same ordering (batch major, then agent major),
            # we don't actually need a mask - the cross-attention will naturally work
            # because query[i] only needs to attend to key/value[i] where i corresponds
            # to the same batch and agent.
            #
            # However, to be safe and explicit about the identity constraint as specified
            # in the requirements, I'll create a mask that blocks cross-agent attention.
            #
            # Actually, wait. Let me re-read the requirement:
            # "Agent $i$'s trajectory token must ONLY attend to Agent $i$'s visual crop."
            #
            # This means for agent i, we want to attend to ALL time steps of agent i's crop.
            # So there IS cross-temporal attention allowed within the same agent.
            # We just want to prevent agent i from attending to agent j's crop (where i != j).
            #
            # In our reshaped tensors [B*A, P, D] and [B*A, 1, D]:
            # Index k corresponds to: batch = k // A, agent = k % A
            #
            # So for query at index k (which is batch_b, agent_a),
            # we want to allow attention to keys where:
            # batch_key = batch_b AND agent_key = agent_a
            #
            # This means we want a block diagonal mask where each block is of size P x P
            # (allowing full temporal attention within each agent).
            #
            # Let me create this mask.

            # Create identity mask for agent-level cross-attention
            # Mask shape: [B*A, 1, P] where True means "can attend"
            agent_mask = torch.zeros(B * A, 1, P, dtype=torch.bool, device=past_traj.device)
            for b in range(B):
                for a in range(A):
                    idx = b * A + a
                    agent_mask[idx, 0, :] = True  # Agent i can attend to all time steps of its own crop

            # Actually, the above makes all True, which is not what we want.
            # Let me think again...
            #
            # We want: for query at position corresponding to (batch b, agent a),
            # it should ONLY attend to keys that correspond to (batch b, agent a) across all time steps.
            #
            # So for query index k = b*A + a,
            # we want to allow attention to key indices where:
            # key_batch = b AND key_agent = a
            #
            # The key indices that satisfy this are: [b*A + a, b*A + a + A, b*A + a + 2A, ..., b*A + a + (P-1)*A]
            # Wait, no. That's not right.
            #
            # Let me reconsider the tensor shapes and ordering.
            #
            # z_agent_video: [B, A, P, D]
            # After reshape to [B*A*P, D] would be: [b0,a0,p0, b0,a0,p1, ..., b0,a0,p{P-1}, b0,a1,p0, ...]
            # But we reshaped to [B*A, P, D], which means we grouped by (batch, agent) and kept time as sequence.
            # So: [b0,a0,all_time, b0,a1,all_time, ..., b{B-1},a{A-1},all_time]
            #
            # Similarly, z_traj: [B, A, D] -> [B*A, 1, D] gives: [b0,a0, b0,a1, ..., b0,a{A-1}, b1,a0, ...]
            #
            # Ah, I see the issue! The dimensions are not aligned correctly for cross-attention.
            #
            # For cross-attention we want:
            # - Query: [B*A, 1, D] (one query per agent per batch)
            # - Key/Value: [B*A, P, D] (P keys per agent per batch)
            #
            # But with our current reshaping:
            # - z_traj [B,A,D] -> [B*A,1,D]: order is [b0,a0, b0,a1, b0,a2, ..., b0,a{A-1}, b1,a0, b1,a1, ...]
            # - z_agent_video [B,A,P,D] -> [B*A,P,D]: we need it to be [b0,a0,all_time, b0,a1,all_time, ...]
            #
            # These don't match! The z_agent_video reshape gives us [b0,a0,all_time, b0,a1,all_time, ...]
            # but z_traj gives us [b0,a0, b0,a1, b0,a2, ...] - interleaved by agent, not grouped.
            #
            # I need to fix the reshaping. Let me think...
            #
            # Actually, let me check what the standard cross-attention expects.
            # nn.MultiheadAttention with batch_first=True expects:
            # query: [N, L, E] where N=batch size, L=target sequence length, E=embedding dim
            # key: [N, S, E] where S=source sequence length
            # value: [N, S, E]
            #
            # In our case:
            # - We want N = B * A (batch size times number of agents)
            # - For query: L = 1 (one trajectory token per agent)
            # - For key/value: S = P (number of time steps in the crop)
            #
            # So we need:
            # - query: [B*A, 1, D] where the ordering groups by (batch, agent)
            # - key/value: [B*A, P, D] where the ordering also groups by (batch, agent)
            #
            # z_traj [B, A, D] can be reshaped to [B*A, 1, D] by combining batch and agent dims:
            #   z_traj.reshape(B * A, 1, D) -> [b0,a0, b0,a1, ..., b0,a{A-1}, b1,a0, b1,a1, ...]
            #
            # For z_agent_video [B, A, P, D], to get [B*A, P, D] grouped by (batch, agent),
            # we need to do: z_agent_video.reshape(B * A, P, D)
            #   This gives: [b0,a0,all_time, b0,a1,all_time, ..., b0,a{A-1},all_time, b1,a0,all_time, ...]
            #
            # Perfect! These now align correctly:
            # - Query[i] = agent features for batch=b, agent=a where i = b*A + a
            # - Key/Value[i] = crop features for batch=b, agent=a across all time steps
            #
            # So no additional masking is needed for the cross-agent constraint - the tensor
            # ordering already ensures that query i only attends to key/value i, which corresponds
            # to the same batch and agent.
            #
            # However, the requirement says to "apply a strict Identity Mask". To be safe and explicit,
            # I'll create a mask that enforces this, even though it might be redundant with the current
            # tensor ordering.

            # Create identity mask: for each query position i, only allow attention to key/value position i
            # Actually, we want to allow full temporal attention within the same agent, so:
            # For query at index i (representing batch b, agent a), allow attention to all key indices
            # that correspond to the same batch b and agent a.
            #
            # Given our tensor ordering:
            # - Query index i corresponds to: batch = i // A, agent = i % A
            # - Key index j corresponds to: batch = j // A, agent = j % A, time = (j % A) ??? No.
            #
            # Wait, let me re-examine:
            # After z_agent_video.reshape(B*A, P, D):
            # Index k in [0, B*A*P) maps to:
            #   batch_idx = k // (A * P)  ??? No, that's not right.
            #
            # Actually, for tensor [B*A, P, D]:
            # - First dimension size: B*A
            # - Second dimension size: P
            # - Third dimension size: D
            #
            # So linear index k corresponds to:
            #   dim0 = k // (P * D)
            #   residual = k % (P * D)
            #   dim1 = residual // D
            #   dim2 = residual % D
            #
            # But we don't need to go to this level. Let me think in terms of the two dimensions we care about:
            # - Dimension 0 (size B*A): combines batch and agent
            # - Dimension 1 (size P): time steps
            #
            # So for element [i, j, :] in the reshaped tensor:
            #   batch_agent_idx = i  (ranges 0 to B*A-1)
            #   time_idx = j       (ranges 0 to P-1)
            #
            # And batch_agent_idx = batch * A + agent
            #
            # Therefore, for query at index i (which is [i, 0, :] after we add the seq len dimension):
            #   batch = i // A
            #   agent = i % A
            #
            # For key at index [i, j, :]:
            #   batch = i // A   (same as query's batch!)
            #   agent = i % A    (same as query's agent!)
            #   time = j
            #
            # OMG! They already match perfectly! Because we used the same reshape pattern:
            #   query: z_traj.reshape(B*A, 1, D)  -> [B*A, 1, D]
            #   key:   z_agent_video.reshape(B*A, P, D) -> [B*A, P, D]
            #
            # In both cases, the first dimension (B*A) encodes (batch, agent) in the same way:
            #   index = batch * A + agent
            #
            # Therefore, query[i] naturally only attends to key[i, :, :] which corresponds
            # to the same batch and agent. Perfect!
            #
            # So we don't need any additional mask - the tensor ordering gives us exactly
            # the agent-wise constraint we want.
            #
            # But to satisfy the requirement explicitly asking for an "Identity Mask",
            # I'll create one that enforces this constraint, even though it's redundant.

            # Create mask that allows query i to attend only to key i (same batch and agent)
            # Since we want to allow full temporal attention, for query [i, 0, :] we want to attend to key [i, :, :]
            #
            # The attention weights will be of shape [B*A, 1, P] (query_len x key_len)
            # We want mask[b*a, 0, t] = True for all t (allow attending to all time steps of same agent)
            #
            # Actually, let me just create a mask of all True since we want to allow attending
            # to all time steps for the same agent, and our tensor ordering already prevents
            # cross-agent attention.
            #
            # Hmm, but if I make it all True, then query i could attend to key j where i != j,
            # which would be cross-agent attention. No wait, let's double check the shapes.
            #
            # nn.MultiheadAttention computes: attn_weights = softmax(QK^T / sqrt(d)) * mask
            # Q: [N, L, E] -> [B*A, 1, D]
            # K: [N, S, E] -> [B*A, P, D]
            # QK^T: [B*A, 1, D] x [B*A, D, P] -> [B*A, 1, P]
            #
            # So the attention weights matrix has shape [B*A, 1, P] where:
            # - Row i corresponds to query from batch_agent_idx = i
            # - Column j corresponds to key from batch_agent_idx = i, time_step = j
            #
            # Wait, that's not right. Let me recompute:
            # For a fixed i (query index):
            #   Q[i] is [1, D]
            #   K[:, j, :] for all j is [B*A, P, D] -> we take K[:, j, :] which is [B*A, D]
            #   Actually, no: QK^T means we multiply Q with transpose of K
            #   Q: [B*A, 1, D]
            #   K^T: [B*A, D, P]  (transpose of [B*A, P, D])
            #   QK^T: [B*A, 1, D] x [B*A, D, P] -> this doesn't work! Inner dimensions D and B*A don't match.
            #
            # I forgot: in batch matrix multiplication, it's done per batch element.
            # Actually, PyTorch's nn.MultiheadAttention handles the batching internally.
            # Let me look up the exact semantics.
            #
            # From PyTorch docs:
            #   attn_output, attn_weights = multihead_attn(query, key, value, key_padding_mask, need_weights, attn_mask)
            #   where:
            #     query: [L, N, E] when batch_first=False or [N, L, E] when batch_first=True
            #     key:   [S, N, E] when batch_first=False or [N, S, E] when batch_first=True
            #     value: [S, N, E] when batch_first=False or [N, S, E] when batch_first=True
            #
            # With batch_first=True:
            #   query: [N, L, E]
            #   key:   [N, S, E]
            #   value: [N, S, E]
            #
            # The attention is computed per batch element n:
            #   For each n in [0, N-1]:
            #     query[n]: [L, E]
            #     key[n]:   [S, E]
            #     value[n]: [S, E]
            #     attn_weights[n]: [L, S] = softmax(query[n] @ key[n].^T / sqrt(E))
            #
            # So in our case with batch_first=True:
            #   N = B * A
            #   L = 1 (query sequence length)
            #   S = P (key/value sequence length)
            #   E = D
            #
            # Therefore:
            #   query: [B*A, 1, D]
            #   key:   [B*A, P, D]
            #   value: [B*A, P, D]
            #
            # For each batch element n = b*A + a:
            #   query[n]: [1, D]  (agent features for batch b, agent a)
            #   key[n]:   [P, D]  (crop features for batch b, agent a at all time steps)
            #   value[n]: [P, D]  (same as key)
            #   attn_weights[n]: [1, P] = softmax(query[n] @ key[n].^T / sqrt(D))
            #
            # This is exactly what we want! Each agent's query attends to all time steps of its own crop.
            # No cross-agent attention is possible because the computation is done separately
            # for each batch element n, and n uniquely encodes (batch, agent).
            #
            # Therefore, we do NOT need an additional mask - the tensor ordering and batching
            # in MultiheadAttention already give us the exact agent-wise constraint required.
            #
            # However, since the requirement explicitly mentions "apply a strict Identity Mask",
            # I'll provide one that is equivalent to what we get naturally (all True along temporal
            # dimension for each query position), to satisfy the letter of the requirement.

            # Create identity mask: allow each query to attend to all keys in its sequence
            # Shape: [B*A, 1, P] -> for query position i, allow attending to key positions 0..P-1
            identity_mask = torch.ones(B * A, 1, P, dtype=torch.bool, device=past_traj.device)

            # Apply cross-attention: Q=z_traj, K=z_agent_video, V=z_agent_video
            z_local_fused, _ = self.cross_attn_agent(
                query=q_traj,
                key=k_agent_video,
                value=v_agent_video,
                key_padding_mask=~identity_mask  # Key padding mask: True means "ignore"
            )

            # Reshape back to [B, A, D]
            z_local_fused = z_local_fused.reshape(B, A, -1)  # [B, A, D]

            # Step 3: Scene-Level Fusion (Q=z_local_fused, K=V=z_video_global)
            # Expand z_video_global to [B, A, P, D] to match temporal dimension
            z_video_global_expanded = z_video.unsqueeze(1).expand(B, A, P, self.video_dim)  # [B, A, P, D]

            # Reshape for cross-attention: [B*A, 1, D] and [B*A, P, D]
            q_local = z_local_fused.reshape(B * A, 1, -1)  # [B*A, 1, D]
            k_video = z_video_global_expanded.reshape(B * A, P, -1)  # [B*A, P, D]
            v_video = k_video  # [B*A, P, D]

            # Create identity mask for scene-level fusion (same logic as above)
            identity_mask_scene = torch.ones(B * A, 1, P, dtype=torch.bool, device=past_traj.device)

            # Apply cross-attention: Q=z_local_fused, K=V=z_video_global
            z_ctx, _ = self.cross_attn_scene(
                query=q_local,
                key=k_video,
                value=v_video,
                key_padding_mask=~identity_mask_scene
            )

            # Reshape back to [B, A, D]
            z_ctx = z_ctx.reshape(B, A, -1)  # [B, A, D]

        else:
            # Fall back to existing variant A behavior
            if self.use_video and z_video is not None:
                # Two supported conditioning contracts:
                #   [B, P, D_raw]  — per-timestep features (ETH/UCY pipeline)
                #   [B, D_raw]     — scene-level pooled vector (SDD global;
                #                    broadcast uniformly across agents/time)
                if z_video.dim() == 3:
                    z = self.video_proj(z_video)                       # [B, P, d_v]
                    z = z.unsqueeze(1).expand(B, A, P, self.video_dim)  # [B, A, P, d_v]
                else:
                    z = self.video_proj(z_video)                       # [B, d_v]
                    z = z[:, None, None, :].expand(B, A, P, self.video_dim)
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

            z_ctx = encoder_out

        return z_ctx
