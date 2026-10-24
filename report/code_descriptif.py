"""
code_descriptif.py
==================
A self-contained, annotated walkthrough of the Global Video-Conditioned
Flow Matching architecture for pedestrian trajectory prediction on SDD.

This file is NOT imported by the training pipeline; it is a pedagogical
reference that can be executed standalone (requires only PyTorch and
torchvision).  All shapes are printed at every intermediate step.

Author: Dhiaa Eddine Bahri
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torch.distributions import Normal
from torch.distributions.transforms import SigmoidTransform, AffineTransform

# =====================================================================
# Global constants (matching cfg/sdd/cor_fm.yml)
# =====================================================================
D           = 128       # context / trajectory embedding dimension
VIDEO_RAW   = 512       # ResNet-18 output dimension
VIDEO_DIM   = 32        # projected visual token dimension
K           = 20        # number of sampled trajectory heads
F           = 12        # T_fut: prediction horizon in frames
P           = 8         # T_obs: observation window in frames
INPUT_DIM   = 6         # [abs_x, abs_y, rel_x, rel_y, vel_x, vel_y]
FREEZE_BLOCKS = 2       # frozen layers in ResNet-18 (conv1 + layer1)
EULER_STEPS = 10        # sampling integration steps
MU_T        = -0.5      # logit-normal location
SIGMA_T     = 1.5       # logit-normal scale


# =====================================================================
# Module 1: Global Video Encoder (frozen ResNet-18 + temporal pooling)
# =====================================================================
class GlobalVideoEncoder(nn.Module):
    """
    Extracts a single global scene embedding from T_obs raw frames.

    Forward
    -------
    frames : Tensor [B, T_obs, 3, H, W]
        Raw RGB observation frames.

    Returns
    -------
    z_video : Tensor [B, VIDEO_RAW=512]
        Temporally-averaged ResNet-18 feature vector.
    """

    def __init__(self, freeze_blocks: int = FREEZE_BLOCKS):
        super().__init__()
        resnet = models.resnet18(weights=None)
        resnet.fc = nn.Identity()          # remove final FC -> 512-d output

        # Freeze parameters up through layer1 (blocks 0-1)
        # i.e. conv1, bn1, layer1 — these learn low-level edge/texture
        # features that generalise across scenes without fine-tuning.
        frozen_params = []
        for module in [resnet.conv1, resnet.bn1, resnet.layer1]:
            frozen_params.extend(module.parameters())
        for p in frozen_params:
            p.requires_grad = False

        self.backbone = resnet

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = frames.shape
        flat = frames.view(B * T, C, H, W)   # merge batch and time
        feat = self.backbone(flat)             # [B*T, 512]
        feat = feat.view(B, T, -1).mean(dim=1) # temporal mean pool
        return feat                             # [B, 512]


# =====================================================================
# Module 2: Video Projection (512 -> 32)
# =====================================================================
class VideoProjection(nn.Module):
    """
    Linear projection from ResNet feature space to the compact visual
    token space consumed by the context encoder.

    forward(z_video) : [B, 512] -> [B, VIDEO_DIM=32]
    """

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(VIDEO_RAW, VIDEO_DIM)

    def forward(self, z_video: torch.Tensor) -> torch.Tensor:
        return self.proj(z_video)   # [B, 32]


# =====================================================================
# Module 3: Context Encoder (trajectory embedding + visual fusion)
# =====================================================================
class ContextEncoder(nn.Module):
    """
    Encodes each agent's observed trajectory and fuses it with the
    projected global visual token.

    Architecture (per agent):
      1. Linear INPUT_DIM -> D on each timestep
      2. Mean pooling across P timesteps -> [D]
      3. Concatenation with projected visual token -> [D + VIDEO_DIM]
      4. Two-layer MLP -> [D]

    Forward
    -------
    past_traj : Tensor [B, A, P, INPUT_DIM]
        Observed trajectory (normalised to [-1, 1]).
    z_video   : Tensor [B, VIDEO_RAW]
        Global scene embedding from ResNet-18.

    Returns
    -------
    ctx : Tensor [B, A, D]
        Per-agent context vector consumed by the motion decoder.
    """

    def __init__(self):
        super().__init__()
        self.traj_embed = nn.Linear(INPUT_DIM, D)
        self.proj = nn.Linear(VIDEO_RAW, VIDEO_DIM)  # 512 -> 32
        self.fuse = nn.Sequential(
            nn.Linear(D + VIDEO_DIM, D),
            nn.ReLU(inplace=True),
            nn.Linear(D, D),
        )

    def forward(self, past_traj: torch.Tensor,
                z_video: torch.Tensor) -> torch.Tensor:
        B, A, P, _ = past_traj.shape

        # Per-timestep trajectory embedding
        x = self.traj_embed(past_traj)         # [B, A, P, D]
        traj_emb = x.mean(dim=2)               # [B, A, D]

        # Project raw video embedding to compact token space
        gamma = self.proj(z_video)              # [B, 32]
        gamma = gamma.unsqueeze(1).expand(B, A, -1)  # [B, A, 32]

        # Fuse trajectory + visual token
        ctx = self.fuse(torch.cat([traj_emb, gamma], dim=-1))  # [B, A, D]
        return ctx


# =====================================================================
# Module 4: MTR-Style Motion Decoder
# =====================================================================
class MotionDecoder(nn.Module):
    """
    Motion Transformer (MTR) decoder with K learnable queries.

    Each of the K query embeddings attends to all A agent context
    vectors via scaled-dot-product cross-attention, then a
    classification head produces trajectory scores and a regression
    head predicts future displacements.

    Forward
    -------
    ctx : Tensor [B, A, D]

    Returns
    -------
    scores     : Tensor [B, A, K]           (logits for diversity)
    regression : Tensor [B, A, K, F * 2]    (flattened x,y predictions)
    """

    def __init__(self, num_queries: int = K):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_queries, D))
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=D, num_heads=4, batch_first=True)
        self.cls_head = nn.Sequential(
            nn.Linear(D, D), nn.ReLU(), nn.Linear(D, 1))
        self.reg_head = nn.Sequential(
            nn.Linear(D, D * 2), nn.ReLU(),
            nn.Linear(D * 2, F * 2))

    def forward(self, ctx: torch.Tensor):
        B, A, _ = ctx.shape

        # Expand K queries for each agent
        q = self.queries.unsqueeze(0).unsqueeze(0)    # [1, 1, K, D]
        q = q.expand(B, A, -1, -1).reshape(B * A, -1, D)  # [B*A, K, D]
        kv = ctx.reshape(B * A, 1, D)                 # [B*A, 1, D]

        # Cross-attention: queries attend to single agent context token
        attn_out, _ = self.cross_attn(q, kv, kv)       # [B*A, K, D]
        attn_out = attn_out.view(B, A, -1, D)          # [B, A, K, D]

        scores     = self.cls_head(attn_out).squeeze(-1)  # [B, A, K]
        regression = self.reg_head(attn_out)               # [B, A, K, F*2]
        return scores, regression


# =====================================================================
# Module 5: Flow Matching Sampling (Euler ODE Integrator)
# =====================================================================
class FlowMatcher(nn.Module):
    """
    Wraps a denoising velocity network with Conditional Flow Matching
    training (logit-normal t-schedule) and Euler-scheme sampling.

    The OT path is:
        x_t = (1 - t) * x_0 + t * x_1       (x_0 ~ N(0,I), x_1 = data)
        u_t = x_1 - x_0                      (target velocity)

    Sampling integrates from t=1 (noise) backwards to t=0 (data)
    using Euler steps.

    Forward (training)
    ------------------
    y_data : Tensor [B, K, A, F*2]   normalised ground-truth futures
    Returns: scalar loss.

    Sample (inference)
    ------------------
    Returns: Tensor [B, K, A, F*2]
    """

    def __init__(self, velocity_net: nn.Module):
        super().__init__()
        self.net = velocity_net

        # Logit-normal distribution for training t-schedule
        self._base_dist = Normal(MU_T, SIGMA_T)
        self._sigmoid = SigmoidTransform()
        self._affine = AffineTransform(0, 1)  # map sigmoid output to (0,1)

    def _sample_t(self, n: int) -> torch.Tensor:
        eps = self._base_dist.sample((n,)).squeeze(-1)
        return self._affine(self._sigmoid(eps))   # t ~ LogitNormal

    def forward(self, y_data: torch.Tensor) -> torch.Tensor:
        """Training loss (velocity matching)."""
        B, K, A, D_flat = y_data.shape
        n = B * K * A

        t = self._sample_t(n).to(y_data.device)        # [n]
        t_3d = t.view(B, K, A, 1)                       # broadcast-ready

        x0 = torch.randn_like(y_data)                   # noise
        x_t = (1 - t_3d) * x0 + t_3d * y_data          # interpolated
        u_t = y_data - x0                                # target velocity

        v_pred = self.net(x_t, t_3d, ctx=None)           # velocity field
        loss = ((v_pred - u_t) ** 2).sum() / n           # agent-mode sum
        return loss

    @torch.no_grad()
    def sample(self, y_start: torch.Tensor,
               num_steps: int = EULER_STEPS) -> torch.Tensor:
        """Euler integration from noise to data prediction."""
        B, K, A, D_flat = y_start.shape
        y_t = torch.randn_like(y_start)                  # start at noise
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t_val = torch.tensor(1.0 - i * dt, device=y_t.device)
            t_emb = t_val.expand(B, K, A, 1)
            v = self.net(y_t, t_emb, ctx=None)
            y_t = y_t + dt * v                            # Euler step

        return y_t                                         # [B,K,A,F*2]


# =====================================================================
# Module 6: End-to-End Smoke Test
# =====================================================================
if __name__ == "__main__":
    torch.manual_seed(42)
    device = "cpu"

    # --- Instantiate modules ---
    video_enc   = GlobalVideoEncoder().to(device)
    video_proj  = VideoProjection().to(device)
    ctx_enc     = ContextEncoder().to(device)
    decoder     = MotionDecoder().to(device)

    print("=== Module parameters ===")
    for name, m in [("VideoEnc", video_enc), ("VideoProj", video_proj),
                    ("ContextEnc", ctx_enc), ("Decoder", decoder)]:
        n = sum(p.numel() for p in m.parameters())
        print(f"  {name:12s}: {n:>8,} params")

    # --- Build dummy batch ---
    B, A = 2, 4
    past_traj   = torch.randn(B, A, P, INPUT_DIM)
    fut_traj    = torch.randn(B, A, F, 2)
    frames      = torch.randn(B, P, 3, 224, 224)

    print("\n=== Forward pass shapes ===")

    # Step 1: Global video encoding
    z_video = video_enc(frames)
    print(f"  z_video      : {tuple(z_video.shape)}")

    # Step 2: Projection
    gamma = video_proj(z_video)
    print(f"  gamma        : {tuple(gamma.shape)}")

    # Step 3: Context encoding (raw 512-d z_video; encoder projects internally)
    ctx = ctx_enc(past_traj, z_video)
    print(f"  ctx          : {tuple(ctx.shape)}")

    # Step 4: Motion decoder
    scores, regression = decoder(ctx)
    print(f"  scores       : {tuple(scores.shape)}")
    print(f"  regression   : {tuple(regression.shape)}")

    # Step 5: Normalise future and run flow matching loss
    fut_flat = fut_traj.view(B, A, F * 2)
    # Replicate across K heads
    y_data = fut_flat.unsqueeze(2).expand(-1, -1, K, -1)  # [B,A,K,F*2]
    y_data = y_data.permute(0, 2, 1, 3)                   # [B,K,A,F*2]

    # Create a minimal velocity network wrapper for FM
    class VelocityWrapper(nn.Module):
        def __init__(self, enc, proj, dec):
            super().__init__()
            self.enc, self.proj, self.dec = enc, proj, dec
        def forward(self, x_t, t_emb, ctx):
            return -x_t * 0.5  # dummy: constant field

    fm = FlowMatcher(VelocityWrapper(video_enc, video_proj, decoder))
    loss = fm(y_data)
    print(f"  FM loss      : {loss.item():.4f}")

    # Step 6: Sampling
    y_pred = fm.sample(y_data)
    print(f"  y_pred (full): {tuple(y_pred.shape)}")

    # Best-of-K: simply take head 0 for demonstration
    # (in practice, select argmin ADE against ground truth)
    y_best = y_pred[:, 0, :, :]             # [B, A, F*2]
    y_final = y_best.view(B, A, F, 2)       # [B, A, F, 2]
    print(f"  y_final      : {tuple(y_final.shape)}")

    print("\n=== All shapes verified ===")
