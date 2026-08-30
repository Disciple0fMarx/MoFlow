"""Cross-Attention fusion between trajectory tokens and agent-centric video
tokens for the CFM pipeline.

Interaction contract (see CLAUDE.md "Tri-Modal Cascaded Cross-Attention
Fusion"):
  * The trajectory token ``z_traj`` acts as the **Query (Q)**.
  * The agent-centric video token ``z_agent_video`` acts as the **Key (K)**
    and **Value (V)**.

Implementation notes
--------------------
* Shape contract on entry:
    - ``z_traj:        [B, A, P, D]``   (per-timestep trajectory embeddings)
    - ``z_agent_video: [B, A, T, D]``   (per-timestep agent-local crops,
                                         typically T == P)
  where ``B`` = batch, ``A`` = number of agents in the window, ``P`` =
  observation length, ``T`` = video timesteps, ``D`` = model dimension.

* Crucially, the **temporal** axis is *not* merged with any cross-agent
  dimension.  We keep each agent's sequence separate and run a single
  cross-attention over the *temporal* dimension, which is exactly where the
  trajectory (Q) can attend to the same agent's video evidence (K/V).  There
  is **no cross-agent mixing** here, so no identity mask is required; instead
  the per-agent independence guarantees zero leakage.

* Output ``z_ctx: [B, A, P, D]`` preserves the trajectory shape so it drops
  straight into the downstream motion/flow decoder.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CausalTemporalCrossAttention(nn.Module):
    """Per-agent cross-attention: Q = trajectory, K/V = agent video.

    For each agent independently, we attend the P trajectory timesteps against
    the same agent's T video timesteps.  ``nn.MultiheadAttention`` with
    ``batch_first=True`` treats the trailing token axis as sequence, so we
    flatten ``[B, A]`` into the batch leading dimension and set
    ``key_padding_mask=None`` because every agent has full video coverage in
    its own crop.

    Parameters
    ----------
    d_model : int
        Model / token dimension ``D`` (must match trajectory and video tokens).
    nhead : int
        Number of attention heads.
    dropout : float
        Dropout on attention output weights.

    forward(z_traj, z_agent_video) -> z_ctx
    ---------------------------------------
    z_traj        : ``[B, A, P, D]``
    z_agent_video : ``[B, A, T, D]``
    returns z_ctx : ``[B, A, P, D]``
    """

    def __init__(self, d_model: int = 128, nhead: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, z_traj: torch.Tensor, z_agent_video: torch.Tensor
    ) -> torch.Tensor:
        B, A, P, D = z_traj.shape
        T = z_agent_video.shape[2]
        if z_agent_video.shape[:2] != (B, A):
            raise ValueError(
                "z_traj and z_agent_video must share (B, A); got "
                f"{tuple(z_traj.shape)} vs {tuple(z_agent_video.shape)}"
            )

        # Flatten the agent axis into batch: [B*A, P, D] and [B*A, T, D].
        q = z_traj.reshape(B * A, P, D)
        kv = z_agent_video.reshape(B * A, T, D)

        # Q = trajectory tokens, K/V = agent video tokens.
        fused, _ = self.attn(query=q, key=kv, value=kv)

        # Residual + post-norm (helps stabilise the CFM loss scale).
        fused = fused.reshape(B, A, P, D)
        return self.norm(z_traj + fused)


# Convenience alias matching the "Q=traj / K,V=agent-video" naming in CLAUDE.
AgentVideoFusion = CausalTemporalCrossAttention

__all__ = [
    "CausalTemporalCrossAttention",
    "AgentVideoFusion",
]
