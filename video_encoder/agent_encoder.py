"""
Agent-Centric Video Encoder for MoFlow.
Extracts per-agent visual features from crops around agent positions.

Two public entry points:

* :class:`AgentVideoEncoder` — the core time-shared ResNet-18 encoder that
  turns ``[B, A, T_obs, C, H, W]`` raw crops into ``[B, A, T_obs, D]``
  agent-specific visual tokens.
* :class:`CompactAgentVideoEncoder` — a lightweight, fully-convolutional
  alternative intended to run on smaller crops with lower compute.  Uses the
  same ``[B, A, T_obs, C, H, W] -> [B, A, T_obs, D]`` contract so it is a
  drop-in replacement for :class:`AgentVideoEncoder`.

The crop tensors themselves are produced by
:mod:`data.agent_crop_sdd` (:func:`extract_agent_crops` /
:class:`SDDAgentCropDataset`) from the raw ``annotations.txt`` bounding boxes.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from einops import rearrange


class AgentVideoEncoder(nn.Module):
    def __init__(
        self,
        d_model=128,
        pretrained=True,
        freeze_blocks=2,
        spatial_dropout_rate=0.5,
    ):
        super().__init__()

        # Load pre-trained ResNet-18
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)

        # Extract the feature backbone (everything before the final avgpool and fc)
        # Output shape: [B*T, 512, 7, 7] for 224x224 input
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        self.feature_dim = 512
        self.d_model = d_model

        # Freeze early blocks if specified
        if freeze_blocks > 0:
            self._freeze_resnet_blocks(freeze_blocks)

        # Spatial dropout layer
        self.spatial_dropout = nn.Dropout2d(p=spatial_dropout_rate) if spatial_dropout_rate > 0 else nn.Identity()

        # Project visual features to match the trajectory embedding dimension (D)
        self.token_projection = nn.Linear(self.feature_dim, d_model)

    def _freeze_resnet_blocks(self, num_blocks):
        """Freeze the first num_blocks of ResNet layers.

        ResNet-18 structure:
        - conv1, bn1, relu, maxpool (initial layers)
        - layer1, layer2, layer3, layer4 (the 4 main blocks)
        """
        # Freeze initial layers (conv1, bn1)
        if num_blocks >= 1:
            for param in self.backbone[0].parameters():  # conv1
                param.requires_grad = False
            for param in self.backbone[1].parameters():  # bn1
                param.requires_grad = False
            for param in self.backbone[2].parameters():  # relu
                param.requires_grad = False
            for param in self.backbone[3].parameters():  # maxpool
                param.requires_grad = False

        # Freeze specified number of residual layers
        # layer indexing: [0]=conv1, [1]=bn1, [2]=relu, [3]=maxpool, [4]=layer1, [5]=layer2, [6]=layer3, [7]=layer4
        for i in range(4, 4 + num_blocks):
            if i < len(self.backbone):
                for param in self.backbone[i].parameters():
                    param.requires_grad = False

    def forward(self, video_crops):
        """
        Input:  [Batch, Agents, T_obs(8), Channels(3), H, W]
        Output: [Batch, Agents, T_obs, D]
        """
        B, A, T, C, H, W = video_crops.shape

        # 1. Flatten Batch, Agents, and Time dimensions: (B*A*T, 3, H, W)
        x = rearrange(video_crops, 'b a t c h w -> (b a t) c h w')

        # Resize to the backbone's canonical input size (224x224) if needed.
        if H != 224 or W != 224:
            x = nn.functional.interpolate(
                x, size=(224, 224), mode="bilinear", align_corners=False
            )

        # 2. Extract spatial features via ResNet-18 backbone: (B*A*T, 512, 7, 7)
        spatial_features = self.backbone(x)

        # 3. Apply spatial dropout
        spatial_features = self.spatial_dropout(spatial_features)

        # 4. Global average pool over spatial dimensions: (B*A*T, 512)
        pooled_features = torch.mean(spatial_features, dim=[2, 3])

        # 5. Project to D_MODEL: (B*A*T, D)
        projected_features = self.token_projection(pooled_features)

        # 6. Reshape back to (B, A, T, D)
        agent_video_features = rearrange(
            projected_features, '(b a t) d -> b a t d', b=B, a=A, t=T
        )

        return agent_video_features


class CompactAgentVideoEncoder(nn.Module):
    """Lightweight fully-convolutional agent video encoder (drop-in for #VE).

    A compact stack of 3x3 2D convolutions + max-pool + global-average-pool
    that maps small crops to ``[B, A, T_obs, D]`` visual tokens.  No external
    ImageNet weights, so it is robust when pretraining weights are unavailable
    on the lab machine.  Suitable for running directly on the 64x64 crops from
    :func:`data.agent_crop_sdd.extract_agent_crops`.

    ``forward(video_crops)`` accepts any crop resolution >= 8 and returns
    ``[B, A, T_obs, d_model]``.
    """

    def __init__(self, d_model: int = 128, in_channels: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),               # 64 -> 32
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),               # 32 -> 16
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),               # 16 -> 8
            nn.AdaptiveAvgPool2d(1),       # -> [.., 128, 1, 1]
        )
        self.token_projection = nn.Linear(128, d_model)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, video_crops: torch.Tensor) -> torch.Tensor:
        """[B, A, T, C, H, W] -> [B, A, T, d_model]."""
        B, A, T, C, H, W = video_crops.shape
        x = rearrange(video_crops, "b a t c h w -> (b a t) c h w")
        x = self.features(x)                    # [B*A*T, 128, 1, 1]
        x = x.flatten(start_dim=1)              # [B*A*T, 128]
        x = self.dropout(x)
        x = self.token_projection(x)            # [B*A*T, d_model]
        return rearrange(x, "(b a t) d -> b a t d", b=B, a=A, t=T)
