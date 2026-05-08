import numpy as np
import torch
import torch.nn as nn
from .context_encoder import build_context_encoder
from .motion_decoder import build_decoder
from .motion_decoder.mtr_decoder import modulate
from .utils.common_layers import build_mlps
from einops import repeat, rearrange
from models.context_encoder.mtr_encoder import SinusoidalPosEmb

class ETHMotionTransformer(nn.Module):
    def __init__(self, model_config, logger, config):
        super().__init__()
        self.model_cfg = model_config
        # Ensure D_MODEL is correctly pulled from config (128) [1, 2]
        self.dim = self.model_cfg.CONTEXT_ENCODER.D_MODEL
        self.config = config
        
        # Build the context encoder (which we integrated with video support) [3]
        self.context_encoder = build_context_encoder(self.model_cfg.CONTEXT_ENCODER, use_pre_norm=True)
        self.motion_decoder = build_decoder(self.model_cfg.MOTION_DECODER, use_pre_norm=True)

    def forward(self, batch):
        """
        Multimodal Forward Pass for the Teacher Model (Module 7)
        """
        # 1. Extract inputs from the multimodal dataloader [4]
        # past_traj contains kinematic features (e.g., [Batch, Agents, 48])
        past_traj = batch['pre_motion_3D'] 
        
        # Extract synchronized video tensor [Batch, T_obs(8), 3, 224, 224]
        # Using .get() for backward compatibility with kinematic-only batches
        video = batch.get('video_frames', None) 

        # 2. Multimodal Context Encoding (Étape 2)
        # Inside ETHEncoder, z_traj will now query z_video via Cross-Attention [3, 5]
        # Output z_ctx shape: [Batch, Agents, 128]
        z_ctx = self.context_encoder(past_traj, video_tensor=video)

        # 3. Conditional Flow Matching Logic
        # The fused z_ctx is passed to the decoder to modulate the velocity field v_theta
        # based on visual scene constraints [6, 7]
        
        # (The original MoFlow flow matching logic continues here...)
        # e.g., sampling flow time t, noisy trajectories Y_t, etc.
        # predictions = self.motion_decoder(..., z_ctx=z_ctx, ...)
        
        # return predictions

class ETHIMLETransformer(nn.Module):
    def __init__(self, model_config, logger, config):
        super().__init__()
        self.model_cfg = model_config
        self.dim = self.model_cfg.CONTEXT_ENCODER.D_MODEL
        self.cfg = config
        
        # Student model shares the same multimodal encoder architecture [8, 9]
        self.context_encoder = build_context_encoder(self.model_cfg.CONTEXT_ENCODER, use_pre_norm=True)

    def forward(self, batch):
        """
        Multimodal Forward Pass for the One-Step Student Model (Module 8)
        """
        # 1. Extract inputs
        past_traj = batch['pre_motion_3D']
        video = batch.get('video_frames', None)

        # 2. Multimodal Context Encoding
        # Student learns to approximate the teacher's video-conditioned context [10, 11]
        z_ctx = self.context_encoder(past_traj, video_tensor=video)

        # 3. One-Step Student Generation (IMLE Distillation)
        # return student_predictions_conditioned_on_video
