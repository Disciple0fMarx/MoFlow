import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from einops import rearrange

class GlobalVideoEncoder(nn.Module):
    def __init__(self, d_model=128, pretrained=True):
        super().__init__()
        # Use the modern 'weights' parameter to avoid deprecation warnings
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)
        
        # Extract the feature backbone (output shape: [B*T, 512, 7, 7])
        self.backbone = nn.Sequential(*list(resnet.children())[:-2]) 
        
        self.feature_dim = 512
        # Project visual features to match the trajectory embedding dimension (D)
        self.token_projection = nn.Linear(self.feature_dim, d_model)
        
    def forward(self, video_tensor):
        """
        Input:  [Batch, T_obs(8), Channels(3), H(224), W(224)]
        Output: [Batch, L_global(8), D(128)]
        """
        B, T, C, H, W = video_tensor.shape
        
        # 1. Flatten Batch and Time: (32, 3, 224, 224)
        x = rearrange(video_tensor, 'b t c h w -> (b t) c h w')
        
        # 2. Extract spatial maps: (32, 512, 7, 7)
        spatial_features = self.backbone(x)
        
        # 3. FIX: Pool over Height (2) and Width (3)
        # Resulting shape: (32, 512)
        pooled_features = torch.mean(spatial_features, dim=[2, 3])
        
        # 4. Unflatten to (4, 8, 512)
        temporal_features = rearrange(pooled_features, '(b t) d -> b t d', b=B, t=T)
        
        # 5. Project to D_MODEL (128): (4, 8, 128)
        # This will now work: (32 x 512) * (512 x 128) -> (32 x 128)
        z_video = self.token_projection(temporal_features)
        
        return z_video
