"""
Tiny 3D CNN crop encoder for appearance embedding.

~50K parameters. Trained jointly with the GNN — no separate
triplet pre-training. The edge classification loss supervises
what appearance information matters.

Crop size: (8, 16, 16) voxels → 48-dim embedding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CropEncoder(nn.Module):
    """
    Lightweight 3D CNN for cell appearance embedding.
    
    Input: (B, 1, 8, 16, 16) — small 3D crops centered on detections.
    Output: (B, embed_dim) — appearance embedding vectors.
    
    Architecture:
        Conv3d(1→16, k=3) → BN → ReLU → MaxPool(2)
        Conv3d(16→32, k=3) → BN → ReLU → AdaptiveAvgPool3d(1)
        Linear(32 → embed_dim)
    
    ~50K parameters. FP16 compatible.
    """

    def __init__(self, embed_dim: int = 48):
        super().__init__()
        self.embed_dim = embed_dim

        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(16)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(32)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(32, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, 8, 16, 16) float tensor, normalized crops.
            
        Returns:
            (B, embed_dim) embedding vectors.
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool3d(x, 2)  # (B, 16, 4, 8, 8)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)  # (B, 32, 1, 1, 1)
        x = x.flatten(1)  # (B, 32)
        x = self.fc(x)    # (B, embed_dim)
        return x

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def normalize_crop(crop: torch.Tensor) -> torch.Tensor:
    """
    Normalize a batch of crops to zero-mean, unit-variance.
    Handles the case where a crop is all zeros (boundary padding).
    
    Args:
        crop: (B, 1, D, H, W) raw uint16 values cast to float.
    
    Returns:
        Normalized tensor of same shape.
    """
    # Per-crop normalization
    B = crop.shape[0]
    flat = crop.view(B, -1)
    mean = flat.mean(dim=1, keepdim=True)
    std = flat.std(dim=1, keepdim=True).clamp(min=1e-6)
    flat_norm = (flat - mean) / std
    return flat_norm.view_as(crop)
