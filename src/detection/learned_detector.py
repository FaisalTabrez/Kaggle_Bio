"""
Learned 3D cell detector — CenterNet-style heatmap regression.

LoG recall topped out at 52%. This module trains a lightweight 3D U-Net
to predict Gaussian heatmaps centered on cell nuclei.

Architecture: ~800K params, fits easily on T4.
  Input:  (B, 1, 64, 256, 256) — single frame
  Output: (B, 1, 64, 256, 256) — heatmap with Gaussian peaks

Training data: sparse GT centroids → Gaussian target heatmaps.
~10K annotated cells across 199 training samples.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional


class ConvBlock(nn.Module):
    """3D Conv → InstanceNorm → LeakyReLU."""
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size, padding=padding)
        self.norm = nn.InstanceNorm3d(out_ch)
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class DownBlock(nn.Module):
    """Downsample: MaxPool → 2x ConvBlock."""
    def __init__(self, in_ch, out_ch, pool_size=(2, 2, 2)):
        super().__init__()
        self.pool = nn.MaxPool3d(pool_size)
        self.conv1 = ConvBlock(in_ch, out_ch)
        self.conv2 = ConvBlock(out_ch, out_ch)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UpBlock(nn.Module):
    """Upsample: Trilinear up → concat skip → 2x ConvBlock."""
    def __init__(self, in_ch, skip_ch, out_ch, scale_factor=(2, 2, 2)):
        super().__init__()
        self.scale_factor = scale_factor
        self.conv1 = ConvBlock(in_ch + skip_ch, out_ch)
        self.conv2 = ConvBlock(out_ch, out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='trilinear', align_corners=False)
        # Handle size mismatches from odd dimensions
        if x.shape != skip.shape:
            x = F.pad(x, [
                0, skip.shape[4] - x.shape[4],
                0, skip.shape[3] - x.shape[3],
                0, skip.shape[2] - x.shape[2],
            ])
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class CellDetectorUNet(nn.Module):
    """
    Lightweight 3D U-Net for cell centroid heatmap prediction.
    
    ~800K params. Input: (B, 1, D, H, W). Output: (B, 1, D, H, W).
    
    Architecture:
        Encoder: 1→16→32→64 (3 levels)
        Bottleneck: 64→128
        Decoder: 128→64→32→16
        Head: 16→1 (sigmoid)
    
    Anisotropy-aware: Z is pooled less aggressively since
    there are only 64 Z slices vs 256 XY.
    """

    def __init__(self, in_channels=1, base_channels=16):
        super().__init__()
        c = base_channels  # 16

        # Encoder
        self.enc1_conv1 = ConvBlock(in_channels, c)
        self.enc1_conv2 = ConvBlock(c, c)
        # Z=64, Y=256, X=256

        self.enc2 = DownBlock(c, c * 2, pool_size=(2, 2, 2))
        # Z=32, Y=128, X=128

        self.enc3 = DownBlock(c * 2, c * 4, pool_size=(2, 2, 2))
        # Z=16, Y=64, X=64

        # Bottleneck
        self.bottleneck = DownBlock(c * 4, c * 8, pool_size=(2, 2, 2))
        # Z=8, Y=32, X=32

        # Decoder
        self.dec3 = UpBlock(c * 8, c * 4, c * 4, scale_factor=(2, 2, 2))
        self.dec2 = UpBlock(c * 4, c * 2, c * 2, scale_factor=(2, 2, 2))
        self.dec1 = UpBlock(c * 2, c, c, scale_factor=(2, 2, 2))

        # Output head
        self.head = nn.Conv3d(c, 1, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x: (B, 1, D, H, W) normalized input volume.
        Returns:
            (B, 1, D, H, W) heatmap with values in [0, 1].
        """
        # Encoder
        e1 = self.enc1_conv2(self.enc1_conv1(x))  # (B, 16, 64, 256, 256)
        e2 = self.enc2(e1)   # (B, 32, 32, 128, 128)
        e3 = self.enc3(e2)   # (B, 64, 16, 64, 64)

        # Bottleneck
        bn = self.bottleneck(e3)  # (B, 128, 8, 32, 32)

        # Decoder
        d3 = self.dec3(bn, e3)  # (B, 64, 16, 64, 64)
        d2 = self.dec2(d3, e2)  # (B, 32, 32, 128, 128)
        d1 = self.dec1(d2, e1)  # (B, 16, 64, 256, 256)

        # Heatmap
        out = torch.sigmoid(self.head(d1))  # (B, 1, 64, 256, 256)
        return out

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================
# Training utilities
# ============================================================

def make_gaussian_heatmap(
    volume_shape: Tuple[int, int, int],
    centers_zyx: List[Tuple[int, int, int]],
    sigma_voxels: Tuple[float, float, float] = (2.0, 4.0, 4.0),
) -> np.ndarray:
    """
    Create a Gaussian heatmap target from cell centroid positions.
    
    Args:
        volume_shape: (Z, Y, X) dimensions.
        centers_zyx: List of (z, y, x) centroid positions.
        sigma_voxels: Gaussian sigma per axis in voxels.
                      Z is smaller because of anisotropy (fewer slices).
    
    Returns:
        Heatmap array of shape volume_shape with values in [0, 1].
    """
    heatmap = np.zeros(volume_shape, dtype=np.float32)
    Z, Y, X = volume_shape
    sz, sy, sx = sigma_voxels

    for cz, cy, cx in centers_zyx:
        cz, cy, cx = int(round(cz)), int(round(cy)), int(round(cx))

        # Compute local region (3*sigma is sufficient)
        rz = int(np.ceil(3 * sz))
        ry = int(np.ceil(3 * sy))
        rx = int(np.ceil(3 * sx))

        z0, z1 = max(0, cz - rz), min(Z, cz + rz + 1)
        y0, y1 = max(0, cy - ry), min(Y, cy + ry + 1)
        x0, x1 = max(0, cx - rx), min(X, cx + rx + 1)

        zz, yy, xx = np.meshgrid(
            np.arange(z0, z1), np.arange(y0, y1), np.arange(x0, x1),
            indexing='ij',
        )

        gaussian = np.exp(-(
            ((zz - cz) ** 2) / (2 * sz ** 2) +
            ((yy - cy) ** 2) / (2 * sy ** 2) +
            ((xx - cx) ** 2) / (2 * sx ** 2)
        ))

        # Max with existing (handles overlapping Gaussians)
        heatmap[z0:z1, y0:y1, x0:x1] = np.maximum(
            heatmap[z0:z1, y0:y1, x0:x1], gaussian
        )

    return heatmap


class FocalMSELoss(nn.Module):
    """
    Focal MSE loss for heatmap regression with extreme class imbalance.
    
    Most voxels are background (target=0). Standard MSE wastes capacity
    on easy negatives. Focal weighting emphasizes:
      - Hard negatives (background predicted as cell)
      - All positives (cell centers)
    """

    def __init__(self, alpha: float = 2.0, beta: float = 4.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, 1, D, H, W) predicted heatmap.
            target: (B, 1, D, H, W) Gaussian target heatmap.
        """
        pos_mask = (target >= 0.01)  # Near cell centers
        neg_mask = ~pos_mask

        # Positive loss: standard MSE weighted by (1 - pred)^alpha
        pos_loss = torch.zeros_like(pred)
        if pos_mask.any():
            pos_weight = (1 - pred[pos_mask]) ** self.alpha
            pos_loss[pos_mask] = pos_weight * (pred[pos_mask] - target[pos_mask]) ** 2

        # Negative loss: down-weight easy negatives, focus on hard ones
        neg_loss = torch.zeros_like(pred)
        if neg_mask.any():
            neg_weight = pred[neg_mask] ** self.alpha * (1 - target[neg_mask]) ** self.beta
            neg_loss[neg_mask] = neg_weight * pred[neg_mask] ** 2

        loss = (pos_loss.sum() + neg_loss.sum()) / max(pos_mask.sum().item(), 1)
        return loss


def extract_peaks_from_heatmap(
    heatmap: np.ndarray,
    threshold: float = 0.3,
    min_distance_voxels: Tuple[int, int, int] = (2, 5, 5),
) -> List[Tuple[float, float, float, float]]:
    """
    Extract cell centroids from predicted heatmap via peak detection.
    
    Args:
        heatmap: 3D array (Z, Y, X) with values in [0, 1].
        threshold: Minimum peak value.
        min_distance_voxels: Minimum distance between peaks per axis.
    
    Returns:
        List of (z, y, x, confidence) tuples.
    """
    from scipy.ndimage import maximum_filter

    footprint_size = tuple(2 * d + 1 for d in min_distance_voxels)
    local_max = maximum_filter(heatmap, size=footprint_size)
    peaks_mask = (heatmap == local_max) & (heatmap > threshold)

    coords = np.array(np.nonzero(peaks_mask)).T  # (N, 3)
    if len(coords) == 0:
        return []

    results = []
    for z, y, x in coords:
        conf = float(heatmap[z, y, x])
        results.append((float(z), float(y), float(x), conf))

    # Sort by confidence descending
    results.sort(key=lambda r: -r[3])
    return results


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    """Normalize uint16 volume to [0, 1] float32."""
    v = volume.astype(np.float32)
    vmin, vmax = v.min(), v.max()
    if vmax > vmin:
        v = (v - vmin) / (vmax - vmin)
    return v
