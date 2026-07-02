"""
Anisotropy-aware 3D LoG blob detector for cell nuclei detection.

Uses Laplacian of Gaussian (LoG) across multiple scales to find
cell centroids. No training required — classical signal processing.

The detector handles the competition's anisotropic voxel spacing
(Z: 1.625 µm, YX: 0.40625 µm) by scaling sigma per axis.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from scipy.ndimage import gaussian_laplace, maximum_filter, label
from scipy.spatial import cKDTree

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.coords import Detection, VOXEL_SPACING_UM


def detect_cells_log(
    volume: np.ndarray,
    frame_idx: int,
    voxel_spacing: np.ndarray = VOXEL_SPACING_UM,
    sigma_um_range: Tuple[float, ...] = (2.0, 3.0, 4.0, 5.0),
    threshold_abs: float = 0.0,
    threshold_rel: float = 0.1,
    min_distance_um: float = 5.0,
    exclude_border: int = 2,
) -> List[Detection]:
    """
    Detect cell nuclei using multi-scale LoG blob detection.

    Args:
        volume: 3D array (Z, Y, X), single timepoint.
        frame_idx: Frame index t for the returned Detection objects.
        voxel_spacing: (Z, Y, X) spacing in µm.
        sigma_um_range: LoG sigma values in µm to search.
        threshold_abs: Absolute threshold on LoG response.
        threshold_rel: Relative threshold (fraction of max response).
        min_distance_um: Minimum distance between detections in µm.
        exclude_border: Exclude peaks within this many voxels of border.

    Returns:
        List of Detection objects with centroids and intensity stats.
    """
    vol = volume.astype(np.float32)

    # Normalize intensity to [0, 1] for consistent thresholding
    vmin, vmax = vol.min(), vol.max()
    if vmax > vmin:
        vol_norm = (vol - vmin) / (vmax - vmin)
    else:
        return []

    # Multi-scale LoG responses
    log_responses = []
    for sigma_um in sigma_um_range:
        # Convert isotropic sigma (µm) to anisotropic sigma (voxels)
        sigma_voxels = np.array([
            sigma_um / voxel_spacing[0],  # Z: larger spacing → smaller sigma
            sigma_um / voxel_spacing[1],  # Y
            sigma_um / voxel_spacing[2],  # X
        ])

        # LoG response (negated because LoG gives negative at bright blobs)
        log_resp = -gaussian_laplace(vol_norm, sigma=sigma_voxels)

        # Scale-normalize: multiply by sigma^2 for fair comparison across scales
        mean_sigma = np.mean(sigma_voxels)
        log_resp *= mean_sigma ** 2

        log_responses.append(log_resp)

    # Stack and take max across scales
    log_stack = np.stack(log_responses, axis=0)  # (n_scales, Z, Y, X)
    log_max = np.max(log_stack, axis=0)  # (Z, Y, X) — best scale per voxel
    scale_idx = np.argmax(log_stack, axis=0)  # which scale was best

    # Find threshold
    effective_threshold = max(
        threshold_abs,
        threshold_rel * log_max.max(),
    )

    # Local maxima detection with anisotropy-aware neighborhood
    min_dist_voxels = np.array([
        max(1, int(np.round(min_distance_um / voxel_spacing[0]))),
        max(1, int(np.round(min_distance_um / voxel_spacing[1]))),
        max(1, int(np.round(min_distance_um / voxel_spacing[2]))),
    ])

    # Use maximum filter for local maxima detection
    footprint_size = 2 * min_dist_voxels + 1
    local_max = maximum_filter(log_max, size=footprint_size)
    peaks_mask = (log_max == local_max) & (log_max > effective_threshold)

    # Exclude border
    if exclude_border > 0:
        b = exclude_border
        peaks_mask[:b, :, :] = False
        peaks_mask[-b:, :, :] = False
        peaks_mask[:, :b, :] = False
        peaks_mask[:, -b:, :] = False
        peaks_mask[:, :, :b] = False
        peaks_mask[:, :, -b:] = False

    # Extract peak coordinates
    peak_coords = np.array(np.nonzero(peaks_mask)).T  # (N, 3) as [z, y, x]

    if len(peak_coords) == 0:
        return []

    # NMS in physical space
    peak_coords = _nms_physical(
        peak_coords, voxel_spacing, min_distance_um,
        scores=log_max[peaks_mask],
    )

    # Build Detection objects
    detections = []
    for i, (z, y, x) in enumerate(peak_coords):
        z, y, x = int(z), int(y), int(x)

        # Get local intensity stats
        stats = _local_intensity_stats(volume, z, y, x, radius=4)

        # Get the best scale for this peak
        best_scale = int(scale_idx[z, y, x])
        blob_radius_um = sigma_um_range[best_scale] * np.sqrt(3)

        detections.append(Detection(
            node_id=i,  # Will be renumbered later
            t=frame_idx,
            z=float(z),
            y=float(y),
            x=float(x),
            confidence=float(log_max[z, y, x]),
            intensity_mean=stats["mean"],
            intensity_std=stats["std"],
            intensity_max=stats["max"],
            intensity_min=stats["min"],
            blob_scale=float(blob_radius_um),
        ))

    return detections


def detect_all_frames(
    zarr_loader,
    voxel_spacing: Optional[np.ndarray] = None,
    **kwargs,
) -> Dict[int, List[Detection]]:
    """
    Run detection on all frames of a volume.

    Args:
        zarr_loader: A ZarrLoader instance.
        voxel_spacing: Override voxel spacing (uses loader's default if None).
        **kwargs: Additional arguments passed to detect_cells_log.

    Returns:
        Dict mapping frame index → list of Detection objects.
        Node IDs are globally unique across frames.
    """
    if voxel_spacing is None:
        voxel_spacing = zarr_loader.voxel_spacing

    all_detections = {}
    global_id = 0

    for t in range(zarr_loader.n_frames):
        frame = zarr_loader.get_frame(t)
        frame_dets = detect_cells_log(
            frame, frame_idx=t, voxel_spacing=voxel_spacing, **kwargs
        )

        # Assign globally unique IDs
        for det in frame_dets:
            det.node_id = global_id
            global_id += 1

        all_detections[t] = frame_dets

    return all_detections


def _nms_physical(
    coords: np.ndarray,
    spacing: np.ndarray,
    min_dist_um: float,
    scores: np.ndarray,
) -> np.ndarray:
    """
    Non-maximum suppression in physical (µm) coordinate space.
    Keeps highest-scoring peak when two are within min_dist_um.
    """
    if len(coords) == 0:
        return coords

    # Convert to physical space for distance computation
    physical = coords.astype(np.float64) * spacing

    # Sort by score (descending)
    order = np.argsort(-scores)
    keep = []
    suppressed = set()

    # Build KD-tree for efficient neighbor search
    tree = cKDTree(physical)

    for idx in order:
        if idx in suppressed:
            continue
        keep.append(idx)

        # Find neighbors within min_dist
        neighbors = tree.query_ball_point(physical[idx], r=min_dist_um)
        for n in neighbors:
            if n != idx:
                suppressed.add(n)

    return coords[keep]


def _local_intensity_stats(
    volume: np.ndarray,
    z: int,
    y: int,
    x: int,
    radius: int = 4,
) -> Dict[str, float]:
    """Compute intensity statistics in a local cubic neighborhood."""
    Z, Y, X = volume.shape
    z0 = max(0, z - radius)
    z1 = min(Z, z + radius + 1)
    y0 = max(0, y - radius)
    y1 = min(Y, y + radius + 1)
    x0 = max(0, x - radius)
    x1 = min(X, x + radius + 1)

    patch = volume[z0:z1, y0:y1, x0:x1].astype(np.float32)
    if patch.size == 0:
        return {"mean": 0.0, "std": 0.0, "max": 0.0, "min": 0.0}

    return {
        "mean": float(np.mean(patch)),
        "std": float(np.std(patch)),
        "max": float(np.max(patch)),
        "min": float(np.min(patch)),
    }
