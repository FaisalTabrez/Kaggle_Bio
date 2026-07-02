"""
Handcrafted feature extraction — lean ~32 dimensions.

Extracts: coordinates, intensity stats, local density, velocity estimate,
blob scale, frame position. No heavy image processing.
"""

import numpy as np
from typing import List, Dict, Optional
from scipy.spatial import cKDTree

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.coords import (
    Detection, VOXEL_SPACING_UM, voxel_to_physical,
    scale_coords_for_kdtree,
)


def extract_handcrafted_features(
    detections_by_frame: Dict[int, List[Detection]],
    n_frames_total: int,
    volume_shape: tuple = (64, 256, 256),
    voxel_spacing: np.ndarray = VOXEL_SPACING_UM,
    density_k: int = 5,
) -> Dict[int, np.ndarray]:
    """
    Extract handcrafted features for all detections.

    Features per detection (~32 dims):
        - Normalized physical coordinates (z, y, x):  3
        - Normalized frame position (t / T):           1
        - Intensity stats (mean, std, max, min):       4
        - Blob scale (radius in µm):                   1
        - Confidence (LoG response):                   1
        - Local density (mean, std of k-NN dists):     2
        - Velocity estimate (Δz, Δy, Δx in µm):       3
        - Velocity magnitude:                          1
        - Padding to 32:                               remaining

    Args:
        detections_by_frame: Dict mapping frame_idx → list of Detection.
        n_frames_total: Total number of frames in the volume.
        volume_shape: (Z, Y, X) spatial dimensions.
        voxel_spacing: (Z, Y, X) voxel spacing in µm.
        density_k: Number of neighbors for density computation.

    Returns:
        Dict mapping node_id → feature vector (np.ndarray of shape [32]).
    """
    # Physical extent for normalization
    max_physical = np.array(volume_shape, dtype=np.float64) * voxel_spacing

    # Pre-compute velocity: match each detection to nearest in previous frame
    velocities = _compute_velocities(detections_by_frame, voxel_spacing)

    # Pre-compute local densities per frame
    densities = _compute_local_densities(detections_by_frame, voxel_spacing, density_k)

    features = {}

    for t, det_list in detections_by_frame.items():
        for det in det_list:
            feat = np.zeros(32, dtype=np.float32)

            # Normalized physical coordinates [0, 1]
            phys = voxel_to_physical(np.array([det.z, det.y, det.x]), voxel_spacing)
            feat[0:3] = phys / max_physical

            # Normalized frame position
            feat[3] = t / max(n_frames_total - 1, 1)

            # Intensity stats (log-scaled for better range)
            feat[4] = np.log1p(det.intensity_mean) / 10.0
            feat[5] = np.log1p(det.intensity_std) / 10.0
            feat[6] = np.log1p(det.intensity_max) / 10.0
            feat[7] = np.log1p(det.intensity_min) / 10.0

            # Blob scale (normalized by typical cell radius ~5 µm)
            feat[8] = det.blob_scale / 10.0

            # Detection confidence (LoG response, normalized)
            feat[9] = min(det.confidence / 1.0, 5.0) / 5.0

            # Local density
            density_info = densities.get(det.node_id, (0.0, 0.0))
            feat[10] = density_info[0] / 20.0  # mean k-NN distance, normalized
            feat[11] = density_info[1] / 20.0  # std k-NN distance

            # Velocity estimate
            vel = velocities.get(det.node_id, np.zeros(3))
            feat[12:15] = vel / 10.0  # µm, normalized

            # Velocity magnitude
            feat[15] = np.linalg.norm(vel) / 15.0

            # Reserved for future features (16-31)
            # Keeping zeros — easy to add more later without resizing

            features[det.node_id] = feat

    return features


def _compute_velocities(
    detections_by_frame: Dict[int, List[Detection]],
    voxel_spacing: np.ndarray,
) -> Dict[int, np.ndarray]:
    """
    Estimate velocity for each detection by matching to nearest
    detection in the previous frame.

    Returns: Dict[node_id → velocity_vector_um (Δz, Δy, Δx)].
    """
    velocities = {}
    sorted_frames = sorted(detections_by_frame.keys())

    for i, t in enumerate(sorted_frames):
        if i == 0:
            # No previous frame — zero velocity
            for det in detections_by_frame[t]:
                velocities[det.node_id] = np.zeros(3)
            continue

        t_prev = sorted_frames[i - 1]
        prev_dets = detections_by_frame[t_prev]
        curr_dets = detections_by_frame[t]

        if len(prev_dets) == 0:
            for det in curr_dets:
                velocities[det.node_id] = np.zeros(3)
            continue

        # Build KD-tree in physical space for previous frame
        prev_coords = scale_coords_for_kdtree(
            np.array([[d.z, d.y, d.x] for d in prev_dets]),
            voxel_spacing,
        )
        tree = cKDTree(prev_coords)

        for det in curr_dets:
            curr_phys = voxel_to_physical(
                np.array([det.z, det.y, det.x]), voxel_spacing
            )
            dist, idx = tree.query(curr_phys, k=1)

            if dist < 30.0:  # Only if reasonably close (30 µm)
                prev_det = prev_dets[idx]
                prev_phys = voxel_to_physical(
                    np.array([prev_det.z, prev_det.y, prev_det.x]), voxel_spacing
                )
                velocities[det.node_id] = curr_phys - prev_phys
            else:
                velocities[det.node_id] = np.zeros(3)

    return velocities


def _compute_local_densities(
    detections_by_frame: Dict[int, List[Detection]],
    voxel_spacing: np.ndarray,
    k: int = 5,
) -> Dict[int, tuple]:
    """
    Compute local density: mean and std of k-nearest-neighbor distances
    in physical space, per frame.

    Returns: Dict[node_id → (mean_knn_dist, std_knn_dist)].
    """
    densities = {}

    for t, det_list in detections_by_frame.items():
        if len(det_list) <= 1:
            for det in det_list:
                densities[det.node_id] = (0.0, 0.0)
            continue

        coords = scale_coords_for_kdtree(
            np.array([[d.z, d.y, d.x] for d in det_list]),
            voxel_spacing,
        )
        tree = cKDTree(coords)

        actual_k = min(k + 1, len(det_list))
        dists, _ = tree.query(coords, k=actual_k)

        for i, det in enumerate(det_list):
            # Skip self-distance (first column)
            knn_dists = dists[i, 1:actual_k]
            densities[det.node_id] = (
                float(np.mean(knn_dists)),
                float(np.std(knn_dists)),
            )

    return densities
