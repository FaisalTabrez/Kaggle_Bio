"""
Edge feature computation for the spatio-temporal graph.

Compact edge features (~20 dims) capturing:
- Physical distance and displacement
- Direction vector
- Intensity and size ratios
- Cosine similarity of embeddings (computed later in GNN forward)
- Temporal distance
- Local density difference
- Motion consistency
- Distance rank among candidates
"""

import numpy as np
from typing import List, Dict, Tuple

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.coords import Detection, VOXEL_SPACING_UM, voxel_to_physical


def compute_edge_features(
    edge_list: List[Tuple[int, int]],
    detections: Dict[int, Detection],
    densities: Dict[int, Tuple[float, float]] = None,
    velocities: Dict[int, np.ndarray] = None,
    voxel_spacing: np.ndarray = VOXEL_SPACING_UM,
    edge_dim: int = 20,
) -> np.ndarray:
    """
    Compute feature vectors for all candidate edges.

    Edge features (20 dims):
        [0]    Physical distance (µm)
        [1:4]  Displacement (Δz, Δy, Δx) in µm
        [4:7]  Normalized direction vector
        [7]    Intensity ratio (src / dst)
        [8]    Size ratio (src / dst)
        [9]    Temporal distance (frame gap, usually 1 or 2)
        [10]   Local density difference
        [11]   Motion consistency (distance from predicted position)
        [12:15] Relative midpoint position (normalized z, y, x)
        [15]   Distance rank (1st nearest, 2nd, etc.)
        [16]   Number of candidates for this source node
        [17]   Log distance
        [18]   Intensity difference (absolute)
        [19]   Reserved

    Args:
        edge_list: List of (source_id, target_id).
        detections: Dict[node_id → Detection].
        densities: Optional dict[node_id → (mean_knn_dist, std_knn_dist)].
        velocities: Optional dict[node_id → velocity_vector_um].
        voxel_spacing: Voxel spacing in µm.
        edge_dim: Output dimension per edge.

    Returns:
        np.ndarray of shape (E, edge_dim).
    """
    n_edges = len(edge_list)
    features = np.zeros((n_edges, edge_dim), dtype=np.float32)

    if n_edges == 0:
        return features

    # Pre-compute per-source candidate counts and distance ranks
    src_edges = {}
    for i, (src, dst) in enumerate(edge_list):
        src_edges.setdefault(src, []).append(i)

    for i, (src_id, dst_id) in enumerate(edge_list):
        src = detections.get(src_id)
        dst = detections.get(dst_id)

        if src is None or dst is None:
            continue

        src_phys = voxel_to_physical(
            np.array([src.z, src.y, src.x]), voxel_spacing
        )
        dst_phys = voxel_to_physical(
            np.array([dst.z, dst.y, dst.x]), voxel_spacing
        )

        # Displacement and distance
        displacement = dst_phys - src_phys
        distance = float(np.linalg.norm(displacement))

        features[i, 0] = distance / 20.0  # Normalize by ~max expected

        features[i, 1:4] = displacement / 20.0  # Signed displacement

        # Direction vector
        if distance > 1e-6:
            features[i, 4:7] = displacement / distance
        # else: remains zero

        # Intensity ratio
        if dst.intensity_mean > 0:
            features[i, 7] = np.clip(
                src.intensity_mean / dst.intensity_mean, 0.1, 10.0
            ) / 5.0
        else:
            features[i, 7] = 1.0

        # Size ratio
        if dst.blob_scale > 0:
            features[i, 8] = np.clip(
                src.blob_scale / dst.blob_scale, 0.1, 10.0
            ) / 5.0
        else:
            features[i, 8] = 1.0

        # Temporal distance
        frame_gap = abs(dst.t - src.t)
        features[i, 9] = frame_gap / 3.0

        # Local density difference
        if densities is not None:
            src_dens = densities.get(src_id, (0.0, 0.0))[0]
            dst_dens = densities.get(dst_id, (0.0, 0.0))[0]
            features[i, 10] = abs(src_dens - dst_dens) / 20.0

        # Motion consistency
        if velocities is not None:
            vel = velocities.get(src_id)
            if vel is not None:
                predicted = src_phys + vel * frame_gap
                pred_error = float(np.linalg.norm(predicted - dst_phys))
                features[i, 11] = pred_error / 15.0

        # Relative midpoint
        midpoint = (src_phys + dst_phys) / 2.0
        # Rough normalization by volume extent (~100 µm per dim)
        features[i, 12:15] = midpoint / 100.0

        # Candidate count for source
        n_candidates = len(src_edges.get(src_id, []))
        features[i, 16] = n_candidates / 10.0

        # Log distance
        features[i, 17] = np.log1p(distance) / 4.0

        # Absolute intensity difference
        features[i, 18] = abs(
            src.intensity_mean - dst.intensity_mean
        ) / 1000.0

    # Compute distance ranks per source
    for src_id, edge_indices in src_edges.items():
        dists = [(features[idx, 0], idx) for idx in edge_indices]
        dists.sort()
        for rank, (_, idx) in enumerate(dists):
            features[idx, 15] = rank / max(len(dists) - 1, 1)

    return features
