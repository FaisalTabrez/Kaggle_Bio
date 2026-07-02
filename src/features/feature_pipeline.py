"""
Unified feature extraction pipeline.

Combines handcrafted features (~32 dims) with CNN crop embeddings (~48 dims)
into a single ~80-dim node feature vector.

The crop encoder is part of the model and trained jointly with the GNN.
Handcrafted features are pre-computed and fixed.
"""

import numpy as np
import torch
from typing import List, Dict, Optional, Tuple

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.coords import Detection, VOXEL_SPACING_UM
from src.features.handcrafted import extract_handcrafted_features


def build_node_features(
    detections_by_frame: Dict[int, List[Detection]],
    n_frames_total: int,
    volume_shape: Tuple[int, int, int] = (64, 256, 256),
    voxel_spacing: np.ndarray = VOXEL_SPACING_UM,
    handcrafted_dim: int = 32,
) -> Tuple[Dict[int, np.ndarray], List[Detection]]:
    """
    Build handcrafted node features for all detections.
    
    The CNN embedding is NOT computed here — it happens inside the GNN
    forward pass so gradients flow through.

    Args:
        detections_by_frame: Frame → detection list mapping.
        n_frames_total: Total frames in volume.
        volume_shape: (Z, Y, X) spatial dims.
        voxel_spacing: Voxel spacing in µm.
        handcrafted_dim: Dimension of handcrafted feature vector.

    Returns:
        Tuple of:
        - features: Dict[node_id → np.ndarray of shape (handcrafted_dim,)]
        - flat_detections: Flat list of all detections (sorted by node_id)
    """
    # Extract handcrafted features
    features = extract_handcrafted_features(
        detections_by_frame,
        n_frames_total=n_frames_total,
        volume_shape=volume_shape,
        voxel_spacing=voxel_spacing,
    )

    # Flatten detections sorted by node_id
    flat_detections = []
    for t in sorted(detections_by_frame.keys()):
        flat_detections.extend(detections_by_frame[t])
    flat_detections.sort(key=lambda d: d.node_id)

    return features, flat_detections


def features_to_tensor(
    features: Dict[int, np.ndarray],
    node_ids: List[int],
) -> torch.Tensor:
    """
    Convert feature dict to a tensor ordered by node ID.

    Args:
        features: Dict[node_id → feature_vector].
        node_ids: Ordered list of node IDs.

    Returns:
        Float tensor of shape (N, feature_dim).
    """
    feat_list = [features[nid] for nid in node_ids]
    return torch.tensor(np.stack(feat_list), dtype=torch.float32)
