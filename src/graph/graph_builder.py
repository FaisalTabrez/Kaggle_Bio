"""
Biology-informed spatio-temporal graph construction.

Instead of blind k-NN, candidate edges are filtered by:
  1. Physical distance < threshold
  2. Velocity plausibility
  3. Intensity similarity
  4. Cell size similarity

This reduces candidates from ~8 to ~2-4 per cell, making
the classification problem much easier for the GNN.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.spatial import cKDTree

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.coords import (
    Detection, VOXEL_SPACING_UM,
    voxel_to_physical, scale_coords_for_kdtree,
)


def build_graph(
    detections_by_frame: Dict[int, List[Detection]],
    voxel_spacing: np.ndarray = VOXEL_SPACING_UM,
    max_displacement_um: float = 15.0,
    max_frame_gap: int = 1,
    k_max: int = 6,
    intensity_ratio_bounds: Tuple[float, float] = (0.4, 2.5),
    size_ratio_bounds: Tuple[float, float] = (0.4, 2.5),
    velocity_tolerance_um: float = 12.0,
    use_biology_filters: bool = True,
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Build a spatio-temporal graph connecting detections across frames.

    Each detection = node. Candidate edges connect nodes in frame t
    to nodes in frames t+1 (and optionally t+2 for gap-closing).

    Biology-informed filtering reduces false candidates dramatically.

    Args:
        detections_by_frame: Frame → list of Detection objects.
        voxel_spacing: (Z, Y, X) µm per voxel.
        max_displacement_um: Maximum physical distance for an edge.
        max_frame_gap: Connect frames up to this gap (1 or 2).
        k_max: Maximum candidates per source node after filtering.
        intensity_ratio_bounds: (min, max) intensity ratio for valid edge.
        size_ratio_bounds: (min, max) blob scale ratio for valid edge.
        velocity_tolerance_um: Max deviation from predicted position.
        use_biology_filters: Whether to apply intensity/size/velocity filters.

    Returns:
        Tuple of:
        - edge_index: np.ndarray of shape (2, E) with (source, target) node IDs.
        - edge_list: List of (source_id, target_id) tuples.
    """
    sorted_frames = sorted(detections_by_frame.keys())
    all_edges = []

    # Pre-compute velocity estimates for velocity plausibility check
    velocities = {}
    for i, t in enumerate(sorted_frames):
        if i == 0:
            for det in detections_by_frame[t]:
                velocities[det.node_id] = None
            continue

        t_prev = sorted_frames[i - 1]
        prev_dets = detections_by_frame[t_prev]
        curr_dets = detections_by_frame[t]

        if len(prev_dets) > 0:
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
                if dist < 30.0:
                    prev_det = prev_dets[idx]
                    prev_phys = voxel_to_physical(
                        np.array([prev_det.z, prev_det.y, prev_det.x]),
                        voxel_spacing,
                    )
                    velocities[det.node_id] = curr_phys - prev_phys
                else:
                    velocities[det.node_id] = None
        else:
            for det in curr_dets:
                velocities[det.node_id] = None

    # Build edges
    for frame_idx, t_src in enumerate(sorted_frames):
        src_dets = detections_by_frame[t_src]
        if len(src_dets) == 0:
            continue

        # Connect to frames within max_frame_gap
        for gap in range(1, max_frame_gap + 1):
            next_frame_idx = frame_idx + gap
            if next_frame_idx >= len(sorted_frames):
                continue

            t_dst = sorted_frames[next_frame_idx]
            dst_dets = detections_by_frame[t_dst]
            if len(dst_dets) == 0:
                continue

            # Scale displacement budget for gap size
            max_dist = max_displacement_um * gap

            # Build KD-tree for target frame
            dst_coords = scale_coords_for_kdtree(
                np.array([[d.z, d.y, d.x] for d in dst_dets]),
                voxel_spacing,
            )
            tree = cKDTree(dst_coords)

            for src_det in src_dets:
                src_phys = voxel_to_physical(
                    np.array([src_det.z, src_det.y, src_det.x]),
                    voxel_spacing,
                )

                # Find all candidates within distance budget
                candidate_indices = tree.query_ball_point(src_phys, r=max_dist)

                # Score and filter candidates
                scored_candidates = []
                for idx in candidate_indices:
                    dst_det = dst_dets[idx]
                    dst_phys = dst_coords[idx]

                    dist = float(np.linalg.norm(src_phys - dst_phys))

                    if use_biology_filters:
                        # Filter 1: Intensity similarity
                        if src_det.intensity_mean > 0 and dst_det.intensity_mean > 0:
                            i_ratio = src_det.intensity_mean / dst_det.intensity_mean
                            if i_ratio < intensity_ratio_bounds[0] or i_ratio > intensity_ratio_bounds[1]:
                                continue

                        # Filter 2: Size similarity
                        if src_det.blob_scale > 0 and dst_det.blob_scale > 0:
                            s_ratio = src_det.blob_scale / dst_det.blob_scale
                            if s_ratio < size_ratio_bounds[0] or s_ratio > size_ratio_bounds[1]:
                                continue

                        # Filter 3: Velocity plausibility
                        vel = velocities.get(src_det.node_id)
                        if vel is not None:
                            predicted_pos = src_phys + vel * gap
                            prediction_error = float(np.linalg.norm(
                                predicted_pos - dst_phys
                            ))
                            if prediction_error > velocity_tolerance_um * gap:
                                continue

                    scored_candidates.append((dist, src_det.node_id, dst_det.node_id))

                # Keep top-k by distance
                scored_candidates.sort(key=lambda x: x[0])
                for dist, src_id, dst_id in scored_candidates[:k_max]:
                    all_edges.append((src_id, dst_id))

    if len(all_edges) == 0:
        return np.zeros((2, 0), dtype=np.int64), []

    edge_array = np.array(all_edges, dtype=np.int64).T  # (2, E)
    return edge_array, all_edges


def compute_gt_edge_coverage(
    candidate_edges: List[Tuple[int, int]],
    gt_edges: List[Tuple[int, int]],
) -> Dict[str, float]:
    """
    Check how many ground truth edges are covered by the candidate graph.
    
    This validates that graph construction isn't too aggressive — 
    if GT edges are missing from candidates, the GNN can never recover them.

    Args:
        candidate_edges: List of (src, dst) candidate edges.
        gt_edges: List of (src, dst) ground truth edges.

    Returns:
        Dict with coverage, n_covered, n_missing.
    """
    candidate_set = set(candidate_edges)
    covered = sum(1 for e in gt_edges if e in candidate_set)

    return {
        "coverage": covered / max(len(gt_edges), 1),
        "n_covered": covered,
        "n_missing": len(gt_edges) - covered,
        "n_gt_edges": len(gt_edges),
        "n_candidate_edges": len(candidate_edges),
        "ratio": len(candidate_edges) / max(len(gt_edges), 1),
    }
