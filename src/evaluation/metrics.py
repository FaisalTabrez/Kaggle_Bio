"""
Local implementation of the competition evaluation metrics.

Score = Adjusted_Edge_Jaccard + 0.1 × Division_Jaccard

- Edge-Jaccard: bipartite node matching (≤7µm), edge TP/FP/FN → Jaccard
- Division-Jaccard: micro-averaged Jaccard over division events
"""

import numpy as np
from typing import List, Dict, Tuple
from scipy.optimize import linear_sum_assignment
from collections import defaultdict

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.coords import physical_distance_batch, VOXEL_SPACING_UM, MATCHING_TOLERANCE_UM


def compute_score(
    pred_nodes: List[Dict],
    pred_edges: List[Tuple[int, int]],
    gt_nodes: List[Dict],
    gt_edges: List[Tuple[int, int]],
    tolerance_um: float = MATCHING_TOLERANCE_UM,
    voxel_spacing: np.ndarray = VOXEL_SPACING_UM,
) -> Dict[str, float]:
    """
    Compute the competition score locally.

    Args:
        pred_nodes: List of dicts with 'id', 't', 'z', 'y', 'x'.
        pred_edges: List of (source_id, target_id).
        gt_nodes: List of dicts with 'id', 't', 'z', 'y', 'x'.
        gt_edges: List of (source_id, target_id).
        tolerance_um: Maximum matching distance (7.0 µm).
        voxel_spacing: Voxel spacing in µm.

    Returns:
        Dict with 'score', 'edge_jaccard', 'division_jaccard', and details.
    """
    # Step 1: Match pred nodes to GT nodes via bipartite assignment
    node_matching = _match_nodes(
        pred_nodes, gt_nodes, tolerance_um, voxel_spacing
    )

    # Step 2: Edge-Jaccard
    edge_result = _compute_edge_jaccard(
        pred_edges, gt_edges, node_matching
    )

    # Step 3: Division-Jaccard
    div_result = _compute_division_jaccard(
        pred_edges, gt_edges, node_matching
    )

    # Combined score
    score = edge_result["jaccard"] + 0.1 * div_result["jaccard"]

    return {
        "score": score,
        "edge_jaccard": edge_result["jaccard"],
        "division_jaccard": div_result["jaccard"],
        "edge_tp": edge_result["tp"],
        "edge_fp": edge_result["fp"],
        "edge_fn": edge_result["fn"],
        "div_tp": div_result["tp"],
        "div_fp": div_result["fp"],
        "div_fn": div_result["fn"],
        "n_matched_nodes": len(node_matching),
        "n_pred_nodes": len(pred_nodes),
        "n_gt_nodes": len(gt_nodes),
    }


def _match_nodes(
    pred_nodes: List[Dict],
    gt_nodes: List[Dict],
    tolerance_um: float,
    voxel_spacing: np.ndarray,
) -> Dict[int, int]:
    """
    Match predicted nodes to GT nodes using bipartite assignment.
    
    Returns: Dict[pred_node_id → gt_node_id] for matched pairs.
    """
    if not pred_nodes or not gt_nodes:
        return {}

    # Group by frame for per-frame matching
    pred_by_frame = defaultdict(list)
    for n in pred_nodes:
        pred_by_frame[n["t"]].append(n)

    gt_by_frame = defaultdict(list)
    for n in gt_nodes:
        gt_by_frame[n["t"]].append(n)

    matching = {}

    for t in gt_by_frame:
        p_list = pred_by_frame.get(t, [])
        g_list = gt_by_frame[t]

        if not p_list or not g_list:
            continue

        p_coords = np.array([[n["z"], n["y"], n["x"]] for n in p_list])
        g_coords = np.array([[n["z"], n["y"], n["x"]] for n in g_list])

        dist_matrix = physical_distance_batch(
            p_coords, g_coords, is_voxel=True, spacing=voxel_spacing
        )

        row_ind, col_ind = linear_sum_assignment(dist_matrix)

        for r, c in zip(row_ind, col_ind):
            if dist_matrix[r, c] <= tolerance_um:
                matching[p_list[r]["id"]] = g_list[c]["id"]

    return matching


def _compute_edge_jaccard(
    pred_edges: List[Tuple[int, int]],
    gt_edges: List[Tuple[int, int]],
    node_matching: Dict[int, int],
) -> Dict[str, float]:
    """Compute Edge-Jaccard using node matching."""
    # Map predicted edges to GT space
    pred_mapped = set()
    for src, dst in pred_edges:
        gt_src = node_matching.get(src)
        gt_dst = node_matching.get(dst)
        if gt_src is not None and gt_dst is not None:
            pred_mapped.add((gt_src, gt_dst))

    gt_edge_set = set(gt_edges)

    tp = len(pred_mapped & gt_edge_set)
    fp = len(pred_mapped - gt_edge_set)
    fn = len(gt_edge_set - pred_mapped)

    jaccard = tp / max(tp + fp + fn, 1)

    return {"jaccard": jaccard, "tp": tp, "fp": fp, "fn": fn}


def _compute_division_jaccard(
    pred_edges: List[Tuple[int, int]],
    gt_edges: List[Tuple[int, int]],
    node_matching: Dict[int, int],
) -> Dict[str, float]:
    """
    Compute Division-Jaccard.
    
    A division = a node with ≥2 outgoing edges.
    """
    # Find pred divisions (in GT space via matching)
    pred_out = defaultdict(set)
    for src, dst in pred_edges:
        gt_src = node_matching.get(src)
        gt_dst = node_matching.get(dst)
        if gt_src is not None and gt_dst is not None:
            pred_out[gt_src].add(gt_dst)

    pred_div_nodes = {
        node for node, targets in pred_out.items() if len(targets) >= 2
    }

    # Find GT divisions
    gt_out = defaultdict(set)
    for src, dst in gt_edges:
        gt_out[src].add(dst)

    gt_div_nodes = {
        node for node, targets in gt_out.items() if len(targets) >= 2
    }

    tp = len(pred_div_nodes & gt_div_nodes)
    fp = len(pred_div_nodes - gt_div_nodes)
    fn = len(gt_div_nodes - pred_div_nodes)

    jaccard = tp / max(tp + fp + fn, 1)

    return {"jaccard": jaccard, "tp": tp, "fp": fp, "fn": fn}
