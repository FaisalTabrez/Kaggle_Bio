"""
Detection recall diagnostic — measures how well the detector
finds ground truth cell centroids.

This is the FIRST thing to run before building the graph/GNN.
If recall < 85%, invest in a learned detector. If ≥ 85%, proceed.
"""

import numpy as np
from typing import List, Dict, Tuple
from scipy.optimize import linear_sum_assignment

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.coords import Detection, physical_distance_batch, VOXEL_SPACING_UM, MATCHING_TOLERANCE_UM
from src.data.geff_loader import GeffNode


def measure_detection_recall(
    detections: List[Detection],
    gt_nodes: List[GeffNode],
    tolerance_um: float = MATCHING_TOLERANCE_UM,
    voxel_spacing: np.ndarray = VOXEL_SPACING_UM,
) -> Dict[str, float]:
    """
    Measure detection recall/precision against sparse GT annotations.

    Uses bipartite matching (Hungarian) with physical distance ≤ tolerance_um
    to match predicted detections to ground truth nodes, frame by frame.

    Args:
        detections: All detections across all frames.
        gt_nodes: Ground truth nodes from GEFF.
        tolerance_um: Maximum matching distance in µm (competition: 7.0).
        voxel_spacing: (Z, Y, X) voxel spacing in µm.

    Returns:
        Dict with recall, precision, f1, matched_count, 
        avg_distance, per_frame stats.
    """
    # Group by frame
    det_by_frame = {}
    for d in detections:
        det_by_frame.setdefault(d.t, []).append(d)

    gt_by_frame = {}
    for n in gt_nodes:
        gt_by_frame.setdefault(n.t, []).append(n)

    total_tp = 0
    total_fp = 0
    total_fn = 0
    matched_distances = []
    per_frame = {}

    # Only evaluate frames that have GT annotations
    all_frames = sorted(set(gt_by_frame.keys()))

    for t in all_frames:
        gt_list = gt_by_frame.get(t, [])
        det_list = det_by_frame.get(t, [])

        if len(gt_list) == 0:
            # No GT in this frame — skip (sparse annotations)
            continue

        if len(det_list) == 0:
            # Missed everything
            total_fn += len(gt_list)
            per_frame[t] = {"tp": 0, "fp": 0, "fn": len(gt_list), "recall": 0.0}
            continue

        # Build cost matrix: physical distance
        det_coords = np.array([[d.z, d.y, d.x] for d in det_list])
        gt_coords = np.array([[n.z, n.y, n.x] for n in gt_list])

        dist_matrix = physical_distance_batch(
            det_coords, gt_coords, is_voxel=True, spacing=voxel_spacing
        )  # (N_det, N_gt)

        # Hungarian assignment
        row_ind, col_ind = linear_sum_assignment(dist_matrix)

        # Count matches within tolerance
        frame_tp = 0
        for r, c in zip(row_ind, col_ind):
            if dist_matrix[r, c] <= tolerance_um:
                frame_tp += 1
                matched_distances.append(dist_matrix[r, c])

        frame_fn = len(gt_list) - frame_tp
        frame_fp = len(det_list) - frame_tp

        total_tp += frame_tp
        total_fn += frame_fn
        total_fp += frame_fp

        frame_recall = frame_tp / max(len(gt_list), 1)
        per_frame[t] = {
            "tp": frame_tp,
            "fp": frame_fp,
            "fn": frame_fn,
            "recall": frame_recall,
            "n_det": len(det_list),
            "n_gt": len(gt_list),
        }

    # Aggregate metrics
    recall = total_tp / max(total_tp + total_fn, 1)
    precision = total_tp / max(total_tp + total_fp, 1)
    f1 = 2 * recall * precision / max(recall + precision, 1e-8)

    return {
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "avg_match_distance_um": float(np.mean(matched_distances)) if matched_distances else 0.0,
        "max_match_distance_um": float(np.max(matched_distances)) if matched_distances else 0.0,
        "n_gt_nodes": total_tp + total_fn,
        "n_detections": total_tp + total_fp,
        "n_evaluated_frames": len(all_frames),
        "per_frame": per_frame,
    }


def print_recall_report(
    results: Dict,
    sample_name: str = "",
) -> None:
    """Print a formatted detection recall report."""
    print(f"\n{'=' * 60}")
    print(f"  DETECTION RECALL REPORT  {sample_name}")
    print(f"{'=' * 60}")
    print(f"  Recall:    {results['recall']:.1%}  ({results['tp']}/{results['tp'] + results['fn']} GT nodes matched)")
    print(f"  Precision: {results['precision']:.1%}  ({results['tp']}/{results['tp'] + results['fp']} detections valid)")
    print(f"  F1 Score:  {results['f1']:.3f}")
    print(f"  Avg Match Distance: {results['avg_match_distance_um']:.2f} µm")
    print(f"  Max Match Distance: {results['max_match_distance_um']:.2f} µm")
    print(f"  Evaluated Frames:   {results['n_evaluated_frames']}")
    print(f"{'=' * 60}")

    # Decision gate
    if results['recall'] >= 0.90:
        print("  [PASS] PROCEED: Recall >= 90% -- LoG detector is sufficient.")
    elif results['recall'] >= 0.85:
        print("  [WARN] MARGINAL: Recall 85-90% -- LoG may suffice, monitor closely.")
    else:
        print("  [FAIL] INSUFFICIENT: Recall < 85% -- consider a learned detector.")
    print()
