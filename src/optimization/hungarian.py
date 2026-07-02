"""
Hungarian assignment + division pass for track construction.

Two-phase approach:
  1. Frame-by-frame Hungarian assignment (1-to-1)
     using -log(confidence) as costs
  2. Division pass: check remaining high-confidence edges
     for 1-to-2 patterns → assign as divisions

Simple and effective. Global optimization (MCF/ILP) deferred
until this plateaus.
"""

import numpy as np
from typing import List, Dict, Tuple, Set
from scipy.optimize import linear_sum_assignment
from collections import defaultdict


def hungarian_tracking(
    edge_predictions: List[Dict],
    n_frames: int,
    min_track_confidence: float = 0.3,
    min_division_confidence: float = 0.5,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int, int]]]:
    """
    Build tracks using frame-by-frame Hungarian assignment.

    Args:
        edge_predictions: List of dicts, each with:
            'src_id': source node ID
            'dst_id': target node ID
            'src_frame': source frame index
            'dst_frame': target frame index
            'confidence': P(correct edge)
            'class_probs': [P(false), P(track), P(division)]
        n_frames: Total number of frames.
        min_track_confidence: Minimum confidence for track edges.
        min_division_confidence: Minimum confidence for division edges.

    Returns:
        Tuple of:
        - track_edges: List of (src_id, dst_id) for confirmed tracks
        - divisions: List of (parent_id, child1_id, child2_id)
    """
    # Group edges by frame pair
    edges_by_pair = defaultdict(list)
    for pred in edge_predictions:
        key = (pred["src_frame"], pred["dst_frame"])
        edges_by_pair[key].append(pred)

    track_edges = []
    matched_targets = set()  # Prevent double-matching
    matched_sources = set()

    # Phase 1: Hungarian assignment for each frame pair
    for (t_src, t_dst) in sorted(edges_by_pair.keys()):
        frame_edges = edges_by_pair[(t_src, t_dst)]

        # Filter by minimum confidence
        viable = [
            e for e in frame_edges
            if e["confidence"] > min_track_confidence
        ]
        if not viable:
            continue

        # Build bipartite graph
        src_ids = sorted(set(e["src_id"] for e in viable))
        dst_ids = sorted(set(e["dst_id"] for e in viable))

        src_idx_map = {sid: i for i, sid in enumerate(src_ids)}
        dst_idx_map = {did: i for i, did in enumerate(dst_ids)}

        # Cost matrix: -log(confidence) so lower = better
        cost_matrix = np.full(
            (len(src_ids), len(dst_ids)), fill_value=1e6, dtype=np.float64
        )

        for e in viable:
            si = src_idx_map[e["src_id"]]
            di = dst_idx_map[e["dst_id"]]
            conf = max(e["confidence"], 1e-8)
            cost_matrix[si, di] = -np.log(conf)

        # Hungarian assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < 1e5:  # Valid assignment
                src_id = src_ids[r]
                dst_id = dst_ids[c]

                # Skip if already matched
                if dst_id in matched_targets:
                    continue

                track_edges.append((src_id, dst_id))
                matched_targets.add(dst_id)
                matched_sources.add(src_id)

    # Phase 2: Division pass
    divisions = _find_divisions(
        edge_predictions, track_edges, matched_targets,
        min_division_confidence,
    )

    # Add division edges to track_edges
    for parent, child1, child2 in divisions:
        track_edges.append((parent, child1))
        track_edges.append((parent, child2))

    return track_edges, divisions


def _find_divisions(
    edge_predictions: List[Dict],
    existing_tracks: List[Tuple[int, int]],
    matched_targets: Set[int],
    min_confidence: float,
) -> List[Tuple[int, int, int]]:
    """
    Find division events: parent → two daughters.

    Look for source nodes that have:
    1. One existing track edge (already matched)
    2. Another high-confidence edge to an unmatched target
    AND both edges have high division class probability.
    """
    divisions = []

    # Build index: source → matched target
    src_to_dst = defaultdict(list)
    for src, dst in existing_tracks:
        src_to_dst[src].append(dst)

    # Group remaining edges by source
    remaining = defaultdict(list)
    for pred in edge_predictions:
        if pred["dst_id"] not in matched_targets:
            if pred["confidence"] > min_confidence:
                remaining[pred["src_id"]].append(pred)

    # Check each source with exactly 1 existing match + 1 remaining candidate
    for src_id, existing_dsts in src_to_dst.items():
        if len(existing_dsts) != 1:
            continue  # Already has 0 or multiple matches

        child1 = existing_dsts[0]

        # Check if source has high P(division)
        src_remaining = remaining.get(src_id, [])
        if not src_remaining:
            continue

        # Find best remaining candidate
        best = max(src_remaining, key=lambda e: e["class_probs"][2])

        if best["class_probs"][2] > 0.3:  # P(division) threshold
            child2 = best["dst_id"]
            divisions.append((src_id, child1, child2))
            matched_targets.add(child2)

    return divisions
