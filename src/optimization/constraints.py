"""
Biological constraint enforcement for track post-processing.

Filters:
  1. Maximum displacement between linked cells
  2. Minimum track length
  3. Division validation (daughters near parent)
  4. No merging (max in-degree = 1)
  5. Temporal monotonicity (edges go forward only)
"""

import numpy as np
from typing import List, Dict, Tuple, Set
from collections import defaultdict

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.coords import Detection, physical_distance, VOXEL_SPACING_UM


def enforce_constraints(
    track_edges: List[Tuple[int, int]],
    divisions: List[Tuple[int, int, int]],
    detections: Dict[int, Detection],
    voxel_spacing: np.ndarray = VOXEL_SPACING_UM,
    max_displacement_um: float = 15.0,
    min_track_length: int = 3,
    max_daughter_dist_um: float = 10.0,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int, int]]]:
    """
    Apply biological constraints to filtered tracks.

    Args:
        track_edges: List of (src_id, dst_id) edges.
        divisions: List of (parent, child1, child2) tuples.
        detections: Dict[node_id → Detection].
        voxel_spacing: Voxel spacing in µm.
        max_displacement_um: Max allowed displacement per edge.
        min_track_length: Min number of edges in a track.
        max_daughter_dist_um: Max distance from parent to daughter.

    Returns:
        Filtered (track_edges, divisions).
    """
    # 1. Filter by displacement
    valid_edges = []
    for src_id, dst_id in track_edges:
        src = detections.get(src_id)
        dst = detections.get(dst_id)
        if src is None or dst is None:
            continue

        dist = physical_distance(
            np.array([src.z, src.y, src.x]),
            np.array([dst.z, dst.y, dst.x]),
            is_voxel=True, spacing=voxel_spacing,
        )
        frame_gap = abs(dst.t - src.t)
        if dist <= max_displacement_um * frame_gap:
            valid_edges.append((src_id, dst_id))

    # 2. Resolve merges (max in-degree = 1)
    valid_edges = _resolve_merges(valid_edges, detections, voxel_spacing)

    # 3. Enforce temporal monotonicity
    valid_edges = [
        (s, d) for s, d in valid_edges
        if detections.get(s) and detections.get(d)
        and detections[s].t < detections[d].t
    ]

    # 4. Filter short tracks
    valid_edges = _filter_short_tracks(valid_edges, min_track_length)

    # 5. Validate divisions
    valid_divisions = []
    division_edges = set()
    for parent, child1, child2 in divisions:
        p = detections.get(parent)
        c1 = detections.get(child1)
        c2 = detections.get(child2)

        if p is None or c1 is None or c2 is None:
            continue

        d1 = physical_distance(
            np.array([p.z, p.y, p.x]),
            np.array([c1.z, c1.y, c1.x]),
            is_voxel=True, spacing=voxel_spacing,
        )
        d2 = physical_distance(
            np.array([p.z, p.y, p.x]),
            np.array([c2.z, c2.y, c2.x]),
            is_voxel=True, spacing=voxel_spacing,
        )

        if d1 <= max_daughter_dist_um and d2 <= max_daughter_dist_um:
            valid_divisions.append((parent, child1, child2))
            division_edges.add((parent, child1))
            division_edges.add((parent, child2))

    # Remove any track edges that conflict with division edges
    valid_edges = [
        e for e in valid_edges if e not in division_edges
    ]
    # Re-add division edges
    for parent, child1, child2 in valid_divisions:
        valid_edges.append((parent, child1))
        valid_edges.append((parent, child2))

    return valid_edges, valid_divisions


def _resolve_merges(
    edges: List[Tuple[int, int]],
    detections: Dict[int, Detection],
    voxel_spacing: np.ndarray,
) -> List[Tuple[int, int]]:
    """If a node has in-degree > 1, keep only the closest incoming edge."""
    incoming = defaultdict(list)
    for src, dst in edges:
        incoming[dst].append(src)

    resolved = []
    for dst_id, src_ids in incoming.items():
        if len(src_ids) <= 1:
            resolved.append((src_ids[0], dst_id))
        else:
            # Keep the closest source
            dst = detections.get(dst_id)
            if dst is None:
                continue

            best_src = None
            best_dist = float("inf")
            for src_id in src_ids:
                src = detections.get(src_id)
                if src is None:
                    continue
                dist = physical_distance(
                    np.array([src.z, src.y, src.x]),
                    np.array([dst.z, dst.y, dst.x]),
                    is_voxel=True, spacing=voxel_spacing,
                )
                if dist < best_dist:
                    best_dist = dist
                    best_src = src_id

            if best_src is not None:
                resolved.append((best_src, dst_id))

    return resolved


def _filter_short_tracks(
    edges: List[Tuple[int, int]],
    min_length: int,
) -> List[Tuple[int, int]]:
    """Remove tracks (connected components) shorter than min_length edges."""
    if min_length <= 1:
        return edges

    # Build adjacency
    forward = defaultdict(list)
    backward = defaultdict(set)
    all_nodes = set()

    for src, dst in edges:
        forward[src].append(dst)
        backward[dst].add(src)
        all_nodes.add(src)
        all_nodes.add(dst)

    # Find track starts (no incoming edges)
    starts = [n for n in all_nodes if n not in backward]

    # Trace each track and measure length
    keep_edges = set()
    for start in starts:
        track = []
        current = start
        while current in forward:
            nexts = forward[current]
            if len(nexts) == 1:
                track.append((current, nexts[0]))
                current = nexts[0]
            elif len(nexts) >= 2:
                # Division: each branch is a separate track continuation
                for nxt in nexts:
                    track.append((current, nxt))
                break
            else:
                break

        if len(track) >= min_length:
            keep_edges.update(track)

    return [e for e in edges if e in keep_edges]
