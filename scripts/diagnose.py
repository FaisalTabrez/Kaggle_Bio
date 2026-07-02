"""
Pipeline Diagnostics v2 -- Run BEFORE any training.

Measures:
  1. Detection Recall across multiple LoG parameter settings
  2. Candidate Edge Recall across multiple graph configs

Usage:
    python scripts/diagnose.py
"""

import os
import sys
import time
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.zarr_loader import ZarrLoader, discover_samples
from src.data.geff_loader import GeffLoader
from src.detection.blob_detector import detect_cells_log
from src.detection.recall_check import measure_detection_recall
from src.graph.graph_builder import build_graph
from src.utils.coords import Detection, VOXEL_SPACING_UM, physical_distance_batch

from scipy.optimize import linear_sum_assignment


# ============================================================
# Detector configs to benchmark
# ============================================================
DETECTOR_CONFIGS = [
    {
        "name": "default",
        "sigma_um_range": (2.0, 3.0, 4.0, 5.0),
        "threshold_rel": 0.10,
        "min_distance_um": 5.0,
    },
    {
        "name": "permissive",
        "sigma_um_range": (1.5, 2.0, 3.0, 4.0, 5.0, 6.0),
        "threshold_rel": 0.05,
        "min_distance_um": 4.0,
    },
    {
        "name": "very_permissive",
        "sigma_um_range": (1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0),
        "threshold_rel": 0.02,
        "min_distance_um": 3.5,
    },
    {
        "name": "aggressive",
        "sigma_um_range": (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0),
        "threshold_rel": 0.01,
        "min_distance_um": 3.0,
    },
]

GRAPH_CONFIGS = [
    {"name": "bio_k6_15", "k_max": 6, "max_displacement_um": 15.0,
     "use_biology_filters": True, "max_frame_gap": 1},
    {"name": "knn_k8_20", "k_max": 8, "max_displacement_um": 20.0,
     "use_biology_filters": False, "max_frame_gap": 1},
    {"name": "knn_k12_25", "k_max": 12, "max_displacement_um": 25.0,
     "use_biology_filters": False, "max_frame_gap": 1},
]


def detect_single_volume(loader, det_cfg, voxel_spacing):
    """Run detection on all frames with given config."""
    all_dets_by_frame = {}
    global_id = 0
    for t in range(loader.n_frames):
        frame = loader.get_frame(t)
        dets = detect_cells_log(
            frame, frame_idx=t, voxel_spacing=voxel_spacing,
            sigma_um_range=det_cfg["sigma_um_range"],
            threshold_rel=det_cfg["threshold_rel"],
            min_distance_um=det_cfg["min_distance_um"],
        )
        for d in dets:
            d.node_id = global_id
            global_id += 1
        all_dets_by_frame[t] = dets
    return all_dets_by_frame


def match_dets_to_gt(all_dets, gt_nodes, voxel_spacing):
    """Bipartite match detections to GT nodes. Returns mappings."""
    det_to_gt = {}
    gt_to_det = {}
    for t in set(d.t for d in all_dets):
        fd = [d for d in all_dets if d.t == t]
        fg = [n for n in gt_nodes if n.t == t]
        if not fd or not fg:
            continue
        dc = np.array([[d.z, d.y, d.x] for d in fd])
        gc = np.array([[n.z, n.y, n.x] for n in fg])
        dm = physical_distance_batch(dc, gc, is_voxel=True, spacing=voxel_spacing)
        ri, ci = linear_sum_assignment(dm)
        for r, c in zip(ri, ci):
            if dm[r, c] <= 7.0:
                det_to_gt[fd[r].node_id] = fg[c].node_id
                gt_to_det[fg[c].node_id] = fd[r].node_id
    return det_to_gt, gt_to_det


def measure_edge_recall(dets_by_frame, gt, det_to_gt, gt_to_det, voxel_spacing, graph_cfg):
    """Measure candidate edge recall for a specific graph config."""
    edge_index, edge_list = build_graph(
        dets_by_frame, voxel_spacing=voxel_spacing,
        max_frame_gap=graph_cfg["max_frame_gap"],
        k_max=graph_cfg["k_max"],
        max_displacement_um=graph_cfg["max_displacement_um"],
        use_biology_filters=graph_cfg["use_biology_filters"],
    )
    # Map candidate edges to GT space
    cand_gt = set()
    for s, d in edge_list:
        gs = det_to_gt.get(s)
        gd = det_to_gt.get(d)
        if gs is not None and gd is not None:
            cand_gt.add((gs, gd))

    gt_edges = {(e.source_id, e.target_id) for e in gt.edges}
    covered = len(cand_gt & gt_edges)
    missing = len(gt_edges) - covered
    
    # Diagnose why edges are missing
    reasons = defaultdict(int)
    for gs, gd in gt_edges:
        if (gs, gd) not in cand_gt:
            src_ok = gs in gt_to_det
            dst_ok = gd in gt_to_det
            if not src_ok and not dst_ok:
                reasons["both_undetected"] += 1
            elif not src_ok:
                reasons["src_undetected"] += 1
            elif not dst_ok:
                reasons["dst_undetected"] += 1
            else:
                reasons["filtered_out"] += 1

    # Avg out-degree
    out_deg = defaultdict(int)
    for s, d in edge_list:
        out_deg[s] += 1
    avg_deg = np.mean(list(out_deg.values())) if out_deg else 0

    return {
        "recall": covered / max(len(gt_edges), 1),
        "covered": covered,
        "missing": missing,
        "n_gt": len(gt_edges),
        "n_cand": len(edge_list),
        "avg_deg": avg_deg,
        "reasons": dict(reasons),
    }


def run_diagnostics(data_dir, n_samples=5):
    """Main diagnostic runner."""
    samples = discover_samples(data_dir, split="train")
    print(f"Found {len(samples)} training samples")
    print(f"Running diagnostics on {min(n_samples, len(samples))} samples\n")

    all_results = []

    for i, info in enumerate(samples[:n_samples]):
        if info["geff_path"] is None:
            continue

        name = info["name"]
        print(f"{'='*70}")
        print(f"  Sample {i+1}/{n_samples}: {name}")
        print(f"{'='*70}")

        loader = ZarrLoader(info["zarr_path"])
        gt = GeffLoader(info["geff_path"]).load()
        gs = gt.summary()
        print(f"  Volume: {loader.shape}, GT: {gs['n_nodes']} nodes, {gs['n_edges']} edges, {gs['n_divisions']} divs")

        if gs["n_nodes"] == 0:
            print("  -- No GT nodes, skipping")
            continue

        sample_result = {"name": name, "gt_nodes": gs["n_nodes"], "gt_edges": gs["n_edges"]}

        # ============ DETECTION SWEEP ============
        print(f"\n  --- Detection Sweep ---")
        best_det_cfg = None
        best_recall = 0
        best_dets = None

        for det_cfg in DETECTOR_CONFIGS:
            t0 = time.time()
            dets_by_frame = detect_single_volume(loader, det_cfg, loader.voxel_spacing)
            n_dets = sum(len(v) for v in dets_by_frame.values())
            all_dets = [d for ds in dets_by_frame.values() for d in ds]
            recall_res = measure_detection_recall(all_dets, gt.nodes, voxel_spacing=loader.voxel_spacing)
            elapsed = time.time() - t0

            print(f"    {det_cfg['name']:<20} | Recall: {recall_res['recall']:5.1%} "
                  f"({recall_res['tp']}/{recall_res['tp']+recall_res['fn']}) | "
                  f"Dets: {n_dets:6d} | "
                  f"Avg dist: {recall_res['avg_match_distance_um']:.2f}um | "
                  f"{elapsed:.0f}s")

            sample_result[f"det_{det_cfg['name']}_recall"] = recall_res["recall"]
            sample_result[f"det_{det_cfg['name']}_n_dets"] = n_dets

            if recall_res["recall"] > best_recall:
                best_recall = recall_res["recall"]
                best_det_cfg = det_cfg
                best_dets = dets_by_frame

        print(f"    >> Best: {best_det_cfg['name']} ({best_recall:.1%})")
        sample_result["best_det"] = best_det_cfg["name"]
        sample_result["best_det_recall"] = best_recall

        # ============ GRAPH SWEEP (using best detector) ============
        print(f"\n  --- Graph Sweep (using '{best_det_cfg['name']}' detector) ---")
        all_dets = [d for ds in best_dets.values() for d in ds]
        det_to_gt, gt_to_det = match_dets_to_gt(all_dets, gt.nodes, loader.voxel_spacing)

        for graph_cfg in GRAPH_CONFIGS:
            t0 = time.time()
            er = measure_edge_recall(best_dets, gt, det_to_gt, gt_to_det,
                                     loader.voxel_spacing, graph_cfg)
            elapsed = time.time() - t0

            print(f"    {graph_cfg['name']:<20} | Edge Recall: {er['recall']:5.1%} "
                  f"({er['covered']}/{er['n_gt']}) | "
                  f"Cands: {er['n_cand']:6d} | "
                  f"Avg deg: {er['avg_deg']:.1f} | "
                  f"Missing reasons: {er['reasons']} | {elapsed:.0f}s")

            sample_result[f"graph_{graph_cfg['name']}_recall"] = er["recall"]
            sample_result[f"graph_{graph_cfg['name']}_n_cand"] = er["n_cand"]
            sample_result[f"graph_{graph_cfg['name']}_reasons"] = er["reasons"]

        all_results.append(sample_result)
        print()

    # ============ SUMMARY ============
    if not all_results:
        print("No results collected!")
        return

    print(f"\n{'='*80}")
    print("  SUMMARY")
    print(f"{'='*80}")

    # Detection summary
    print("\n  Detection Recall by Config:")
    for det_cfg in DETECTOR_CONFIGS:
        key = f"det_{det_cfg['name']}_recall"
        vals = [r[key] for r in all_results if key in r]
        if vals:
            print(f"    {det_cfg['name']:<20} | Mean: {np.mean(vals):5.1%} | "
                  f"Min: {np.min(vals):5.1%} | Max: {np.max(vals):5.1%}")

    # Graph summary
    print("\n  Candidate Edge Recall by Config:")
    for graph_cfg in GRAPH_CONFIGS:
        key = f"graph_{graph_cfg['name']}_recall"
        vals = [r[key] for r in all_results if key in r]
        if vals:
            print(f"    {graph_cfg['name']:<20} | Mean: {np.mean(vals):5.1%} | "
                  f"Min: {np.min(vals):5.1%} | Max: {np.max(vals):5.1%}")

    # Decisions
    best_det_key = max(
        DETECTOR_CONFIGS,
        key=lambda c: np.mean([r.get(f"det_{c['name']}_recall", 0) for r in all_results])
    )
    best_det_mean = np.mean([r.get(f"det_{best_det_key['name']}_recall", 0) for r in all_results])

    print(f"\n  DECISIONS:")
    if best_det_mean >= 0.90:
        print(f"    [PASS] Best detector '{best_det_key['name']}' recall {best_det_mean:.1%} >= 90%")
        print(f"           --> LoG is SUFFICIENT. No learned detector needed.")
    elif best_det_mean >= 0.85:
        print(f"    [WARN] Best detector '{best_det_key['name']}' recall {best_det_mean:.1%} -- marginal")
    else:
        print(f"    [FAIL] Best detector '{best_det_key['name']}' recall {best_det_mean:.1%} < 85%")
        print(f"           --> Consider a learned detector.")

    # Save
    out_path = str(PROJECT_ROOT / "outputs" / "diagnostics.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    DATA_DIR = r"D:\kaggle_biohub\biohub-cell-tracking-during-development"
    run_diagnostics(DATA_DIR, n_samples=3)
