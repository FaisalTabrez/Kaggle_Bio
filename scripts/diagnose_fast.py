"""
Fast Pipeline Diagnostic -- only processes GT-annotated frames.

Since GT is sparse (~52 nodes across ~52 frames out of 100),
we only need to detect in frames that HAVE annotations.
This cuts detection time by ~50% and is what actually matters.

Usage:
    python scripts/diagnose_fast.py
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


def detect_gt_frames_only(loader, gt_graph, det_cfg, voxel_spacing):
    """Only detect in frames that have GT annotations + their neighbors."""
    gt_frames = set()
    for n in gt_graph.nodes:
        gt_frames.add(n.t)
        # Also detect in t-1 and t+1 for edge construction
        if n.t > 0:
            gt_frames.add(n.t - 1)
        if n.t < loader.n_frames - 1:
            gt_frames.add(n.t + 1)

    all_dets_by_frame = {}
    global_id = 0

    for t in sorted(gt_frames):
        frame = loader.get_frame(t)
        dets = detect_cells_log(
            frame, frame_idx=t, voxel_spacing=voxel_spacing,
            sigma_um_range=det_cfg.get("sigma_um_range", (2.0, 3.0, 4.0, 5.0)),
            threshold_rel=det_cfg.get("threshold_rel", 0.10),
            min_distance_um=det_cfg.get("min_distance_um", 5.0),
        )
        for d in dets:
            d.node_id = global_id
            global_id += 1
        all_dets_by_frame[t] = dets

    return all_dets_by_frame


def run_fast_diagnostic(data_dir, n_samples=3):
    """Fast diagnostic: only GT frames, fewer configs."""
    samples = discover_samples(data_dir, split="train")
    print(f"Found {len(samples)} training samples")

    # Detector configs
    det_configs = [
        {"name": "default",        "sigma_um_range": (2.0, 3.0, 4.0, 5.0),
         "threshold_rel": 0.10, "min_distance_um": 5.0},
        {"name": "permissive",     "sigma_um_range": (1.5, 2.0, 3.0, 4.0, 5.0, 6.0),
         "threshold_rel": 0.05, "min_distance_um": 4.0},
        {"name": "very_permissive","sigma_um_range": (1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0),
         "threshold_rel": 0.02, "min_distance_um": 3.5},
    ]

    # Graph configs
    graph_configs = [
        {"name": "bio_k6_15",  "k_max": 6,  "max_displacement_um": 15.0,
         "use_biology_filters": True,  "max_frame_gap": 1},
        {"name": "knn_k8_20",  "k_max": 8,  "max_displacement_um": 20.0,
         "use_biology_filters": False, "max_frame_gap": 1},
        {"name": "knn_k12_25", "k_max": 12, "max_displacement_um": 25.0,
         "use_biology_filters": False, "max_frame_gap": 1},
    ]

    all_results = []

    for i, info in enumerate(samples[:n_samples]):
        if info["geff_path"] is None:
            continue
        name = info["name"]

        print(f"\n{'='*70}")
        print(f"  [{i+1}/{n_samples}] {name}")
        print(f"{'='*70}")

        loader = ZarrLoader(info["zarr_path"])
        gt = GeffLoader(info["geff_path"]).load()
        gs = gt.summary()
        print(f"  Volume: {loader.shape}")
        print(f"  GT: {gs['n_nodes']} nodes, {gs['n_edges']} edges, "
              f"{gs['n_divisions']} divisions, frames {gs['frame_range']}")

        if gs["n_nodes"] == 0:
            print("  -- No GT, skipping")
            continue

        gt_frames = sorted(set(n.t for n in gt.nodes))
        n_gt_frames = len(gt_frames)
        print(f"  GT frames: {n_gt_frames} (will detect ~{n_gt_frames * 2} frames incl. neighbors)")

        sample_res = {
            "name": name,
            "gt_nodes": gs["n_nodes"],
            "gt_edges": gs["n_edges"],
            "gt_divs": gs["n_divisions"],
        }

        # === DETECTION SWEEP ===
        print(f"\n  --- Detection ---")
        best_dets = None
        best_recall = 0
        best_det_name = ""

        for dcfg in det_configs:
            t0 = time.time()
            dets_by_frame = detect_gt_frames_only(loader, gt, dcfg, loader.voxel_spacing)
            n_dets = sum(len(v) for v in dets_by_frame.values())
            all_dets = [d for ds in dets_by_frame.values() for d in ds]

            recall_res = measure_detection_recall(
                all_dets, gt.nodes, voxel_spacing=loader.voxel_spacing
            )
            elapsed = time.time() - t0

            r = recall_res["recall"]
            tp = recall_res["tp"]
            fn = recall_res["fn"]
            avg_d = recall_res["avg_match_distance_um"]

            print(f"    {dcfg['name']:<20} Recall: {r:5.1%} ({tp}/{tp+fn})  "
                  f"Dets: {n_dets:5d}  AvgDist: {avg_d:.2f}um  {elapsed:.0f}s")

            sample_res[f"det_{dcfg['name']}"] = {
                "recall": r, "n_dets": n_dets, "avg_dist": avg_d
            }

            if r > best_recall:
                best_recall = r
                best_dets = dets_by_frame
                best_det_name = dcfg["name"]

        print(f"    >> Best: {best_det_name} ({best_recall:.1%})")

        # === EDGE RECALL SWEEP ===
        print(f"\n  --- Candidate Edge Recall (using '{best_det_name}') ---")
        all_dets = [d for ds in best_dets.values() for d in ds]

        # Match dets to GT
        det_to_gt = {}
        gt_to_det = {}
        for t in set(d.t for d in all_dets):
            fd = [d for d in all_dets if d.t == t]
            fg = [n for n in gt.nodes if n.t == t]
            if not fd or not fg:
                continue
            dc = np.array([[d.z, d.y, d.x] for d in fd])
            gc = np.array([[n.z, n.y, n.x] for n in fg])
            dm = physical_distance_batch(dc, gc, is_voxel=True, spacing=loader.voxel_spacing)
            ri, ci = linear_sum_assignment(dm)
            for r, c in zip(ri, ci):
                if dm[r, c] <= 7.0:
                    det_to_gt[fd[r].node_id] = fg[c].node_id
                    gt_to_det[fg[c].node_id] = fd[r].node_id

        gt_edge_set = {(e.source_id, e.target_id) for e in gt.edges}

        for gcfg in graph_configs:
            t0 = time.time()
            _, edge_list = build_graph(
                best_dets, voxel_spacing=loader.voxel_spacing,
                max_frame_gap=gcfg["max_frame_gap"],
                k_max=gcfg["k_max"],
                max_displacement_um=gcfg["max_displacement_um"],
                use_biology_filters=gcfg["use_biology_filters"],
            )

            # Map to GT space
            cand_gt = set()
            for s, d in edge_list:
                gs_ = det_to_gt.get(s)
                gd_ = det_to_gt.get(d)
                if gs_ is not None and gd_ is not None:
                    cand_gt.add((gs_, gd_))

            covered = len(cand_gt & gt_edge_set)
            missing = len(gt_edge_set) - covered
            er = covered / max(len(gt_edge_set), 1)

            # Why missing?
            reasons = defaultdict(int)
            for gs_, gd_ in gt_edge_set:
                if (gs_, gd_) not in cand_gt:
                    s_ok = gs_ in gt_to_det
                    d_ok = gd_ in gt_to_det
                    if not s_ok and not d_ok:
                        reasons["both_undetected"] += 1
                    elif not s_ok:
                        reasons["src_undetected"] += 1
                    elif not d_ok:
                        reasons["dst_undetected"] += 1
                    else:
                        reasons["filtered"] += 1

            # Avg degree
            out_deg = defaultdict(int)
            for s, d in edge_list:
                out_deg[s] += 1
            avg_deg = np.mean(list(out_deg.values())) if out_deg else 0

            elapsed = time.time() - t0
            print(f"    {gcfg['name']:<15} EdgeRecall: {er:5.1%} ({covered}/{len(gt_edge_set)})  "
                  f"Cands: {len(edge_list):5d}  Deg: {avg_deg:.1f}  "
                  f"Missing: {dict(reasons)}  {elapsed:.0f}s")

            sample_res[f"graph_{gcfg['name']}"] = {
                "edge_recall": er, "n_cands": len(edge_list),
                "avg_deg": avg_deg, "reasons": dict(reasons),
            }

        all_results.append(sample_res)

    # === SUMMARY ===
    if not all_results:
        print("\nNo results!")
        return

    print(f"\n\n{'='*70}")
    print("  AGGREGATE SUMMARY")
    print(f"{'='*70}")

    print("\n  Detection Recall:")
    for dcfg in det_configs:
        key = f"det_{dcfg['name']}"
        vals = [r[key]["recall"] for r in all_results if key in r]
        if vals:
            print(f"    {dcfg['name']:<20} Mean: {np.mean(vals):5.1%}  "
                  f"Min: {np.min(vals):5.1%}  Max: {np.max(vals):5.1%}")

    print("\n  Candidate Edge Recall:")
    for gcfg in graph_configs:
        key = f"graph_{gcfg['name']}"
        vals = [r[key]["edge_recall"] for r in all_results if key in r]
        if vals:
            print(f"    {gcfg['name']:<15} Mean: {np.mean(vals):5.1%}  "
                  f"Min: {np.min(vals):5.1%}  Max: {np.max(vals):5.1%}")

    # Decision
    best_det = max(det_configs,
                   key=lambda c: np.mean([r.get(f"det_{c['name']}", {}).get("recall", 0)
                                          for r in all_results]))
    best_r = np.mean([r.get(f"det_{best_det['name']}", {}).get("recall", 0) for r in all_results])

    print(f"\n  DECISION:")
    if best_r >= 0.90:
        print(f"    [PASS] Detection recall {best_r:.1%} >= 90% --> LoG sufficient")
    elif best_r >= 0.85:
        print(f"    [WARN] Detection recall {best_r:.1%} -- marginal, may need tuning")
    else:
        print(f"    [FAIL] Detection recall {best_r:.1%} < 85% --> need learned detector")

    # Save
    out = str(PROJECT_ROOT / "outputs" / "diagnostics.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Saved: {out}")


if __name__ == "__main__":
    run_fast_diagnostic(
        r"D:\kaggle_biohub\biohub-cell-tracking-during-development",
        n_samples=3,
    )
