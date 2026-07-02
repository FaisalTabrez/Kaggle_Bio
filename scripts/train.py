"""
Main training loop for the GNN cell tracker.

Integrates: curriculum learning, hard negative mining, mixed precision,
joint CNN+GNN training, and validation scoring.

Usage:
    python scripts/train.py
"""

import os
import sys
import time
import json
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.zarr_loader import ZarrLoader, discover_samples
from src.data.geff_loader import GeffLoader
from src.detection.blob_detector import detect_all_frames
from src.detection.recall_check import measure_detection_recall, print_recall_report
from src.features.handcrafted import extract_handcrafted_features
from src.graph.graph_builder import build_graph, compute_gt_edge_coverage
from src.graph.edge_features import compute_edge_features
from src.models.edge_mpn import EdgeMPN
from src.models.losses import CombinedLoss
from src.training.curriculum import CurriculumScheduler
from src.training.hard_negatives import HardNegativeSampler
from src.utils.coords import Detection, VOXEL_SPACING_UM


def prepare_sample(
    zarr_path: str,
    geff_path: str,
    voxel_spacing: np.ndarray = VOXEL_SPACING_UM,
    max_frame_gap: int = 1,
    use_biology_filters: bool = True,
) -> Dict:
    """
    Prepare a single sample: detect → features → graph → labels.

    Returns dict with all tensors needed for training.
    """
    # Load data
    loader = ZarrLoader(zarr_path)
    gt = GeffLoader(geff_path).load()

    # Detect cells
    detections_by_frame = detect_all_frames(
        loader, voxel_spacing=voxel_spacing,
    )

    # Check detection recall
    all_dets = [d for dets in detections_by_frame.values() for d in dets]
    recall_result = measure_detection_recall(all_dets, gt.nodes, voxel_spacing=voxel_spacing)

    # Build detection lookup
    det_by_id = {d.node_id: d for d in all_dets}

    # Extract features
    features = extract_handcrafted_features(
        detections_by_frame,
        n_frames_total=loader.n_frames,
        volume_shape=loader.volume_shape,
        voxel_spacing=voxel_spacing,
    )

    # Build graph
    edge_index, edge_list = build_graph(
        detections_by_frame,
        voxel_spacing=voxel_spacing,
        max_frame_gap=max_frame_gap,
        use_biology_filters=use_biology_filters,
    )

    # Compute edge features
    edge_feats = compute_edge_features(
        edge_list, det_by_id, voxel_spacing=voxel_spacing,
    )

    # Match detections to GT and label edges
    # First: match each detection to nearest GT node (within tolerance)
    from src.detection.recall_check import measure_detection_recall
    from scipy.optimize import linear_sum_assignment
    from src.utils.coords import physical_distance_batch

    # Build detection-to-GT matching
    det_to_gt = {}
    gt_to_det = {}

    for t in set(d.t for d in all_dets):
        frame_dets = [d for d in all_dets if d.t == t]
        frame_gt = [n for n in gt.nodes if n.t == t]

        if not frame_dets or not frame_gt:
            continue

        det_coords = np.array([[d.z, d.y, d.x] for d in frame_dets])
        gt_coords = np.array([[n.z, n.y, n.x] for n in frame_gt])

        dist_matrix = physical_distance_batch(
            det_coords, gt_coords, is_voxel=True, spacing=voxel_spacing,
        )

        row_ind, col_ind = linear_sum_assignment(dist_matrix)
        for r, c in zip(row_ind, col_ind):
            if dist_matrix[r, c] <= 7.0:
                det_to_gt[frame_dets[r].node_id] = frame_gt[c].node_id
                gt_to_det[frame_gt[c].node_id] = frame_dets[r].node_id

    # Label edges using GT matching
    gt_edge_set = {(e.source_id, e.target_id) for e in gt.edges}
    gt_division_sources = {
        src for src, edges_list in gt._outgoing_edges.items()
        if len(edges_list) >= 2
    }

    edge_labels = []
    for src_det, dst_det in edge_list:
        gt_src = det_to_gt.get(src_det)
        gt_dst = det_to_gt.get(dst_det)

        if gt_src is not None and gt_dst is not None and (gt_src, gt_dst) in gt_edge_set:
            if gt_src in gt_division_sources:
                edge_labels.append(2)  # Division
            else:
                edge_labels.append(1)  # Track
        else:
            edge_labels.append(0)  # False

    # Build tensors
    node_ids = sorted(features.keys())
    node_features = torch.tensor(
        np.stack([features[nid] for nid in node_ids]),
        dtype=torch.float32,
    )

    # Extract crops for CNN encoder
    crops = []
    for nid in node_ids:
        det = det_by_id[nid]
        crop = loader.get_crop(det.t, (det.z, det.y, det.x), crop_size=(8, 16, 16))
        crops.append(crop.astype(np.float32))
    crops_tensor = torch.tensor(np.stack(crops), dtype=torch.float32).unsqueeze(1)

    # Remap edge indices to be 0-based contiguous
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    remapped_edges = np.array([
        [id_to_idx[src], id_to_idx[dst]]
        for src, dst in edge_list
        if src in id_to_idx and dst in id_to_idx
    ], dtype=np.int64)

    if len(remapped_edges) == 0:
        return None

    return {
        "node_features": node_features,
        "crops": crops_tensor,
        "edge_index": torch.tensor(remapped_edges.T, dtype=torch.long),
        "edge_attr": torch.tensor(edge_feats, dtype=torch.float32),
        "edge_labels": torch.tensor(edge_labels, dtype=torch.long),
        "n_nodes": len(node_ids),
        "n_edges": len(edge_list),
        "recall": recall_result["recall"],
        "dataset_name": loader.dataset_name,
    }


def train_epoch(
    model: EdgeMPN,
    optimizer: torch.optim.Optimizer,
    loss_fn: CombinedLoss,
    samples: List[Dict],
    device: torch.device,
    scaler: torch.amp.GradScaler,
    hard_neg_sampler: HardNegativeSampler,
    epoch: int,
) -> Dict[str, float]:
    """Train for one epoch over all samples."""
    model.train()
    total_loss = 0
    total_correct = 0
    total_edges = 0

    for sample in samples:
        if sample is None:
            continue

        x = sample["node_features"].to(device)
        crops = sample["crops"].to(device)
        edge_index = sample["edge_index"].to(device)
        edge_attr = sample["edge_attr"].to(device)
        labels = sample["edge_labels"].to(device)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):
            output = model(x, edge_attr, edge_index, crops)
            losses = loss_fn(
                output["logits"], output["confidence"],
                labels, edge_index,
            )

        # Hard negative mining: re-weight the loss
        if hard_neg_sampler is not None:
            mask = hard_neg_sampler.sample(labels, output["logits"].detach(), epoch)
            if mask.sum() > 0:
                # Recompute loss on selected edges only
                with torch.amp.autocast("cuda"):
                    masked_losses = loss_fn(
                        output["logits"][mask],
                        output["confidence"][mask],
                        labels[mask],
                        edge_index[:, mask],
                    )
                loss = masked_losses["total"]
            else:
                loss = losses["total"]
        else:
            loss = losses["total"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        preds = output["logits"].argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_edges += len(labels)

    return {
        "loss": total_loss / max(len(samples), 1),
        "accuracy": total_correct / max(total_edges, 1),
    }


@torch.no_grad()
def validate(
    model: EdgeMPN,
    samples: List[Dict],
    loss_fn: CombinedLoss,
    device: torch.device,
) -> Dict[str, float]:
    """Validate on held-out samples."""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_edges = 0
    class_correct = {0: 0, 1: 0, 2: 0}
    class_total = {0: 0, 1: 0, 2: 0}

    for sample in samples:
        if sample is None:
            continue

        x = sample["node_features"].to(device)
        crops = sample["crops"].to(device)
        edge_index = sample["edge_index"].to(device)
        edge_attr = sample["edge_attr"].to(device)
        labels = sample["edge_labels"].to(device)

        with torch.amp.autocast("cuda"):
            output = model(x, edge_attr, edge_index, crops)
            losses = loss_fn(
                output["logits"], output["confidence"],
                labels, edge_index,
            )

        total_loss += losses["total"].item()
        preds = output["logits"].argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_edges += len(labels)

        for c in range(3):
            c_mask = (labels == c)
            class_correct[c] += (preds[c_mask] == c).sum().item()
            class_total[c] += c_mask.sum().item()

    return {
        "loss": total_loss / max(len(samples), 1),
        "accuracy": total_correct / max(total_edges, 1),
        "acc_false": class_correct[0] / max(class_total[0], 1),
        "acc_track": class_correct[1] / max(class_total[1], 1),
        "acc_division": class_correct[2] / max(class_total[2], 1),
    }


def main():
    """Main training entry point."""
    # Configuration
    DATA_DIR = r"D:\kaggle_biohub\biohub-cell-tracking-during-development"
    OUTPUT_DIR = r"c:\Users\FAISAL TABREZ\Documents\Kaggle_Bio\outputs"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Discover samples
    samples_info = discover_samples(DATA_DIR, split="train")
    print(f"Found {len(samples_info)} training samples")

    # Split by embryo prefix for validation
    embryo_44b6 = [s for s in samples_info if s["embryo_prefix"] == "44b6"]
    embryo_6bba = [s for s in samples_info if s["embryo_prefix"] == "6bba"]

    # Use last few 44b6 samples for validation
    n_val = max(5, len(embryo_44b6) // 10)
    val_infos = embryo_44b6[-n_val:]
    train_infos = embryo_44b6[:-n_val] + embryo_6bba

    print(f"Train: {len(train_infos)} samples, Val: {len(val_infos)} samples")

    # Prepare a small subset first (for speed)
    print("\nPreparing training samples (first 10)...")
    train_samples = []
    for info in train_infos[:10]:
        print(f"  Processing {info['name']}...")
        sample = prepare_sample(info["zarr_path"], info["geff_path"])
        if sample:
            print(f"    Nodes: {sample['n_nodes']}, Edges: {sample['n_edges']}, "
                  f"Recall: {sample['recall']:.1%}")
            train_samples.append(sample)

    print("\nPreparing validation samples...")
    val_samples = []
    for info in val_infos[:3]:
        print(f"  Processing {info['name']}...")
        sample = prepare_sample(info["zarr_path"], info["geff_path"])
        if sample:
            val_samples.append(sample)

    if not train_samples:
        print("ERROR: No valid training samples!")
        return

    # Initialize model
    model = EdgeMPN(
        handcrafted_dim=32,
        crop_embed_dim=48,
        edge_input_dim=20,
        hidden_dim=128,
        n_layers=4,
        dropout=0.1,
        use_crops=True,
    ).to(device)

    print(f"\nModel: {model.summary()}")

    # Training components
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)
    loss_fn = CombinedLoss()
    scaler = torch.amp.GradScaler("cuda")
    curriculum = CurriculumScheduler(patience=5, min_epochs_per_phase=8)
    hard_neg_sampler = HardNegativeSampler(hard_ratio=0.7, warmup_epochs=5)

    # Training loop
    best_val_loss = float("inf")
    n_epochs = 50

    print(f"\nStarting training: {n_epochs} epochs")
    print(f"Curriculum: {curriculum.config.description}")

    for epoch in range(n_epochs):
        t0 = time.time()

        # Train
        train_metrics = train_epoch(
            model, optimizer, loss_fn, train_samples,
            device, scaler,
            hard_neg_sampler if curriculum.config.hard_negative_mining else None,
            epoch,
        )

        # Validate
        val_metrics = validate(model, val_samples, loss_fn, device)

        # Learning rate step
        scheduler.step()

        # Curriculum step
        curriculum.step(val_metrics["loss"])

        elapsed = time.time() - t0

        # Print progress
        print(
            f"Epoch {epoch+1:3d}/{n_epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['accuracy']:.3f} | "
            f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.3f} | "
            f"Track: {val_metrics['acc_track']:.3f} Div: {val_metrics['acc_division']:.3f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
            f"{elapsed:.1f}s"
        )

        # Save best model
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "curriculum": curriculum.state_dict(),
            }, os.path.join(OUTPUT_DIR, "best_model.pt"))

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to {OUTPUT_DIR}/best_model.pt")


if __name__ == "__main__":
    main()
