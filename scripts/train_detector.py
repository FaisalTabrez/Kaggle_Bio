"""
Train the learned cell detector on sparse GT annotations.

Strategy:
  - Each training sample provides ~50 annotated frames with ~1 cell each
  - Create Gaussian heatmap targets at GT positions
  - Train 3D U-Net to regress these heatmaps
  - Use FocalMSE loss for extreme foreground/background imbalance
  - ~199 samples × ~50 frames = ~10K training examples
  - Train on T4 with mixed precision, ~1-2 hours

Usage:
    python scripts/train_detector.py
"""

import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List, Dict, Tuple

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.zarr_loader import ZarrLoader, discover_samples
from src.data.geff_loader import GeffLoader
from src.detection.learned_detector import (
    CellDetectorUNet, make_gaussian_heatmap, FocalMSELoss,
    extract_peaks_from_heatmap, normalize_volume,
)
from src.detection.recall_check import measure_detection_recall
from src.utils.coords import Detection, VOXEL_SPACING_UM


def collect_training_frames(
    samples_info: List[Dict],
    max_samples: int = 50,
) -> List[Dict]:
    """
    Collect (volume_frame, GT_centroids) pairs from training data.
    
    Only loads frames that have GT annotations (memory-efficient).
    
    Returns list of dicts with 'zarr_path', 'frame_idx', 'centroids'.
    """
    all_frames = []

    for info in samples_info[:max_samples]:
        if info["geff_path"] is None:
            continue

        gt = GeffLoader(info["geff_path"]).load()
        if gt.n_nodes == 0:
            continue

        # Group GT centroids by frame
        centroids_by_frame = {}
        for node in gt.nodes:
            centroids_by_frame.setdefault(node.t, []).append(
                (node.z, node.y, node.x)
            )

        for frame_idx, centroids in centroids_by_frame.items():
            all_frames.append({
                "zarr_path": info["zarr_path"],
                "frame_idx": frame_idx,
                "centroids": centroids,
                "dataset_name": info["name"],
            })

    random.shuffle(all_frames)
    print(f"Collected {len(all_frames)} annotated frames from {min(max_samples, len(samples_info))} samples")
    return all_frames


def load_frame_and_target(
    frame_info: Dict,
    volume_shape: Tuple[int, int, int] = (64, 256, 256),
    sigma_voxels: Tuple[float, float, float] = (2.0, 4.0, 4.0),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load a single frame and create its heatmap target.
    
    Returns:
        (input_tensor, target_tensor) each of shape (1, 1, Z, Y, X).
    """
    loader = ZarrLoader(frame_info["zarr_path"])
    frame = loader.get_frame(frame_info["frame_idx"])

    # Normalize input
    input_vol = normalize_volume(frame)
    input_tensor = torch.tensor(input_vol, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # Create Gaussian heatmap target
    heatmap = make_gaussian_heatmap(
        volume_shape=volume_shape,
        centers_zyx=frame_info["centroids"],
        sigma_voxels=sigma_voxels,
    )
    target_tensor = torch.tensor(heatmap, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    return input_tensor, target_tensor


def train_detector(
    data_dir: str,
    output_dir: str = None,
    n_train_samples: int = 40,
    n_val_samples: int = 5,
    n_epochs: int = 30,
    lr: float = 1e-3,
    eval_every: int = 5,
):
    """Train the learned cell detector."""
    if output_dir is None:
        output_dir = str(PROJECT_ROOT / "outputs")
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Discover samples — split by embryo for validation
    all_samples = discover_samples(data_dir, split="train")
    embryo_44b6 = [s for s in all_samples if s["embryo_prefix"] == "44b6"]
    embryo_6bba = [s for s in all_samples if s["embryo_prefix"] == "6bba"]

    # Use 44b6 last few for validation
    val_samples = embryo_44b6[-n_val_samples:]
    train_samples = embryo_44b6[:-n_val_samples][:n_train_samples // 2] + \
                    embryo_6bba[:n_train_samples // 2]

    print(f"Train samples: {len(train_samples)}, Val samples: {len(val_samples)}")

    # Collect frames
    print("\nCollecting training frames...")
    train_frames = collect_training_frames(train_samples, max_samples=len(train_samples))
    val_frames = collect_training_frames(val_samples, max_samples=len(val_samples))
    print(f"Train frames: {len(train_frames)}, Val frames: {len(val_frames)}")

    # Model
    model = CellDetectorUNet(in_channels=1, base_channels=16).to(device)
    print(f"Model params: {model.n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    loss_fn = FocalMSELoss(alpha=2.0, beta=4.0)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    best_val_recall = 0

    for epoch in range(n_epochs):
        t0 = time.time()

        # === TRAIN ===
        model.train()
        epoch_loss = 0
        n_train = 0

        # Shuffle frames each epoch
        random.shuffle(train_frames)

        for fi, frame_info in enumerate(train_frames):
            try:
                inp, tgt = load_frame_and_target(frame_info)
                inp = inp.to(device)
                tgt = tgt.to(device)
            except Exception as e:
                continue

            optimizer.zero_grad()

            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    pred = model(inp)
                    loss = loss_fn(pred, tgt)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(inp)
                loss = loss_fn(pred, tgt)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            epoch_loss += loss.item()
            n_train += 1

            # Memory management
            del inp, tgt, pred, loss
            if device.type == "cuda":
                torch.cuda.empty_cache()

            # Progress every 100 frames
            if (fi + 1) % 100 == 0:
                print(f"    [{fi+1}/{len(train_frames)}] loss: {epoch_loss/n_train:.6f}")

        scheduler.step()
        avg_loss = epoch_loss / max(n_train, 1)
        elapsed = time.time() - t0

        print(f"Epoch {epoch+1:3d}/{n_epochs} | "
              f"Loss: {avg_loss:.6f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
              f"{elapsed:.0f}s ({n_train} frames)")

        # === VALIDATION (recall check) ===
        if (epoch + 1) % eval_every == 0 or epoch == n_epochs - 1:
            print(f"  Evaluating recall on {len(val_frames)} val frames...")
            val_recall = evaluate_detector(model, val_frames, device)
            print(f"  Val Recall: {val_recall:.1%}")

            if val_recall > best_val_recall:
                best_val_recall = val_recall
                save_path = os.path.join(output_dir, "detector_best.pt")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_recall": val_recall,
                }, save_path)
                print(f"  Saved best model ({val_recall:.1%}) -> {save_path}")

    print(f"\nTraining complete. Best val recall: {best_val_recall:.1%}")
    return model


@torch.no_grad()
def evaluate_detector(
    model: CellDetectorUNet,
    val_frames: List[Dict],
    device: torch.device,
    threshold: float = 0.3,
) -> float:
    """Evaluate detector recall on validation frames."""
    model.eval()

    all_detections = []
    all_gt_nodes = []

    # Group val_frames by dataset to load volume once
    from collections import defaultdict
    frames_by_dataset = defaultdict(list)
    for f in val_frames:
        frames_by_dataset[f["zarr_path"]].append(f)

    det_id = 0
    for zarr_path, frames in frames_by_dataset.items():
        for frame_info in frames:
            try:
                inp, _ = load_frame_and_target(frame_info)
                inp = inp.to(device)

                if device.type == "cuda":
                    with torch.amp.autocast("cuda"):
                        pred = model(inp)
                else:
                    pred = model(inp)

                heatmap = pred[0, 0].cpu().numpy()

                # Extract peaks
                peaks = extract_peaks_from_heatmap(
                    heatmap, threshold=threshold,
                    min_distance_voxels=(2, 5, 5),
                )

                for z, y, x, conf in peaks:
                    all_detections.append(Detection(
                        node_id=det_id, t=frame_info["frame_idx"],
                        z=z, y=y, x=x, confidence=conf,
                    ))
                    det_id += 1

                # GT nodes
                from src.data.geff_loader import GeffNode
                for cz, cy, cx in frame_info["centroids"]:
                    all_gt_nodes.append(GeffNode(
                        node_id=0, t=frame_info["frame_idx"],
                        z=int(cz), y=int(cy), x=int(cx),
                    ))

                del inp, pred
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            except Exception as e:
                continue

    if not all_gt_nodes:
        return 0.0

    result = measure_detection_recall(
        all_detections, all_gt_nodes,
        voxel_spacing=VOXEL_SPACING_UM,
    )
    return result["recall"]


if __name__ == "__main__":
    DATA_DIR = r"D:\kaggle_biohub\biohub-cell-tracking-during-development"
    train_detector(
        DATA_DIR,
        n_train_samples=40,  # Use 40 samples (enough to start)
        n_val_samples=5,
        n_epochs=30,
        lr=1e-3,
        eval_every=5,
    )
