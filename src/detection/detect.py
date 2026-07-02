"""
Unified detection interface — selects LoG or learned detector.

After training the learned detector, load it here and use it as a
drop-in replacement for LoG in the full pipeline.

Usage:
    from src.detection.detect import create_detector, detect_all_frames

    detector = create_detector(method="learned", checkpoint="outputs/detector_best.pt")
    detections_by_frame = detect_all_frames(loader, detector)
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from src.utils.coords import Detection, VOXEL_SPACING_UM
from src.detection.blob_detector import detect_cells_log


def create_detector(
    method: str = "learned",
    checkpoint: str = None,
    device: str = "auto",
    threshold: float = 0.3,
    **kwargs,
):
    """
    Factory function for detectors.
    
    Args:
        method: "learned" or "log".
        checkpoint: Path to trained model checkpoint (for learned).
        device: "cuda", "cpu", or "auto".
        threshold: Peak detection threshold (for learned).
        
    Returns:
        Detector callable.
    """
    if method == "log":
        return LogDetector(**kwargs)
    elif method == "learned":
        if checkpoint is None:
            raise ValueError("checkpoint path required for learned detector")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return LearnedDetector(checkpoint, device, threshold)
    else:
        raise ValueError(f"Unknown method: {method}")


class LogDetector:
    """Wrapper around LoG blob detection."""
    
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    
    def detect_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        voxel_spacing: np.ndarray = VOXEL_SPACING_UM,
    ) -> List[Detection]:
        return detect_cells_log(
            frame, frame_idx=frame_idx,
            voxel_spacing=voxel_spacing,
            **self.kwargs,
        )


class LearnedDetector:
    """Wrapper around trained 3D U-Net heatmap detector."""
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        threshold: float = 0.3,
    ):
        self.device = torch.device(device)
        self.threshold = threshold
        
        # Import and load model
        from src.detection.learned_detector import (
            CellDetectorUNet, extract_peaks_from_heatmap, normalize_volume,
        )
        self._normalize = normalize_volume
        self._extract_peaks = extract_peaks_from_heatmap
        
        # Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        
        # Get config from checkpoint or use defaults
        config = ckpt.get("config", {})
        base_ch = config.get("base_ch", 16)
        self.threshold = config.get("peak_threshold", threshold)
        
        self.model = CellDetectorUNet(base_channels=base_ch).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        
        val_recall = ckpt.get("val_recall", "unknown")
        print(f"Loaded learned detector (val recall: {val_recall})")
    
    @torch.no_grad()
    def detect_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        voxel_spacing: np.ndarray = VOXEL_SPACING_UM,
    ) -> List[Detection]:
        """Detect cells in a single 3D frame."""
        # Normalize
        inp = self._normalize(frame)
        inp_tensor = torch.tensor(inp, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        inp_tensor = inp_tensor.to(self.device)
        
        # Predict heatmap
        if self.device.type == "cuda":
            with torch.amp.autocast("cuda"):
                pred = self.model(inp_tensor)
        else:
            pred = self.model(inp_tensor)
        
        heatmap = pred[0, 0].cpu().numpy()
        
        # Extract peaks
        peaks = self._extract_peaks(
            heatmap, threshold=self.threshold,
            min_distance_voxels=(2, 5, 5),
        )
        
        # Convert to Detection objects
        detections = []
        for z, y, x, conf in peaks:
            # Get local intensity stats from the original frame
            iz, iy, ix = int(round(z)), int(round(y)), int(round(x))
            patch = frame[
                max(0, iz-1):iz+2,
                max(0, iy-2):iy+3,
                max(0, ix-2):ix+3,
            ].astype(np.float32)
            
            detections.append(Detection(
                node_id=-1,  # Will be assigned later
                t=frame_idx,
                z=z, y=y, x=x,
                confidence=conf,
                intensity_mean=float(patch.mean()) if patch.size > 0 else 0,
                intensity_std=float(patch.std()) if patch.size > 0 else 0,
                intensity_max=float(patch.max()) if patch.size > 0 else 0,
                intensity_min=float(patch.min()) if patch.size > 0 else 0,
            ))
        
        # Memory cleanup
        del inp_tensor, pred
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        
        return detections


def detect_all_frames(
    loader,
    detector=None,
    voxel_spacing: np.ndarray = VOXEL_SPACING_UM,
    frames: List[int] = None,
    verbose: bool = True,
) -> Dict[int, List[Detection]]:
    """
    Run detection on all frames (or specified frames) of a volume.
    
    Args:
        loader: ZarrLoader instance.
        detector: Detector instance (LogDetector or LearnedDetector).
                  If None, uses LogDetector.
        voxel_spacing: Voxel spacing in µm.
        frames: Specific frame indices to detect. None = all frames.
        verbose: Print progress.
    
    Returns:
        Dict[frame_idx → List[Detection]] with globally unique node_ids.
    """
    if detector is None:
        detector = LogDetector()
    
    if frames is None:
        frames = list(range(loader.n_frames))
    
    detections_by_frame = {}
    global_id = 0
    
    for i, t in enumerate(frames):
        frame = loader.get_frame(t)
        dets = detector.detect_frame(frame, t, voxel_spacing)
        
        # Assign globally unique IDs
        for d in dets:
            d.node_id = global_id
            global_id += 1
        
        detections_by_frame[t] = dets
        
        if verbose and (i + 1) % 20 == 0:
            print(f"  Frame {t}: {len(dets)} detections ({i+1}/{len(frames)})")
    
    total = sum(len(d) for d in detections_by_frame.values())
    if verbose:
        print(f"  Total: {total} detections across {len(frames)} frames")
    
    return detections_by_frame
