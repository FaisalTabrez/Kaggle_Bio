# %% [markdown]
# # BioHub Cell Tracking — Inference & Submission (Colab)
#
# **Run AFTER training the detector and GNN.**
#
# Pipeline:
# 1. Load trained detector → detect cells in test volumes
# 2. Extract features + build graphs
# 3. Run GNN edge classification
# 4. Hungarian assignment + constraints → tracks
# 5. Generate submission CSV
#
# **Prerequisites:**
# - `detector_best.pt` in Drive checkpoints
# - `gnn_best.pt` in Drive checkpoints (or use heuristic baseline)

# %% [markdown]
# ## 1. Setup

# %%
!pip install -q zarr scipy scikit-image numpy pandas torch

# %%
from google.colab import drive
drive.mount('/content/drive')

# %%
import os, sys, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Dict
from scipy.ndimage import maximum_filter
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from dataclasses import dataclass
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Paths
DATA_DIR = "/content/drive/MyDrive/kaggle_biohub/biohub-cell-tracking-during-development"
CKPT_DIR = "/content/drive/MyDrive/kaggle_biohub/checkpoints"

# %% [markdown]
# ## 2. Core Modules (same as training notebook)

# %%
VOXEL_SPACING_UM = np.array([1.625, 0.40625, 0.40625])


@dataclass
class Detection:
    node_id: int
    t: int
    z: float
    y: float
    x: float
    confidence: float = 1.0
    intensity_mean: float = 0.0
    intensity_std: float = 0.0
    intensity_max: float = 0.0
    intensity_min: float = 0.0
    blob_scale: float = 1.0


import zarr

class ZarrLoader:
    def __init__(self, zarr_path):
        self.zarr_path = Path(zarr_path)
        self._store = zarr.open(str(self.zarr_path), mode="r")
        self._array = self._store["0"]
        self._spacing = VOXEL_SPACING_UM
        zarr_json = self.zarr_path / "zarr.json"
        if zarr_json.exists():
            try:
                with open(zarr_json) as f:
                    meta = json.load(f)
                ms = meta.get("attributes", {}).get("multiscales", [])
                if ms:
                    ds = ms[0].get("datasets", [])
                    if ds:
                        for t in ds[0].get("coordinateTransformations", []):
                            if t.get("type") == "scale":
                                self._spacing = np.array(t["scale"][1:])
            except Exception:
                pass

    @property
    def shape(self):      return self._array.shape
    @property
    def n_frames(self):   return self._array.shape[0]
    @property
    def volume_shape(self): return tuple(self._array.shape[1:])
    @property
    def voxel_spacing(self): return self._spacing

    def get_frame(self, t):
        return np.asarray(self._array[t])

    def get_crop(self, t, center_zyx, crop_size=(8, 16, 16)):
        frame = self.get_frame(t)
        Z, Y, X = frame.shape
        dz, dy, dx = crop_size
        cz, cy, cx = int(round(center_zyx[0])), int(round(center_zyx[1])), int(round(center_zyx[2]))
        z0, z1 = max(0, cz-dz//2), min(Z, cz+dz//2+dz%2)
        y0, y1 = max(0, cy-dy//2), min(Y, cy+dy//2+dy%2)
        x0, x1 = max(0, cx-dx//2), min(X, cx+dx//2+dx%2)
        oz0 = z0 - (cz-dz//2); oy0 = y0 - (cy-dy//2); ox0 = x0 - (cx-dx//2)
        crop = np.zeros(crop_size, dtype=frame.dtype)
        crop[oz0:oz0+(z1-z0), oy0:oy0+(y1-y0), ox0:ox0+(x1-x0)] = frame[z0:z1, y0:y1, x0:x1]
        return crop

# %%
# === DETECTOR (same U-Net as training notebook) ===

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.norm = nn.InstanceNorm3d(out_ch)
        self.act = nn.LeakyReLU(0.01, inplace=True)
    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool_size=(2,2,2)):
        super().__init__()
        self.pool = nn.MaxPool3d(pool_size)
        self.conv1 = ConvBlock(in_ch, out_ch)
        self.conv2 = ConvBlock(out_ch, out_ch)
    def forward(self, x):
        return self.conv2(self.conv1(self.pool(x)))

class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, scale=(2,2,2)):
        super().__init__()
        self.scale = scale
        self.conv1 = ConvBlock(in_ch + skip_ch, out_ch)
        self.conv2 = ConvBlock(out_ch, out_ch)
    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=self.scale, mode='trilinear', align_corners=False)
        if x.shape != skip.shape:
            x = F.pad(x, [0, skip.shape[4]-x.shape[4], 0, skip.shape[3]-x.shape[3], 0, skip.shape[2]-x.shape[2]])
        return self.conv2(self.conv1(torch.cat([x, skip], 1)))

class CellDetectorUNet(nn.Module):
    def __init__(self, base_ch=16):
        super().__init__()
        c = base_ch
        self.enc1a = ConvBlock(1, c); self.enc1b = ConvBlock(c, c)
        self.enc2 = DownBlock(c, c*2); self.enc3 = DownBlock(c*2, c*4)
        self.bottleneck = DownBlock(c*4, c*8)
        self.dec3 = UpBlock(c*8, c*4, c*4)
        self.dec2 = UpBlock(c*4, c*2, c*2)
        self.dec1 = UpBlock(c*2, c, c)
        self.head = nn.Conv3d(c, 1, 1)
    def forward(self, x):
        e1 = self.enc1b(self.enc1a(x))
        e2 = self.enc2(e1); e3 = self.enc3(e2); bn = self.bottleneck(e3)
        d3 = self.dec3(bn, e3); d2 = self.dec2(d3, e2); d1 = self.dec1(d2, e1)
        return torch.sigmoid(self.head(d1))

def normalize_volume(vol):
    v = vol.astype(np.float32)
    vmin, vmax = v.min(), v.max()
    return (v - vmin) / (vmax - vmin) if vmax > vmin else v

def extract_peaks(heatmap, threshold=0.3, min_dist=(2, 5, 5)):
    footprint = tuple(2*d+1 for d in min_dist)
    local_max = maximum_filter(heatmap, size=footprint)
    mask = (heatmap == local_max) & (heatmap > threshold)
    coords = np.array(np.nonzero(mask)).T
    if len(coords) == 0: return []
    results = [(float(z), float(y), float(x), float(heatmap[z,y,x])) for z,y,x in coords]
    results.sort(key=lambda r: -r[3])
    return results

# %% [markdown]
# ## 3. Load Detector & Run on Test Data

# %%
# Load trained detector
det_path = os.path.join(CKPT_DIR, "detector_best.pt")
ckpt = torch.load(det_path, map_location=device, weights_only=True)
model = CellDetectorUNet(base_ch=16).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print(f"Detector loaded (val recall: {ckpt.get('val_recall', '?')})")

# %%
# Discover test samples
test_path = Path(DATA_DIR) / "test"
test_samples = sorted(test_path.glob("*.zarr"))
print(f"Test samples: {len(test_samples)}")
for s in test_samples:
    print(f"  {s.stem}")

# %%
# Run detection on all test samples
PEAK_THRESHOLD = 0.3

all_test_results = {}

for zarr_path in test_samples:
    name = zarr_path.stem
    print(f"\n{'='*60}")
    print(f"  Processing: {name}")
    print(f"{'='*60}")

    loader = ZarrLoader(str(zarr_path))
    print(f"  Shape: {loader.shape}")
    n_frames = loader.n_frames

    detections_by_frame = {}
    global_id = 0
    t0 = time.time()

    with torch.no_grad():
        for t in range(n_frames):
            frame = loader.get_frame(t)
            inp = torch.tensor(
                normalize_volume(frame), dtype=torch.float32
            ).unsqueeze(0).unsqueeze(0).to(device)

            with torch.amp.autocast("cuda"):
                pred = model(inp)
            hm = pred[0, 0].cpu().numpy()

            peaks = extract_peaks(hm, threshold=PEAK_THRESHOLD)
            frame_dets = []
            for z, y, x, conf in peaks:
                frame_dets.append(Detection(
                    node_id=global_id, t=t, z=z, y=y, x=x, confidence=conf))
                global_id += 1
            detections_by_frame[t] = frame_dets

            del inp, pred, frame
            torch.cuda.empty_cache()

            if (t + 1) % 25 == 0:
                print(f"    Frame {t+1}/{n_frames}: {len(frame_dets)} dets")

    elapsed = time.time() - t0
    total_dets = sum(len(d) for d in detections_by_frame.values())
    print(f"  Total: {total_dets} detections in {elapsed:.0f}s")

    all_test_results[name] = detections_by_frame

# %% [markdown]
# ## 4. Build Tracks (Nearest-Neighbor Baseline)
#
# Until the GNN is trained, use a simple nearest-neighbor tracker
# to generate a valid submission. This establishes the baseline.

# %%
def physical_distance(a_zyx, b_zyx, spacing=VOXEL_SPACING_UM):
    diff = (np.array(a_zyx) - np.array(b_zyx)) * spacing
    return np.sqrt(np.sum(diff**2))


def nearest_neighbor_tracking(
    detections_by_frame,
    max_displacement_um=15.0,
    spacing=VOXEL_SPACING_UM,
):
    """
    Simple nearest-neighbor tracking baseline.
    For each cell in frame t, find the nearest cell in frame t+1.
    """
    frames = sorted(detections_by_frame.keys())
    track_edges = []
    divisions = []

    for i in range(len(frames) - 1):
        t_src = frames[i]
        t_dst = frames[i + 1]
        src_dets = detections_by_frame[t_src]
        dst_dets = detections_by_frame[t_dst]

        if not src_dets or not dst_dets:
            continue

        src_coords = np.array([[d.z, d.y, d.x] for d in src_dets]) * spacing
        dst_coords = np.array([[d.z, d.y, d.x] for d in dst_dets]) * spacing

        # Build cost matrix
        from scipy.spatial.distance import cdist
        dist_matrix = cdist(src_coords, dst_coords)

        # Hungarian assignment
        row_ind, col_ind = linear_sum_assignment(dist_matrix)

        for r, c in zip(row_ind, col_ind):
            if dist_matrix[r, c] <= max_displacement_um:
                track_edges.append((src_dets[r].node_id, dst_dets[c].node_id))

    return track_edges, divisions


# Run tracking on all test samples
all_tracks = {}

for name, dets_by_frame in all_test_results.items():
    print(f"\nTracking: {name}")
    edges, divs = nearest_neighbor_tracking(dets_by_frame)
    all_dets = [d for ds in dets_by_frame.values() for d in ds]
    all_tracks[name] = (all_dets, edges)
    print(f"  Nodes: {len(all_dets)}, Edges: {len(edges)}, Divisions: {len(divs)}")

# %% [markdown]
# ## 5. Generate Submission CSV

# %%
def generate_submission(tracks_by_dataset, output_path="submission.csv"):
    """Generate competition submission CSV."""
    rows = []
    row_id = 0

    for dataset_name in sorted(tracks_by_dataset.keys()):
        nodes, edges = tracks_by_dataset[dataset_name]

        for node in nodes:
            rows.append({
                "id": row_id,
                "dataset": dataset_name,
                "row_type": "node",
                "node_id": int(node.node_id),
                "t": int(node.t),
                "z": int(round(node.z)),
                "y": int(round(node.y)),
                "x": int(round(node.x)),
                "source_id": -1,
                "target_id": -1,
            })
            row_id += 1

        for src_id, dst_id in edges:
            rows.append({
                "id": row_id,
                "dataset": dataset_name,
                "row_type": "edge",
                "node_id": -1,
                "t": -1,
                "z": -1,
                "y": -1,
                "x": -1,
                "source_id": int(src_id),
                "target_id": int(dst_id),
            })
            row_id += 1

    df = pd.DataFrame(rows)
    columns = ["id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]
    df = df[columns]
    df.to_csv(output_path, index=False)
    print(f"Submission: {output_path} ({len(df)} rows)")
    return df

# %%
# Generate submission
submission_path = os.path.join(CKPT_DIR, "submission.csv")
df = generate_submission(all_tracks, output_path=submission_path)
print(f"\nSubmission saved to: {submission_path}")

# Quick stats
for ds in df["dataset"].unique():
    ds_df = df[df["dataset"] == ds]
    n_nodes = len(ds_df[ds_df["row_type"] == "node"])
    n_edges = len(ds_df[ds_df["row_type"] == "edge"])
    print(f"  {ds}: {n_nodes} nodes, {n_edges} edges")

# %%
# Validate format
print("\nValidation:")
print(f"  Columns: {list(df.columns)}")
print(f"  Datasets: {sorted(df['dataset'].unique())}")
print(f"  Row types: {dict(df['row_type'].value_counts())}")
print(f"  Duplicate IDs: {df['id'].duplicated().any()}")

# Check edge references
for ds in df["dataset"].unique():
    ds_df = df[df["dataset"] == ds]
    node_ids = set(ds_df[ds_df["row_type"] == "node"]["node_id"])
    edge_df = ds_df[ds_df["row_type"] == "edge"]
    bad_src = edge_df[~edge_df["source_id"].isin(node_ids)]
    bad_dst = edge_df[~edge_df["target_id"].isin(node_ids)]
    if len(bad_src) > 0 or len(bad_dst) > 0:
        print(f"  [WARN] {ds}: {len(bad_src)} bad source refs, {len(bad_dst)} bad target refs")
    else:
        print(f"  [OK] {ds}: All edge references valid")

print(f"\nDownload submission from: {submission_path}")
print("Upload to Kaggle to get your first leaderboard score!")
