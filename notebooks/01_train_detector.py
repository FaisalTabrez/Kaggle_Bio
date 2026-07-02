# %% [markdown]
# # BioHub Cell Tracking — Detector Training (Colab)
#
# **Goal:** Train the 3D U-Net heatmap detector to achieve >85% recall.
#
# LoG blob detection capped at 52.4% recall. This notebook trains a learned
# detector (~1.4M params) on sparse GT annotations using FocalMSE loss.
#
# **Runtime:** T4 GPU, ~3-4 hours for 30 epochs on 40 training samples.
# **No mixed precision** — model is small enough for pure float32.

# %% [markdown]
# ## 1. Setup & Data

# %%
# Install dependencies
!pip install -q zarr scipy scikit-image numpy pandas matplotlib

# %%
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# %%
import os, shutil

# Your zip on Drive:
ZIP_PATH = "/content/drive/MyDrive/kaggle_biohub/biohub-cell-tracking-during-development.zip"
DRIVE_DATA_DIR = "/content/drive/MyDrive/kaggle_biohub/biohub-cell-tracking-during-development"

if os.path.exists(DRIVE_DATA_DIR) and os.path.isdir(DRIVE_DATA_DIR):
    train_path = os.path.join(DRIVE_DATA_DIR, "train")
    if os.path.exists(train_path):
        print(f"[OK] Data unzipped: {DRIVE_DATA_DIR}")
        print(f"     train/ has {len(os.listdir(train_path))} items")
    else:
        print(f"[WARN] Directory exists but no train/ folder.")
else:
    print(f"Data not yet unzipped. Run the next cell.")

disk = shutil.disk_usage("/content")
print(f"Colab disk: {disk.free / 1e9:.1f} GB free / {disk.total / 1e9:.1f} GB total")

# %%
# === UNZIP (only if needed — one-time, ~60 min) ===
import subprocess, time as _time

DATA_DIR = DRIVE_DATA_DIR
if not os.path.exists(os.path.join(DRIVE_DATA_DIR, "train")):
    print("Extracting to Google Drive...")
    t0 = _time.time()
    result = subprocess.run(
        ["unzip", "-n", "-q", ZIP_PATH, "-d", os.path.dirname(DRIVE_DATA_DIR)],
        capture_output=True, text=True, timeout=7200)
    print(f"Done in {(_time.time()-t0)/60:.0f} min (exit: {result.returncode})")
else:
    print("Already extracted.")

# Find train/ (zip may nest directories)
for root, dirs, files in os.walk(DRIVE_DATA_DIR):
    if "train" in dirs:
        DATA_DIR = root
        print(f"[OK] train/ at: {DATA_DIR}")
        break
else:
    print("[ERROR] train/ not found!")

# %% [markdown]
# ## 2. All Core Definitions
#
# **Run this single cell.** It defines ALL imports, constants, data loaders,
# model architecture, loss function, and utilities.

# %%
import os, sys, time, random, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import zarr
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Dict, Optional
from scipy.ndimage import maximum_filter
from scipy.optimize import linear_sum_assignment
from dataclasses import dataclass

# --- Constants ---
VOXEL_SPACING_UM = np.array([1.625, 0.40625, 0.40625])  # Z, Y, X in microns
MATCHING_TOLERANCE_UM = 7.0

if 'DATA_DIR' not in dir():
    DATA_DIR = "/content/drive/MyDrive/kaggle_biohub/biohub-cell-tracking-during-development"

random.seed(42); np.random.seed(42); torch.manual_seed(42)


# --- Data Classes ---

@dataclass
class Detection:
    node_id: int
    t: int
    z: float
    y: float
    x: float
    confidence: float = 1.0


@dataclass
class GeffNode:
    node_id: int
    t: int
    z: int
    y: int
    x: int


def physical_distance_batch(pos_a, pos_b, is_voxel=True, spacing=VOXEL_SPACING_UM):
    a = np.asarray(pos_a, dtype=np.float64)
    b = np.asarray(pos_b, dtype=np.float64)
    if is_voxel:
        a = a * spacing
        b = b * spacing
    diff = a[:, np.newaxis, :] - b[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1))


# --- Data Loading ---

class ZarrLoader:
    """Lazy Zarr v3 volume loader."""
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
    def shape(self):
        return self._array.shape

    @property
    def n_frames(self):
        return self._array.shape[0]

    @property
    def volume_shape(self):
        return tuple(self._array.shape[1:])

    @property
    def voxel_spacing(self):
        return self._spacing

    def get_frame(self, t):
        return np.asarray(self._array[t])


class GeffLoader:
    """Load sparse GT annotations from GEFF format."""
    def __init__(self, geff_path):
        self.geff_path = Path(geff_path)

    def load_nodes(self):
        store = zarr.open(str(self.geff_path), mode="r")
        node_ids = np.asarray(store["nodes"]["ids"])
        t_vals = np.asarray(store["nodes"]["props"]["t"]["values"])
        z_vals = np.asarray(store["nodes"]["props"]["z"]["values"])
        y_vals = np.asarray(store["nodes"]["props"]["y"]["values"])
        x_vals = np.asarray(store["nodes"]["props"]["x"]["values"])
        return [GeffNode(int(node_ids[i]), int(t_vals[i]),
                         int(z_vals[i]), int(y_vals[i]), int(x_vals[i]))
                for i in range(len(node_ids))]


def discover_samples(data_dir, split="train"):
    data_path = Path(data_dir) / split
    if not data_path.exists():
        data_path = Path(data_dir)
    samples = []
    for zarr_dir in sorted(data_path.glob("*.zarr")):
        name = zarr_dir.stem
        geff_dir = zarr_dir.parent / f"{name}.geff"
        samples.append({
            "name": name,
            "zarr_path": str(zarr_dir),
            "geff_path": str(geff_dir) if geff_dir.exists() else None,
            "embryo_prefix": name.split("_")[0],
        })
    return samples


# --- 3D U-Net Detector ---

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.norm = nn.InstanceNorm3d(out_ch)
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool_size=(2, 2, 2)):
        super().__init__()
        self.pool = nn.MaxPool3d(pool_size)
        self.conv1 = ConvBlock(in_ch, out_ch)
        self.conv2 = ConvBlock(out_ch, out_ch)

    def forward(self, x):
        return self.conv2(self.conv1(self.pool(x)))


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, scale=(2, 2, 2)):
        super().__init__()
        self.scale = scale
        self.conv1 = ConvBlock(in_ch + skip_ch, out_ch)
        self.conv2 = ConvBlock(out_ch, out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=self.scale, mode='trilinear',
                          align_corners=False)
        if x.shape != skip.shape:
            x = F.pad(x, [0, skip.shape[4] - x.shape[4],
                          0, skip.shape[3] - x.shape[3],
                          0, skip.shape[2] - x.shape[2]])
        return self.conv2(self.conv1(torch.cat([x, skip], 1)))


class CellDetectorUNet(nn.Module):
    """Lightweight 3D U-Net for heatmap prediction. ~1.4M params, T4-friendly."""
    def __init__(self, base_ch=16):
        super().__init__()
        c = base_ch
        self.enc1a = ConvBlock(1, c)
        self.enc1b = ConvBlock(c, c)
        self.enc2 = DownBlock(c, c * 2)
        self.enc3 = DownBlock(c * 2, c * 4)
        self.bottleneck = DownBlock(c * 4, c * 8)
        self.dec3 = UpBlock(c * 8, c * 4, c * 4)
        self.dec2 = UpBlock(c * 4, c * 2, c * 2)
        self.dec1 = UpBlock(c * 2, c, c)
        self.head = nn.Conv3d(c, 1, 1)

    def forward(self, x):
        e1 = self.enc1b(self.enc1a(x))
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        bn = self.bottleneck(e3)
        d3 = self.dec3(bn, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        return torch.sigmoid(self.head(d1))

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters())


# --- Loss ---

class FocalMSELoss(nn.Module):
    """Focal MSE for extreme fg/bg imbalance in heatmaps."""
    def __init__(self, alpha=2.0, beta=4.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, target):
        pred = pred.float()
        target = target.float()
        pos_mask = (target >= 0.01)
        pos_weight = (1 - pred) ** self.alpha
        pos_loss = torch.where(
            pos_mask, pos_weight * (pred - target) ** 2, torch.zeros_like(pred))
        neg_weight = pred ** self.alpha * (1 - target) ** self.beta
        neg_loss = torch.where(
            ~pos_mask, neg_weight * pred ** 2, torch.zeros_like(pred))
        return (pos_loss.sum() + neg_loss.sum()) / max(pos_mask.sum().item(), 1)


# --- Utilities ---

def make_gaussian_heatmap(volume_shape, centers_zyx, sigma_voxels=(2.0, 4.0, 4.0)):
    heatmap = np.zeros(volume_shape, dtype=np.float32)
    Z, Y, X = volume_shape
    sz, sy, sx = sigma_voxels
    for cz, cy, cx in centers_zyx:
        cz, cy, cx = int(round(cz)), int(round(cy)), int(round(cx))
        rz = int(np.ceil(3 * sz))
        ry = int(np.ceil(3 * sy))
        rx = int(np.ceil(3 * sx))
        z0, z1 = max(0, cz - rz), min(Z, cz + rz + 1)
        y0, y1 = max(0, cy - ry), min(Y, cy + ry + 1)
        x0, x1 = max(0, cx - rx), min(X, cx + rx + 1)
        zz, yy, xx = np.meshgrid(
            np.arange(z0, z1), np.arange(y0, y1), np.arange(x0, x1),
            indexing='ij')
        g = np.exp(-(
            (zz - cz) ** 2 / (2 * sz ** 2) +
            (yy - cy) ** 2 / (2 * sy ** 2) +
            (xx - cx) ** 2 / (2 * sx ** 2)))
        heatmap[z0:z1, y0:y1, x0:x1] = np.maximum(
            heatmap[z0:z1, y0:y1, x0:x1], g)
    return heatmap


def extract_peaks(heatmap, threshold=0.3, min_dist=(2, 5, 5)):
    footprint = tuple(2 * d + 1 for d in min_dist)
    local_max = maximum_filter(heatmap, size=footprint)
    mask = (heatmap == local_max) & (heatmap > threshold)
    coords = np.array(np.nonzero(mask)).T
    if len(coords) == 0:
        return []
    results = [(float(z), float(y), float(x), float(heatmap[z, y, x]))
               for z, y, x in coords]
    results.sort(key=lambda r: -r[3])
    return results


def normalize_volume(vol):
    v = vol.astype(np.float32)
    vmin, vmax = v.min(), v.max()
    return (v - vmin) / (vmax - vmin) if vmax > vmin else v


def measure_recall(detections, gt_nodes, spacing=VOXEL_SPACING_UM, tol=7.0):
    det_by_t = defaultdict(list)
    for d in detections:
        det_by_t[d.t].append(d)
    gt_by_t = defaultdict(list)
    for n in gt_nodes:
        gt_by_t[n.t].append(n)
    tp, fn, dists = 0, 0, []
    for t in gt_by_t:
        gl = gt_by_t[t]
        dl = det_by_t.get(t, [])
        if not dl:
            fn += len(gl)
            continue
        dc = np.array([[d.z, d.y, d.x] for d in dl])
        gc = np.array([[n.z, n.y, n.x] for n in gl])
        dm = physical_distance_batch(dc, gc, is_voxel=True, spacing=spacing)
        ri, ci = linear_sum_assignment(dm)
        for r, c in zip(ri, ci):
            if dm[r, c] <= tol:
                tp += 1
                dists.append(dm[r, c])
        fn += len(gl) - sum(1 for r, c in zip(ri, ci) if dm[r, c] <= tol)
    return {"recall": tp / max(tp + fn, 1), "tp": tp, "fn": fn,
            "avg_dist": np.mean(dists) if dists else 0}


# --- Print status ---
print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}, "
          f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
print(f"DATA_DIR: {DATA_DIR}")
print("All definitions loaded successfully!")

# %% [markdown]
# ## 3. Discover & Prepare Training Data

# %%
all_samples = discover_samples(DATA_DIR, split="train")
print(f"Total training samples: {len(all_samples)}")

embryo_44b6 = [s for s in all_samples if s["embryo_prefix"] == "44b6"]
embryo_6bba = [s for s in all_samples if s["embryo_prefix"] == "6bba"]
print(f"  44b6: {len(embryo_44b6)} samples")
print(f"  6bba: {len(embryo_6bba)} samples")

N_TRAIN_PER_EMBRYO = 20
N_VAL = 5

val_samples = embryo_44b6[-N_VAL:]
train_samples = (embryo_44b6[:-N_VAL][:N_TRAIN_PER_EMBRYO] +
                 embryo_6bba[:N_TRAIN_PER_EMBRYO])

print(f"\nTrain: {len(train_samples)} samples")
print(f"Val:   {len(val_samples)} samples")

# %%
def collect_annotated_frames(samples_info, label=""):
    frames = []
    skipped = 0
    for info in samples_info:
        if info["geff_path"] is None:
            skipped += 1
            continue
        try:
            gt_nodes = GeffLoader(info["geff_path"]).load_nodes()
        except Exception as e:
            print(f"  [WARN] {info['name']}: {e}")
            skipped += 1
            continue
        if not gt_nodes:
            skipped += 1
            continue
        centroids_by_t = defaultdict(list)
        for n in gt_nodes:
            centroids_by_t[n.t].append((n.z, n.y, n.x))
        for t, centroids in centroids_by_t.items():
            frames.append({
                "zarr_path": info["zarr_path"],
                "frame_idx": t,
                "centroids": centroids,
                "name": info["name"],
            })
    print(f"  {label}: {len(frames)} annotated frames "
          f"(from {len(samples_info) - skipped} samples, {skipped} skipped)")
    return frames


print("Collecting annotated frames...")
train_frames = collect_annotated_frames(train_samples, "Train")
val_frames = collect_annotated_frames(val_samples, "Val")
random.shuffle(train_frames)

print(f"\nTotal training frames: {len(train_frames)}")
print(f"Total validation frames: {len(val_frames)}")

# Sanity check
test_loader = ZarrLoader(train_frames[0]["zarr_path"])
test_frame = test_loader.get_frame(train_frames[0]["frame_idx"])
print(f"\nSample frame shape: {test_frame.shape}, dtype: {test_frame.dtype}")
print(f"Range: [{test_frame.min()}, {test_frame.max()}]")
del test_frame

# %%
# === SPEED OPTIMIZATION: Copy training data to fast local disk ===
# Google Drive FUSE is ~10x slower than local SSD.
# 45 samples x ~450 MB = ~20 GB, fits in Colab's free disk.

import shutil

LOCAL_DATA = "/content/local_data/train"
os.makedirs(LOCAL_DATA, exist_ok=True)

# Collect unique zarr/geff paths we need
all_paths = set()
for f in train_frames + val_frames:
    all_paths.add(f["zarr_path"])
    geff = f["zarr_path"].replace(".zarr", ".geff")
    if os.path.exists(geff):
        all_paths.add(geff)

print(f"Copying {len(all_paths)} dirs to local disk...")
t0 = time.time()
copied = 0
for src_path in sorted(all_paths):
    name = os.path.basename(src_path)
    dst_path = os.path.join(LOCAL_DATA, name)
    if os.path.exists(dst_path):
        continue  # Already copied (e.g. from a previous run)
    shutil.copytree(src_path, dst_path)
    copied += 1
    if copied % 10 == 0:
        print(f"  Copied {copied}/{len(all_paths)}...")
print(f"Done: {copied} new copies in {(time.time()-t0)/60:.1f} min")

# Update frame paths to point to local copies
for f in train_frames:
    name = os.path.basename(f["zarr_path"])
    f["zarr_path"] = os.path.join(LOCAL_DATA, name)
for f in val_frames:
    name = os.path.basename(f["zarr_path"])
    f["zarr_path"] = os.path.join(LOCAL_DATA, name)

disk = shutil.disk_usage("/content")
print(f"Local disk: {disk.free / 1e9:.1f} GB remaining")

# %% [markdown]
# ## 4. Train the Detector
#
# **Pure float32** — no mixed precision. Model is only 1.4M params,
# uses ~3 GB VRAM on (64, 256, 256) volumes. T4 has 16 GB.

# %%
N_EPOCHS = 30
LR = 1e-3
EVAL_EVERY = 5
SIGMA_VOXELS = (2.0, 4.0, 4.0)
PEAK_THRESHOLD = 0.3
FRAMES_PER_EPOCH = 800  # Subsample for speed (3680 total, see all over ~5 epochs)

SAVE_DIR = "/content/drive/MyDrive/kaggle_biohub/checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model = CellDetectorUNet(base_ch=16).to(device)
print(f"Model parameters: {model.n_params:,}")

# Resume from checkpoint if available
start_epoch = 0
best_val_recall = 0
resume_path = os.path.join(SAVE_DIR, "detector_best.pt")
if os.path.exists(resume_path):
    ckpt = torch.load(resume_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    start_epoch = ckpt.get("epoch", 0) + 1
    best_val_recall = ckpt.get("val_recall", 0)
    print(f"Resumed from epoch {start_epoch} (best recall: {best_val_recall:.1%})")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=N_EPOCHS,
    last_epoch=start_epoch - 1 if start_epoch > 0 else -1)
loss_fn = FocalMSELoss(alpha=2.0, beta=4.0)

# %%
# === TRAINING LOOP (pure float32, no autocast) ===
history = {"epoch": [], "loss": [], "val_recall": [], "n_frames": []}

for epoch in range(start_epoch, N_EPOCHS):
    t0 = time.time()
    model.train()
    epoch_loss = 0
    n_trained = 0
    n_errors = 0

    random.shuffle(train_frames)
    epoch_frames = train_frames[:FRAMES_PER_EPOCH]  # Subsample for speed

    for fi, frame_info in enumerate(epoch_frames):
        try:
            loader = ZarrLoader(frame_info["zarr_path"])
            frame = loader.get_frame(frame_info["frame_idx"])

            inp = torch.tensor(
                normalize_volume(frame), dtype=torch.float32
            ).unsqueeze(0).unsqueeze(0).to(device)

            heatmap = make_gaussian_heatmap(
                frame.shape, frame_info["centroids"], SIGMA_VOXELS)
            tgt = torch.tensor(
                heatmap, dtype=torch.float32
            ).unsqueeze(0).unsqueeze(0).to(device)

            optimizer.zero_grad()
            pred = model(inp)
            loss = loss_fn(pred, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_trained += 1

            del inp, tgt, pred, loss, frame
            torch.cuda.empty_cache()

        except Exception as e:
            n_errors += 1
            if n_errors <= 3:
                print(f"  [ERROR] {frame_info['name']} t={frame_info['frame_idx']}: {e}")
            continue

        if (fi + 1) % 200 == 0:
            avg = epoch_loss / max(n_trained, 1)
            eta = (time.time() - t0) / (fi + 1) * (len(epoch_frames) - fi - 1)
            print(f"    [{fi+1}/{len(epoch_frames)}] "
                  f"loss: {avg:.6f} | ETA: {eta/60:.0f}min")

    scheduler.step()
    avg_loss = epoch_loss / max(n_trained, 1)
    elapsed = time.time() - t0

    # === VALIDATION ===
    val_recall = None
    if (epoch + 1) % EVAL_EVERY == 0 or epoch == N_EPOCHS - 1:
        model.eval()
        all_dets = []
        all_gt = []
        det_id = 0

        with torch.no_grad():
            for frame_info in val_frames:
                try:
                    loader = ZarrLoader(frame_info["zarr_path"])
                    frame = loader.get_frame(frame_info["frame_idx"])
                    inp = torch.tensor(
                        normalize_volume(frame), dtype=torch.float32
                    ).unsqueeze(0).unsqueeze(0).to(device)

                    pred = model(inp)
                    hm = pred[0, 0].cpu().numpy()
                    peaks = extract_peaks(hm, threshold=PEAK_THRESHOLD)

                    for z, y, x, conf in peaks:
                        all_dets.append(Detection(
                            node_id=det_id, t=frame_info["frame_idx"],
                            z=z, y=y, x=x, confidence=conf))
                        det_id += 1

                    for cz, cy, cx in frame_info["centroids"]:
                        all_gt.append(GeffNode(
                            node_id=0, t=frame_info["frame_idx"],
                            z=int(cz), y=int(cy), x=int(cx)))

                    del inp, pred, frame
                    torch.cuda.empty_cache()
                except Exception:
                    continue

        if all_gt:
            result = measure_recall(all_dets, all_gt)
            val_recall = result["recall"]

            if val_recall > best_val_recall:
                best_val_recall = val_recall
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_recall": val_recall,
                    "config": {"base_ch": 16, "threshold": PEAK_THRESHOLD,
                               "sigma_voxels": SIGMA_VOXELS},
                }, os.path.join(SAVE_DIR, "detector_best.pt"))
                print(f"  ** New best! Saved (recall: {val_recall:.1%})")

    # Crash recovery checkpoint
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "val_recall": val_recall or best_val_recall,
    }, os.path.join(SAVE_DIR, "detector_latest.pt"))

    # Log
    history["epoch"].append(epoch + 1)
    history["loss"].append(avg_loss)
    history["val_recall"].append(val_recall)
    history["n_frames"].append(n_trained)

    recall_str = f"Recall: {val_recall:.1%}" if val_recall is not None else ""
    err_str = f"({n_errors} errors)" if n_errors > 0 else ""
    print(f"Epoch {epoch+1:3d}/{N_EPOCHS} | "
          f"Loss: {avg_loss:.6f} | "
          f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
          f"{recall_str} | "
          f"{elapsed/60:.1f}min ({n_trained} frames) {err_str}")

print(f"\n{'='*60}")
print(f"Training complete! Best validation recall: {best_val_recall:.1%}")
print(f"{'='*60}")

# %% [markdown]
# ## 5. Training Curves

# %%
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(history["epoch"], history["loss"], 'b-o', markersize=3)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("FocalMSE Loss")
ax1.set_title("Training Loss")
ax1.grid(True, alpha=0.3)

recall_epochs = [e for e, r in zip(history["epoch"], history["val_recall"])
                 if r is not None]
recall_vals = [r for r in history["val_recall"] if r is not None]
if recall_vals:
    ax2.plot(recall_epochs, recall_vals, 'r-o', markersize=5, linewidth=2)
    ax2.axhline(y=0.85, color='green', linestyle='--', alpha=0.7, label='85% gate')
    ax2.axhline(y=0.90, color='blue', linestyle='--', alpha=0.7, label='90% target')
    ax2.axhline(y=0.524, color='gray', linestyle=':', alpha=0.5,
                label='LoG baseline (52.4%)')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Detection Recall")
    ax2.set_title("Validation Recall")
    ax2.set_ylim(0, 1.05)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "training_curves.png"), dpi=150,
            bbox_inches='tight')
plt.show()

with open(os.path.join(SAVE_DIR, "training_history.json"), "w") as f:
    json.dump(history, f, indent=2)
print("Curves + history saved to Drive.")

# %% [markdown]
# ## 6. Full Evaluation on Held-Out Samples

# %%
# Load best model
ckpt = torch.load(os.path.join(SAVE_DIR, "detector_best.pt"),
                  map_location=device, weights_only=True)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print(f"Loaded best model from epoch {ckpt['epoch']+1} "
      f"(recall: {ckpt['val_recall']:.1%})")

# %%
print("\n--- Detailed Evaluation ---\n")

eval_samples = embryo_44b6[-10:]
eval_results = []

for info in eval_samples:
    if info["geff_path"] is None:
        continue
    name = info["name"]
    loader = ZarrLoader(info["zarr_path"])
    gt_nodes = GeffLoader(info["geff_path"]).load_nodes()
    if not gt_nodes:
        continue

    gt_frames = sorted(set(n.t for n in gt_nodes))
    all_dets = []
    det_id = 0

    with torch.no_grad():
        for t in gt_frames:
            try:
                frame = loader.get_frame(t)
                inp = torch.tensor(
                    normalize_volume(frame), dtype=torch.float32
                ).unsqueeze(0).unsqueeze(0).to(device)
                pred = model(inp)
                hm = pred[0, 0].cpu().numpy()
                peaks = extract_peaks(hm, threshold=PEAK_THRESHOLD)
                for z, y, x, conf in peaks:
                    all_dets.append(Detection(det_id, t, z, y, x, conf))
                    det_id += 1
                del inp, pred, frame
                torch.cuda.empty_cache()
            except Exception:
                continue

    result = measure_recall(all_dets, gt_nodes)
    eval_results.append({"name": name, **result})
    print(f"  {name}: Recall {result['recall']:5.1%} "
          f"({result['tp']}/{result['tp']+result['fn']}) "
          f"AvgDist: {result['avg_dist']:.2f}um  "
          f"Dets: {len(all_dets)}")

recalls = [r["recall"] for r in eval_results]
print(f"\n{'='*50}")
print(f"  Mean Recall: {np.mean(recalls):.1%}")
print(f"  Std:         {np.std(recalls):.1%}")
print(f"  Min:         {np.min(recalls):.1%}")
print(f"  Max:         {np.max(recalls):.1%}")
print(f"{'='*50}")

if np.mean(recalls) >= 0.90:
    print("\n  [PASS] >= 90% -- Excellent! Proceed to GNN training.")
elif np.mean(recalls) >= 0.85:
    print("\n  [PASS] >= 85% -- Good. Proceed to GNN training.")
elif np.mean(recalls) >= 0.70:
    print("\n  [WARN] 70-85% -- Marginal. May need tuning.")
else:
    print(f"\n  [FAIL] {np.mean(recalls):.1%} -- Need more training.")

# %% [markdown]
# ## 7. Export

# %%
final_path = os.path.join(SAVE_DIR, "detector_final.pt")
torch.save({
    "model_state_dict": model.state_dict(),
    "config": {"base_ch": 16, "peak_threshold": PEAK_THRESHOLD,
               "sigma_voxels": SIGMA_VOXELS},
    "eval_results": eval_results,
    "mean_recall": float(np.mean(recalls)),
}, final_path)

print(f"Final model saved to: {final_path}")
print(f"\nTo use locally:")
print(f"  1. Download {SAVE_DIR}/detector_best.pt")
print(f"  2. Place in Kaggle_Bio/outputs/detector_best.pt")
print(f"  3. Use src/detection/detect.py with method='learned'")
