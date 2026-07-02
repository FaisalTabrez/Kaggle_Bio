# BioHub Cell Tracking — Graph Neural Network Solution

A graph-first approach to the [Kaggle BioHub Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development) competition.

## Philosophy

**Graph-first, not segmentation-first.** Cell tracking is naturally a graph prediction problem. This pipeline converts 3D+T microscopy volumes into sparse spatio-temporal graphs and uses an Edge-oriented Message Passing Network (EdgeMPN) to classify edges as valid tracks, cell divisions, or false connections.

## Pipeline

```
3D+T Zarr Volume
  → LoG Detection (classical, no training)
  → Lean Feature Extraction (~80 dims)
  → Biology-Informed Graph Construction
  → EdgeMPN (GNN edge classification)
  → Hungarian Assignment + Biological Constraints
  → Submission CSV
```

## Key Design Decisions

- **No detector training initially** — classical LoG blob detection; upgrade only if recall < 85%
- **Joint CNN+GNN training** — no separate triplet pre-training; edge loss supervises appearance learning
- **Biology-informed graph construction** — candidate edges filtered by distance, velocity, intensity, and size priors
- **Hard negative mining** — focus on difficult B-vs-C decisions, not obvious far-away negatives
- **Confidence calibration** — temperature-scaled probabilities → `-log(P)` costs for Hungarian
- **Curriculum learning** — t→t+1 first, then divisions, then gap-closing

## Project Structure

```
Kaggle_Bio/
├── configs/          # Hydra configuration
├── src/
│   ├── data/         # Zarr/GEFF loading, PyG dataset
│   ├── detection/    # LoG blob detector, recall diagnostic
│   ├── features/     # Crop CNN + handcrafted features
│   ├── graph/        # Biology-filtered graph construction
│   ├── models/       # EdgeMPN, classifier head, losses
│   ├── training/     # Curriculum, hard negatives, calibration
│   ├── optimization/ # Hungarian + biological constraints
│   ├── evaluation/   # Edge/Division-Jaccard metrics
│   ├── submission/   # CSV generation + validation
│   └── utils/        # Coordinate transforms, visualization
├── notebooks/        # Colab-ready notebooks
├── scripts/          # CLI entry points
└── tests/            # Unit tests
```

## Quick Start

```bash
pip install -r requirements.txt

# 1. Check detection recall
python scripts/evaluate.py --mode=detection --data=data/train/

# 2. Train GNN
python scripts/train.py experiment=baseline

# 3. Run inference
python scripts/inference.py --checkpoint=best.pt --data=data/test/
```

## Hardware Requirements

- **Target:** Google Colab Free (Tesla T4, 16 GB VRAM)
- **Memory:** ~40 MB per graph (fits easily on T4)
- **Training:** ~2–4 hours per fold with curriculum learning
