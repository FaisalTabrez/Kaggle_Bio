"""
Zarr v3 volume loader for the BioHub Cell Tracking competition.

Loads 3D+T microscopy volumes stored in Zarr v3 format.
Volume shape: (T, Z, Y, X) — typically (100, 64, 256, 256), uint16.
Each timepoint is chunked as (1, 64, 256, 256) with blosc/zstd compression.

Usage:
    loader = ZarrLoader("path/to/sample.zarr")
    frame = loader.get_frame(t=0)        # shape: (64, 256, 256)
    volume = loader.get_volume()          # shape: (100, 64, 256, 256)
    metadata = loader.metadata            # dict with shape, spacing, stats
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

try:
    import zarr
except ImportError:
    raise ImportError("zarr>=3.0.0 required. Install with: pip install zarr")


class ZarrLoader:
    """
    Lazy loader for Zarr v3 3D+T microscopy volumes.
    
    Supports frame-by-frame access (memory-efficient for Colab)
    and full volume loading when needed.
    """

    def __init__(self, zarr_path: str):
        """
        Args:
            zarr_path: Path to the .zarr directory.
        """
        self.zarr_path = Path(zarr_path)
        if not self.zarr_path.exists():
            raise FileNotFoundError(f"Zarr directory not found: {self.zarr_path}")

        # Open the zarr store
        self._store = zarr.open(str(self.zarr_path), mode="r")

        # Access the array at path "0" (as per competition format)
        self._array = self._store["0"]

        # Parse metadata from zarr.json
        self._metadata = self._parse_metadata()

    def _parse_metadata(self) -> Dict[str, Any]:
        """Parse the root zarr.json for volume metadata."""
        zarr_json_path = self.zarr_path / "zarr.json"
        metadata = {}

        if zarr_json_path.exists():
            with open(zarr_json_path, "r") as f:
                root_meta = json.load(f)

            attrs = root_meta.get("attributes", {})

            # Extract voxel spacing from multiscales
            multiscales = attrs.get("multiscales", [])
            if multiscales:
                ms = multiscales[0]
                datasets = ms.get("datasets", [])
                if datasets:
                    transforms = datasets[0].get("coordinateTransformations", [])
                    for t in transforms:
                        if t.get("type") == "scale":
                            metadata["voxel_spacing"] = t["scale"]  # [T, Z, Y, X]

                # Extract axis info
                axes = ms.get("axes", [])
                metadata["axes"] = [
                    {"name": a["name"], "type": a.get("type"), "unit": a.get("unit")}
                    for a in axes
                ]

            # Extract image statistics
            image_stats = attrs.get("image_statistics", {})
            if image_stats:
                metadata["image_statistics"] = image_stats

        # Array shape and dtype
        metadata["shape"] = self._array.shape  # (T, Z, Y, X)
        metadata["dtype"] = str(self._array.dtype)
        metadata["n_frames"] = self._array.shape[0]
        metadata["volume_shape"] = self._array.shape[1:]  # (Z, Y, X)

        return metadata

    @property
    def metadata(self) -> Dict[str, Any]:
        """Volume metadata including shape, spacing, and image statistics."""
        return self._metadata

    @property
    def shape(self) -> Tuple[int, ...]:
        """Full volume shape (T, Z, Y, X)."""
        return self._array.shape

    @property
    def n_frames(self) -> int:
        """Number of timepoints."""
        return self._array.shape[0]

    @property
    def volume_shape(self) -> Tuple[int, int, int]:
        """Spatial shape (Z, Y, X)."""
        return tuple(self._array.shape[1:])

    @property
    def voxel_spacing(self) -> np.ndarray:
        """
        Voxel spacing in µm for (Z, Y, X).
        Defaults to competition standard if not in metadata.
        """
        spacing = self._metadata.get("voxel_spacing", [1.0, 1.625, 0.40625, 0.40625])
        return np.array(spacing[1:])  # Skip time dimension

    @property
    def dataset_name(self) -> str:
        """Extract dataset name from path (e.g., '44b6_0113de3b')."""
        return self.zarr_path.stem.replace(".zarr", "")

    def get_frame(self, t: int) -> np.ndarray:
        """
        Load a single timepoint.

        Args:
            t: Frame index (0-indexed).

        Returns:
            3D array of shape (Z, Y, X), dtype uint16.
        """
        if t < 0 or t >= self.n_frames:
            raise IndexError(
                f"Frame {t} out of range [0, {self.n_frames - 1}]"
            )
        return np.asarray(self._array[t])

    def get_frames(self, t_start: int, t_end: int) -> np.ndarray:
        """
        Load a range of timepoints.

        Args:
            t_start: First frame (inclusive).
            t_end: Last frame (exclusive).

        Returns:
            4D array of shape (t_end - t_start, Z, Y, X).
        """
        return np.asarray(self._array[t_start:t_end])

    def get_volume(self) -> np.ndarray:
        """
        Load the entire 4D volume into memory.

        Warning: For (100, 64, 256, 256) uint16, this is ~3.2 GB.
        Prefer get_frame() for memory-constrained environments.

        Returns:
            4D array of shape (T, Z, Y, X).
        """
        return np.asarray(self._array[:])

    def get_crop(
        self,
        t: int,
        center_zyx: Tuple[float, float, float],
        crop_size: Tuple[int, int, int] = (8, 16, 16),
    ) -> np.ndarray:
        """
        Extract a 3D crop centered at a given position in a frame.

        Handles boundary padding with zeros.

        Args:
            t: Frame index.
            center_zyx: Center position in voxel coordinates (z, y, x).
            crop_size: Crop dimensions (dz, dy, dx).

        Returns:
            3D array of shape crop_size.
        """
        frame = self.get_frame(t)
        Z, Y, X = frame.shape
        dz, dy, dx = crop_size
        cz, cy, cx = int(round(center_zyx[0])), int(round(center_zyx[1])), int(round(center_zyx[2]))

        # Compute source and destination slices with boundary handling
        z0 = max(0, cz - dz // 2)
        z1 = min(Z, cz + dz // 2 + dz % 2)
        y0 = max(0, cy - dy // 2)
        y1 = min(Y, cy + dy // 2 + dy % 2)
        x0 = max(0, cx - dx // 2)
        x1 = min(X, cx + dx // 2 + dx % 2)

        # Destination offsets
        oz0 = z0 - (cz - dz // 2)
        oy0 = y0 - (cy - dy // 2)
        ox0 = x0 - (cx - dx // 2)

        crop = np.zeros(crop_size, dtype=frame.dtype)
        crop[
            oz0 : oz0 + (z1 - z0),
            oy0 : oy0 + (y1 - y0),
            ox0 : ox0 + (x1 - x0),
        ] = frame[z0:z1, y0:y1, x0:x1]

        return crop

    def get_intensity_stats(
        self,
        t: int,
        center_zyx: Tuple[float, float, float],
        radius: int = 4,
    ) -> Dict[str, float]:
        """
        Compute intensity statistics in a spherical neighborhood.

        Args:
            t: Frame index.
            center_zyx: Center in voxels (z, y, x).
            radius: Neighborhood radius in voxels (isotropic).

        Returns:
            Dict with mean, std, max, min.
        """
        crop = self.get_crop(t, center_zyx, crop_size=(2 * radius,) * 3)
        values = crop[crop > 0].astype(np.float32)
        if len(values) == 0:
            values = crop.flatten().astype(np.float32)

        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "max": float(np.max(values)),
            "min": float(np.min(values)),
        }

    def __repr__(self) -> str:
        return (
            f"ZarrLoader('{self.dataset_name}', "
            f"shape={self.shape}, "
            f"spacing={self.voxel_spacing.tolist()})"
        )


def discover_samples(
    data_dir: str,
    split: str = "train",
) -> list:
    """
    Discover all Zarr samples in a data directory.

    Args:
        data_dir: Root data directory containing train/ and test/ folders.
        split: "train" or "test".

    Returns:
        List of dicts with 'name', 'zarr_path', 'geff_path' (None for test).
    """
    data_path = Path(data_dir) / split
    if not data_path.exists():
        # Maybe the path itself is the split directory
        data_path = Path(data_dir)

    samples = []
    for zarr_dir in sorted(data_path.glob("*.zarr")):
        name = zarr_dir.stem
        geff_dir = zarr_dir.parent / f"{name}.geff"

        samples.append({
            "name": name,
            "zarr_path": str(zarr_dir),
            "geff_path": str(geff_dir) if geff_dir.exists() else None,
            "embryo_prefix": name.split("_")[0],  # e.g., "44b6" or "6bba"
        })

    return samples
