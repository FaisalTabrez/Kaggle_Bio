"""
Coordinate transforms and anisotropy handling.

The competition data is anisotropic:
  - Z spacing: 1.625 µm/voxel
  - Y spacing: 0.40625 µm/voxel
  - X spacing: 0.40625 µm/voxel

All distance computations for matching use physical (µm) coordinates.
The matching tolerance is 7.0 µm.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


# Competition-defined constants
VOXEL_SPACING_UM = np.array([1.625, 0.40625, 0.40625])  # Z, Y, X in µm
MATCHING_TOLERANCE_UM = 7.0
Z_ANISOTROPY_FACTOR = VOXEL_SPACING_UM[0] / VOXEL_SPACING_UM[1]  # 4.0


@dataclass
class Detection:
    """A single detected cell in one frame."""
    node_id: int
    t: int
    z: float  # voxel coordinates (sub-voxel precision)
    y: float
    x: float
    confidence: float = 1.0
    intensity_mean: float = 0.0
    intensity_std: float = 0.0
    intensity_max: float = 0.0
    intensity_min: float = 0.0
    blob_scale: float = 1.0  # estimated radius in µm

    @property
    def pos_voxel(self) -> np.ndarray:
        """Position in voxel coordinates (z, y, x)."""
        return np.array([self.z, self.y, self.x])

    @property
    def pos_physical(self) -> np.ndarray:
        """Position in physical coordinates (µm)."""
        return voxel_to_physical(self.pos_voxel)


def voxel_to_physical(
    coords_voxel: np.ndarray,
    spacing: np.ndarray = VOXEL_SPACING_UM,
) -> np.ndarray:
    """
    Convert voxel coordinates to physical (µm) coordinates.

    Args:
        coords_voxel: Array of shape (..., 3) with (z, y, x) in voxels.
        spacing: Voxel spacing in µm, shape (3,).

    Returns:
        Array of same shape in µm.
    """
    return np.asarray(coords_voxel, dtype=np.float64) * spacing


def physical_to_voxel(
    coords_physical: np.ndarray,
    spacing: np.ndarray = VOXEL_SPACING_UM,
) -> np.ndarray:
    """
    Convert physical (µm) coordinates back to voxel coordinates.

    Args:
        coords_physical: Array of shape (..., 3) with (z, y, x) in µm.
        spacing: Voxel spacing in µm, shape (3,).

    Returns:
        Array of same shape in voxels.
    """
    return np.asarray(coords_physical, dtype=np.float64) / spacing


def physical_distance(
    pos_a: np.ndarray,
    pos_b: np.ndarray,
    is_voxel: bool = True,
    spacing: np.ndarray = VOXEL_SPACING_UM,
) -> float:
    """
    Compute Euclidean distance in physical (µm) space.

    Args:
        pos_a: Position (z, y, x), shape (3,).
        pos_b: Position (z, y, x), shape (3,).
        is_voxel: If True, positions are in voxel coords and will be scaled.
        spacing: Voxel spacing if is_voxel is True.

    Returns:
        Distance in µm (scalar).
    """
    a = np.asarray(pos_a, dtype=np.float64)
    b = np.asarray(pos_b, dtype=np.float64)
    if is_voxel:
        a = a * spacing
        b = b * spacing
    return float(np.linalg.norm(a - b))


def physical_distance_batch(
    pos_a: np.ndarray,
    pos_b: np.ndarray,
    is_voxel: bool = True,
    spacing: np.ndarray = VOXEL_SPACING_UM,
) -> np.ndarray:
    """
    Compute pairwise Euclidean distances in physical (µm) space.

    Args:
        pos_a: Positions, shape (N, 3) in (z, y, x).
        pos_b: Positions, shape (M, 3) in (z, y, x).
        is_voxel: If True, scale by voxel spacing.
        spacing: Voxel spacing.

    Returns:
        Distance matrix, shape (N, M) in µm.
    """
    a = np.asarray(pos_a, dtype=np.float64)
    b = np.asarray(pos_b, dtype=np.float64)
    if is_voxel:
        a = a * spacing
        b = b * spacing
    # Efficient pairwise distance: ||a_i - b_j||
    diff = a[:, np.newaxis, :] - b[np.newaxis, :, :]  # (N, M, 3)
    return np.sqrt(np.sum(diff ** 2, axis=-1))  # (N, M)


def scale_coords_for_kdtree(
    coords_voxel: np.ndarray,
    spacing: np.ndarray = VOXEL_SPACING_UM,
) -> np.ndarray:
    """
    Scale voxel coordinates so that Euclidean distance in the scaled
    space equals physical distance. Used for KD-tree nearest neighbor
    searches.

    Args:
        coords_voxel: Array of shape (N, 3) with (z, y, x) in voxels.
        spacing: Voxel spacing in µm.

    Returns:
        Scaled coordinates, shape (N, 3).
    """
    return np.asarray(coords_voxel, dtype=np.float64) * spacing


def normalize_coordinates(
    coords: np.ndarray,
    volume_shape: Tuple[int, int, int],
    spacing: np.ndarray = VOXEL_SPACING_UM,
) -> np.ndarray:
    """
    Normalize spatial coordinates to [0, 1] range based on volume dimensions.

    Args:
        coords: Array of shape (N, 3) with (z, y, x) in voxels.
        volume_shape: (Z, Y, X) dimensions of the volume.
        spacing: Voxel spacing in µm.

    Returns:
        Normalized coordinates, shape (N, 3) in [0, 1].
    """
    max_physical = np.array(volume_shape, dtype=np.float64) * spacing
    physical = np.asarray(coords, dtype=np.float64) * spacing
    return physical / max_physical


def matching_tolerance_voxels(
    tolerance_um: float = MATCHING_TOLERANCE_UM,
    spacing: np.ndarray = VOXEL_SPACING_UM,
) -> np.ndarray:
    """
    Convert matching tolerance from µm to voxels per axis.

    Returns:
        Array of shape (3,) with tolerance in voxels for (z, y, x).
    """
    return tolerance_um / spacing
