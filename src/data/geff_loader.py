"""
GEFF (Graph Exchange File Format) loader for ground truth annotations.

GEFF is a Zarr v3-based format containing:
  - nodes/ids: uint64 array of node IDs, shape (N,)
  - nodes/props/{t,z,y,x}/values: int64 centroid coordinates per node
  - edges/ids: uint64 array of shape (E, 2) with (source_id, target_id)

The ground truth is SPARSE — not every cell in every frame is labeled.
"""

import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set

try:
    import zarr
except ImportError:
    raise ImportError("zarr>=3.0.0 required. Install with: pip install zarr")


@dataclass
class GeffNode:
    """A single ground truth node (cell centroid)."""
    node_id: int
    t: int
    z: int
    y: int
    x: int


@dataclass
class GeffEdge:
    """A single ground truth edge (temporal link)."""
    source_id: int
    target_id: int


@dataclass
class GeffGraph:
    """
    Complete ground truth graph from a GEFF file.
    
    Contains sparse annotations: only a subset of cells are labeled.
    """
    nodes: List[GeffNode]
    edges: List[GeffEdge]
    metadata: Dict

    # Lazily computed caches
    _node_by_id: Dict[int, GeffNode] = field(default_factory=dict, repr=False)
    _nodes_by_frame: Dict[int, List[GeffNode]] = field(default_factory=dict, repr=False)
    _outgoing_edges: Dict[int, List[GeffEdge]] = field(default_factory=dict, repr=False)
    _incoming_edges: Dict[int, List[GeffEdge]] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self._build_indices()

    def _build_indices(self):
        """Build lookup indices for fast access."""
        self._node_by_id = {n.node_id: n for n in self.nodes}

        self._nodes_by_frame = {}
        for n in self.nodes:
            self._nodes_by_frame.setdefault(n.t, []).append(n)

        self._outgoing_edges = {}
        self._incoming_edges = {}
        for e in self.edges:
            self._outgoing_edges.setdefault(e.source_id, []).append(e)
            self._incoming_edges.setdefault(e.target_id, []).append(e)

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    @property
    def frames(self) -> List[int]:
        """Sorted list of annotated frame indices."""
        return sorted(self._nodes_by_frame.keys())

    @property
    def n_frames(self) -> int:
        """Number of frames with at least one annotated node."""
        return len(self._nodes_by_frame)

    @property
    def frame_range(self) -> Tuple[int, int]:
        """(min_frame, max_frame) of annotated nodes."""
        frames = self.frames
        return (frames[0], frames[-1]) if frames else (0, 0)

    def get_node(self, node_id: int) -> Optional[GeffNode]:
        """Look up a node by ID."""
        return self._node_by_id.get(node_id)

    def get_nodes_at_frame(self, t: int) -> List[GeffNode]:
        """Get all annotated nodes at frame t."""
        return self._nodes_by_frame.get(t, [])

    def get_outgoing_edges(self, node_id: int) -> List[GeffEdge]:
        """Get all edges originating from this node."""
        return self._outgoing_edges.get(node_id, [])

    def get_incoming_edges(self, node_id: int) -> List[GeffEdge]:
        """Get all edges pointing to this node."""
        return self._incoming_edges.get(node_id, [])

    @property
    def division_nodes(self) -> List[GeffNode]:
        """Nodes with 2+ outgoing edges (cell divisions)."""
        divs = []
        for node_id, edges in self._outgoing_edges.items():
            if len(edges) >= 2:
                node = self._node_by_id.get(node_id)
                if node is not None:
                    divs.append(node)
        return divs

    @property
    def n_divisions(self) -> int:
        return len(self.division_nodes)

    def get_edge_labels(
        self,
        candidate_edges: List[Tuple[int, int]],
    ) -> List[int]:
        """
        Label candidate edges against ground truth.
        
        Args:
            candidate_edges: List of (source_id, target_id) tuples.
            
        Returns:
            List of labels: 0 = false, 1 = valid track, 2 = division.
        """
        gt_edge_set = {(e.source_id, e.target_id) for e in self.edges}
        division_sources = {
            node_id for node_id, edges in self._outgoing_edges.items()
            if len(edges) >= 2
        }

        labels = []
        for src, dst in candidate_edges:
            if (src, dst) in gt_edge_set:
                if src in division_sources:
                    labels.append(2)  # Division edge
                else:
                    labels.append(1)  # Valid track edge
            else:
                labels.append(0)  # False edge
        return labels

    def track_lengths(self) -> List[int]:
        """Compute the length of each track (connected component)."""
        # Build adjacency for forward traversal
        visited = set()
        lengths = []

        # Find track starts: nodes with no incoming edges
        all_targets = {e.target_id for e in self.edges}
        starts = [n.node_id for n in self.nodes if n.node_id not in all_targets]

        for start in starts:
            length = 0
            current = start
            while current is not None and current not in visited:
                visited.add(current)
                length += 1
                # Follow outgoing edges (take first for non-division)
                out_edges = self._outgoing_edges.get(current, [])
                if len(out_edges) == 1:
                    current = out_edges[0].target_id
                elif len(out_edges) >= 2:
                    # Division: count each branch separately later
                    for edge in out_edges:
                        if edge.target_id not in visited:
                            # Start a new track from each daughter
                            sub_length = 0
                            sub_current = edge.target_id
                            while sub_current is not None and sub_current not in visited:
                                visited.add(sub_current)
                                sub_length += 1
                                sub_out = self._outgoing_edges.get(sub_current, [])
                                sub_current = sub_out[0].target_id if len(sub_out) == 1 else None
                            if sub_length > 0:
                                lengths.append(length + sub_length)
                    current = None
                else:
                    current = None
            if length > 0 and current is None:
                lengths.append(length)

        return lengths

    def summary(self) -> Dict:
        """Summary statistics of the ground truth graph."""
        track_lens = self.track_lengths()
        nodes_per_frame = [
            len(self.get_nodes_at_frame(t)) for t in self.frames
        ]
        return {
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
            "n_divisions": self.n_divisions,
            "n_annotated_frames": self.n_frames,
            "frame_range": self.frame_range,
            "avg_nodes_per_frame": np.mean(nodes_per_frame) if nodes_per_frame else 0,
            "n_tracks": len(track_lens),
            "avg_track_length": np.mean(track_lens) if track_lens else 0,
            "max_track_length": max(track_lens) if track_lens else 0,
        }


class GeffLoader:
    """
    Load GEFF ground truth files.
    
    Usage:
        loader = GeffLoader("path/to/sample.geff")
        graph = loader.load()
        print(graph.summary())
    """

    def __init__(self, geff_path: str):
        self.geff_path = Path(geff_path)
        if not self.geff_path.exists():
            raise FileNotFoundError(f"GEFF directory not found: {self.geff_path}")

    def load(self) -> GeffGraph:
        """Load the full GEFF graph."""
        store = zarr.open(str(self.geff_path), mode="r")

        # Load node IDs
        node_ids = np.asarray(store["nodes"]["ids"])

        # Load node properties (coordinates)
        t_vals = np.asarray(store["nodes"]["props"]["t"]["values"])
        z_vals = np.asarray(store["nodes"]["props"]["z"]["values"])
        y_vals = np.asarray(store["nodes"]["props"]["y"]["values"])
        x_vals = np.asarray(store["nodes"]["props"]["x"]["values"])

        # Build node list
        nodes = []
        for i in range(len(node_ids)):
            nodes.append(GeffNode(
                node_id=int(node_ids[i]),
                t=int(t_vals[i]),
                z=int(z_vals[i]),
                y=int(y_vals[i]),
                x=int(x_vals[i]),
            ))

        # Load edges
        edge_ids = np.asarray(store["edges"]["ids"])  # shape (E, 2)
        edges = []
        for i in range(len(edge_ids)):
            edges.append(GeffEdge(
                source_id=int(edge_ids[i, 0]),
                target_id=int(edge_ids[i, 1]),
            ))

        # Parse metadata from zarr.json
        metadata = {}
        zarr_json_path = self.geff_path / "zarr.json"
        if zarr_json_path.exists():
            with open(zarr_json_path, "r") as f:
                root_meta = json.load(f)
            geff_meta = root_meta.get("attributes", {}).get("geff", {})
            metadata = {
                "geff_version": geff_meta.get("geff_version"),
                "directed": geff_meta.get("directed", True),
                "axes": geff_meta.get("axes", []),
                "estimated_nodes": geff_meta.get("extra", {}).get(
                    "estimated_number_of_nodes"
                ),
            }

        return GeffGraph(nodes=nodes, edges=edges, metadata=metadata)

    @property
    def dataset_name(self) -> str:
        return self.geff_path.stem.replace(".geff", "")
