"""
Edge-oriented Message Passing Network (EdgeMPN) for cell tracking.

Inspired by Ben-Haim & Raviv (ECCV 2022). Performs mutual updates
of node and edge features through message passing layers, then
classifies each edge as: false (0), valid track (1), or division (2).

Key design:
  - Edge features are first-class citizens (not just attention weights)
  - Node and edge features are updated together in each layer
  - Residual connections for gradient flow
  - Two output heads: 3-class logits + calibrated confidence
  - Crop encoder is integrated for end-to-end training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.mlp import MLP, EdgeMLP, NodeMLP
from src.features.crop_encoder import CropEncoder, normalize_crop


class EdgeMPNLayer(nn.Module):
    """
    Single layer of Edge-oriented Message Passing.
    
    1. Edge update:  h_e' = MLP([h_e || h_v_src || h_v_dst]) + h_e
    2. Aggregate:    m_v  = mean({h_e' : e ∈ neighbors(v)})
    3. Node update:  h_v' = MLP([h_v || m_v]) + h_v
    """

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()

        self.edge_update = EdgeMLP(
            edge_dim=edge_dim, node_dim=node_dim,
            out_dim=edge_dim, hidden_dim=hidden_dim, dropout=dropout,
        )
        self.node_update = NodeMLP(
            node_dim=node_dim, msg_dim=edge_dim,
            out_dim=node_dim, hidden_dim=hidden_dim, dropout=dropout,
        )

        # Layer norms for stability
        self.edge_norm = nn.LayerNorm(edge_dim)
        self.node_norm = nn.LayerNorm(node_dim)

    def forward(
        self,
        h_v: torch.Tensor,      # (N, node_dim)
        h_e: torch.Tensor,      # (E, edge_dim)
        edge_index: torch.Tensor,  # (2, E)
    ) -> tuple:
        """
        Args:
            h_v: Node features (N, node_dim).
            h_e: Edge features (E, edge_dim).
            edge_index: (2, E) with [source_indices, target_indices].

        Returns:
            Tuple of updated (h_v, h_e).
        """
        src, dst = edge_index[0], edge_index[1]

        # 1. Edge update
        h_e_new = self.edge_update(h_e, h_v[src], h_v[dst])
        h_e = self.edge_norm(h_e + h_e_new)  # Residual + norm

        # 2. Aggregate messages to nodes (mean pooling over incoming edges)
        # Aggregate to both source and destination nodes
        msg_dst = scatter_mean(h_e, dst, dim=0, dim_size=h_v.shape[0])
        msg_src = scatter_mean(h_e, src, dim=0, dim_size=h_v.shape[0])
        m_v = msg_dst + msg_src  # Both directions contribute

        # 3. Node update
        h_v_new = self.node_update(h_v, m_v)
        h_v = self.node_norm(h_v + h_v_new)  # Residual + norm

        return h_v, h_e


class EdgeMPN(nn.Module):
    """
    Complete EdgeMPN model for cell tracking edge classification.

    Architecture:
        CropEncoder(crops) → crop_embed
        h_v = MLP_proj_node(concat(handcrafted, crop_embed))
        h_e = MLP_proj_edge(edge_features)
        for layer in EdgeMPN_layers:
            h_v, h_e = layer(h_v, h_e, edge_index)
        logits = MLP_cls([h_e || h_v_src || h_v_dst])  → 3-class
        conf = MLP_conf([h_e || h_v_src || h_v_dst])  → sigmoid
    """

    def __init__(
        self,
        handcrafted_dim: int = 32,
        crop_embed_dim: int = 48,
        edge_input_dim: int = 20,
        hidden_dim: int = 128,
        n_layers: int = 4,
        n_classes: int = 3,
        dropout: float = 0.1,
        use_crops: bool = True,
    ):
        super().__init__()
        self.use_crops = use_crops
        self.n_classes = n_classes

        # Crop encoder (jointly trained)
        if use_crops:
            self.crop_encoder = CropEncoder(embed_dim=crop_embed_dim)
            node_input_dim = handcrafted_dim + crop_embed_dim
        else:
            self.crop_encoder = None
            node_input_dim = handcrafted_dim

        # Input projections
        self.node_proj = MLP(
            node_input_dim, hidden_dim, hidden_dim,
            n_layers=2, dropout=dropout, residual=False,
        )
        self.edge_proj = MLP(
            edge_input_dim, hidden_dim, hidden_dim,
            n_layers=2, dropout=dropout, residual=False,
        )

        # Message passing layers
        self.layers = nn.ModuleList([
            EdgeMPNLayer(hidden_dim, hidden_dim, hidden_dim, dropout)
            for _ in range(n_layers)
        ])

        # Classification head: 3-class (false / track / division)
        classifier_input = hidden_dim * 3  # [h_e || h_v_src || h_v_dst]
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

        # Confidence head: calibrated P(correct edge)
        self.confidence_head = nn.Sequential(
            nn.Linear(classifier_input, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x_handcrafted: torch.Tensor,  # (N, handcrafted_dim)
        edge_attr: torch.Tensor,       # (E, edge_input_dim)
        edge_index: torch.Tensor,      # (2, E)
        crops: torch.Tensor = None,    # (N, 1, 8, 16, 16) or None
    ) -> dict:
        """
        Forward pass.

        Args:
            x_handcrafted: Handcrafted node features (N, 32).
            edge_attr: Edge features (E, 20).
            edge_index: Edge indices (2, E).
            crops: Optional cell image crops (N, 1, 8, 16, 16).

        Returns:
            Dict with:
                'logits': (E, 3) — class logits per edge
                'confidence': (E, 1) — calibrated confidence per edge
                'h_v': (N, hidden_dim) — final node embeddings
                'h_e': (E, hidden_dim) — final edge embeddings
        """
        # Build node features
        if self.use_crops and crops is not None:
            crop_embed = self.crop_encoder(normalize_crop(crops))
            x = torch.cat([x_handcrafted, crop_embed], dim=-1)
        else:
            x = x_handcrafted

        # Project to hidden dimension
        h_v = self.node_proj(x)      # (N, hidden_dim)
        h_e = self.edge_proj(edge_attr)  # (E, hidden_dim)

        # Message passing
        for layer in self.layers:
            h_v, h_e = layer(h_v, h_e, edge_index)

        # Edge classification
        src, dst = edge_index[0], edge_index[1]
        edge_repr = torch.cat([h_e, h_v[src], h_v[dst]], dim=-1)

        logits = self.classifier(edge_repr)       # (E, 3)
        confidence = self.confidence_head(edge_repr)  # (E, 1)

        return {
            "logits": logits,
            "confidence": confidence.squeeze(-1),
            "h_v": h_v,
            "h_e": h_e,
        }

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def summary(self) -> str:
        crop_params = self.crop_encoder.n_params if self.crop_encoder else 0
        total = self.n_params
        return (
            f"EdgeMPN: {total:,} params "
            f"(crop_encoder: {crop_params:,}, "
            f"layers: {len(self.layers)}, "
            f"hidden: {self.layers[0].edge_norm.normalized_shape[0]})"
        )
