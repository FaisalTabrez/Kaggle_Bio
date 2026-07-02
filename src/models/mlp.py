"""
Shared MLP building blocks for the GNN.

All MLPs use BatchNorm + ReLU + Dropout with residual connections
where input and output dimensions match.
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Multi-layer perceptron with batch norm, ReLU, and dropout.
    
    Supports residual connection when in_dim == out_dim.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        n_layers: int = 2,
        dropout: float = 0.1,
        batch_norm: bool = True,
        residual: bool = True,
    ):
        super().__init__()
        self.residual = residual and (in_dim == out_dim)

        layers = []
        dims = [in_dim] + [hidden_dim] * (n_layers - 1) + [out_dim]

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # No activation/norm on last layer
                if batch_norm:
                    layers.append(nn.BatchNorm1d(dims[i + 1]))
                layers.append(nn.ReLU(inplace=True))
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        if self.residual:
            out = out + x
        return out


class EdgeMLP(nn.Module):
    """
    MLP for edge feature updates.
    
    Takes concatenated [h_e || h_v_src || h_v_dst] and outputs
    updated edge features.
    """

    def __init__(
        self,
        edge_dim: int,
        node_dim: int,
        out_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        in_dim = edge_dim + 2 * node_dim  # [h_e || h_v_src || h_v_dst]
        self.mlp = MLP(
            in_dim, hidden_dim, out_dim,
            n_layers=2, dropout=dropout, residual=False,
        )

    def forward(
        self,
        h_e: torch.Tensor,
        h_v_src: torch.Tensor,
        h_v_dst: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            h_e: (E, edge_dim)
            h_v_src: (E, node_dim) — source node features indexed by edge
            h_v_dst: (E, node_dim) — target node features indexed by edge
            
        Returns:
            (E, out_dim) updated edge features.
        """
        x = torch.cat([h_e, h_v_src, h_v_dst], dim=-1)
        return self.mlp(x)


class NodeMLP(nn.Module):
    """
    MLP for node feature updates.
    
    Takes concatenated [h_v || m_v] (node features + aggregated messages)
    and outputs updated node features.
    """

    def __init__(
        self,
        node_dim: int,
        msg_dim: int,
        out_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        in_dim = node_dim + msg_dim  # [h_v || m_v]
        self.mlp = MLP(
            in_dim, hidden_dim, out_dim,
            n_layers=2, dropout=dropout, residual=False,
        )

    def forward(
        self,
        h_v: torch.Tensor,
        m_v: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            h_v: (N, node_dim)
            m_v: (N, msg_dim) — aggregated messages from neighbors
            
        Returns:
            (N, out_dim) updated node features.
        """
        x = torch.cat([h_v, m_v], dim=-1)
        return self.mlp(x)
