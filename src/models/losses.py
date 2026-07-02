"""
Loss functions for the cell tracking GNN.

- FocalLoss: handles extreme class imbalance (most edges are false)
- ConfidenceBCE: trains the dedicated confidence head
- SiblingConsistencyLoss: encourages consistent division predictions
- CombinedLoss: weighted combination of all losses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification with class imbalance.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    For cell tracking:
        class 0 (false edge): very common → low alpha
        class 1 (valid track): common → medium alpha
        class 2 (division): rare → high alpha
    """

    def __init__(
        self,
        alpha: Optional[List[float]] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        if alpha is None:
            alpha = [0.1, 0.6, 0.3]  # Default for cell tracking
        self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (E, C) raw class logits.
            targets: (E,) integer class labels.

        Returns:
            Scalar loss.
        """
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=logits.shape[1]).float()

        # Per-class focal weight
        p_t = (probs * targets_one_hot).sum(dim=1)  # P(true class)
        focal_weight = (1 - p_t) ** self.gamma

        # Alpha weighting
        alpha_t = self.alpha[targets]

        # Cross-entropy
        ce = F.cross_entropy(logits, targets, reduction="none")

        loss = alpha_t * focal_weight * ce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class ConfidenceBCE(nn.Module):
    """
    BCE loss for the confidence head.
    
    Target: 1.0 if the edge is a true positive (class 1 or 2), 0.0 otherwise.
    This trains the model to output calibrated P(correct edge).
    """

    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(
        self,
        confidence: torch.Tensor,  # (E,) sigmoid outputs
        targets: torch.Tensor,     # (E,) integer class labels
    ) -> torch.Tensor:
        binary_targets = (targets > 0).float()
        return self.bce(confidence, binary_targets)


class SiblingConsistencyLoss(nn.Module):
    """
    Encourages consistent confidence predictions for sibling division edges.
    
    If parent → child_1 is a division edge, then parent → child_2
    should also be predicted as a division edge with similar confidence.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        confidence: torch.Tensor,  # (E,)
        logits: torch.Tensor,       # (E, 3)
        edge_index: torch.Tensor,   # (2, E)
        targets: torch.Tensor,      # (E,)
    ) -> torch.Tensor:
        """Find sibling edge pairs and compute consistency loss."""
        # Find division edges (class 2) and group by source node
        div_mask = (targets == 2)
        if div_mask.sum() < 2:
            return torch.tensor(0.0, device=confidence.device)

        src_nodes = edge_index[0][div_mask]
        div_indices = torch.where(div_mask)[0]

        # Group by source node
        loss = torch.tensor(0.0, device=confidence.device)
        n_pairs = 0

        unique_sources = src_nodes.unique()
        for src in unique_sources:
            mask = (src_nodes == src)
            sibling_indices = div_indices[mask]

            if len(sibling_indices) < 2:
                continue

            # All pairs of siblings should have similar confidence
            for i in range(len(sibling_indices)):
                for j in range(i + 1, len(sibling_indices)):
                    idx_i = sibling_indices[i]
                    idx_j = sibling_indices[j]
                    # MSE between sibling confidences
                    loss = loss + (confidence[idx_i] - confidence[idx_j]) ** 2
                    # Also encourage both to predict division
                    div_prob_i = F.softmax(logits[idx_i], dim=0)[2]
                    div_prob_j = F.softmax(logits[idx_j], dim=0)[2]
                    loss = loss + (div_prob_i - div_prob_j) ** 2
                    n_pairs += 1

        return loss / max(n_pairs, 1)


class CombinedLoss(nn.Module):
    """
    Weighted combination of all loss components.
    
    L = w_focal * FocalLoss + w_conf * ConfidenceBCE + w_sibling * SiblingLoss
    """

    def __init__(
        self,
        focal_alpha: Optional[List[float]] = None,
        focal_gamma: float = 2.0,
        w_focal: float = 1.0,
        w_conf: float = 0.5,
        w_sibling: float = 0.3,
    ):
        super().__init__()
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.conf_bce = ConfidenceBCE()
        self.sibling = SiblingConsistencyLoss()

        self.w_focal = w_focal
        self.w_conf = w_conf
        self.w_sibling = w_sibling

    def forward(
        self,
        logits: torch.Tensor,       # (E, 3)
        confidence: torch.Tensor,   # (E,)
        targets: torch.Tensor,      # (E,)
        edge_index: torch.Tensor,   # (2, E)
    ) -> dict:
        """
        Compute combined loss.

        Returns:
            Dict with 'total', 'focal', 'confidence', 'sibling' losses.
        """
        l_focal = self.focal(logits, targets)
        l_conf = self.conf_bce(confidence, targets)
        l_sibling = self.sibling(confidence, logits, edge_index, targets)

        total = (
            self.w_focal * l_focal
            + self.w_conf * l_conf
            + self.w_sibling * l_sibling
        )

        return {
            "total": total,
            "focal": l_focal.detach(),
            "confidence": l_conf.detach(),
            "sibling": l_sibling.detach(),
        }
