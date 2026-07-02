"""
Hard negative mining for edge classification.

Most candidate edges are obviously wrong. Training on all edges
wastes capacity. Instead, emphasize:
  - False edges the model predicts as correct (hard negatives)
  - True edges the model predicts as incorrect (hard positives)

Schedule:
  - Epochs 1-5: all edges (warm-up)
  - Epochs 6+: 70% hard negatives, 30% random
"""

import torch
import numpy as np
from typing import Tuple, Optional


class HardNegativeSampler:
    """
    Re-samples training edges to emphasize hard cases after each epoch.
    
    All positive edges (track + division) are always kept.
    Negative edges are re-sampled: 70% hard (high false-positive score),
    30% random.
    """

    def __init__(
        self,
        hard_ratio: float = 0.7,
        warmup_epochs: int = 5,
        min_negatives: int = 100,
    ):
        self.hard_ratio = hard_ratio
        self.warmup_epochs = warmup_epochs
        self.min_negatives = min_negatives

    def sample(
        self,
        edge_labels: torch.Tensor,  # (E,) — 0, 1, or 2
        edge_logits: torch.Tensor,  # (E, 3)
        epoch: int,
    ) -> torch.Tensor:
        """
        Compute a sampling mask for the edges.

        During warm-up: all edges are used.
        After warm-up: keep all positives + 70% hard negatives + 30% random.

        Args:
            edge_labels: Ground truth labels.
            edge_logits: Model predictions (detached).
            epoch: Current training epoch.

        Returns:
            Boolean mask of shape (E,) indicating which edges to train on.
        """
        E = len(edge_labels)
        mask = torch.ones(E, dtype=torch.bool, device=edge_labels.device)

        # During warm-up, use all edges
        if epoch < self.warmup_epochs:
            return mask

        # Find positive and negative edges
        pos_mask = (edge_labels > 0)
        neg_mask = (edge_labels == 0)

        n_pos = pos_mask.sum().item()
        n_neg = neg_mask.sum().item()

        if n_neg == 0 or n_pos == 0:
            return mask

        # Target: keep roughly 2:1 neg:pos ratio, but at least min_negatives
        target_neg = max(self.min_negatives, min(n_neg, n_pos * 2))

        # Compute "hardness" of negative edges:
        # high P(track or division) = hard negative
        with torch.no_grad():
            probs = torch.softmax(edge_logits, dim=1)
            # P(positive) = P(track) + P(division)
            neg_hardness = probs[neg_mask, 1] + probs[neg_mask, 2]

        # Split into hard and random
        n_hard = int(self.hard_ratio * target_neg)
        n_random = target_neg - n_hard

        neg_indices = torch.where(neg_mask)[0]

        # Top-k hardest negatives
        if n_hard > 0 and len(neg_hardness) > 0:
            k_hard = min(n_hard, len(neg_hardness))
            hard_local_indices = neg_hardness.topk(k_hard).indices
            hard_global_indices = neg_indices[hard_local_indices]
        else:
            hard_global_indices = torch.tensor([], dtype=torch.long, device=edge_labels.device)

        # Random sample of remaining negatives
        remaining_local = set(range(len(neg_indices))) - set(hard_local_indices.tolist()) if len(hard_global_indices) > 0 else set(range(len(neg_indices)))
        remaining_local = list(remaining_local)

        if n_random > 0 and len(remaining_local) > 0:
            k_random = min(n_random, len(remaining_local))
            random_local = np.random.choice(remaining_local, size=k_random, replace=False)
            random_global_indices = neg_indices[random_local]
        else:
            random_global_indices = torch.tensor([], dtype=torch.long, device=edge_labels.device)

        # Build final mask: all positives + selected negatives
        mask = torch.zeros(E, dtype=torch.bool, device=edge_labels.device)
        mask[pos_mask] = True
        if len(hard_global_indices) > 0:
            mask[hard_global_indices] = True
        if len(random_global_indices) > 0:
            mask[random_global_indices] = True

        return mask

    def stats(
        self,
        mask: torch.Tensor,
        edge_labels: torch.Tensor,
    ) -> dict:
        """Compute statistics about the sampling."""
        selected = mask.sum().item()
        total = len(mask)
        pos_selected = (mask & (edge_labels > 0)).sum().item()
        neg_selected = (mask & (edge_labels == 0)).sum().item()

        return {
            "selected": selected,
            "total": total,
            "ratio": selected / max(total, 1),
            "pos_selected": pos_selected,
            "neg_selected": neg_selected,
            "neg_pos_ratio": neg_selected / max(pos_selected, 1),
        }
