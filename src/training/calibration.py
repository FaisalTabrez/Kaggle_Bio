"""
Post-hoc temperature scaling for confidence calibration.

After GNN training, optimize a single temperature parameter T
on the validation set so that softmax(logits / T) produces
well-calibrated probabilities.

These calibrated probabilities are then used as -log(P) costs
in the Hungarian assignment, which is much more stable than raw logits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import LBFGS
from typing import Tuple


class TemperatureScaler(nn.Module):
    """
    Learn a single temperature T on validation logits.
    
    calibrated_probs = softmax(logits / T)
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits by learned temperature."""
        return logits / self.temperature

    def calibrate(
        self,
        val_logits: torch.Tensor,
        val_labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 50,
    ) -> float:
        """
        Optimize temperature on validation data.

        Args:
            val_logits: (E, C) logits from the GNN.
            val_labels: (E,) integer labels.
            lr: Learning rate for LBFGS.
            max_iter: Maximum optimization iterations.

        Returns:
            Optimal temperature value.
        """
        nll_criterion = nn.CrossEntropyLoss()
        optimizer = LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval_closure():
            optimizer.zero_grad()
            scaled = self.forward(val_logits)
            loss = nll_criterion(scaled, val_labels)
            loss.backward()
            return loss

        optimizer.step(eval_closure)

        return self.temperature.item()

    def calibrated_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """Get calibrated class probabilities."""
        with torch.no_grad():
            return F.softmax(self.forward(logits), dim=1)
