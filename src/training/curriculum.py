"""
Curriculum learning scheduler for GNN training.

Phases:
  1. t→t+1 only (basic matching)
  2. t→t+1 + divisions
  3. t→t+1 + t→t+2 (gap-closing)
  4. Full graph + hard negative mining
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class CurriculumConfig:
    """Configuration for each curriculum phase."""
    max_frame_gap: int = 1
    include_divisions: bool = False
    hard_negative_mining: bool = False
    description: str = ""


CURRICULUM_PHASES = [
    CurriculumConfig(
        max_frame_gap=1, include_divisions=False,
        hard_negative_mining=False,
        description="Phase 1: t→t+1 only, basic cell matching",
    ),
    CurriculumConfig(
        max_frame_gap=1, include_divisions=True,
        hard_negative_mining=False,
        description="Phase 2: t→t+1 + divisions",
    ),
    CurriculumConfig(
        max_frame_gap=2, include_divisions=True,
        hard_negative_mining=False,
        description="Phase 3: t→t+1 + t→t+2 gap-closing",
    ),
    CurriculumConfig(
        max_frame_gap=2, include_divisions=True,
        hard_negative_mining=True,
        description="Phase 4: Full graph + hard negative mining",
    ),
]


class CurriculumScheduler:
    """
    Manages curriculum phase transitions during training.
    
    Advances to the next phase when validation loss plateaus.
    """

    def __init__(
        self,
        patience: int = 5,
        min_epochs_per_phase: int = 8,
    ):
        self.phase = 0
        self.patience = patience
        self.min_epochs_per_phase = min_epochs_per_phase

        self._best_loss = float("inf")
        self._wait = 0
        self._epochs_in_phase = 0

    @property
    def config(self) -> CurriculumConfig:
        """Current phase configuration."""
        return CURRICULUM_PHASES[min(self.phase, len(CURRICULUM_PHASES) - 1)]

    @property
    def is_final_phase(self) -> bool:
        return self.phase >= len(CURRICULUM_PHASES) - 1

    def step(self, val_loss: float) -> bool:
        """
        Check if we should advance to the next phase.

        Args:
            val_loss: Current validation loss.

        Returns:
            True if phase was advanced.
        """
        self._epochs_in_phase += 1

        if val_loss < self._best_loss:
            self._best_loss = val_loss
            self._wait = 0
        else:
            self._wait += 1

        # Advance phase if:
        # 1. Loss has plateaued (patience exceeded)
        # 2. We've spent enough epochs in this phase
        # 3. We're not in the final phase
        if (
            self._wait >= self.patience
            and self._epochs_in_phase >= self.min_epochs_per_phase
            and not self.is_final_phase
        ):
            self.phase += 1
            self._best_loss = float("inf")  # Reset for new phase
            self._wait = 0
            self._epochs_in_phase = 0
            print(f"\n→ Advancing to {self.config.description}")
            return True

        return False

    def state_dict(self) -> Dict:
        return {
            "phase": self.phase,
            "best_loss": self._best_loss,
            "wait": self._wait,
            "epochs_in_phase": self._epochs_in_phase,
        }

    def load_state_dict(self, state: Dict):
        self.phase = state["phase"]
        self._best_loss = state["best_loss"]
        self._wait = state["wait"]
        self._epochs_in_phase = state["epochs_in_phase"]
