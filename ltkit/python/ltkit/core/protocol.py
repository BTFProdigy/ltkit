from __future__ import annotations

from enum import Enum
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class PrunableModel(Protocol):
    """Backend contract for the six verbs: parameter_groups, scores, apply_mask, snapshot, restore, fit, evaluate."""

    def parameter_groups(self) -> dict[str, object]:
        """Enumerate prunable parameter groups."""

    def scores(self, name: str, criterion: "Criterion") -> np.ndarray:
        """Return per-element importance scores for one group."""

    def apply_mask(self, name: str, mask: np.ndarray) -> None:
        """Apply a mask to one group and keep pruned elements zero."""

    def snapshot(self) -> object:
        """Capture an opaque model state for rewind."""

    def restore(self, state: object) -> None:
        """Restore a prior snapshot state."""

    def fit(self, data, epochs: int) -> None:
        """Train the model in place for the given epochs."""

    def evaluate(self, data) -> float:
        """Evaluate the model and return a metric."""


class Criterion(Enum):
    """Score criterion for the scores verb."""

    MAGNITUDE = "magnitude"
    GATE = "gate"
    RANDOM = "random"
    SNIP = "snip"


class RewindPolicy(Enum):
    """Restore target for the snapshot/restore verbs."""

    INIT = "init"
    EARLY_K = "early_k"
    NONE = "none"
