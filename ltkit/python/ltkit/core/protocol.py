# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Michael King — part of ltkit (https://github.com/BTFProdigy/ltkit)
# Licensed under the Apache License, Version 2.0. See the LICENSE and NOTICE
# files in the project root. Attribution must be retained in derivative works.
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

    def reinit(self, seed: int | None = None) -> None:
        """Optional: re-draw all weights from a fresh random initialization
        (re-applying any active masks). Required only for the RANDOM_REINIT
        rewind control; backends that omit it cannot use that policy."""


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
    RANDOM_REINIT = "random_reinit"
    NONE = "none"