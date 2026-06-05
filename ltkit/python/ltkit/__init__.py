"""LTKit — backend-agnostic lottery-ticket / IMP framework.

The engine drives any model through six verbs (see ../CONTRACT.md). Backends
for torch, keras/tf and jax live in ``ltkit.backends``; Rust (candle/tch) and
C++ (libtorch) ports mirror the same contract.
"""
from .core import (
    PrunableModel,
    Criterion,
    RewindPolicy,
    IMPConfig,
    IMPResult,
    run_imp,
)

__all__ = [
    "PrunableModel",
    "Criterion",
    "RewindPolicy",
    "IMPConfig",
    "IMPResult",
    "run_imp",
]
