# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Michael King — part of ltkit (https://github.com/BTFProdigy/ltkit)
# Licensed under the Apache License, Version 2.0. See the LICENSE and NOTICE
# files in the project root. Attribution must be retained in derivative works.
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