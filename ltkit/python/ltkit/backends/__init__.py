# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Michael King — part of ltkit (https://github.com/BTFProdigy/ltkit)
# Licensed under the Apache License, Version 2.0. See the LICENSE and NOTICE
# files in the project root. Attribution must be retained in derivative works.
"""Backend adapters implementing the PrunableModel contract.

Imports are lazy: each backend pulls in a heavy optional dependency (torch,
tensorflow, jax), so they are only imported on demand to keep ``import ltkit``
free of hard framework requirements.
"""

__all__ = ["TorchBackend", "KerasBackend", "JaxBackend"]


def __getattr__(name):
    if name == "TorchBackend":
        from .torch_backend import TorchBackend
        return TorchBackend
    if name == "KerasBackend":
        from .keras_backend import KerasBackend
        return KerasBackend
    if name == "JaxBackend":
        from .jax_backend import JaxBackend
        return JaxBackend
    raise AttributeError(name)