# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Michael King — part of ltkit (https://github.com/BTFProdigy/ltkit)
# Licensed under the Apache License, Version 2.0. See the LICENSE and NOTICE
# files in the project root. Attribution must be retained in derivative works.
"""End-to-end smoke test: IMP engine + TorchBackend on a tiny classification task.

Verifies the contract holds: sparsity rises monotonically, pruned weights stay
exactly zero across rounds (mask-persistence invariant), and the run produces a
result without touching a tensor inside the engine.
"""
import numpy as np
import torch
import torch.nn as nn

from ltkit import IMPConfig, RewindPolicy, Criterion, run_imp
from ltkit.backends import TorchBackend


def _make_W(d=16, k=3):
    return torch.randn(d, k, generator=torch.Generator().manual_seed(42))


def _make_data(W, seed=0, n=256):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, W.shape[0], generator=g)
    y = (X @ W).argmax(dim=1)
    return X, y


def test_imp_torch_smoke():
    torch.manual_seed(0)
    np.random.seed(0)
    W = _make_W()
    X, y = _make_data(W, seed=0)
    Xv, yv = _make_data(W, seed=1)
    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 3))
    backend = TorchBackend(model, task="classification", lr=1e-2, batch_size=64)

    cfg = IMPConfig(rounds=4, prune_rate=0.3, epochs_per=15,
                    criterion=Criterion.MAGNITUDE, rewind=RewindPolicy.INIT,
                    scope="global")
    res = run_imp(backend, (X, y), (Xv, yv), cfg)

    sparsities = [h["sparsity"] for h in res.history]
    assert sparsities[0] == 0.0, sparsities
    assert all(b >= a for a, b in zip(sparsities, sparsities[1:])), sparsities
    assert sparsities[-1] > 0.5, sparsities

    # mask-persistence invariant: every pruned weight is exactly zero
    for name, param in backend.parameter_groups().items():
        m = res.masks[name].reshape(param.shape)
        pruned = param.detach().cpu().numpy()[~m]
        assert np.all(pruned == 0.0), (name, pruned[pruned != 0][:5])

    # ticket should still classify above chance (0.33)
    assert res.history[-1]["metric"] > 0.4, res.history


if __name__ == "__main__":
    test_imp_torch_smoke()
    print("smoke OK")