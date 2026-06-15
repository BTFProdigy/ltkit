# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Michael King — part of ltkit (https://github.com/BTFProdigy/ltkit)
# Licensed under the Apache License, Version 2.0. See the LICENSE and NOTICE
# files in the project root. Attribution must be retained in derivative works.
"""End-to-end smoke test: IMP engine + KerasBackend (TensorFlow) on a tiny task.

Same invariants as the torch test: sparsity rises, the returned ticket agrees
with zeroed kernels, and no pruned weight is revived by training/rewind.
"""
import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import keras
import numpy as np

from ltkit import IMPConfig, RewindPolicy, Criterion, run_imp
from ltkit.backends import KerasBackend


def _make(W, seed, n=256):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, W.shape[0])).astype("float32")
    y = (X @ W).argmax(1).astype("int64")
    return X, y


def test_imp_keras_smoke():
    keras.utils.set_random_seed(0)
    d, k = 16, 3
    W = np.random.default_rng(42).standard_normal((d, k)).astype("float32")
    X, y = _make(W, 0)
    Xv, yv = _make(W, 1)

    model = keras.Sequential([
        keras.Input(shape=(d,)),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(k),
    ])
    backend = KerasBackend(model, task="classification", lr=1e-2, batch_size=64)

    cfg = IMPConfig(rounds=4, prune_rate=0.3, epochs_per=15,
                    criterion=Criterion.MAGNITUDE, rewind=RewindPolicy.INIT,
                    scope="global")
    res = run_imp(backend, (X, y), (Xv, yv), cfg)

    sp = [h["sparsity"] for h in res.history]
    assert sp[0] == 0.0, sp
    assert all(b >= a for a, b in zip(sp, sp[1:])), sp
    assert sp[-1] > 0.5, sp

    for name, kernel in backend.parameter_groups().items():
        m = res.masks[name].reshape(kernel.shape)
        pruned = kernel.numpy()[~m]
        assert np.all(pruned == 0.0), (name, pruned[pruned != 0][:5])

    assert res.history[-1]["metric"] > 0.45, res.history


if __name__ == "__main__":
    test_imp_keras_smoke()
    print("keras smoke OK")