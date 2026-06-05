"""
Task loaders for all experimental tasks.
Each loader returns (X_train, y_train, X_val, y_val, task_type)
  task_type in {"classification", "regression"}
"""
import torch
import numpy as np
from typing import Tuple

Tensor = torch.Tensor
Split  = Tuple[Tensor, Tensor, Tensor, Tensor, str]


def _split(X, y, val_frac=0.2, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_val = max(1, int(len(X) * val_frac))
    vi, ti = idx[:n_val], idx[n_val:]
    return X[ti], y[ti], X[vi], y[vi]


# ── XOR (2-bit) ──────────────────────────────────────────────────────────────

def xor_task(seed: int = 0) -> Split:
    X = torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
    y = torch.tensor([0, 1, 1, 0], dtype=torch.long)
    # Repeat to have more samples
    X = X.repeat(50, 1); y = y.repeat(50)
    Xtr, ytr, Xv, yv = _split(X.numpy(), y.numpy(), seed=seed)
    return (torch.tensor(Xtr), torch.tensor(ytr), 
            torch.tensor(Xv),  torch.tensor(yv), "classification")


# ── Parity (5-bit) ───────────────────────────────────────────────────────────

def parity_task(n_bits: int = 5, seed: int = 0) -> Split:
    all_bits = np.array([[int(b) for b in f"{i:0{n_bits}b}"]
                          for i in range(2**n_bits)], dtype=np.float32)
    labels   = (all_bits.sum(axis=1) % 2).astype(np.int64)
    # Repeat
    all_bits = np.tile(all_bits, (10, 1))
    labels   = np.tile(labels, 10)
    Xtr, ytr, Xv, yv = _split(all_bits, labels, seed=seed)
    return (torch.tensor(Xtr), torch.tensor(ytr),
            torch.tensor(Xv),  torch.tensor(yv), "classification")


# ── Modular arithmetic: predict (a+b) mod N ──────────────────────────────────

def modular_task(N: int = 5, seed: int = 0) -> Split:
    pairs = [(a, b) for a in range(N) for b in range(N)]
    X = np.array([[a/N, b/N] for a,b in pairs], dtype=np.float32)
    y = np.array([(a+b) % N for a,b in pairs], dtype=np.int64)
    X = np.tile(X, (20, 1)); y = np.tile(y, 20)
    Xtr, ytr, Xv, yv = _split(X, y, seed=seed)
    return (torch.tensor(Xtr), torch.tensor(ytr),
            torch.tensor(Xv),  torch.tensor(yv), "classification")


# ── Symbolic regression: y = sin(x) + x^2 ───────────────────────────────────

def symbolic_task(n: int = 500, seed: int = 0) -> Split:
    rng = np.random.default_rng(seed)
    x   = rng.uniform(-3, 3, size=(n,)).astype(np.float32)
    y   = (np.sin(x) + x**2).astype(np.float32)
    X   = x[:, None]
    y   = y[:, None]
    Xtr, ytr, Xv, yv = _split(X, y, seed=seed)
    return (torch.tensor(Xtr), torch.tensor(ytr),
            torch.tensor(Xv),  torch.tensor(yv), "regression")


# ── Simple PDE regression: solution of y'=xy  →  y=exp(x²/2) ────────────────

def pde_task(n: int = 500, seed: int = 0) -> Split:
    rng = np.random.default_rng(seed)
    x   = rng.uniform(-2, 2, size=(n,)).astype(np.float32)
    y   = np.exp(x**2 / 2).astype(np.float32)
    # Normalise output
    y   = (y - y.mean()) / (y.std() + 1e-8)
    X   = x[:, None]
    y   = y[:, None]
    Xtr, ytr, Xv, yv = _split(X, y, seed=seed)
    return (torch.tensor(Xtr), torch.tensor(ytr),
            torch.tensor(Xv),  torch.tensor(yv), "regression")


# ── MNIST subset ─────────────────────────────────────────────────────────────

def mnist_task(n_samples: int = 1000, seed: int = 0) -> Split:
    proj_rng = np.random.default_rng(12345)
    proj_mat = (proj_rng.standard_normal((784, 64)).astype(np.float32)
                / np.sqrt(784.0))

    try:
        from torchvision import datasets, transforms
        ds = datasets.MNIST(
            root="/tmp/mnist", train=True, download=True,
            transform=transforms.ToTensor()
        )
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(ds), size=min(n_samples, len(ds)), replace=False)
        X   = ds.data[idx].float().reshape(-1, 784) / 255.0
        X   = (X.numpy() @ proj_mat).astype(np.float32)
        y   = ds.targets[idx].numpy()
        Xtr, ytr, Xv, yv = _split(X, y, seed=seed)
        return (torch.tensor(Xtr), torch.tensor(ytr, dtype=torch.long),
                torch.tensor(Xv),  torch.tensor(yv,  dtype=torch.long),
                "classification")
    except Exception as e:
        print(f"[WARN] MNIST unavailable ({e}), falling back to synthetic 64-dim.")
        rng = np.random.default_rng(seed)
        X   = rng.standard_normal((n_samples, 784)).astype(np.float32)
        X   = (X @ proj_mat).astype(np.float32)
        y   = rng.integers(0, 10, size=n_samples).astype(np.int64)
        Xtr, ytr, Xv, yv = _split(X, y, seed=seed)
        return (torch.tensor(Xtr), torch.tensor(ytr),
                torch.tensor(Xv),  torch.tensor(yv), "classification")


# ── Gaussian regression (control / overfitting check) ────────────────────────

def gaussian_task(n: int = 300, seed: int = 0) -> Split:
    rng = np.random.default_rng(seed)
    X   = rng.standard_normal((n, 4)).astype(np.float32)
    # Random GP-like target
    w   = rng.standard_normal((4, 1)).astype(np.float32)
    y   = (X @ w + 0.1*rng.standard_normal((n, 1))).astype(np.float32)
    Xtr, ytr, Xv, yv = _split(X, y, seed=seed)
    return (torch.tensor(Xtr), torch.tensor(ytr),
            torch.tensor(Xv),  torch.tensor(yv), "regression")


# ── Registry ─────────────────────────────────────────────────────────────────

TASKS = {
    "xor":        lambda seed=0: xor_task(seed),
    "parity":     lambda seed=0: parity_task(seed=seed),
    "modular":    lambda seed=0: modular_task(seed=seed),
    "symbolic":   lambda seed=0: symbolic_task(seed=seed),
    "pde":        lambda seed=0: pde_task(seed=seed),
    "mnist":      lambda seed=0: mnist_task(seed=seed),
    "gaussian":   lambda seed=0: gaussian_task(seed=seed),
}

TASK_DIMS = {
    # (in_features, out_features)
    "xor":      (2,  2),
    "parity":   (5,  2),
    "modular":  (2,  5),
    "symbolic": (1,  1),
    "pde":      (1,  1),
    "mnist":    (64, 10),
    "gaussian": (4,  1),
}
