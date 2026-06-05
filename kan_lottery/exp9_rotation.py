"""EXP-9: axis-discovery front-end. Tests H1 (basis mismatch / rotation):
KAN's univariate-per-axis Chebyshev edges may fail when informative dirs are
rotated off the coordinate axes (sklearn make_classification, random-proj MNIST).
Variants: plain | pca-whiten | learned full-precision pre-mix R^{dxd} init~=I."""
import json, os, sys, time, math
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_classification

sys.path.insert(0, os.path.dirname(__file__))
from models import TernaryMLP, TernaryKAN
from training.trainer import train_iterative
from tasks import _split

OUT = os.path.join(os.path.dirname(__file__), "results", "exp9_rotation.json")


def pca_whiten(Xtr, Xv):
    mu = Xtr.mean(axis=0, keepdims=True)
    Xc = Xtr - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    # whitening: project then scale by 1/singular_value (with floor)
    scale = 1.0 / (S / math.sqrt(max(Xc.shape[0] - 1, 1)) + 1e-6)
    W = Vt.T * scale  # (d, d)
    Xtr_w = (Xtr - mu) @ W
    Xv_w  = (Xv  - mu) @ W
    return Xtr_w.astype(np.float32), Xv_w.astype(np.float32)


def tabular_task(seed):
    X, y = make_classification(n_samples=5000, n_features=50, n_informative=20,
                               n_redundant=10, n_classes=2, class_sep=1.0,
                               random_state=seed)
    X = X.astype(np.float32)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    y = y.astype(np.int64)
    Xtr, ytr, Xv, yv = _split(X, y, seed=seed)
    return Xtr, ytr, Xv, yv, "classification", [64, 32], 2


def mnist_d64_task(seed):
    from torchvision import datasets, transforms
    ds = datasets.MNIST(root="/tmp/mnist", train=True, download=True,
                        transform=transforms.ToTensor())
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), size=1000, replace=False)
    X = ds.data[idx].float().reshape(-1, 784).numpy() / 255.0
    proj_rng = np.random.default_rng(12345)
    P = proj_rng.standard_normal((784, 64)).astype(np.float32) / math.sqrt(784.0)
    X = (X @ P).astype(np.float32)
    y = ds.targets[idx].numpy().astype(np.int64)
    Xtr, ytr, Xv, yv = _split(X, y, seed=seed)
    return Xtr, ytr, Xv, yv, "classification", [64, 32], 10


def run_one(arch, variant, seed, Xtr, ytr, Xv, yv, hidden, out_dim):
    if variant == "pca":
        Xtr, Xv = pca_whiten(Xtr, Xv)
    in_d = Xtr.shape[1]
    Xt_tr = torch.tensor(Xtr)
    Xt_v  = torch.tensor(Xv)
    yt_tr = torch.tensor(ytr, dtype=torch.long)
    yt_v  = torch.tensor(yv,  dtype=torch.long)

    arch_cls = TernaryMLP if arch == "mlp" else TernaryKAN
    kwargs = {"dims": [in_d] + hidden + [out_dim]}
    if arch == "kan":
        kwargs["basis_order"] = 6

    if variant == "premix":
        # Full-precision learned R^{dxd} init=I+N(0,0.01). NOT ternary — axis-discovery only.
        class _Wrapped(arch_cls):
            def __init__(self, **kw):
                super().__init__(**kw)
                d = kw["dims"][0]
                self.premix = nn.Linear(d, d, bias=False)
                with torch.no_grad():
                    self.premix.weight.copy_(torch.eye(d) + 0.01 * torch.randn(d, d))
            def forward(self, x):
                return super().forward(self.premix(x))
        cls_to_use = _Wrapped
    else:
        cls_to_use = arch_cls

    t0 = time.time()
    run = train_iterative(cls_to_use, kwargs, Xt_tr, yt_tr, Xt_v, yt_v,
                          "classification", n_rounds=4, epochs_per=50,
                          verbose=False, lambda1=1e-3)
    final_val = float(run["val_curve"][-1])
    final_sp  = float(run["sparsity_curve"][-1])
    elapsed   = time.time() - t0
    return final_val, final_sp, elapsed, in_d


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tasks    = [("tabular", tabular_task), ("mnist_d64", mnist_d64_task)]
    variants = ["plain", "pca", "premix"]
    archs    = ["mlp", "kan"]
    seeds    = list(range(5))
    results  = []
    total    = len(tasks) * len(variants) * len(archs) * len(seeds)
    i = 0
    for task_name, loader in tasks:
        for variant in variants:
            for arch in archs:
                for seed in seeds:
                    i += 1
                    torch.manual_seed(seed); np.random.seed(seed)
                    Xtr, ytr, Xv, yv, _ttype, hidden, out_dim = loader(seed)
                    final_val, final_sp, elapsed, in_d = run_one(
                        arch, variant, seed, Xtr, ytr, Xv, yv, hidden, out_dim
                    )
                    rec = {"task": task_name, "variant": variant, "arch": arch,
                           "seed": seed, "in_dim": in_d,
                           "final_val": final_val, "final_sparsity": final_sp,
                           "elapsed_s": elapsed}
                    results.append(rec)
                    with open(OUT, "w") as f:
                        json.dump(results, f, indent=2)
                    print(f"[{i}/{total}] {task_name} {variant} {arch} seed={seed} "
                          f"in={in_d} val={final_val:.4f} sp={final_sp:.3f} "
                          f"t={elapsed:.1f}s", flush=True)
    print()
    agg = {}
    for r in results:
        agg.setdefault((r["task"], r["variant"], r["arch"]), []).append(r["final_val"])
    for (t, v, a), vals in sorted(agg.items()):
        print(f"{t:12s} {v:7s} {a:4s} mean={np.mean(vals):.4f} std={np.std(vals):.4f}")


if __name__ == "__main__":
    main()
