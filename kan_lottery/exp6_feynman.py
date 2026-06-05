"""EXP-6: Feynman-style multivariate compositional regression.
y = sin(x1*x2) + 0.3*x3^2 + log(|x4|+1) + 0.5*x5*x6   (6 active of 10 inputs)
Tests #8: KAN vs MLP on a nontrivial multivariate regression where compositional
spline structure should advantage KAN.
"""
import json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from models import TernaryMLP, TernaryKAN
from training.trainer import train_online
from tasks import _split

OUT = os.path.join(os.path.dirname(__file__), "results", "exp6_feynman.json")


def feynman_task(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.5, 1.5, size=(n, 10)).astype(np.float32)
    y = (np.sin(X[:, 0] * X[:, 1])
         + 0.3 * X[:, 2] ** 2
         + np.log(np.abs(X[:, 3]) + 1.0)
         + 0.5 * X[:, 4] * X[:, 5]).astype(np.float32)
    y = (y - y.mean()) / (y.std() + 1e-8)
    y = y[:, None]
    Xtr, ytr, Xv, yv = _split(X, y, seed=seed)
    return (torch.tensor(Xtr), torch.tensor(ytr),
            torch.tensor(Xv), torch.tensor(yv))


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    archs = ["mlp", "kan"]
    seeds = list(range(5))
    hidden = [50, 20]
    dims = [10] + hidden + [1]
    results = []
    total = len(archs) * len(seeds)
    i = 0
    for arch in archs:
        for seed in seeds:
            i += 1
            torch.manual_seed(seed)
            np.random.seed(seed)
            Xtr, ytr, Xv, yv = feynman_task(seed=seed)
            if arch == "mlp":
                model = TernaryMLP(dims)
                lam2 = 0.0
            else:
                model = TernaryKAN(dims, basis_order=6)
                lam2 = 1e-5
            t0 = time.time()
            hist = train_online(
                model, Xtr, ytr, Xv, yv, "regression",
                epochs=150, verbose=False,
                lambda1_start=1e-4, lambda1_end=5e-3, lambda2=lam2,
            )
            elapsed = time.time() - t0
            rec = {
                "arch": arch, "seed": seed,
                "final_val": float(hist["val_metric"][-1]),
                "final_sparsity": float(model.sparsity()),
                "val_curve": [float(v) for v in hist["val_metric"]],
                "sp_curve": [float(s) for s in hist["sparsity"]],
                "elapsed_s": elapsed,
            }
            results.append(rec)
            with open(OUT, "w") as f:
                json.dump(results, f, indent=2)
            print(f"[{i}/{total}] {arch} seed={seed} "
                  f"val={rec['final_val']:.5f} sp={rec['final_sparsity']:.3f} "
                  f"t={elapsed:.1f}s", flush=True)
    print()
    for arch in archs:
        vals = [r["final_val"] for r in results if r["arch"] == arch]
        print(f"{arch}: mean={np.mean(vals):.5f} std={np.std(vals):.5f}")


if __name__ == "__main__":
    main()
