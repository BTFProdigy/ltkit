"""EXP-7: High-dim tabular classification (50 features).
Resolves item C: KAN vs MLP on non-image, non-toy tabular data.
Uses sklearn make_classification (50 features, 20 informative, 5000 samples, binary).
"""
import json, os, sys, time
import numpy as np
import torch
from sklearn.datasets import make_classification

sys.path.insert(0, os.path.dirname(__file__))
from models import TernaryMLP, TernaryKAN
from training.trainer import train_iterative
from tasks import _split

OUT = os.path.join(os.path.dirname(__file__), "results", "exp7_tabular.json")


def tabular_task(seed=0):
    X, y = make_classification(
        n_samples=5000, n_features=50, n_informative=20,
        n_redundant=10, n_classes=2, class_sep=1.0, random_state=seed,
    )
    X = X.astype(np.float32)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    y = y.astype(np.int64)
    Xtr, ytr, Xv, yv = _split(X, y, seed=seed)
    return (torch.tensor(Xtr), torch.tensor(ytr, dtype=torch.long),
            torch.tensor(Xv), torch.tensor(yv, dtype=torch.long))


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    archs = ["mlp", "kan"]
    seeds = list(range(5))
    hidden = [64, 32]
    dims = [50] + hidden + [2]
    arch_map = {"mlp": TernaryMLP, "kan": TernaryKAN}
    results = []
    total = len(archs) * len(seeds)
    i = 0
    for arch in archs:
        for seed in seeds:
            i += 1
            torch.manual_seed(seed)
            np.random.seed(seed)
            Xtr, ytr, Xv, yv = tabular_task(seed=seed)
            kwargs = {"dims": dims}
            if arch == "kan":
                kwargs["basis_order"] = 6
            t0 = time.time()
            run = train_iterative(
                arch_map[arch], kwargs, Xtr, ytr, Xv, yv, "classification",
                n_rounds=4, epochs_per=50, verbose=False, lambda1=1e-3,
            )
            elapsed = time.time() - t0
            rec = {
                "arch": arch, "seed": seed,
                "final_val": float(run["val_curve"][-1]),
                "final_sparsity": float(run["sparsity_curve"][-1]),
                "val_curve": [float(v) for v in run["val_curve"]],
                "sp_curve": [float(s) for s in run["sparsity_curve"]],
                "elapsed_s": elapsed,
            }
            results.append(rec)
            with open(OUT, "w") as f:
                json.dump(results, f, indent=2)
            print(f"[{i}/{total}] {arch} seed={seed} "
                  f"val={rec['final_val']:.4f} sp={rec['final_sparsity']:.3f} "
                  f"t={elapsed:.1f}s", flush=True)
    print()
    for arch in archs:
        vals = [r["final_val"] for r in results if r["arch"] == arch]
        print(f"{arch}: mean={np.mean(vals):.4f} std={np.std(vals):.4f}")


if __name__ == "__main__":
    main()
