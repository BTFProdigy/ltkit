import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from models import TernaryMLP, TernaryKAN
from training.trainer import train_iterative
from tasks import _split


def modular_n11(seed):
    pairs = np.array([[a, b] for a in range(11) for b in range(11)], dtype=np.float32)
    X = pairs / 11.0
    y = ((pairs[:, 0] + pairs[:, 1]) % 11).astype(np.int64)
    Xtr, ytr, Xv, yv = _split(X, y, val_frac=0.2, seed=seed)
    return (torch.tensor(Xtr), torch.tensor(ytr, dtype=torch.long),
            torch.tensor(Xv), torch.tensor(yv, dtype=torch.long))


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "exp3_modular.json")
    arch_map = {"mlp": TernaryMLP, "kan": TernaryKAN}
    archs = ["mlp", "kan"]
    seeds = list(range(15))
    hidden = [30, 10]
    dims = [2] + hidden + [11]
    configs = [(a, s) for a in archs for s in seeds]
    total = len(configs)
    results = []

    for i, (arch, seed) in enumerate(configs, start=1):
        torch.manual_seed(seed)
        np.random.seed(seed)
        Xtr, ytr, Xv, yv = modular_n11(seed)
        kwargs = {"dims": dims}
        if arch == "kan":
            kwargs["basis_order"] = 6
        t0 = time.time()
        run = train_iterative(
            arch_map[arch], kwargs, Xtr, ytr, Xv, yv, "classification",
            n_rounds=4, epochs_per=50, verbose=False, lambda1=1e-3,
        )
        elapsed_s = time.time() - t0
        record = {
            "arch": arch, "seed": seed,
            "final_val": float(run["val_curve"][-1]),
            "final_sparsity": float(run["sparsity_curve"][-1]),
            "val_curve": [float(v) for v in run["val_curve"]],
            "sp_curve": [float(s) for s in run["sparsity_curve"]],
            "elapsed_s": elapsed_s,
        }
        results.append(record)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[{i}/{total}] {arch} seed={seed} "
              f"val={record['final_val']:.4f} sp={record['final_sparsity']:.4f} "
              f"t={elapsed_s:.1f}s", flush=True)

    print()
    for arch in archs:
        vals = [r["final_val"] for r in results if r["arch"] == arch]
        print(f"{arch}: mean={np.mean(vals):.4f} std={np.std(vals):.4f} n={len(vals)}")


if __name__ == "__main__":
    main()
