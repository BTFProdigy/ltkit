"""EXP-5: extra seeds (5-14) on parity-IMP and MNIST-d256 to probe variance.
Resolves #7 (small-n statistical fragility).
"""
import json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from models import TernaryMLP, TernaryKAN
from training.trainer import train_iterative
from tasks import parity_task, _split

OUT = os.path.join(os.path.dirname(__file__), "results", "exp5_more_seeds.json")
EXTRA_SEEDS = list(range(5, 15))


def mnist_d256(seed, n_samples=1000):
    from torchvision import datasets, transforms
    ds = datasets.MNIST(root="/tmp/mnist", train=True, download=True,
                        transform=transforms.ToTensor())
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), size=n_samples, replace=False)
    X = ds.data[idx].float().reshape(-1, 784).numpy() / 255.0
    proj_rng = np.random.default_rng(12345)
    P = proj_rng.standard_normal((784, 256)).astype(np.float32) / np.sqrt(784.0)
    X = (X @ P).astype(np.float32)
    y = ds.targets[idx].numpy().astype(np.int64)
    Xtr, ytr, Xv, yv = _split(X, y, seed=seed)
    return (torch.tensor(Xtr), torch.tensor(ytr, dtype=torch.long),
            torch.tensor(Xv), torch.tensor(yv, dtype=torch.long))


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    arch_map = {"mlp": TernaryMLP, "kan": TernaryKAN}
    tasks = [
        ("parity", lambda s: parity_task(seed=s)[:4], [5, 30, 10, 2], [30, 10]),
        ("mnist_d256", mnist_d256, [256, 64, 32, 10], [64, 32]),
    ]
    results = []
    total = len(tasks) * 2 * len(EXTRA_SEEDS)
    i = 0
    for task_name, loader, dims, _hidden in tasks:
        for arch in ["mlp", "kan"]:
            for seed in EXTRA_SEEDS:
                i += 1
                torch.manual_seed(seed); np.random.seed(seed)
                if task_name == "parity":
                    Xtr, ytr, Xv, yv = loader(seed)
                else:
                    Xtr, ytr, Xv, yv = loader(seed)
                kwargs = {"dims": dims}
                if arch == "kan":
                    kwargs["basis_order"] = 6
                t0 = time.time()
                r = train_iterative(
                    arch_map[arch], kwargs, Xtr, ytr, Xv, yv, "classification",
                    n_rounds=4, epochs_per=50, verbose=False, lambda1=1e-3,
                )
                elapsed = time.time() - t0
                rec = {
                    "task": task_name, "arch": arch, "seed": seed,
                    "final_val": float(r["val_curve"][-1]),
                    "final_sparsity": float(r["sparsity_curve"][-1]),
                    "elapsed_s": elapsed,
                }
                results.append(rec)
                with open(OUT, "w") as f:
                    json.dump(results, f, indent=2)
                print(f"[{i}/{total}] {task_name} {arch} seed={seed} "
                      f"val={rec['final_val']:.4f} sp={rec['final_sparsity']:.3f} "
                      f"t={elapsed:.1f}s", flush=True)
    print()
    by = {}
    for r in results:
        by.setdefault((r["task"], r["arch"]), []).append(r["final_val"])
    for (t, a), v in sorted(by.items()):
        print(f"{t:12s} {a:4s} n={len(v):2d} mean={np.mean(v):.4f} std={np.std(v):.4f}")


if __name__ == "__main__":
    main()
