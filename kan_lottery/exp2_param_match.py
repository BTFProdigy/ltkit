"""EXP-2: Param-matched wide MLP vs KAN.
Resolves #5: is KAN's edge over MLP just capacity (3.5x param count)?
Wide MLP is sized to match KAN trainable-param count exactly.
Tasks: modular (N=11) and MNIST (d=64). 5 seeds each, IMP 4 rounds.
"""
import json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from models import TernaryMLP, TernaryKAN
from training.trainer import train_iterative
from tasks import _split

OUT = os.path.join(os.path.dirname(__file__), "results", "exp2_param_match.json")


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def modular_n11(seed):
    pairs = np.array([[a, b] for a in range(11) for b in range(11)], dtype=np.float32)
    X = pairs / 11.0
    y = ((pairs[:, 0] + pairs[:, 1]) % 11).astype(np.int64)
    Xtr, ytr, Xv, yv = _split(X, y, seed=seed)
    return (torch.tensor(Xtr), torch.tensor(ytr, dtype=torch.long),
            torch.tensor(Xv), torch.tensor(yv, dtype=torch.long), 11)


def mnist_d64(seed, n_samples=1000):
    from torchvision import datasets, transforms
    ds = datasets.MNIST(root="/tmp/mnist", train=True, download=True,
                        transform=transforms.ToTensor())
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), size=n_samples, replace=False)
    X = ds.data[idx].float().reshape(-1, 784).numpy() / 255.0
    proj_rng = np.random.default_rng(12345)
    P = proj_rng.standard_normal((784, 64)).astype(np.float32) / np.sqrt(784.0)
    X = (X @ P).astype(np.float32)
    y = ds.targets[idx].numpy().astype(np.int64)
    Xtr, ytr, Xv, yv = _split(X, y, seed=seed)
    return (torch.tensor(Xtr), torch.tensor(ytr, dtype=torch.long),
            torch.tensor(Xv), torch.tensor(yv, dtype=torch.long), 10)


def find_matched_dims(in_dim, out_dim, target_params):
    # Search H1 >= H2 grid to match target ~ KAN param count
    best = None
    for h1 in range(32, 401, 8):
        for h2 in range(16, h1 + 1, 8):
            m = TernaryMLP([in_dim, h1, h2, out_dim])
            n = count_params(m)
            err = abs(n - target_params)
            if best is None or err < best[0]:
                best = (err, h1, h2, n)
    return best[1], best[2], best[3]


def run_task(name, loader, hidden_kan, basis):
    results = []
    archs_cfg = []
    in_dim = out_dim = None
    Xtr0, ytr0, Xv0, yv0, out_dim = loader(0)
    in_dim = Xtr0.shape[1]
    dims_kan = [in_dim] + hidden_kan + [out_dim]
    kan = TernaryKAN(dims_kan, basis_order=basis)
    kan_params = count_params(kan)
    h1, h2, mlp_match_params = find_matched_dims(in_dim, out_dim, kan_params)
    dims_mlp_narrow = [in_dim] + hidden_kan + [out_dim]
    dims_mlp_wide = [in_dim, h1, h2, out_dim]
    archs_cfg = [
        ("kan", TernaryKAN, {"dims": dims_kan, "basis_order": basis}, kan_params),
        ("mlp_narrow", TernaryMLP, {"dims": dims_mlp_narrow},
         count_params(TernaryMLP(dims_mlp_narrow))),
        ("mlp_wide", TernaryMLP, {"dims": dims_mlp_wide}, mlp_match_params),
    ]
    print(f"\n=== task={name} in={in_dim} out={out_dim} ===")
    for tag, cls, kwargs, np_ in archs_cfg:
        print(f"  {tag}: dims={kwargs['dims']} params={np_}")
    for seed in range(5):
        torch.manual_seed(seed); np.random.seed(seed)
        Xtr, ytr, Xv, yv, _ = loader(seed)
        for tag, cls, kwargs, np_ in archs_cfg:
            torch.manual_seed(seed); np.random.seed(seed)
            t0 = time.time()
            r = train_iterative(
                cls, kwargs, Xtr, ytr, Xv, yv, "classification",
                n_rounds=4, epochs_per=50, verbose=False, lambda1=1e-3,
            )
            elapsed = time.time() - t0
            rec = {
                "task": name, "arch": tag, "seed": seed,
                "dims": kwargs["dims"], "n_params": np_,
                "final_val": float(r["val_curve"][-1]),
                "final_sparsity": float(r["sparsity_curve"][-1]),
                "elapsed_s": elapsed,
            }
            results.append(rec)
            with open(OUT, "w") as f:
                json.dump(_all_results + results, f, indent=2)
            print(f"  [{name}] {tag} seed={seed} val={rec['final_val']:.4f} "
                  f"sp={rec['final_sparsity']:.3f} t={elapsed:.1f}s", flush=True)
    return results


_all_results = []


def main():
    global _all_results
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    r1 = run_task("modular_n11", modular_n11, [30, 10], 6)
    _all_results = r1
    r2 = run_task("mnist_d64", mnist_d64, [64, 32], 6)
    _all_results = r1 + r2
    with open(OUT, "w") as f:
        json.dump(_all_results, f, indent=2)
    print("\n--- SUMMARY ---")
    by = {}
    for r in _all_results:
        by.setdefault((r["task"], r["arch"]), []).append(r["final_val"])
    for (t, a), v in sorted(by.items()):
        print(f"{t:15s} {a:10s} mean={np.mean(v):.4f} std={np.std(v):.4f}")


if __name__ == "__main__":
    main()
