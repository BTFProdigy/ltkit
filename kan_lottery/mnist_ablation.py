"""MNIST fairness ablation: sweep input dim × projection seed × architecture × model seed.

Tests whether KAN's MNIST deficit (KAN .699 vs MLP .804 at d=64) survives:
  - higher input dim (128, 256, 784=identity)
  - different random projections
"""
import os, sys, json, time
import numpy as np
import torch
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(__file__))
from models import TernaryMLP, TernaryKAN
from training.trainer import train_iterative
from tasks import _split

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
OUT_JSON = os.path.join(RESULTS_DIR, "mnist_ablation.json")

# ── Sweep grid ───────────────────────────────────────────────────────────────
INPUT_DIMS    = [64, 128, 256, 784]
PROJ_SEEDS    = [12345, 67890]   # ignored for dim=784 (identity)
MODEL_SEEDS   = [0, 1, 2]
ARCHS         = ["mlp", "kan"]
N_SAMPLES     = 1000
HIDDEN        = [64, 32]
IMP_ROUNDS    = 4
EPOCHS_PER    = 50
LAMBDA1       = 1e-3
BASIS_ORDER   = 6


def load_mnist_raw(n_samples, model_seed):
    ds = datasets.MNIST(root="/tmp/mnist", train=True, download=True,
                        transform=transforms.ToTensor())
    rng = np.random.default_rng(model_seed)
    idx = rng.choice(len(ds), size=min(n_samples, len(ds)), replace=False)
    X = ds.data[idx].float().reshape(-1, 784).numpy() / 255.0
    y = ds.targets[idx].numpy().astype(np.int64)
    return X.astype(np.float32), y


def project(X, in_dim, proj_seed):
    if in_dim == 784:
        return X
    proj_rng = np.random.default_rng(proj_seed)
    P = (proj_rng.standard_normal((784, in_dim)).astype(np.float32)
         / np.sqrt(784.0))
    return (X @ P).astype(np.float32)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_one(arch, in_dim, proj_seed, model_seed):
    torch.manual_seed(model_seed)
    np.random.seed(model_seed)

    X_raw, y = load_mnist_raw(N_SAMPLES, model_seed)
    X = project(X_raw, in_dim, proj_seed)
    Xtr, ytr, Xv, yv = _split(X, y, seed=model_seed)
    Xtr = torch.tensor(Xtr); ytr = torch.tensor(ytr, dtype=torch.long)
    Xv  = torch.tensor(Xv);  yv  = torch.tensor(yv,  dtype=torch.long)

    dims = [in_dim] + HIDDEN + [10]
    if arch == "mlp":
        cls, kwargs = TernaryMLP, {"dims": dims}
    else:
        cls, kwargs = TernaryKAN, {"dims": dims, "basis_order": BASIS_ORDER}

    t0 = time.time()
    res = train_iterative(
        cls, kwargs, Xtr, ytr, Xv, yv, "classification",
        n_rounds=IMP_ROUNDS, epochs_per=EPOCHS_PER,
        verbose=False, lambda1=LAMBDA1,
    )
    elapsed = time.time() - t0
    model = res["final_model"]
    return {
        "arch": arch, "in_dim": in_dim, "proj_seed": proj_seed,
        "model_seed": model_seed,
        "dims": dims,
        "n_params": count_params(model),
        "final_val": float(res["val_curve"][-1]),
        "final_sparsity": float(res["sparsity_curve"][-1]),
        "val_curve": [float(v) for v in res["val_curve"]],
        "sp_curve":  [float(s) for s in res["sparsity_curve"]],
        "elapsed_s": elapsed,
    }


def main():
    all_runs = []
    configs = []
    for arch in ARCHS:
        for in_dim in INPUT_DIMS:
            proj_seeds = [0] if in_dim == 784 else PROJ_SEEDS
            for ps in proj_seeds:
                for ms in MODEL_SEEDS:
                    configs.append((arch, in_dim, ps, ms))

    print(f"Total configs: {len(configs)}")
    print(f"Grid: arch={ARCHS}, dim={INPUT_DIMS}, proj_seeds={PROJ_SEEDS}, model_seeds={MODEL_SEEDS}\n")

    t_global = time.time()
    for i, (arch, in_dim, ps, ms) in enumerate(configs, 1):
        tag = f"[{i:2d}/{len(configs)}] arch={arch} d={in_dim:3d} proj={ps:5d} seed={ms}"
        print(tag, end="", flush=True)
        try:
            r = run_one(arch, in_dim, ps, ms)
            all_runs.append(r)
            print(f"  val={r['final_val']:.4f}  sp={r['final_sparsity']:.3f}  "
                  f"params={r['n_params']:>6d}  t={r['elapsed_s']:.1f}s")
        except Exception as e:
            print(f"  ERROR: {e}")

        # Periodic save
        with open(OUT_JSON, "w") as f:
            json.dump(all_runs, f, indent=2)

    print(f"\nTotal time: {(time.time()-t_global)/60:.1f} min")
    print(f"Saved → {OUT_JSON}")


if __name__ == "__main__":
    main()
