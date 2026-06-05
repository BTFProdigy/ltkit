"""EXP-15 (H5): Chebyshev basis_order sweep with global BitNet ternary on KAN, mnist_d64+orth_id."""
import json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import TernaryKAN
from models.rotation import _make_factory
from training.trainer import train_iterative
import exp9_rotation as e9

OUT = os.path.join(os.path.dirname(__file__), "results", "exp15_basis_order.json")
ORDERS = [4, 6, 8, 12]
SEEDS = list(range(5))


def run_one(order, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, ytr, Xv, yv, _tt, hidden, out_dim = e9.mnist_d64_task(seed)
    in_d = Xtr.shape[1]
    rk = {"mode": "orthogonal", "init": "identity", "ternary": False,
          "warmup_epochs": 0, "out_dim": in_d}
    bk = {"dims": [in_d] + hidden + [out_dim], "basis_order": order, "quant": "bitnet"}
    factory = _make_factory(TernaryKAN, bk, in_d, rk, None)
    Xt_tr = torch.tensor(Xtr); Xt_v = torch.tensor(Xv)
    yt_tr = torch.tensor(ytr, dtype=torch.long); yt_v = torch.tensor(yv, dtype=torch.long)
    t0 = time.time()
    run = train_iterative(factory, {}, Xt_tr, yt_tr, Xt_v, yt_v,
                          "classification", n_rounds=4, epochs_per=50,
                          verbose=False, lambda1=1e-3)
    return float(run["val_curve"][-1]), float(run["sparsity_curve"][-1]), time.time() - t0


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    results = []
    total = len(ORDERS) * len(SEEDS)
    i = 0
    for order in ORDERS:
        for seed in SEEDS:
            i += 1
            fv, fs, el = run_one(order, seed)
            rec = {"basis_order": order, "seed": seed, "final_val": fv,
                   "final_sparsity": fs, "elapsed_s": el}
            results.append(rec)
            with open(OUT, "w") as f:
                json.dump(results, f, indent=2)
            print(f"[{i}/{total}] basis_order={order} seed={seed} val={fv:.4f} "
                  f"sp={fs:.3f} t={el:.1f}s", flush=True)
    print()
    agg = {}
    for r in results:
        agg.setdefault(r["basis_order"], []).append(r["final_val"])
    for order, vals in sorted(agg.items()):
        print(f"{order:4d} mean={np.mean(vals):.4f} std={np.std(vals):.4f}")


if __name__ == "__main__":
    main()
