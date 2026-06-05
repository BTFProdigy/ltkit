"""EXP-1: lambda2 (KAN smoothness regularizer) confound sweep.
Resolves #3 + A: original sweep gave KAN lambda2=1e-5 but MLP lambda2=0.
Forces both archs to use the SAME lambda2 ∈ {0, 1e-6, 1e-5, 1e-4}.
Tasks: symbolic (regression), pde (regression), parity-online (classification).
5 seeds × 2 archs × 4 lambda2 × 3 tasks = 120 runs.
"""
import json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from models import TernaryMLP, TernaryKAN
from training.trainer import train_online
from tasks import symbolic_task, pde_task, parity_task

OUT = os.path.join(os.path.dirname(__file__), "results", "exp1_lambda2.json")
LAMBDAS = [0.0, 1e-6, 1e-5, 1e-4]
SEEDS = list(range(5))


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    arch_map = {"mlp": TernaryMLP, "kan": TernaryKAN}
    tasks = [
        ("symbolic", lambda s: symbolic_task(seed=s), [1, 50, 20, 1], "regression"),
        ("pde", lambda s: pde_task(seed=s), [1, 50, 20, 1], "regression"),
        ("parity_online", lambda s: parity_task(seed=s)[:4] + ("classification",),
         [5, 30, 10, 2], "classification"),
    ]
    results = []
    total = len(tasks) * 2 * len(LAMBDAS) * len(SEEDS)
    i = 0
    for task_name, loader, dims, ttype in tasks:
        for arch in ["mlp", "kan"]:
            for lam2 in LAMBDAS:
                for seed in SEEDS:
                    i += 1
                    torch.manual_seed(seed); np.random.seed(seed)
                    out = loader(seed)
                    Xtr, ytr, Xv, yv = out[0], out[1], out[2], out[3]
                    if arch == "mlp":
                        model = TernaryMLP(dims)
                    else:
                        model = TernaryKAN(dims, basis_order=6)
                    t0 = time.time()
                    hist = train_online(
                        model, Xtr, ytr, Xv, yv, ttype,
                        epochs=150, verbose=False,
                        lambda1_start=1e-4, lambda1_end=5e-3,
                        lambda2=lam2,
                    )
                    elapsed = time.time() - t0
                    rec = {
                        "task": task_name, "arch": arch, "lambda2": lam2, "seed": seed,
                        "final_val": float(hist["val_metric"][-1]),
                        "final_sparsity": float(model.sparsity()),
                        "elapsed_s": elapsed,
                    }
                    results.append(rec)
                    with open(OUT, "w") as f:
                        json.dump(results, f, indent=2)
                    print(f"[{i}/{total}] {task_name} {arch} lam2={lam2:.0e} "
                          f"seed={seed} val={rec['final_val']:.4f} "
                          f"sp={rec['final_sparsity']:.3f} t={elapsed:.1f}s",
                          flush=True)
    print()
    by = {}
    for r in results:
        by.setdefault((r["task"], r["arch"], r["lambda2"]), []).append(r["final_val"])
    for (t, a, l), v in sorted(by.items()):
        print(f"{t:15s} {a:4s} lam2={l:.0e}  mean={np.mean(v):.4f} std={np.std(v):.4f}")


if __name__ == "__main__":
    main()
