"""EXP-8: bilinear feature-lift front-end. Tests whether prepending pairwise
products closes the KAN gap on Feynman / tabular / MNIST."""
import json, os, sys, time, math
import numpy as np
import torch
from sklearn.datasets import make_classification

sys.path.insert(0, os.path.dirname(__file__))
from models import TernaryMLP, TernaryKAN
from training.trainer import train_online, train_iterative
from tasks import _split

OUT = os.path.join(os.path.dirname(__file__), "results", "exp8_bilinear.json")
BUDGET_CAP = 64
LIFT_SEED = 12345


def bilinear_lift(Xtr, Xv, budget_cap=64, lift_seed=12345):
    d = Xtr.shape[1]
    rows, cols = np.triu_indices(d, k=1)
    pairs_tr = Xtr[:, rows] * Xtr[:, cols]
    pairs_v = Xv[:, rows] * Xv[:, cols]
    pair_mean = pairs_tr.mean(0)
    pair_std = pairs_tr.std(0) + 1e-8
    pairs_tr_n = (pairs_tr - pair_mean) / pair_std
    pairs_v_n = (pairs_v - pair_mean) / pair_std
    npairs = pairs_tr.shape[1]
    if npairs <= budget_cap:
        proj_tr = pairs_tr_n
        proj_v = pairs_v_n
    else:
        rng_lift = np.random.default_rng(lift_seed)
        R = rng_lift.standard_normal((npairs, budget_cap)).astype(np.float32) / math.sqrt(npairs)
        proj_tr = pairs_tr_n @ R
        proj_v = pairs_v_n @ R
    Xtr_lift = np.concatenate([Xtr, proj_tr.astype(np.float32)], axis=1)
    Xv_lift = np.concatenate([Xv, proj_v.astype(np.float32)], axis=1)
    return Xtr_lift, Xv_lift, Xtr_lift.shape[1]


def feynman_task(seed):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.5, 1.5, size=(1000, 10)).astype(np.float32)
    y = (np.sin(X[:,0]*X[:,1]) + 0.3*X[:,2]**2
         + np.log(np.abs(X[:,3])+1.0) + 0.5*X[:,4]*X[:,5]).astype(np.float32)
    y = (y - y.mean()) / (y.std() + 1e-8); y = y[:, None]
    Xtr, ytr, Xv, yv = _split(X, y, seed=seed)
    return Xtr, ytr, Xv, yv, "regression", [50,20], 1


def tabular_task(seed):
    X, y = make_classification(n_samples=5000, n_features=50, n_informative=20,
                               n_redundant=10, n_classes=2, class_sep=1.0,
                               random_state=seed)
    X = X.astype(np.float32)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    y = y.astype(np.int64)
    Xtr, ytr, Xv, yv = _split(X, y, seed=seed)
    return Xtr, ytr, Xv, yv, "classification", [64,32], 2


def mnist_d64_task(seed):
    from torchvision import datasets, transforms
    ds = datasets.MNIST(root="/tmp/mnist", train=True, download=True,
                        transform=transforms.ToTensor())
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), size=1000, replace=False)
    X = ds.data[idx].float().reshape(-1,784).numpy() / 255.0
    proj_rng = np.random.default_rng(12345)
    P = proj_rng.standard_normal((784, 64)).astype(np.float32) / math.sqrt(784.0)
    X = (X @ P).astype(np.float32)
    y = ds.targets[idx].numpy().astype(np.int64)
    Xtr, ytr, Xv, yv = _split(X, y, seed=seed)
    return Xtr, ytr, Xv, yv, "classification", [64,32], 10


def run_one(arch, variant, task_name, seed, Xtr,ytr,Xv,yv, task_type, hidden, out_dim):
    if variant == "bilinear":
        Xtr, Xv, _ = bilinear_lift(Xtr, Xv)
    in_d = Xtr.shape[1]
    dims = [in_d] + hidden + [out_dim]
    y_dtype = torch.long if task_type=="classification" else torch.float32
    if task_type == "regression":
        yt_tr = torch.tensor(ytr)
        yt_v = torch.tensor(yv)
    else:
        yt_tr = torch.tensor(ytr, dtype=torch.long)
        yt_v = torch.tensor(yv, dtype=torch.long)
    Xt_tr = torch.tensor(Xtr)
    Xt_v = torch.tensor(Xv)
    t0 = time.time()
    if task_name == "feynman":
        lam2 = 1e-5 if arch=="kan" else 0.0
        if arch == "mlp":
            model = TernaryMLP(dims)
        else:
            model = TernaryKAN(dims, basis_order=6)
        hist = train_online(model, Xt_tr, yt_tr, Xt_v, yt_v, "regression",
                            epochs=150, verbose=False,
                            lambda1_start=1e-4, lambda1_end=5e-3, lambda2=lam2)
        final_val = float(hist["val_metric"][-1])
        final_sp  = float(model.sparsity())
    else:  # tabular or mnist -> iterative IMP
        arch_map = {"mlp": TernaryMLP, "kan": TernaryKAN}
        kwargs = {"dims": dims}
        if arch == "kan":
            kwargs["basis_order"] = 6
        run = train_iterative(arch_map[arch], kwargs, Xt_tr, yt_tr, Xt_v, yt_v,
                              "classification", n_rounds=4, epochs_per=50,
                              verbose=False, lambda1=1e-3)
        final_val = float(run["val_curve"][-1])
        final_sp  = float(run["sparsity_curve"][-1])
    elapsed = time.time() - t0
    return final_val, final_sp, elapsed, in_d


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tasks = [("feynman", feynman_task), ("tabular", tabular_task), ("mnist_d64", mnist_d64_task)]
    variants = ["plain", "bilinear"]
    archs = ["mlp", "kan"]
    seeds = list(range(5))
    results = []
    total = len(tasks)*len(variants)*len(archs)*len(seeds)
    i = 0
    for task_name, loader in tasks:
        for variant in variants:
            for arch in archs:
                for seed in seeds:
                    i += 1
                    torch.manual_seed(seed); np.random.seed(seed)
                    Xtr,ytr,Xv,yv,task_type,hidden,out_dim = loader(seed)
                    final_val, final_sp, elapsed, in_d = run_one(
                        arch, variant, task_name, seed,
                        Xtr,ytr,Xv,yv, task_type, hidden, out_dim,
                    )
                    rec = {"task":task_name, "variant":variant, "arch":arch,
                           "seed":seed, "in_dim":in_d,
                           "final_val":final_val, "final_sparsity":final_sp,
                           "elapsed_s":elapsed}
                    results.append(rec)
                    with open(OUT,"w") as f: json.dump(results, f, indent=2)
                    print(f"[{i}/{total}] {task_name} {variant} {arch} seed={seed} "
                          f"in={in_d} val={final_val:.4f} sp={final_sp:.3f} "
                          f"t={elapsed:.1f}s", flush=True)
    print()
    agg = {}
    for r in results:
        agg.setdefault((r["task"], r["variant"], r["arch"]), []).append(r["final_val"])
    for (t,v,a), vals in sorted(agg.items()):
        print(f"{t:12s} {v:9s} {a:4s} mean={np.mean(vals):.4f} std={np.std(vals):.4f}")


if __name__ == "__main__":
    main()
