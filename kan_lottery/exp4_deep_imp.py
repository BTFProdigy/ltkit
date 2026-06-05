"""EXP-4: IMP 8 rounds with per-round masks.
Resolves #2 (terminal sparsity floor), #4 (κ dynamics), D, E.
Tasks: parity, mnist_d64. Both archs, 5 seeds, 8 rounds (terminal sparsity ~0.83).
Tracks val + masks at every round → can compute κ per round across seeds.
"""
import json, os, sys, copy, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from models import TernaryMLP, TernaryKAN
from training.trainer import train_online, _magnitude_prune_masks
from tasks import parity_task, _split

OUT = os.path.join(os.path.dirname(__file__), "results", "exp4_deep_imp.json")
N_ROUNDS = 8
PRUNE_RATE = 0.20
EPOCHS_PER = 50


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
            torch.tensor(Xv), torch.tensor(yv, dtype=torch.long))


def imp_with_masks(model_cls, kwargs, Xtr, ytr, Xv, yv, task_type):
    model = model_cls(**kwargs)
    init_state = copy.deepcopy(model.state_dict())
    keep_frac = 1.0
    masks = None
    val_per_round, sp_per_round, masks_per_round = [], [], []
    for rnd in range(N_ROUNDS):
        model.load_state_dict(copy.deepcopy(init_state))
        if masks is not None:
            model.apply_masks(masks)
        hist = train_online(
            model, Xtr, ytr, Xv, yv, task_type,
            epochs=EPOCHS_PER, lr=5e-3, batch_size=32,
            lambda1_start=1e-3, lambda1_end=1e-3, verbose=False,
        )
        keep_frac *= (1.0 - PRUNE_RATE)
        masks = _magnitude_prune_masks(model, keep_frac)
        model.apply_masks(masks)
        val_per_round.append(float(hist["val_metric"][-1]))
        sp_per_round.append(float(model.sparsity()))
        masks_per_round.append([m.cpu().numpy().astype(np.uint8).tolist() for m in masks])
    return val_per_round, sp_per_round, masks_per_round


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    arch_map = {"mlp": TernaryMLP, "kan": TernaryKAN}
    tasks = [
        ("parity", lambda s: parity_task(seed=s)[:4], [5, 30, 10, 2]),
        ("mnist_d64", mnist_d64, [64, 64, 32, 10]),
    ]
    results = []
    total = len(tasks) * 2 * 5
    i = 0
    for task_name, loader, dims in tasks:
        for arch in ["mlp", "kan"]:
            for seed in range(5):
                i += 1
                torch.manual_seed(seed); np.random.seed(seed)
                Xtr, ytr, Xv, yv = loader(seed)
                kwargs = {"dims": dims}
                if arch == "kan":
                    kwargs["basis_order"] = 6
                t0 = time.time()
                val_pr, sp_pr, masks_pr = imp_with_masks(
                    arch_map[arch], kwargs, Xtr, ytr, Xv, yv, "classification",
                )
                elapsed = time.time() - t0
                rec = {
                    "task": task_name, "arch": arch, "seed": seed,
                    "dims": dims,
                    "val_per_round": val_pr,
                    "sp_per_round": sp_pr,
                    "masks_per_round": masks_pr,
                    "elapsed_s": elapsed,
                }
                results.append(rec)
                with open(OUT, "w") as f:
                    json.dump(results, f)
                print(f"[{i}/{total}] {task_name} {arch} seed={seed} "
                      f"val_final={val_pr[-1]:.4f} sp_final={sp_pr[-1]:.3f} "
                      f"t={elapsed:.1f}s", flush=True)
    print()
    by = {}
    for r in results:
        by.setdefault((r["task"], r["arch"]), []).append(r)
    for (t, a), rs in sorted(by.items()):
        finals = [r["val_per_round"][-1] for r in rs]
        print(f"{t:12s} {a:4s} final_val mean={np.mean(finals):.4f} "
              f"std={np.std(finals):.4f}  sp={rs[0]['sp_per_round'][-1]:.3f}")


if __name__ == "__main__":
    main()
