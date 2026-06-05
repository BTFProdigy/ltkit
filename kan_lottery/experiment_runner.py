"""
Experiment runner: sweeps all (model, task, pruning_mode) configs × seeds.
Saves results to results/ as JSON-able dicts.
"""
import os, sys, json, copy, time
import torch
import numpy as np

# Make sure project root is on path
sys.path.insert(0, os.path.dirname(__file__))

from models import TernaryMLP, TernaryKAN
from tasks  import TASKS, TASK_DIMS
from training.trainer import train_online, train_iterative
from metrics import (
    mean_pairwise_jaccard,
    pruning_resilience,
    summarise_seeds,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Experiment matrix (mirrors Table 3 in the paper) ─────────────────────────

HIDDEN_SMALL  = [30, 10]   # for logic tasks
HIDDEN_MEDIUM = [50, 20]   # for regression tasks
HIDDEN_MNIST  = [64, 32]   # for MNIST (reduced for runtime)

EXPERIMENTS = [
    # id, model, hidden, basis, pruning, tasks
    (1, "mlp", HIDDEN_SMALL,  None, "iterative", ["xor",    "parity",   "modular"]),
    (2, "kan", HIDDEN_SMALL,  6,    "iterative", ["xor",    "parity",   "modular"]),
    (3, "mlp", HIDDEN_MEDIUM, None, "online",    ["symbolic","pde"]),
    (4, "kan", HIDDEN_MEDIUM, 6,    "online",    ["symbolic","pde"]),
    (5, "mlp", HIDDEN_MNIST,  None, "iterative", ["mnist"]),
    (6, "kan", HIDDEN_MNIST,  6,    "iterative", ["mnist"]),
    (7, "mlp", HIDDEN_SMALL,  None, "online",    ["xor",    "parity"]),
    (8, "kan", HIDDEN_SMALL,  6,    "online",    ["xor",    "parity"]),
]

N_SEEDS   = 5
EPOCHS_ONLINE = 150
EPOCHS_IMP    = 50
IMP_ROUNDS    = 4
EPOCHS_ONLINE_QUICK = 80
EPOCHS_IMP_QUICK    = 30


def build_model(model_type, dims, basis_order=6):
    if model_type == "mlp":
        return TernaryMLP(dims)
    else:
        return TernaryKAN(dims, basis_order=basis_order)


def run_single(
    model_type, hidden, basis, pruning_mode, task_name, seed,
    verbose=False, epochs_online=EPOCHS_ONLINE, epochs_imp=EPOCHS_IMP,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    Xtr, ytr, Xv, yv, task_type = TASKS[task_name](seed=seed)
    in_dim, out_dim = TASK_DIMS[task_name]
    dims = [in_dim] + hidden + [out_dim]

    result = {
        "model":    model_type,
        "task":     task_name,
        "pruning":  pruning_mode,
        "seed":     seed,
        "dims":     dims,
    }

    if pruning_mode == "online":
        model = build_model(model_type, dims, basis or 6)
        hist  = train_online(
            model, Xtr, ytr, Xv, yv, task_type,
            epochs=epochs_online, verbose=verbose,
            lambda1_start=1e-4, lambda1_end=5e-3,
            lambda2=1e-5 if model_type == "kan" else 0.0,
        )
        masks    = model.get_masks()
        sparsity = model.sparsity()
        val_fin  = hist["val_metric"][-1]
        resil    = pruning_resilience(model, Xv, yv, task_type)

        result.update({
            "final_val":   val_fin,
            "sparsity":    sparsity,
            "masks":       [m.tolist() for m in masks],
            "history_val": hist["val_metric"],
            "history_sp":  hist["sparsity"],
            "resilience":  resil,
        })
        if model_type == "kan":
            result["complexity"] = model.functional_complexity()

    else:  # iterative IMP
        kwargs = {"dims": dims} if model_type == "mlp" else \
                 {"dims": dims, "basis_order": basis or 6}
        cls    = TernaryMLP if model_type == "mlp" else TernaryKAN

        imp_res = train_iterative(
            cls, kwargs, Xtr, ytr, Xv, yv, task_type,
            n_rounds=IMP_ROUNDS, epochs_per=epochs_imp,
            verbose=verbose,
            lambda1=1e-3,
        )
        model    = imp_res["final_model"]
        masks    = model.get_masks()
        resil    = pruning_resilience(model, Xv, yv, task_type)

        result.update({
            "final_val":      imp_res["val_curve"][-1],
            "sparsity":       imp_res["sparsity_curve"][-1],
            "masks":          [m.tolist() for m in masks],
            "val_curve_imp":  imp_res["val_curve"],
            "sp_curve_imp":   imp_res["sparsity_curve"],
            "resilience":     resil,
        })
        if model_type == "kan":
            result["complexity"] = model.functional_complexity()

    return result


def run_all(verbose=False, quick=False):
    """Run all experiment configurations × seeds."""
    all_results = []

    # In quick mode: 2 seeds, fewer epochs, subset of tasks
    seeds       = list(range(2 if quick else N_SEEDS))
    ep_online   = EPOCHS_ONLINE_QUICK if quick else EPOCHS_ONLINE
    ep_imp      = EPOCHS_IMP_QUICK    if quick else EPOCHS_IMP

    for exp_id, model_type, hidden, basis, pruning, tasks in EXPERIMENTS:
        for task_name in tasks:
            seed_results = []
            print(f"\n[EXP {exp_id}] model={model_type} task={task_name} "
                  f"pruning={pruning}")
            t0 = time.time()
            for seed in seeds:
                print(f"  seed={seed}", end="", flush=True)
                try:
                    res = run_single(
                        model_type, hidden, basis, pruning,
                        task_name, seed, verbose=verbose,
                        epochs_online=ep_online, epochs_imp=ep_imp,
                    )
                    seed_results.append(res)
                    print(f"  val={res['final_val']:.4f}  sp={res['sparsity']:.3f}")
                except Exception as e:
                    print(f"  ERROR: {e}")

            # Compute cross-seed Jaccard if enough seeds
            if len(seed_results) >= 2:
                masks_list = [
                    [torch.tensor(m) for m in r["masks"]]
                    for r in seed_results
                ]
                jaccard = mean_pairwise_jaccard(masks_list)
                for r in seed_results:
                    r["jaccard"] = jaccard
                print(f"  → mean Jaccard={jaccard:.4f}  "
                      f"elapsed={time.time()-t0:.1f}s")

            all_results.extend(seed_results)

    # Save
    out_path = os.path.join(RESULTS_DIR, "all_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved → {out_path}")
    return all_results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--quick",   action="store_true", help="Quick smoke-test (2 seeds)")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    run_all(verbose=args.verbose, quick=args.quick)
