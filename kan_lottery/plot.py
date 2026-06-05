"""
Generate figures from results/all_results.json.
Produces:
  - accuracy_vs_sparsity.png
  - mask_overlap_bar.png
  - pruning_resilience.png
  - kan_complexity.png
"""
import os, json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
FIG_DIR     = RESULTS_DIR


def load_results():
    path = os.path.join(RESULTS_DIR, "all_results.json")
    with open(path) as f:
        return json.load(f)


def _groupby(results, keys):
    groups = defaultdict(list)
    for r in results:
        k = tuple(r.get(k_) for k_ in keys)
        groups[k].append(r)
    return groups


# ── 1. Accuracy vs Sparsity (IMP runs) ───────────────────────────────────────

def plot_accuracy_vs_sparsity(results):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].set_title("IMP: Accuracy vs Sparsity")
    axes[1].set_title("Online: Final val vs Sparsity")

    for mode, ax in zip(["iterative", "online"], axes):
        for model_type, color, ls in [("mlp","#E07B39","--"), ("kan","#4477AA","-")]:
            subset = [r for r in results
                      if r["pruning"] == mode and r["model"] == model_type]
            if not subset:
                continue
            if mode == "iterative":
                # collect (sp, val) from IMP rounds
                all_sp  = []
                all_val = []
                for r in subset:
                    sps  = r.get("sp_curve_imp", [r["sparsity"]])
                    vals = r.get("val_curve_imp", [r["final_val"]])
                    all_sp.append(sps)
                    all_val.append(vals)
                min_len = min(len(s) for s in all_sp)
                sp_arr  = np.array([s[:min_len] for s in all_sp]).mean(0)
                val_arr = np.array([v[:min_len] for v in all_val]).mean(0)
                ax.plot(sp_arr, val_arr, color=color, ls=ls,
                        marker="o", label=model_type.upper())
            else:
                sps  = [r["sparsity"]  for r in subset]
                vals = [r["final_val"] for r in subset]
                ax.scatter(sps, vals, color=color, label=model_type.upper(), alpha=0.7)

        ax.set_xlabel("Sparsity (fraction of gates off)")
        ax.set_ylabel("Val metric")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "accuracy_vs_sparsity.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"Saved {out}")


# ── 2. Mask Overlap (Jaccard) bar chart ───────────────────────────────────────

def plot_jaccard(results):
    # Group by (model, task, pruning)
    groups = _groupby(results, ["model", "task", "pruning"])
    labels, mlp_vals, kan_vals = [], [], []

    tasks_seen = set()
    for (model, task, pruning), rs in groups.items():
        key = (task, pruning)
        if key not in tasks_seen:
            tasks_seen.add(key)

    for task_key in sorted(tasks_seen):
        task, pruning = task_key
        mlp_rs = groups.get(("mlp", task, pruning), [])
        kan_rs = groups.get(("kan", task, pruning), [])
        mj = np.nanmean([r.get("jaccard", float("nan")) for r in mlp_rs])
        kj = np.nanmean([r.get("jaccard", float("nan")) for r in kan_rs])
        labels.append(f"{task}\n({pruning})")
        mlp_vals.append(mj)
        kan_vals.append(kj)

    if not labels:
        return

    x   = np.arange(len(labels))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(max(8, 2*len(labels)), 5))
    ax.bar(x - w/2, mlp_vals, w, label="MLP",  color="#E07B39", alpha=0.85)
    ax.bar(x + w/2, kan_vals, w, label="KAN",  color="#4477AA", alpha=0.85)
    ax.axhline(0.5, ls="--", color="gray", alpha=0.5, label="Jaccard=0.5")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Mean Pairwise Jaccard")
    ax.set_title("Mask Overlap (Ticket Stability) across Seeds")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "mask_overlap_bar.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"Saved {out}")


# ── 3. Pruning Resilience ────────────────────────────────────────────────────

def plot_resilience(results):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title("Pruning Resilience: abrupt gate removal")

    for model_type, color, ls in [("mlp","#E07B39","--"), ("kan","#4477AA","-")]:
        subset = [r for r in results if r["model"] == model_type and "resilience" in r]
        if not subset:
            continue
        # Align on common fracs
        all_fracs = subset[0]["resilience"]["frac_pruned"]
        all_vals  = np.array([r["resilience"]["val_metric"] for r in subset])
        mean_v    = all_vals.mean(0)
        std_v     = all_vals.std(0)
        ax.plot(all_fracs, mean_v, color=color, ls=ls, label=model_type.upper(), marker="o")
        ax.fill_between(all_fracs, mean_v-std_v, mean_v+std_v, color=color, alpha=0.15)

    ax.set_xlabel("Fraction of gates pruned")
    ax.set_ylabel("Val metric (mean ± std over seeds & tasks)")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "pruning_resilience.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"Saved {out}")


# ── 4. KAN Functional Complexity ────────────────────────────────────────────

def plot_kan_complexity(results):
    kan_results = [r for r in results if r["model"] == "kan" and "complexity" in r]
    if not kan_results:
        print("No KAN complexity data found.")
        return

    ab   = [r["complexity"]["mean_active_basis"]   for r in kan_results]
    ent  = [r["complexity"]["mean_coeff_entropy"]  for r in kan_results]
    sps  = [r["sparsity"]                          for r in kan_results]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    axes[0].scatter(sps, ab, alpha=0.7, color="#4477AA")
    axes[0].set_xlabel("Sparsity"); axes[0].set_ylabel("Mean active basis / edge")
    axes[0].set_title("KAN: Active Basis vs Sparsity")
    axes[0].grid(alpha=0.3)

    axes[1].scatter(sps, ent, alpha=0.7, color="#44AA77")
    axes[1].set_xlabel("Sparsity"); axes[1].set_ylabel("Mean coefficient entropy")
    axes[1].set_title("KAN: Coefficient Entropy vs Sparsity")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "kan_complexity.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"Saved {out}")


# ── Summary table (printed) ──────────────────────────────────────────────────

def print_summary(results):
    groups = _groupby(results, ["model", "task", "pruning"])
    print("\n" + "="*80)
    print(f"{'Model':<6} {'Task':<10} {'Pruning':<11} "
          f"{'Val (mean)':<12} {'Sparsity':<10} {'Jaccard':<10}")
    print("="*80)
    for (model, task, pruning), rs in sorted(groups.items()):
        vals = np.nanmean([r.get("final_val", float("nan")) for r in rs])
        sps  = np.nanmean([r.get("sparsity",  float("nan")) for r in rs])
        jac  = np.nanmean([r.get("jaccard",   float("nan")) for r in rs])
        print(f"{model:<6} {task:<10} {pruning:<11} "
              f"{vals:<12.4f} {sps:<10.4f} {jac:<10.4f}")
    print("="*80)


if __name__ == "__main__":
    results = load_results()
    plot_accuracy_vs_sparsity(results)
    plot_jaccard(results)
    plot_resilience(results)
    plot_kan_complexity(results)
    print_summary(results)
