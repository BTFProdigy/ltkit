"""
Evaluation metrics for lottery-ticket / sparsity experiments.
"""
import torch
import numpy as np
from itertools import combinations
from typing import List, Dict, Optional


# ── Mask overlap (Jaccard) ────────────────────────────────────────────────────

def jaccard_pair(mask_a: List[torch.Tensor], mask_b: List[torch.Tensor]) -> float:
    """
    Compute Jaccard index between two mask lists (one tensor per layer).
    """
    intersect = sum((a.bool() & b.bool()).sum().item()
                    for a, b in zip(mask_a, mask_b))
    union     = sum((a.bool() | b.bool()).sum().item()
                    for a, b in zip(mask_a, mask_b))
    return intersect / max(union, 1)


def mean_pairwise_jaccard(masks_list: List[List[torch.Tensor]]) -> float:
    """
    Average pairwise Jaccard across a set of masks (one per seed).
    """
    if len(masks_list) < 2:
        return float("nan")
    scores = [jaccard_pair(a, b) for a, b in combinations(masks_list, 2)]
    return float(np.mean(scores))


# ── Accuracy vs sparsity curve ────────────────────────────────────────────────

def accuracy_vs_sparsity(
    sparsities: List[float],
    val_metrics: List[float],
) -> Dict:
    """Pair up sparsity levels with corresponding validation metrics."""
    return {
        "sparsities":   sparsities,
        "val_metrics":  val_metrics,
    }


# ── Pruning resilience: abrupt pruning ────────────────────────────────────────

def pruning_resilience(
    model,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    task_type: str,
    prune_fracs: List[float] = None,
) -> Dict:
    """
    Abruptly prune increasing fractions of smallest-gate edges and
    measure validation metric drop.

    Returns dict with lists of (frac_pruned, val_metric).
    """
    import copy
    from training.trainer import _eval_model

    if prune_fracs is None:
        prune_fracs = [0.0, 0.3, 0.5, 0.7, 0.9, 0.95]

    results = {"frac_pruned": [], "val_metric": []}

    for frac in prune_fracs:
        m = copy.deepcopy(model)
        m.eval()
        # Force gates: keep top (1-frac) by magnitude
        for layer in m.layers:
            gate_vals = layer.gate.forward(deterministic=True).detach()
            flat      = gate_vals.flatten()
            k         = max(1, int(len(flat) * (1 - frac)))
            thresh    = flat.kthvalue(len(flat) - k + 1).values
            mask      = (gate_vals >= thresh).float()
            # Freeze gate by setting log_alpha far
            with torch.no_grad():
                layer.gate.log_alpha.data = torch.where(
                    mask.bool(),
                    torch.full_like(layer.gate.log_alpha, 6.0),
                    torch.full_like(layer.gate.log_alpha, -6.0),
                )
        metric = _eval_model(m, X_val, y_val, task_type)
        results["frac_pruned"].append(frac)
        results["val_metric"].append(metric)

    return results


# ── KAN functional complexity ─────────────────────────────────────────────────

def kan_complexity_over_time(history_complexity: List[Dict]) -> Dict:
    """
    Aggregate complexity dicts across epochs into arrays for plotting.
    """
    keys = list(history_complexity[0].keys())
    return {k: [d[k] for d in history_complexity] for k in keys}


# ── Summary statistics ────────────────────────────────────────────────────────

def summarise_seeds(results_per_seed: List[Dict], key: str) -> Dict:
    """
    Given a list of result dicts (one per seed), summarise a numeric key.
    """
    vals = [r[key] for r in results_per_seed if key in r]
    vals = np.array(vals, dtype=float)
    return {
        "mean": float(np.nanmean(vals)),
        "std":  float(np.nanstd(vals)),
        "min":  float(np.nanmin(vals)),
        "max":  float(np.nanmax(vals)),
        "n":    len(vals),
    }
