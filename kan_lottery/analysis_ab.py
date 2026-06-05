#!/usr/bin/env python3
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
IN_PATH = ROOT / "results" / "all_results.json"
OUT_PATH = ROOT / "results" / "analysis_ab.json"
BOOTSTRAP_RESAMPLES = 10000
RNG = np.random.default_rng(0)

def flatten_masks(masks):
    return np.concatenate([np.asarray(mask, dtype=np.uint8).ravel() for mask in masks])

def pair_metrics(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    jaccard = 1.0 if not union else float(inter / union)
    po = float(np.mean(a == b))
    pa = float(np.mean(a))
    pb = float(np.mean(b))
    pe = pa * pb + (1.0 - pa) * (1.0 - pb)
    if pe >= 1.0:
        return jaccard, 1.0 if po == 1.0 else 0.0
    return jaccard, float((po - pe) / (1.0 - pe))

def bootstrap_mean_ci(values, resamples=BOOTSTRAP_RESAMPLES):
    values = np.asarray(values, dtype=float)
    draws = values[RNG.integers(0, len(values), size=(resamples, len(values)))]
    means = draws.mean(axis=1)
    mean = float(values.mean())
    lo, hi = np.quantile(means, [0.025, 0.975])
    return mean, float(lo), float(hi)

def bootstrap_delta(a, b, resamples=BOOTSTRAP_RESAMPLES):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    da = a[RNG.integers(0, len(a), size=(resamples, len(a)))].mean(axis=1)
    db = b[RNG.integers(0, len(b), size=(resamples, len(b)))].mean(axis=1)
    diffs = da - db
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return float(a.mean() - b.mean()), float(lo), float(hi), not (lo <= 0.0 <= hi)

def format_table(rows, columns, formats):
    widths = {col: len(col) for col in columns}
    rendered_rows = []
    for row in rows:
        rendered = {col: formats[col](row[col]) for col in columns}
        rendered_rows.append(rendered)
        for col in columns:
            widths[col] = max(widths[col], len(rendered[col]))
    out = ["  ".join(col.ljust(widths[col]) for col in columns),
           "  ".join("-" * widths[col] for col in columns)]
    out.extend("  ".join(row[col].rjust(widths[col]) for col in columns) for row in rendered_rows)
    return "\n".join(out)

def main():
    with IN_PATH.open() as f:
        records = json.load(f)
    groups = defaultdict(list)
    for record in records:
        groups[(record["model"], record["task"], record["pruning"])].append(record)
    matched_sparsity, bootstrap_ci = [], []
    grouped_by_task_pruning = defaultdict(lambda: defaultdict(list))
    for (model, task, pruning), items in sorted(groups.items()):
        items = sorted(items, key=lambda r: r["seed"])
        flats = [flatten_masks(r["masks"]) for r in items]
        jaccards, kappas = [], []
        for i, j in combinations(range(len(items)), 2):
            jaccard, kappa = pair_metrics(flats[i], flats[j])
            jaccards.append(jaccard)
            kappas.append(kappa)
        recomputed_jaccard_mean = float(np.mean(jaccards)) if jaccards else 0.0
        matched_sparsity.append({
            "model": model, "task": task, "pruning": pruning, "n": len(items),
            "mean_sparsity": float(np.mean([r["sparsity"] for r in items])),
            "jaccard_mean": recomputed_jaccard_mean,
            "kappa_mean": float(np.mean(kappas)) if kappas else 0.0,
        })
        mean, lo, hi = bootstrap_mean_ci([r["final_val"] for r in items])
        bootstrap_ci.append({
            "model": model, "task": task, "pruning": pruning, "n": len(items),
            "mean_final_val": mean, "ci_lo": lo, "ci_hi": hi,
        })
        grouped_by_task_pruning[(task, pruning)][model] = [r["final_val"] for r in items]
    kan_minus_mlp_deltas = []
    for (task, pruning), model_vals in sorted(grouped_by_task_pruning.items()):
        if "kan" not in model_vals or "mlp" not in model_vals:
            continue
        mean, lo, hi, significant = bootstrap_delta(model_vals["kan"], model_vals["mlp"])
        kan_minus_mlp_deltas.append({
            "task": task, "pruning": pruning,
            "delta_mean": mean, "ci_lo": lo, "ci_hi": hi, "significant": significant,
        })
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        json.dump({
            "matched_sparsity": matched_sparsity,
            "bootstrap_ci": bootstrap_ci,
            "kan_minus_mlp_deltas": kan_minus_mlp_deltas,
        }, f, indent=2)
    print("TABLE A: matched-sparsity Jaccard")
    print(format_table(matched_sparsity,
        ["model", "task", "pruning", "n", "mean_sparsity", "jaccard_mean", "kappa_mean"],
        {"model": str, "task": str, "pruning": str, "n": lambda x: str(int(x)),
         "mean_sparsity": lambda x: f"{x:.4f}", "jaccard_mean": lambda x: f"{x:.4f}",
         "kappa_mean": lambda x: f"{x:.4f}"}))
    print()
    print("TABLE B1: bootstrap 95% CIs for final_val")
    print(format_table(bootstrap_ci,
        ["model", "task", "pruning", "n", "mean_final_val", "ci_lo", "ci_hi"],
        {"model": str, "task": str, "pruning": str, "n": lambda x: str(int(x)),
         "mean_final_val": lambda x: f"{x:.4f}", "ci_lo": lambda x: f"{x:.4f}",
         "ci_hi": lambda x: f"{x:.4f}"}))
    print()
    print("TABLE B2: KAN minus MLP deltas (positive = KAN better for accuracy; for regression MSE, positive = KAN WORSE)")
    print(format_table(kan_minus_mlp_deltas,
        ["task", "pruning", "delta_mean", "ci_lo", "ci_hi", "significant"],
        {"task": str, "pruning": str,
         "delta_mean": lambda x: f"{x:+.4f}", "ci_lo": lambda x: f"{x:+.4f}",
         "ci_hi": lambda x: f"{x:+.4f}", "significant": lambda x: str(bool(x))}))

if __name__ == "__main__":
    main()
