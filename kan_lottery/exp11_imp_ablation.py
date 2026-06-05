"""EXP-11 (H4): IMP-aggressiveness ablation on MNIST-d64 with orth_id rotation.
Tests whether KAN suffers more than MLP under 4-round magnitude pruning."""
import json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import TernaryMLP, TernaryKAN
from models.rotation import _make_factory
from training.trainer import train_iterative
import exp9_rotation as e9

OUT = os.path.join(os.path.dirname(__file__), "results", "exp11_imp_ablation.json")

SCHEDULES = [
    ("noimp", {"n_rounds": 1, "epochs_per": 200}),
    ("imp",   {"n_rounds": 4, "epochs_per": 50}),
]
BACKBONES = ["mlp", "kan"]
SEEDS = list(range(5))


def run_one(backbone_name, sched, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, ytr, Xv, yv, _tt, hidden, out_dim = e9.mnist_d64_task(seed)
    in_d = Xtr.shape[1]
    rk = {"mode": "orthogonal", "init": "identity", "ternary": False,
          "warmup_epochs": 0, "out_dim": in_d}
    backbone_cls = TernaryMLP if backbone_name == "mlp" else TernaryKAN
    bk = {"dims": [in_d] + hidden + [out_dim]}
    if backbone_name == "kan":
        bk["basis_order"] = 6
    factory = _make_factory(backbone_cls, bk, in_d, rk, None)
    Xt_tr = torch.tensor(Xtr); Xt_v = torch.tensor(Xv)
    yt_tr = torch.tensor(ytr, dtype=torch.long); yt_v = torch.tensor(yv, dtype=torch.long)
    t0 = time.time()
    run = train_iterative(factory, {}, Xt_tr, yt_tr, Xt_v, yt_v,
                          "classification", verbose=False, lambda1=1e-3, **sched)
    return float(run["val_curve"][-1]), float(run["sparsity_curve"][-1]), time.time() - t0


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    results = []
    total = len(SCHEDULES) * len(BACKBONES) * len(SEEDS)
    i = 0
    for sname, sched in SCHEDULES:
        for backbone in BACKBONES:
            for seed in SEEDS:
                i += 1
                fv, fs, el = run_one(backbone, sched, seed)
                rec = {"schedule": sname, "backbone": backbone, "seed": seed,
                       "final_val": fv, "final_sparsity": fs, "elapsed_s": el}
                results.append(rec)
                with open(OUT, "w") as f:
                    json.dump(results, f, indent=2)
                print(f"[{i}/{total}] {sname} {backbone} seed={seed} "
                      f"val={fv:.4f} sp={fs:.3f} t={el:.1f}s", flush=True)
    print()
    agg = {}
    for r in results:
        agg.setdefault((r["schedule"], r["backbone"]), []).append(r["final_val"])
    for (s, a), vals in sorted(agg.items()):
        print(f"{s:6s} {a:4s} mean={np.mean(vals):.4f} std={np.std(vals):.4f}")


if __name__ == "__main__":
    main()
