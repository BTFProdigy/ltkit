"""EXP-10: sweep the rotation tool across modes/inits on tabular + mnist_d64."""
import json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import TernaryMLP, TernaryKAN
from models.rotation import _make_factory
from training.trainer import train_iterative
import exp9_rotation as e9

OUT = os.path.join(os.path.dirname(__file__), "results", "exp10_rotation_sweep.json")

VARIANTS = [
    ("orth_id",   {"mode": "orthogonal", "init": "identity",    "ternary": False, "warmup_epochs": 0}),
    ("orth_pca",  {"mode": "orthogonal", "init": "pca",         "ternary": False, "warmup_epochs": 0}),
    ("lowrank20", {"mode": "lowrank",    "init": "identity",    "ternary": False, "warmup_epochs": 0, "out_dim_override": 20}),
    ("dense_tern",{"mode": "dense",      "init": "identity",    "ternary": True,  "warmup_epochs": 10}),
]


def run_one(loader, backbone_name, vname, rk_template, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, ytr, Xv, yv, _tt, hidden, out_dim = loader(seed)
    in_d = Xtr.shape[1]
    rk = {k: v for k, v in rk_template.items() if k != "out_dim_override"}
    rot_out = rk_template.get("out_dim_override", in_d)
    rk["out_dim"] = rot_out
    backbone_cls = TernaryMLP if backbone_name == "mlp" else TernaryKAN
    bk = {"dims": [in_d] + hidden + [out_dim]}
    if backbone_name == "kan":
        bk["basis_order"] = 6
    pca_data = Xtr if rk["init"] == "pca" else None
    factory = _make_factory(backbone_cls, bk, in_d, rk, pca_data)
    Xt_tr = torch.tensor(Xtr); Xt_v = torch.tensor(Xv)
    yt_tr = torch.tensor(ytr, dtype=torch.long); yt_v = torch.tensor(yv, dtype=torch.long)
    t0 = time.time()
    run = train_iterative(factory, {}, Xt_tr, yt_tr, Xt_v, yt_v,
                          "classification", n_rounds=4, epochs_per=50,
                          verbose=False, lambda1=1e-3)
    return float(run["val_curve"][-1]), float(run["sparsity_curve"][-1]), time.time() - t0, in_d, rot_out


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tasks = [("tabular", e9.tabular_task), ("mnist_d64", e9.mnist_d64_task)]
    backbones = ["kan"]
    seeds = list(range(5))
    results = []
    total = len(tasks) * len(VARIANTS) * len(backbones) * len(seeds)
    i = 0
    for task_name, loader in tasks:
        for vname, rk in VARIANTS:
            for backbone in backbones:
                for seed in seeds:
                    i += 1
                    final_val, final_sp, elapsed, in_d, rot_out = run_one(loader, backbone, vname, rk, seed)
                    rec = {"task": task_name, "variant": vname, "backbone": backbone,
                           "seed": seed, "in_dim": in_d, "rot_out": rot_out,
                           "final_val": final_val, "final_sparsity": final_sp,
                           "elapsed_s": elapsed}
                    results.append(rec)
                    with open(OUT, "w") as f:
                        json.dump(results, f, indent=2)
                    print(f"[{i}/{total}] {task_name} {vname} {backbone} seed={seed} "
                          f"in={in_d} rot={rot_out} val={final_val:.4f} sp={final_sp:.3f} "
                          f"t={elapsed:.1f}s", flush=True)
    print()
    agg = {}
    for r in results:
        agg.setdefault((r["task"], r["variant"], r["backbone"]), []).append(r["final_val"])
    for (t, v, a), vals in sorted(agg.items()):
        print(f"{t:12s} {v:11s} {a:4s} mean={np.mean(vals):.4f} std={np.std(vals):.4f}")


if __name__ == "__main__":
    main()
