"""EXP-16: basis-coefficient lottery ticket (edge_only vs basis_only vs both) on KAN+BitNet, mnist_d64+orth_id."""
import json, os, sys, time, copy
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import TernaryKAN
from models.rotation import _make_factory
from training.trainer import train_online, train_iterative, _magnitude_prune_masks
import exp9_rotation as e9

OUT = os.path.join(os.path.dirname(__file__), "results", "exp16_basis_imp.json")
REGIMES = ["edge_only", "basis_only", "both"]
SEEDS = list(range(5))
N_ROUNDS = 4
EPOCHS_PER = 50


def _coeff_masks(model, keep_frac):
    mags = torch.cat([l.coeffs.detach().abs().flatten() for l in model.layers])
    k = max(1, int(mags.numel() * keep_frac))
    thresh = mags.kthvalue(mags.numel() - k + 1).values
    return [(l.coeffs.detach().abs() >= thresh).float() for l in model.layers]


def _run_manual(seed, regime):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, ytr, Xv, yv, _tt, hidden, out_dim = e9.mnist_d64_task(seed)
    in_d = Xtr.shape[1]
    rk = {"mode": "orthogonal", "init": "identity", "ternary": False, "warmup_epochs": 0, "out_dim": in_d}
    bk = {"dims": [in_d] + hidden + [out_dim], "basis_order": 6, "quant": "bitnet"}
    factory = _make_factory(TernaryKAN, bk, in_d, rk, None)
    Xt_tr = torch.tensor(Xtr); Xt_v = torch.tensor(Xv)
    yt_tr = torch.tensor(ytr, dtype=torch.long); yt_v = torch.tensor(yv, dtype=torch.long)
    model = factory()
    init_state = copy.deepcopy(model.state_dict())
    keep_frac = 1.0
    gate_masks = None
    coeff_masks = None
    t0 = time.time()
    final_val = 0.0
    for _ in range(N_ROUNDS):
        model.load_state_dict(copy.deepcopy(init_state))
        if gate_masks is not None:
            model.apply_masks(gate_masks)
        if coeff_masks is not None:
            for layer, mask in zip(model.layers, coeff_masks):
                layer.apply_coeff_mask(mask)
        hist = train_online(model, Xt_tr, yt_tr, Xt_v, yt_v, "classification",
                            epochs=EPOCHS_PER, lr=5e-3, batch_size=32,
                            lambda1_start=1e-3, lambda1_end=1e-3, verbose=False)
        final_val = float(hist["val_metric"][-1])
        keep_frac *= 0.8
        if regime in ("edge_only", "both"):
            gate_masks = _magnitude_prune_masks(model, keep_frac)
            model.apply_masks(gate_masks)
        if regime in ("basis_only", "both"):
            coeff_masks = _coeff_masks(model, keep_frac)
            for layer, mask in zip(model.layers, coeff_masks):
                layer.apply_coeff_mask(mask)
    elapsed = time.time() - t0
    if regime == "edge_only":
        sparsity = float(model.sparsity())
    else:
        sparsity = float(model.backbone.coeff_sparsity())
    return final_val, sparsity, elapsed


def _run_edge_only(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, ytr, Xv, yv, _tt, hidden, out_dim = e9.mnist_d64_task(seed)
    in_d = Xtr.shape[1]
    rk = {"mode": "orthogonal", "init": "identity", "ternary": False, "warmup_epochs": 0, "out_dim": in_d}
    bk = {"dims": [in_d] + hidden + [out_dim], "basis_order": 6, "quant": "bitnet"}
    factory = _make_factory(TernaryKAN, bk, in_d, rk, None)
    Xt_tr = torch.tensor(Xtr); Xt_v = torch.tensor(Xv)
    yt_tr = torch.tensor(ytr, dtype=torch.long); yt_v = torch.tensor(yv, dtype=torch.long)
    t0 = time.time()
    run = train_iterative(factory, {}, Xt_tr, yt_tr, Xt_v, yt_v, "classification",
                          n_rounds=N_ROUNDS, epochs_per=EPOCHS_PER, verbose=False,
                          lambda1=1e-3)
    return float(run["val_curve"][-1]), float(run["sparsity_curve"][-1]), time.time() - t0


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    results = []
    total = len(REGIMES) * len(SEEDS)
    i = 0
    for regime in REGIMES:
        for seed in SEEDS:
            i += 1
            if regime == "edge_only":
                fv, sp, el = _run_edge_only(seed)
            else:
                fv, sp, el = _run_manual(seed, regime)
            rec = {"regime": regime, "seed": seed, "final_val": fv, "coeff_sparsity": sp, "elapsed_s": el}
            results.append(rec)
            with open(OUT, "w") as f:
                json.dump(results, f, indent=2)
            print(f"[{i}/{total}] regime={regime} seed={seed} final_val={fv:.4f} sparsity={sp:.3f} elapsed={el:.1f}s", flush=True)
    agg = {}
    for r in results:
        agg.setdefault(r["regime"], []).append(r["final_val"])
    for regime, vals in sorted(agg.items()):
        print(f"{regime:10s} mean={np.mean(vals):.4f} std={np.std(vals):.4f}")


if __name__ == "__main__":
    main()
