"""Reusable axis-discovery (rotation) front-end. Wraps any backbone with a learned R."""
import json, os, sys, time, argparse
import numpy as np
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))


def _ternary_ste(w, thresh_mult=0.7):
    thresh = w.abs().mean() * thresh_mult
    w_t = torch.where(w.abs() > thresh, torch.sign(w), torch.zeros_like(w))
    return w + (w_t - w).detach()


class RotationFrontEnd(nn.Module):
    def __init__(self, in_dim, out_dim=None, mode="dense", init="identity",
                 ternary=False, warmup_epochs=0, pca_data=None, noise=0.01):
        super().__init__()
        assert mode in ("dense", "orthogonal", "lowrank")
        assert init in ("identity", "pca", "random_orth")
        self.in_dim = in_dim
        self.out_dim = out_dim if out_dim is not None else in_dim
        self.mode = mode
        self.ternary = ternary
        self.warmup_epochs = warmup_epochs
        self._epoch = 0
        if mode == "lowrank":
            assert self.out_dim < in_dim, "lowrank requires out_dim < in_dim"
        self.linear = nn.Linear(in_dim, self.out_dim, bias=False)
        with torch.no_grad():
            if init == "identity":
                if self.out_dim == in_dim:
                    self.linear.weight.copy_(torch.eye(in_dim) + noise * torch.randn(in_dim, in_dim))
                else:
                    W = torch.zeros(self.out_dim, in_dim)
                    k = min(self.out_dim, in_dim)
                    W[:k, :k] = torch.eye(k)
                    self.linear.weight.copy_(W + noise * torch.randn(self.out_dim, in_dim))
            elif init == "pca":
                assert pca_data is not None, "init=pca needs pca_data"
                Xc = pca_data - pca_data.mean(axis=0, keepdims=True)
                _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
                R = Vt[: self.out_dim].astype(np.float32)
                self.linear.weight.copy_(torch.from_numpy(R))
            elif init == "random_orth":
                A = torch.randn(max(in_dim, self.out_dim), max(in_dim, self.out_dim))
                Q, _ = torch.linalg.qr(A)
                self.linear.weight.copy_(Q[: self.out_dim, :in_dim].contiguous())
        if mode == "orthogonal":
            from torch.nn.utils.parametrizations import orthogonal
            orthogonal(self.linear, "weight")

    def step_epoch(self):
        self._epoch += 1

    def in_warmup(self):
        return self._epoch < self.warmup_epochs

    def forward(self, x):
        if self.ternary and not self.in_warmup():
            w = _ternary_ste(self.linear.weight)
            return x @ w.T
        return self.linear(x)

    def extract_R(self):
        return self.linear.weight.detach().cpu().numpy().copy()


class WrappedBackbone(nn.Module):
    def __init__(self, backbone, rotation):
        super().__init__()
        self.backbone = backbone
        self.rotation = rotation

    def forward(self, x):
        return self.backbone(self.rotation(x))

    @property
    def layers(self):
        return self.backbone.layers

    def sparsity(self):
        return self.backbone.sparsity()

    def l0_penalty(self):
        return self.backbone.l0_penalty()

    def complexity_penalty(self):
        if hasattr(self.backbone, "complexity_penalty"):
            return self.backbone.complexity_penalty()
        return torch.tensor(0.0)

    def total_gates(self):
        return self.backbone.total_gates()

    def active_gates(self):
        return self.backbone.active_gates()

    def get_masks(self):
        return self.backbone.get_masks()

    def apply_masks(self, masks):
        return self.backbone.apply_masks(masks)


def build_wrapped(backbone_cls, backbone_kwargs, in_dim, rot_kwargs, pca_data=None):
    rk = dict(rot_kwargs)
    bk = dict(backbone_kwargs)
    if rk.get("mode") == "lowrank":
        bk["dims"] = [rk["out_dim"]] + list(bk["dims"][1:])
    rot = RotationFrontEnd(in_dim=in_dim, pca_data=pca_data, **rk)
    backbone = backbone_cls(**bk)
    return WrappedBackbone(backbone, rot)


def _make_factory(backbone_cls, backbone_kwargs, in_dim, rot_kwargs, pca_data):
    def factory(**ignored):
        return build_wrapped(backbone_cls, backbone_kwargs, in_dim, rot_kwargs, pca_data)
    return factory


def _fit_cli(args):
    from models import TernaryMLP, TernaryKAN
    from training.trainer import train_iterative
    import exp9_rotation as e9
    loader = {"tabular": e9.tabular_task, "mnist_d64": e9.mnist_d64_task}[args.task]
    backbone_cls = TernaryMLP if args.backbone == "mlp" else TernaryKAN
    results = []
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    for seed in range(args.seeds):
        torch.manual_seed(seed); np.random.seed(seed)
        Xtr, ytr, Xv, yv, _tt, hidden, out_dim = loader(seed)
        in_d = Xtr.shape[1]
        rot_out = args.out_dim if args.out_dim else in_d
        rot_kwargs = {"out_dim": rot_out, "mode": args.mode, "init": args.init,
                      "ternary": args.ternary, "warmup_epochs": args.warmup}
        bk = {"dims": [in_d] + hidden + [out_dim]}
        if args.backbone == "kan":
            bk["basis_order"] = args.basis_order
        factory = _make_factory(backbone_cls, bk, in_d, rot_kwargs,
                                Xtr if args.init == "pca" else None)
        Xt_tr = torch.tensor(Xtr); Xt_v = torch.tensor(Xv)
        yt_tr = torch.tensor(ytr, dtype=torch.long); yt_v = torch.tensor(yv, dtype=torch.long)
        t0 = time.time()
        run = train_iterative(factory, {}, Xt_tr, yt_tr, Xt_v, yt_v,
                              "classification", n_rounds=args.rounds,
                              epochs_per=args.epochs_per, verbose=False,
                              lambda1=args.lambda1)
        elapsed = time.time() - t0
        rec = {"seed": seed, "task": args.task, "backbone": args.backbone,
               "mode": args.mode, "init": args.init, "ternary": args.ternary,
               "warmup": args.warmup, "in_dim": in_d, "rot_out": rot_out,
               "final_val": float(run["val_curve"][-1]),
               "final_sparsity": float(run["sparsity_curve"][-1]),
               "elapsed_s": elapsed}
        results.append(rec)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"seed={seed} val={rec['final_val']:.4f} sp={rec['final_sparsity']:.3f} t={elapsed:.1f}s", flush=True)
    vals = [r["final_val"] for r in results]
    print(f"mean={np.mean(vals):.4f} std={np.std(vals):.4f}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--task", choices=["tabular", "mnist_d64"], required=True)
    f.add_argument("--backbone", choices=["mlp", "kan"], required=True)
    f.add_argument("--mode", choices=["dense", "orthogonal", "lowrank"], default="dense")
    f.add_argument("--init", choices=["identity", "pca", "random_orth"], default="identity")
    f.add_argument("--ternary", action="store_true")
    f.add_argument("--warmup", type=int, default=0)
    f.add_argument("--out-dim", type=int, default=None)
    f.add_argument("--seeds", type=int, default=5)
    f.add_argument("--rounds", type=int, default=4)
    f.add_argument("--epochs-per", type=int, default=50)
    f.add_argument("--lambda1", type=float, default=1e-3)
    f.add_argument("--basis-order", type=int, default=6)
    f.add_argument("--output", default="results/rotation_fit.json")
    args = p.parse_args()
    if args.cmd == "fit":
        _fit_cli(args)


if __name__ == "__main__":
    main()
