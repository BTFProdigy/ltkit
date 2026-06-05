"""
Training loop: online gating and IMP (iterative magnitude pruning with rewind).
"""
import copy, torch, torch.nn as nn, torch.optim as optim
from typing import Dict, List


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_loader(X, y, batch_size=32, shuffle=True):
    ds = torch.utils.data.TensorDataset(X, y)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

def _accuracy(logits, labels):
    return (logits.argmax(1) == labels).float().mean().item()

def _mse(pred, target):
    return ((pred - target)**2).mean().item()

def _eval_model(model, X_val, y_val, task_type, batch_size=512):
    model.eval()
    loader = _make_loader(X_val, y_val, batch_size, shuffle=False)
    total, n = 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            out = model(xb)
            if task_type == "classification":
                total += _accuracy(out, yb) * len(xb)
            else:
                total += _mse(out, yb.float()) * len(xb)
            n += len(xb)
    model.train()
    return total / n

def _make_optimizer(model, lr: float, gate_lr_mult: float = 8.0):
    """Separate parameter groups: higher LR for gate log_alpha."""
    gate_p   = [p for n,p in model.named_parameters() if "log_alpha" in n]
    weight_p = [p for n,p in model.named_parameters() if "log_alpha" not in n]
    groups = [{"params": weight_p, "lr": lr, "weight_decay": 0.0}]
    if gate_p:
        groups.append({"params": gate_p, "lr": lr * gate_lr_mult, "weight_decay": 0.0})
    return optim.Adam(groups)

def _compute_loss(model, out, y, task_type, lambda1=0.0, lambda2=0.0):
    if task_type == "classification":
        task_loss = nn.CrossEntropyLoss()(out, y)
    else:
        task_loss = nn.MSELoss()(out, y.float())
    l0    = model.l0_penalty()    * lambda1 if lambda1 > 0 else 0.0
    compl = (model.complexity_penalty() * lambda2
             if (hasattr(model, "complexity_penalty") and lambda2 > 0) else 0.0)
    return task_loss + l0 + compl


# ── Online training ───────────────────────────────────────────────────────────

def train_online(
    model,
    X_train, y_train, X_val, y_val,
    task_type: str,
    epochs: int          = 300,
    lr: float            = 5e-3,
    batch_size: int      = 32,
    lambda1_start: float = 2e-3,
    lambda1_end:   float = 1e-2,
    lambda2:       float = 1e-5,
    verbose: bool        = False,
) -> Dict:
    optimizer = _make_optimizer(model, lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    loader    = _make_loader(X_train, y_train, batch_size)

    history = {"train_loss": [], "val_metric": [], "sparsity": [], "epoch": []}

    for epoch in range(1, epochs + 1):
        t       = (epoch - 1) / max(epochs - 1, 1)
        lambda1 = lambda1_start + t * (lambda1_end - lambda1_start)

        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            out  = model(xb)
            loss = _compute_loss(model, out, yb, task_type, lambda1, lambda2)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        val_metric = _eval_model(model, X_val, y_val, task_type)
        sparsity   = model.sparsity()

        history["epoch"].append(epoch)
        history["train_loss"].append(epoch_loss / max(len(loader), 1))
        history["val_metric"].append(val_metric)
        history["sparsity"].append(sparsity)

        if verbose and epoch % 50 == 0:
            print(f"  ep {epoch:3d} | loss {epoch_loss/max(len(loader),1):.4f}"
                  f" | val {val_metric:.4f} | sp {sparsity:.3f}")

    return history


# ── IMP (iterative magnitude pruning with rewind) ─────────────────────────────

def _magnitude_prune_masks(model, keep_frac: float) -> list:
    masks = []
    for layer in model.layers:
        gate_vals = layer.gate.forward(deterministic=True)
        flat      = gate_vals.detach().flatten()
        k         = max(1, int(len(flat) * keep_frac))
        thresh    = flat.kthvalue(len(flat) - k + 1).values
        masks.append((gate_vals.detach() >= thresh).float())
    return masks

def train_iterative(
    model_cls, model_kwargs: dict,
    X_train, y_train, X_val, y_val,
    task_type: str,
    n_rounds:   int   = 4,
    epochs_per: int   = 100,
    prune_rate: float = 0.20,
    lr:         float = 5e-3,
    batch_size: int   = 32,
    lambda1:    float = 0.0,
    verbose:    bool  = False,
) -> Dict:
    model      = model_cls(**model_kwargs)
    init_state = copy.deepcopy(model.state_dict())
    keep_frac  = 1.0
    masks      = None
    all_history, sparsity_curve, val_curve = [], [], []

    for rnd in range(n_rounds):
        if verbose:
            print(f"\n=== IMP Round {rnd+1}/{n_rounds}  keep_frac={keep_frac:.2f} ===")
        model.load_state_dict(copy.deepcopy(init_state))
        if masks is not None:
            model.apply_masks(masks)

        hist = train_online(
            model, X_train, y_train, X_val, y_val, task_type,
            epochs=epochs_per, lr=lr, batch_size=batch_size,
            lambda1_start=lambda1, lambda1_end=lambda1, verbose=verbose,
        )
        all_history.append(hist)

        keep_frac *= (1.0 - prune_rate)
        masks      = _magnitude_prune_masks(model, keep_frac)
        model.apply_masks(masks)
        sparsity_curve.append(model.sparsity())
        val_curve.append(hist["val_metric"][-1])

    return {
        "rounds": all_history, "sparsity_curve": sparsity_curve,
        "val_curve": val_curve, "final_model": model, "final_masks": masks,
    }
